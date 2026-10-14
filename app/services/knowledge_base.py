"""The knowledge base: one place that keeps Pinecone, the local chunk store and the keyword
index in sync.

* ChunkStore (SQLite) holds every chunk's text, which BM25 keyword search needs, and records
  which source each chunk came from.
* KnowledgeBase.ingest() makes re-ingesting a document safe:
    new chunks are uploaded, unchanged chunks are skipped, and chunks that no longer exist in
    the new version are deleted from Pinecone too (no stale answers from old versions).
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from app.core.config import Settings
from app.rag.retrieval import HybridRetriever
from app.services.ingestion import SUPPORTED_EXTENSIONS, IngestionError, chunk_markdown, load_file, load_url

logger = logging.getLogger(__name__)


def default_targets(settings: Settings) -> list[str]:
    """What a plain `cli.py ingest` loads: SOURCE_URLS plus every supported file in data/sample_kb."""
    files = sorted(str(p) for p in settings.sample_kb_dir.glob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    return list(settings.source_urls) + files


class ChunkStore:
    """SQLite table of chunks: id, source, text and metadata."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS chunks (
                       id TEXT PRIMARY KEY,
                       source TEXT NOT NULL,
                       content TEXT NOT NULL,
                       metadata TEXT NOT NULL,
                       created_at TEXT NOT NULL)"""
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")

    @contextmanager
    def _connect(self):
        """Open a connection, commit on success (roll back on error), always close."""
        con = sqlite3.connect(self.path, timeout=30)
        try:
            with con:
                yield con
        finally:
            con.close()

    def ids_for_source(self, source: str) -> set[str]:
        with self._connect() as con:
            return {row[0] for row in con.execute("SELECT id FROM chunks WHERE source = ?", (source,))}

    def replace_source(self, source: str, chunks: list[Document]) -> None:
        """Make the stored chunks for `source` exactly `chunks` (one transaction)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as con:
            con.execute("DELETE FROM chunks WHERE source = ?", (source,))
            con.executemany(
                "INSERT OR REPLACE INTO chunks (id, source, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                [(c.id, source, c.page_content, json.dumps(c.metadata), now) for c in chunks],
            )

    def delete_source(self, source: str) -> list[str]:
        with self._lock, self._connect() as con:
            ids = [row[0] for row in con.execute("SELECT id FROM chunks WHERE source = ?", (source,))]
            con.execute("DELETE FROM chunks WHERE source = ?", (source,))
        return ids

    def all_documents(self) -> list[Document]:
        with self._connect() as con:
            rows = con.execute("SELECT id, content, metadata FROM chunks ORDER BY source, id").fetchall()
        return [Document(id=r[0], page_content=r[1], metadata=json.loads(r[2])) for r in rows]

    def sources(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT source, json_extract(MIN(metadata), '$.title'), COUNT(*), MAX(created_at) "
                "FROM chunks GROUP BY source ORDER BY MAX(created_at) DESC"
            ).fetchall()
        return [{"source": r[0], "title": r[1], "chunks": r[2], "updated_at": r[3]} for r in rows]

    def version(self) -> tuple:
        """Changes whenever chunks are added or removed (by this process or another one)."""
        with self._connect() as con:
            return tuple(con.execute("SELECT COUNT(*), MAX(created_at), TOTAL(LENGTH(id)) FROM chunks").fetchone())

    def count(self) -> int:
        with self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


@dataclass
class IngestResult:
    source: str
    title: str
    chunks: int  # chunks in the new version
    uploaded: int  # newly embedded and uploaded
    unchanged: int  # already in Pinecone, skipped
    removed: int  # stale chunks deleted from Pinecone

    def as_dict(self) -> dict:
        return asdict(self)


class KnowledgeBase:
    def __init__(self, settings: Settings, store: ChunkStore, vector_index, retriever: Optional[HybridRetriever]):
        self.settings = settings
        self.store = store
        self.vector_index = vector_index  # app.rag.vectorstore.VectorIndex (or a fake in tests)
        self.retriever = retriever
        self._lock = threading.Lock()  # one ingestion at a time keeps the three stores consistent
        self._index_version: tuple | None = None

    def ingest(self, md: str, source: str, title: str) -> IngestResult:
        chunks = chunk_markdown(md, source=source, title=title, settings=self.settings)
        if not chunks:
            raise IngestionError(f"{source} produced no chunks (too little text).")
        with self._lock:
            old_ids = self.store.ids_for_source(source)
            new_ids = {c.id for c in chunks}
            uploaded = self.vector_index.upsert(chunks)  # 1) vectors first: a failure here changes nothing
            stale = sorted(old_ids - new_ids)
            if stale:
                self.vector_index.delete(stale)  # 2) remove chunks that no longer exist
            self.store.replace_source(source, chunks)  # 3) local text store
            self._refresh_keyword_index()  # 4) keyword index sees the new text immediately
        result = IngestResult(source, title, len(chunks), uploaded, len(chunks) - uploaded, len(stale))
        logger.info("Ingested %s", result.as_dict())
        return result

    def ingest_url(self, url: str) -> IngestResult:
        md, title = load_url(url, allow_private=self.settings.allow_private_urls)
        return self.ingest(md, source=url, title=title)

    def ingest_file(self, path: Path, display_name: Optional[str] = None) -> IngestResult:
        md, title = load_file(path, display_name)
        return self.ingest(md, source=display_name or path.name, title=title)

    def ingest_target(self, target: str) -> IngestResult:
        """Ingest a URL (http/https) or a local file path."""
        if target.startswith(("http://", "https://")):
            return self.ingest_url(target)
        return self.ingest_file(Path(target))

    def ingest_defaults(self) -> int:
        """Ingest default_targets(); returns how many failed (each failure is logged, not raised)."""
        failures = 0
        for target in default_targets(self.settings):
            try:
                self.ingest_target(target)
            except (IngestionError, FileNotFoundError) as exc:
                failures += 1
                logger.warning("Could not ingest %s: %s", target, exc)
        return failures

    def delete_source(self, source: str) -> int:
        with self._lock:
            ids = self.store.ids_for_source(source)
            if not ids:
                return 0
            self.vector_index.delete(sorted(ids))
            self.store.delete_source(source)
            self._refresh_keyword_index()
        logger.info("Deleted source %s (%d chunks)", source, len(ids))
        return len(ids)

    def refresh_if_stale(self) -> bool:
        """Rebuild the keyword index if another process (e.g. `cli.py ingest`) changed the store."""
        if self.retriever is None or self.store.version() == self._index_version:
            return False
        with self._lock:
            self._refresh_keyword_index()
        return True

    def _refresh_keyword_index(self) -> None:
        if self.retriever is not None:
            version = self.store.version()
            self.retriever.refresh_keyword_index(self.store.all_documents())
            self._index_version = version
