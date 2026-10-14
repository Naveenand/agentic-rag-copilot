"""Audit log: one row per question, recording the decision path the agent took.

Used to debug answers ("why did it go to the web?"), spot slow requests and review usage.
Questions that look like they contain private data are stored redacted.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.rag.workflow import looks_private

logger = logging.getLogger(__name__)
REDACTED = "[redacted: possible private data]"


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS query_audit (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       created_at TEXT NOT NULL,
                       request_id TEXT,
                       thread_id TEXT,
                       question TEXT NOT NULL,
                       route TEXT,
                       source_used TEXT,
                       web_searched INTEGER,
                       grounded INTEGER,
                       latency_s REAL,
                       trace_json TEXT,
                       error TEXT)"""
            )

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.path, timeout=30)
        try:
            with con:
                yield con
        finally:
            con.close()

    def record(
        self,
        *,
        question: str,
        request_id: str,
        thread_id: str,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
        """Never raises: a broken audit log must not break answering."""
        result = result or {}
        check = result.get("check") or {}
        try:
            with self._connect() as con:
                con.execute(
                    "INSERT INTO query_audit (created_at, request_id, thread_id, question, route, source_used, "
                    "web_searched, grounded, latency_s, trace_json, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        request_id,
                        thread_id,
                        REDACTED if looks_private(question) else question[:1000],
                        result.get("route"),
                        result.get("source_used"),
                        int(bool(result.get("web_searched"))),
                        int(bool(check.get("grounded", True))) if result else None,
                        result.get("latency_s"),
                        json.dumps(result.get("trace", [])),
                        error,
                    ),
                )
        except sqlite3.Error:
            logger.exception("Could not write audit record")

    def recent(self, limit: int = 50) -> list[dict]:
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT * FROM query_audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["trace"] = json.loads(item.pop("trace_json") or "[]")
            out.append(item)
        return out

    def summary(self) -> dict:
        with self._connect() as con:
            total, avg_latency = con.execute("SELECT COUNT(*), AVG(latency_s) FROM query_audit").fetchone()
            by_source = dict(
                con.execute("SELECT source_used, COUNT(*) FROM query_audit GROUP BY source_used").fetchall()
            )
            errors = con.execute("SELECT COUNT(*) FROM query_audit WHERE error IS NOT NULL").fetchone()[0]
        return {
            "questions": total,
            "avg_latency_s": round(avg_latency or 0, 2),
            "by_source": by_source,
            "errors": errors,
        }
