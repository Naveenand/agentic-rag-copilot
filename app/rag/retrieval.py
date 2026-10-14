"""Search: hybrid KB retrieval (dense + BM25 → Reciprocal Rank Fusion → cross-encoder rerank)
and Tavily web search.

"Retrieve wide, rerank narrow": two cheap searches collect ~20 candidates, then an accurate
but slower cross-encoder re-scores only those. Web results become Documents too, so KB and
web evidence share the same reranker, grader and citation code.
"""

import logging
import math
import re
import threading

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.core.config import Settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory keyword index over the same chunks that are in Pinecone."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in docs]) if docs else None

    def __len__(self) -> int:
        return len(self.docs)

    def search(self, query: str, k: int) -> list[Document]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.docs[i] for i in ranked if scores[i] > 0]


def doc_key(doc: Document) -> str:
    return doc.metadata.get("chunk_id") or doc.metadata.get("source", "") + "|" + doc.page_content[:200]


def rrf_fuse(result_lists: list[list[Document]], k: int = 60) -> list[Document]:
    """Merge ranked lists using positions only: score = sum(1 / (k + rank))."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = doc_key(doc)
            docs[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


def rerank(reranker, query: str, docs: list[Document], top_n: int, min_score: float) -> list[Document]:
    """Cross-encoder reranking. Adds `rerank_score` (0-1) to each returned document's metadata."""
    if not docs:
        return []
    raw = [float(s) for s in reranker.predict([(query, d.page_content) for d in docs])]
    if min(raw) < 0 or max(raw) > 1:  # logits → 0-1 so the threshold means the same for any model
        raw = [1 / (1 + math.exp(-s)) for s in raw]
    scored = sorted(zip(docs, raw, strict=True), key=lambda pair: pair[1], reverse=True)
    return [
        Document(id=d.id, page_content=d.page_content, metadata={**d.metadata, "rerank_score": round(s, 4)})
        for d, s in scored[:top_n]
        if s >= min_score
    ]


class HybridRetriever:
    """Vector store + BM25 + reranker behind two methods.

    The BM25 index can be swapped while the server is answering questions (after a document
    upload): a new index is built first, then assigned in one step, so readers always see a
    complete index.
    """

    def __init__(self, vectorstore, bm25_index: BM25Index, reranker, settings: Settings):
        self.vectorstore = vectorstore
        self.bm25_index = bm25_index
        self.reranker = reranker
        self.settings = settings
        self._refresh_lock = threading.Lock()

    def refresh_keyword_index(self, docs: list[Document]) -> None:
        with self._refresh_lock:
            new_index = BM25Index(docs)
            self.bm25_index = new_index
        logger.info("BM25 index refreshed: %d chunks", len(docs))

    def retrieve(self, queries: list[str]) -> list[Document]:
        """Dense + BM25 for every query → RRF fusion → rerank against queries[0]."""
        s = self.settings
        bm25 = self.bm25_index  # one consistent snapshot for this request
        result_lists = []
        for q in queries:
            result_lists.append(self.vectorstore.similarity_search(q, k=s.dense_k))
            result_lists.append(bm25.search(q, k=s.sparse_k))
        candidates = rrf_fuse(result_lists, k=s.rrf_k)[: s.max_candidates]
        return self.rerank(queries[0], candidates)

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        s = self.settings
        return rerank(self.reranker, query, docs, top_n=s.rerank_top_n, min_score=s.rerank_min_score)


# ============================================================================
# Web search (Tavily). include_answer=False: Tavily's AI summary has no source, so it is not evidence
# ============================================================================


def tavily_to_docs(result) -> list[Document]:
    if not isinstance(result, dict):
        logger.warning("Unexpected Tavily result type: %s", type(result))
        return []
    if result.get("error"):
        logger.warning("Tavily error: %s", result["error"])
    docs = []
    for item in result.get("results", []):
        content = (item.get("content") or "").strip()
        if content:
            docs.append(
                Document(
                    page_content=content,
                    metadata={"source": item.get("url", ""), "title": item.get("title", ""), "source_type": "web"},
                )
            )
    return docs


class WebSearcher:
    def __init__(self, tool):
        self.tool = tool

    @classmethod
    def from_settings(cls, settings: Settings) -> "WebSearcher":
        from langchain_tavily import TavilySearch

        tool = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=settings.web_max_results,
            topic="general",
            search_depth="advanced",
            include_answer=False,
            include_raw_content=False,
        )
        return cls(tool)

    def search(self, query: str) -> list[Document]:
        try:
            return tavily_to_docs(self.tool.invoke({"query": query}))
        except Exception:  # noqa: BLE001 - an outage means "no web results", not a crash
            logger.exception("Web search failed")
            return []
