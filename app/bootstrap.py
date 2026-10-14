"""Composition root: build every component once, in one place.

The API (at startup) and the CLI both call `build_components`. Tests skip it and build the
same objects from fakes, so nothing here needs to change for testing.
"""

import logging
from dataclasses import dataclass

from app.core.config import Settings
from app.rag.agent import RAGAgent, make_checkpointer, make_llms
from app.rag.retrieval import BM25Index, HybridRetriever, WebSearcher
from app.rag.vectorstore import VectorIndex, connect_vectorstore, make_embeddings
from app.rag.workflow import Services
from app.services.audit import AuditLog
from app.services.knowledge_base import ChunkStore, KnowledgeBase

logger = logging.getLogger(__name__)


@dataclass
class Components:
    settings: Settings
    agent: RAGAgent
    kb: KnowledgeBase
    audit: AuditLog


def build_knowledge_base(settings: Settings, retriever=None) -> tuple[KnowledgeBase, object]:
    """Embeddings + Pinecone + chunk store. Returns (kb, vectorstore)."""
    embeddings = make_embeddings(settings)
    vectorstore, index = connect_vectorstore(settings, embeddings)
    store = ChunkStore(settings.chunk_db_path)
    vector_index = VectorIndex(vectorstore, index, settings.pinecone_namespace)
    return KnowledgeBase(settings, store, vector_index, retriever), vectorstore


def build_components(settings: Settings) -> Components:
    missing = settings.missing_keys()
    if missing:
        raise RuntimeError(f"Missing API keys: {', '.join(missing)}. Set them in .env or the environment.")
    from sentence_transformers import CrossEncoder  # heavy import, only when really starting

    kb, vectorstore = build_knowledge_base(settings)
    retriever = HybridRetriever(
        vectorstore=vectorstore,
        bm25_index=BM25Index([]),  # filled from the chunk store by refresh_if_stale below
        reranker=CrossEncoder(settings.rerank_model),
        settings=settings,
    )
    kb.retriever = retriever
    kb.refresh_if_stale()  # records the store version the index was built from
    if kb.store.count() == 0:
        logger.warning("Knowledge base is empty. Run `python cli.py ingest` or upload a document.")

    llm_fast, llm_strong = make_llms(settings)
    web = WebSearcher.from_settings(settings) if settings.enable_web else None
    services = Services(settings=settings, llm_fast=llm_fast, llm_strong=llm_strong, retriever=retriever, web=web)
    agent = RAGAgent(services, checkpointer=make_checkpointer(settings))
    return Components(settings=settings, agent=agent, kb=kb, audit=AuditLog(settings.audit_db_path))
