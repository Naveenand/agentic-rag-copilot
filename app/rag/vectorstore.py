"""Embeddings + Pinecone: connect to the index, upload new chunks, delete stale ones.

Unlike a "delete and recreate the index on mismatch" approach, this never deletes an index:
a wrong embedding size stops startup with a clear error instead of wiping your data.
"""

import logging
import time

from langchain_core.documents import Document

from app.core.config import Settings

logger = logging.getLogger(__name__)


def make_embeddings(settings: Settings):
    """bge-small-en-v1.5 runs locally (free). BGE expects an instruction in front of queries only."""
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embed_model,
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": settings.bge_query_prompt},
    )
    dim = len(embeddings.embed_query("dimension check"))
    if dim != settings.embed_dim:
        raise RuntimeError(f"{settings.embed_model} returns {dim} dims but EMBED_DIM is {settings.embed_dim}.")
    return embeddings


def connect_vectorstore(settings: Settings, embeddings):
    """Create the serverless index if it doesn't exist; return (vectorstore, raw_index)."""
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=settings.pinecone_api_key)
    name = settings.pinecone_index_name
    if pc.has_index(name):
        existing_dim = pc.describe_index(name).dimension
        if existing_dim != settings.embed_dim:
            raise RuntimeError(
                f"Pinecone index '{name}' has dimension {existing_dim}, but the embedding model needs "
                f"{settings.embed_dim}. Set PINECONE_INDEX_NAME to a new index name."
            )
    else:
        logger.info("Creating Pinecone index %s (dim=%d)", name, settings.embed_dim)
        pc.create_index(
            name=name,
            dimension=settings.embed_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)

    index = pc.Index(name)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, namespace=settings.pinecone_namespace)
    return vectorstore, index


class VectorIndex:
    """Thin wrapper over Pinecone with the three operations ingestion needs."""

    def __init__(self, vectorstore, index, namespace: str, batch_size: int = 100):
        self.vectorstore = vectorstore
        self.index = index
        self.namespace = namespace
        self.batch_size = batch_size

    def existing_ids(self, ids: list[str]) -> set[str]:
        found: set[str] = set()
        for start in range(0, len(ids), self.batch_size):
            response = self.index.fetch(ids=ids[start : start + self.batch_size], namespace=self.namespace)
            found.update(response.vectors.keys())
        return found

    def upsert(self, chunks: list[Document]) -> int:
        """Embed and upload only chunks whose IDs are not in the index yet."""
        if not chunks:
            return 0
        existing = self.existing_ids([c.id for c in chunks])
        new_chunks = [c for c in chunks if c.id not in existing]
        if new_chunks:
            self.vectorstore.add_documents(new_chunks, ids=[c.id for c in new_chunks])
        return len(new_chunks)

    def delete(self, ids: list[str]) -> int:
        for start in range(0, len(ids), self.batch_size):
            self.index.delete(ids=ids[start : start + self.batch_size], namespace=self.namespace)
        return len(ids)
