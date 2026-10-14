"""Shared test fixtures: fake LLM, vector index, reranker and web search.

The real graph, routing, knowledge base and API code run unchanged on top of these fakes,
so every test is offline: no API keys, no network, no model downloads.
"""

import re
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.bootstrap import Components  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.rag.agent import RAGAgent  # noqa: E402
from app.rag.retrieval import BM25Index, HybridRetriever, WebSearcher, tokenize  # noqa: E402
from app.rag.workflow import Services  # noqa: E402
from app.services.audit import AuditLog  # noqa: E402
from app.services.knowledge_base import ChunkStore, KnowledgeBase  # noqa: E402

SAMPLE_MD = """---
title: Build a custom RAG agent with LangGraph
---
> ## Documentation Index
> Fetch the complete documentation index at: /llms.txt

# Build a custom RAG agent with LangGraph

Build a retrieval agent with LangGraph that decides when to search a vector store versus answering directly.

## Grade documents

Add a conditional edge grade_documents to determine whether the retrieved documents are relevant to the question.
We use a model with structured output (GradeDocuments) to score relevance with a binary yes or no.

```python
# this comment must not become a header
def grade_documents(state):
    return "generate_answer"
```

## Rewrite the question

If the documents are not relevant, the rewrite_question node rewrites the original user question to improve retrieval
and then the agent tries retrieval again with the improved question.

## Generate an answer

Build the generate_answer node: if the documents pass the grader, generate the final answer from the retrieved context.
"""
SAMPLE_SOURCE = "https://example.com/agentic-rag"


def overlap(q: str, t: str) -> float:
    qs, ts = set(tokenize(q)), set(tokenize(t))
    return len(qs & ts) / (len(qs) or 1)


class FakeVectorIndex:
    """Plays Pinecone: stores chunks by id and searches them by word overlap."""

    def __init__(self):
        self.docs = {}
        self.upsert_calls = 0
        self.deleted = []

    # vector-store side (used by HybridRetriever)
    def similarity_search(self, q, k=4):
        return sorted(self.docs.values(), key=lambda d: overlap(q, d.page_content), reverse=True)[:k]

    # index side (used by KnowledgeBase)
    def upsert(self, chunks):
        self.upsert_calls += 1
        new = [c for c in chunks if c.id not in self.docs]
        for c in new:
            self.docs[c.id] = c
        return len(new)

    def delete(self, ids):
        for i in ids:
            self.docs.pop(i, None)
        self.deleted.extend(ids)
        return len(ids)


class FakeReranker:
    def predict(self, pairs):
        return [overlap(q, t) * 4 - 1 for q, t in pairs]  # logit-like scores → sigmoid path


class FakeTavily:
    def __init__(self):
        self.empty = False
        self.calls = 0

    def invoke(self, args):
        self.calls += 1
        if self.empty:
            return {"results": []}
        return {
            "answer": "IGNORED: unsourced AI summary",
            "results": [
                {
                    "url": "https://tavily.com",
                    "title": "Tavily",
                    "content": "Tavily is a search API built for AI agents and RAG search.",
                }
            ],
        }


class FakeStructured:
    def __init__(self, schema, llm):
        self.schema, self.llm = schema, llm

    def invoke(self, prompt):
        name = self.schema.__name__
        self.llm.calls.append(name)
        if self.llm.broken:
            raise RuntimeError("LLM down")
        if name == "RouteDecision":
            msg = prompt.split("Latest user message:")[1].split("Decide the route")[0].strip()
            low = msg.lower()
            if low.startswith(("hello", "thanks")):
                return self.schema(route="direct", standalone_question=msg, contains_private_data=False)
            if "tavily" in low:
                return self.schema(route="web", standalone_question=msg, contains_private_data=False)
            if "after that" in low:
                return self.schema(
                    route="retrieve",
                    contains_private_data=False,
                    standalone_question="What happens after grading documents if they are not relevant?",
                )
            # deliberately weak privacy detection, so the regex guard is what gets tested
            return self.schema(route="retrieve", standalone_question=msg, contains_private_data=False)
        if name == "DocumentGrades":
            if self.llm.grader_fail:
                raise RuntimeError("tool call failed")
            question = prompt.split("Question:")[1].split("\n")[0]
            blocks = re.findall(r'<source id="(\d+)"[^>]*>\n(.*?)\n</source>', prompt, re.S)
            relevant = [int(i) for i, body in blocks if overlap(question, body) > 0.3]
            return self.schema(relevant_ids=relevant + [99], sufficient=bool(relevant) and self.llm.sufficient)
        if name == "SearchQueries":
            return self.schema(queries=["grade documents relevance rewrite question"] * 2)
        if name == "AnswerCheck":
            grounded = self.llm.grounded.pop(0) if self.llm.grounded else True
            answers = self.llm.answers.pop(0) if self.llm.answers else True
            return self.schema(
                grounded=grounded, answers_question=answers, unsupported_claims=[] if grounded else ["made-up claim"]
            )
        raise ValueError(name)


class FakeLLM:
    """Plays every LLM role with simple rules. Knobs let tests force specific branches."""

    def __init__(self):
        self.calls = []
        self.sufficient = True
        self.grounded = []  # queue of check results, e.g. [False, True]
        self.answers = []
        self.grader_fail = False
        self.broken = False
        self.chat_error = None

    def with_structured_output(self, schema):
        return FakeStructured(schema, self)

    def invoke(self, messages):
        self.calls.append("chat")
        if self.chat_error:
            raise self.chat_error
        return AIMessage(content="Answer sentence one [1]. Bogus cite [7]. Both [1, 2].")


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "uploads",
        sample_kb_dir=tmp_path / "sample_kb",
        admin_api_key="test-admin-key-123456",
        checkpointer="memory",
        rate_limit_per_minute=0,
        openai_api_key="",
        pinecone_api_key="",
        tavily_api_key="",
    )


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def fake_tavily():
    return FakeTavily()


@pytest.fixture
def vector_index():
    return FakeVectorIndex()


@pytest.fixture
def kb(settings, vector_index):
    """A knowledge base already holding SAMPLE_MD."""
    store = ChunkStore(settings.chunk_db_path)
    retriever = HybridRetriever(vector_index, BM25Index([]), FakeReranker(), settings)
    kb = KnowledgeBase(settings, store, vector_index, retriever)
    kb.ingest(SAMPLE_MD.split("---\n", 2)[2], source=SAMPLE_SOURCE, title="Build a custom RAG agent with LangGraph")
    return kb


@pytest.fixture
def services(settings, kb, fake_llm, fake_tavily):
    return Services(
        settings=settings, llm_fast=fake_llm, llm_strong=fake_llm, retriever=kb.retriever, web=WebSearcher(fake_tavily)
    )


@pytest.fixture
def agent(services):
    from langgraph.checkpoint.memory import InMemorySaver

    return RAGAgent(services, checkpointer=InMemorySaver())


@pytest.fixture
def components(settings, agent, kb):
    return Components(settings=settings, agent=agent, kb=kb, audit=AuditLog(settings.audit_db_path))
