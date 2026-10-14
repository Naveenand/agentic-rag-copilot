"""LangGraph state and the structured decisions the LLM must return.

* Pydantic schemas: each LLM decision is validated JSON (field descriptions are read by the model).
* AgentState: the shared state every graph node reads from and writes to.
"""

from typing import Annotated, Literal

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Structured LLM outputs (the field descriptions are read by the model)
# ---------------------------------------------------------------------------


class RouteDecision(BaseModel):
    route: Literal["retrieve", "web", "direct"] = Field(
        description=(
            "'direct' only for greetings, thanks, small talk or questions about this conversation. "
            "'retrieve' for any question the knowledge base could plausibly answer. "
            "'web' for questions clearly outside the knowledge base or needing current/live information."
        )
    )
    standalone_question: str = Field(
        description="The latest user message rewritten as a self-contained question, resolving "
        "pronouns like 'it' or 'that' from the chat history. Do not answer it."
    )
    contains_private_data: bool = Field(
        description="True if the question includes confidential or personal information (people's "
        "names, customer or employee data, account numbers, internal project names, secrets, "
        "credentials) that must NOT be sent to a public web search engine."
    )


class DocumentGrades(BaseModel):
    relevant_ids: list[int] = Field(
        description="IDs of the sources that contain information useful for answering the question."
    )
    sufficient: bool = Field(description="True only if the relevant sources together fully answer the question.")


class SearchQueries(BaseModel):
    queries: list[str] = Field(
        description="2 to 3 diverse search queries: rephrasings, more specific terms, or sub-questions."
    )


class AnswerCheck(BaseModel):
    grounded: bool = Field(description="True if every factual claim in the answer is supported by the sources.")
    answers_question: bool = Field(description="True if the answer actually addresses the question.")
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Claims in the answer that the sources do not support."
    )


# ---------------------------------------------------------------------------
# Graph state: the "notebook" every node reads from and writes to
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # chat history, kept across turns
    question: str  # raw user message for this turn
    standalone_question: str  # self-contained version used for search
    route: str
    web_allowed: bool
    queries: list[str]  # queries used for KB retrieval
    kb_attempts: int
    kb_docs: list[Document]  # reranked KB candidates
    web_docs: list[Document]  # reranked web candidates
    web_searched: bool  # True once the web was queried this turn
    evidence: list[Document]  # graded, relevant documents the answer may use
    crag_decision: str  # correct / ambiguous / incorrect
    answer: str
    citations: list[dict]
    generation_attempts: int
    check: dict
    source_used: str
    trace: list[str]  # human-readable log of this turn
