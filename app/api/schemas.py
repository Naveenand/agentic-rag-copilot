"""Request and response models: the API's contract, validated by Pydantic and shown at /docs."""

from typing import Optional

from pydantic import BaseModel, Field

THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1, max_length=2000, examples=["What happens when retrieved documents are not relevant?"]
    )
    thread_id: Optional[str] = Field(
        default=None,
        pattern=THREAD_ID_PATTERN,
        description="Conversation id. Reuse it for follow-up questions; omit it to start a new conversation.",
    )


class Citation(BaseModel):
    id: int
    type: str  # "Private KB" or "Web"
    title: str = ""
    section: str = ""
    url: str = ""


class ChatResponse(BaseModel):
    answer: str
    source_used: str  # private_kb | web_search | private_kb + web_search | direct | insufficient_evidence | uncited
    route: str
    standalone_question: str
    citations: list[Citation]
    trace: list[str]
    grounded: bool
    web_searched: bool
    latency_s: float
    thread_id: str
    request_id: str


class UrlIngestRequest(BaseModel):
    url: str = Field(
        min_length=8, max_length=2000, examples=["https://docs.langchain.com/oss/python/langgraph/agentic-rag"]
    )


class IngestResponse(BaseModel):
    source: str
    title: str
    chunks: int
    uploaded: int
    unchanged: int
    removed: int


class SourceInfo(BaseModel):
    source: str
    title: Optional[str] = None
    chunks: int
    updated_at: Optional[str] = None


class ReadyResponse(BaseModel):
    status: str
    chunks: int = 0
    sources: int = 0
    web_enabled: bool = False
    reason: Optional[str] = None
