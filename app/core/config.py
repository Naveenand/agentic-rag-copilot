"""Application settings, loaded from environment variables and the project's .env file.

Every value has a safe default except the API keys. In production, set values as real
environment variables (Docker, cloud secrets manager); the .env file is for local development.
Variable names are the field names in upper case, e.g. OPENAI_API_KEY, FAST_MODEL, ENABLE_WEB.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]  # project root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")

    # --- App ---
    app_name: str = "Agentic RAG Knowledge Copilot"
    app_env: str = "development"  # "production" turns on stricter checks
    log_level: str = "INFO"
    log_json: bool = False  # one JSON object per log line (for log collectors)

    # --- API keys (secrets: never commit them) ---
    openai_api_key: str = ""
    pinecone_api_key: str = ""
    tavily_api_key: str = ""
    admin_api_key: str = "change-me"  # protects ingestion and audit endpoints

    # --- Knowledge base ---
    source_urls: list[str] = Field(
        default=["https://docs.langchain.com/oss/python/langgraph/agentic-rag"],
        description="Web pages ingested by `python cli.py ingest` (JSON list in env).",
    )
    kb_description: str = (
        "Company knowledge documents, including the LangGraph documentation on building a custom "
        "agentic RAG agent: splitting documents, retriever tools, grading retrieved documents, "
        "rewriting questions, generating answers and assembling the graph."
    )
    data_dir: Path = BASE_DIR / "data"
    sample_kb_dir: Path = BASE_DIR / "data" / "sample_kb"
    upload_dir: Path = BASE_DIR / "uploads"
    max_upload_mb: int = 10
    allow_private_urls: bool = False  # True lets admins ingest intranet URLs (disables SSRF guard)
    auto_ingest: bool = False  # at startup, ingest SOURCE_URLS + data/sample_kb if the store is empty

    # --- Embeddings / vector DB ---
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    bge_query_prompt: str = "Represent this sentence for searching relevant passages: "
    pinecone_index_name: str = "industry-agentic-rag-kb"
    pinecone_namespace: str = "langgraph-docs-v2-bge-small"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # --- Chunking ---
    chunk_size: int = 1200
    chunk_overlap: int = 150
    min_chunk_chars: int = 80

    # --- Retrieval ---
    dense_k: int = 10
    sparse_k: int = 10
    rrf_k: int = 60
    max_candidates: int = 20
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 5
    rerank_min_score: float = 0.02

    # --- LLMs (OpenAI) ---
    fast_model: str = "gpt-6-luna"
    strong_model: str = "gpt-6.1-sol"
    fast_reasoning_effort: str = "low"
    strong_reasoning_effort: str = "medium"
    llm_timeout_s: int = 60

    # --- Agent limits ---
    max_kb_attempts: int = 2
    max_generation_attempts: int = 2
    enable_web: bool = True
    web_max_results: int = 5
    max_history_messages: int = 6
    max_source_chars: int = 2000
    recursion_limit: int = 25

    # --- API protection ---
    rate_limit_per_minute: int = 30  # per client IP on /api/chat; 0 disables
    access_code: str = ""  # if set, asking needs this code (X-Access-Code header); share it with your demo users
    daily_question_limit: int = 0  # questions per UTC day for the whole app (caps LLM spend); 0 disables
    checkpointer: str = "sqlite"  # "sqlite" keeps conversations across restarts; "memory" does not
    frame_ancestors: str = "'none'"  # sites allowed to show the UI in an iframe, e.g. https://huggingface.co

    @field_validator("app_env")
    @classmethod
    def _env_name(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"development", "production", "test"}:
            raise ValueError("APP_ENV must be development, production or test")
        return v

    # Derived paths (always absolute, so the working directory never matters)
    @property
    def chunk_db_path(self) -> Path:
        return self.data_dir / "kb_chunks.db"

    @property
    def audit_db_path(self) -> Path:
        return self.data_dir / "audit.db"

    @property
    def checkpoint_db_path(self) -> Path:
        return self.data_dir / "conversations.db"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    def missing_keys(
        self, needed: tuple[str, ...] = ("openai_api_key", "pinecone_api_key", "tavily_api_key")
    ) -> list[str]:
        keys = [k for k in needed if not getattr(self, k).strip()]
        if "tavily_api_key" in keys and not self.enable_web:
            keys.remove("tavily_api_key")  # web search off → Tavily not needed
        return [k.upper() for k in keys]

    def check_production_safety(self) -> None:
        """Refuse to start in production with an unsafe admin key."""
        if self.is_production and (self.admin_api_key in {"", "change-me"} or len(self.admin_api_key) < 16):
            raise RuntimeError("ADMIN_API_KEY must be set to a random value of 16+ characters in production.")


@lru_cache
def get_settings() -> Settings:
    return Settings()
