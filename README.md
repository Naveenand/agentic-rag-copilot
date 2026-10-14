# Agentic RAG Knowledge Copilot

A deployable internal assistant that answers questions from your own documents first, checks
that the evidence is relevant, uses the web only when the documents fall short (never for
questions containing private data), and shows the citations and the decision path behind every
answer.

Built with **LangGraph, FastAPI, OpenAI, Pinecone, Tavily**, local **BGE embeddings** and a
**cross-encoder reranker**, plus a plain HTML/CSS/JavaScript web UI.

---

## What a user sees

- A chat page where every answer carries numbered citations to the exact document section.
- A live panel showing each agent step while it works: route, search, grade, rewrite, web, answer, fact-check.
- A badge saying where the answer came from: knowledge base, web, both, or "not enough evidence".
- An **Add a document** dialog (admin key required) for PDF, Word, Markdown, text, HTML or a web page.

## Architecture

```text
Browser (HTML/CSS/JS)
   │  POST /api/chat/stream  (Server-Sent Events: one event per agent step)
   ▼
FastAPI ── request ids, security headers, rate limit, admin auth, audit log
   │
   ▼
LangGraph agent
   route ─► hybrid retrieve ─► grade (CRAG) ─► rewrite & retry ─► web fallback ─► answer ─► fact-check
              │                                                     │
   Pinecone (dense) + BM25 (keywords)                         Tavily search
   → Reciprocal Rank Fusion → cross-encoder rerank
              │
   SQLite: chunk store · audit log · conversation memory
```

## Project structure

```text
agentic-rag-v2/
├── app/
│   ├── main.py                 # FastAPI app: middleware, error handling, UI route, startup
│   ├── bootstrap.py            # builds every component once (used by API and CLI)
│   ├── api/
│   │   ├── routes.py           # health, ready, chat, chat/stream, sources, ingest, audit
│   │   ├── schemas.py          # request/response models (shown at /docs)
│   │   └── deps.py             # admin auth, rate limiter, component access
│   ├── core/
│   │   ├── config.py           # all settings, from environment variables / .env
│   │   └── logging.py          # text or JSON logs with request ids
│   ├── rag/
│   │   ├── state.py            # LangGraph state + structured LLM decisions
│   │   ├── prompts.py          # prompt templates and formatting
│   │   ├── retrieval.py        # BM25, RRF fusion, reranker, hybrid retriever, web search
│   │   ├── vectorstore.py      # embeddings + Pinecone (upload new, delete stale)
│   │   ├── workflow.py         # graph nodes, routing rules, compiled graph
│   │   └── agent.py            # LLMs, conversation memory, RAGAgent.ask / stream
│   └── services/
│       ├── ingestion.py        # URL/PDF/DOCX/MD/TXT/HTML → clean, section-aware chunks
│       ├── knowledge_base.py   # keeps Pinecone, chunk store and BM25 in sync
│       ├── audit.py            # one row per question with the decision path
│       └── evaluation.py       # fixed question set + scoring
├── templates/index.html        # web UI
├── static/css/style.css
├── static/js/app.js
├── data/sample_kb/             # put company documents here for `cli.py ingest`
├── tests/                      # 56 offline tests (fakes, no API keys)
├── cli.py                      # ingest | chat | eval | sources
├── run.py                      # development server
├── Dockerfile, docker-compose.yml, .github/workflows/ci.yml
├── requirements.txt, requirements-dev.txt, pyproject.toml
└── .env.example
```

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt
copy .env.example .env             # macOS/Linux: cp .env.example .env  → then fill in the keys

python cli.py ingest               # 1. build the knowledge base (SOURCE_URLS + data/sample_kb)
python run.py                      # 2. open http://127.0.0.1:8000   (API docs: /docs)
```

Other commands:

```bash
python cli.py ingest my_policy.pdf https://example.com/page   # add specific files or pages
python cli.py chat                 # terminal chat
python cli.py eval                 # evaluation table
python cli.py sources              # what's in the knowledge base
pytest                             # 56 offline tests
ruff check .                       # lint
```

## Run it in production (Docker)

```bash
# .env: real keys, APP_ENV=production, and a strong admin key:
python -c "import secrets; print(secrets.token_urlsafe(32))"

docker compose up --build -d
docker compose exec copilot python cli.py ingest      # first time only
curl http://localhost:8000/api/ready                  # {"status":"ready", ...}
```

The image uses CPU-only PyTorch, bakes in both models (no Hugging Face downloads at runtime),
runs as a non-root user, has a health check, and keeps `data/` and `uploads/` on volumes.

**Deploying to a cloud host** (Render, Railway, Fly.io, AWS App Runner, Google Cloud Run, Azure
Container Apps): push the image, set the same environment variables as secrets, give it about
**2–3 GB of memory**, mount a persistent disk at `/app/data`, and point the health check at
`/api/health` and the readiness check at `/api/ready`.

## API

| Method | Path | Auth | What it does |
|---|---|---|---|
| GET | `/api/health` | none | Liveness: the process is up |
| GET | `/api/ready` | none | Readiness: models loaded, services connected (503 until then) |
| POST | `/api/chat` | none, rate-limited | `{"question", "thread_id"?}` → answer, citations, trace, source |
| POST | `/api/chat/stream` | none, rate-limited | Same, as Server-Sent Events: `step` events, then `final` |
| GET | `/api/sources` | none | Documents in the knowledge base |
| POST | `/api/ingest/file` | `X-Admin-Key` | Upload PDF/DOCX/MD/TXT/HTML (10 MB max); same name replaces |
| POST | `/api/ingest/url` | `X-Admin-Key` | `{"url"}` → fetch and index a public page |
| DELETE | `/api/sources?source=…` | `X-Admin-Key` | Remove a document everywhere |
| GET | `/api/audit?limit=50` | `X-Admin-Key` | Recent questions with their decision path, plus totals |

Example:

```bash
curl -s localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"question": "What happens when retrieved documents are not relevant?"}'
```

## Production features

| Area | What's in place |
|---|---|
| Configuration | Every setting from environment variables (`pydantic-settings`); no secrets in code |
| Startup | Models load in the background: `/api/health` answers at once, `/api/ready` waits; missing keys give a clear 503 instead of a crash |
| Security | Admin key compared in constant time; production refuses to start with a weak key; strict Content-Security-Policy and security headers; uploads checked for type, size and file name; URL ingestion blocks private network addresses (SSRF); error responses never include internal details |
| Privacy | Questions with IDs, emails, phone numbers or secrets never go to web search, and are stored redacted in the audit log |
| Reliability | Retries with backoff, model fallback, safe defaults for every LLM decision, bounded loops, per-IP rate limit |
| Knowledge base | Re-ingesting a document uploads only new chunks and deletes stale ones from Pinecone; uploads are searchable immediately |
| Memory | Conversations persist in SQLite across restarts (`CHECKPOINTER=sqlite`) |
| Observability | Request id on every response and log line; JSON logs in production; audit log of every decision path; optional LangSmith tracing |
| Quality | 56 offline tests covering every route, limit, failure and endpoint; lint and tests in CI; `cli.py eval` for answer quality |

## Scaling beyond one container

This build runs one worker per container, because the BM25 index, rate limiter and SQLite
stores live in the process. To run several replicas behind a load balancer:

1. Conversations: switch to LangGraph's Postgres checkpointer.
2. Chunk store, audit log: move from SQLite to Postgres.
3. Keyword search: use a Pinecone hybrid (sparse + dense) index or OpenSearch instead of in-memory BM25.
4. Rate limiting: move to Redis or the API gateway.

## Configuration

All settings live in `app/core/config.py`; set any of them in `.env` by its name in upper case.
The most useful:

| Variable | Default | Meaning |
|---|---|---|
| `APP_ENV` | `development` | `production` hides `/docs`, enables JSON logs in Docker, enforces a strong admin key |
| `FAST_MODEL` / `STRONG_MODEL` | `gpt-6-luna` / `gpt-6.1-sol` | Decision model / answer model |
| `STRONG_REASONING_EFFORT` | `medium` | `low` is faster |
| `ENABLE_WEB` | `true` | Turn web fallback off completely |
| `RATE_LIMIT_PER_MINUTE` | `30` | Questions per minute per IP (`0` = off) |
| `MAX_UPLOAD_MB` | `10` | Upload size limit |
| `ALLOW_PRIVATE_URLS` | `false` | Let admins ingest intranet URLs |
| `SOURCE_URLS` | LangGraph agentic RAG page | Pages ingested by `cli.py ingest` (JSON list) |
