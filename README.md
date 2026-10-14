---
title: Agentic RAG Knowledge Copilot
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
short_description: Agentic RAG with LangGraph, hybrid search and citations
---

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
├── tests/                      # 62 offline tests (fakes, no API keys)
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
pytest                             # 62 offline tests
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

## Deploy to AWS EC2 (Docker Compose)

The simplest AWS path: one virtual machine running the same `docker compose` setup as your laptop.

1. **Launch an instance** (EC2 → Launch instance): Ubuntu Server 24.04, type `c7i-flex.large`
   (2 vCPU, 4 GiB; free-plan eligible), a new key pair, storage **30 GB**, and a security group
   allowing **SSH (22)** and **HTTP (80)**.
2. **Connect** (instance → Connect → EC2 Instance Connect) and install Docker:
   ```bash
   curl -fsSL https://get.docker.com | sudo sh
   sudo usermod -aG docker ubuntu && newgrp docker
   git clone https://github.com/<you>/agentic-rag-copilot.git && cd agentic-rag-copilot
   cp .env.example .env && nano .env      # keys, ADMIN_API_KEY, ACCESS_CODE, DAILY_QUESTION_LIMIT,
                                          # plus BIND_ADDR=0.0.0.0 and HOST_PORT=80
   docker compose up --build -d
   docker compose exec copilot python cli.py ingest
   ```
3. Open `http://<instance public IP>`.

Stop the instance when you are not demoing (compute billing stops; the disk and your data stay),
and start it again before an interview. The public IP changes on each start unless you attach an
Elastic IP. Ship a new version with `git pull && docker compose up --build -d`.

## Deploy to Azure Container Apps

Container Apps runs the same Docker image, scales to zero when idle (no charge while scaled to
zero) and includes a monthly free grant. The image is built locally and pushed to Azure Container
Registry, which also works on credit-based subscriptions such as Azure for Students.

```powershell
# One-time setup
winget install -e --id Microsoft.AzureCLI          # then open a new terminal
az login
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az provider register --namespace Microsoft.ContainerRegistry

$RG  = "agentic-rag-rg"
$LOC = "eastus"
$ACR = "agenticrag$(Get-Random -Maximum 99999)"     # registry names are global: letters and digits only
$APP = "agentic-rag-copilot"
$IMG = "$ACR.azurecr.io/agentic-rag-copilot:v1"

az group create --name $RG --location $LOC
az acr create --resource-group $RG --name $ACR --sku Basic --admin-enabled true
az acr login --name $ACR
docker build -t $IMG .
docker push $IMG
az containerapp env create --name agentic-rag-env --resource-group $RG --location $LOC

# Secrets are typed at a prompt, so they never appear in the command history
$OPENAI   = Read-Host "OpenAI API key"
$PINECONE = Read-Host "Pinecone API key"
$TAVILY   = Read-Host "Tavily API key"
$ADMIN    = python -c "import secrets; print(secrets.token_urlsafe(32))"
$CODE     = Read-Host "Demo access code to give interviewers"

az containerapp create --name $APP --resource-group $RG --environment agentic-rag-env `
  --image $IMG --registry-server "$ACR.azurecr.io" `
  --target-port 8000 --ingress external `
  --cpu 1.5 --memory 3.0Gi --min-replicas 0 --max-replicas 1 `
  --secrets "openai-key=$OPENAI" "pinecone-key=$PINECONE" "tavily-key=$TAVILY" "admin-key=$ADMIN" "access-code=$CODE" `
  --env-vars "OPENAI_API_KEY=secretref:openai-key" "PINECONE_API_KEY=secretref:pinecone-key" `
             "TAVILY_API_KEY=secretref:tavily-key" "ADMIN_API_KEY=secretref:admin-key" `
             "ACCESS_CODE=secretref:access-code" "AUTO_INGEST=true" `
             "DAILY_QUESTION_LIMIT=100" "RATE_LIMIT_PER_MINUTE=10"

az containerapp show --name $APP --resource-group $RG --query properties.configuration.ingress.fqdn -o tsv
```

Open `https://<that address>`. The first request after an idle period starts the container
(about a minute: models load and `AUTO_INGEST` rebuilds the local chunk store).

| Task | Command |
|---|---|
| Live logs | `az containerapp logs show --name $APP --resource-group $RG --follow` |
| Ship a new version | `docker build -t <acr>.azurecr.io/agentic-rag-copilot:v2 .`, `docker push …:v2`, then `az containerapp update --name $APP --resource-group $RG --image …:v2` |
| Delete everything | `az group delete --name $RG` |

For a stricter setup, pull from the registry with a managed identity
(`--registry-identity system`) instead of the registry admin account.

## Deploy to Hugging Face Spaces

The block at the top of this README is the Space configuration (`sdk: docker`, `app_port: 8000`).

1. Create a Space at huggingface.co/new-space: SDK **Docker**, template **Blank**, hardware **CPU basic**, visibility
   **Private** (switch to Public later in Settings). Creating Docker Spaces needs a Hugging Face PRO plan;
   CPU basic hardware itself has no hourly cost.
2. In the Space's **Settings → Variables and secrets**, add:
   - Secrets: `OPENAI_API_KEY`, `PINECONE_API_KEY`, `TAVILY_API_KEY`, `ADMIN_API_KEY`
   - Variables: `AUTO_INGEST=true`, `FRAME_ANCESTORS=https://huggingface.co`, `RATE_LIMIT_PER_MINUTE=10`
3. Push this repository to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push space main          # password = a Hugging Face access token with write permission
   ```
4. The Space builds the image (about 5–10 minutes), then serves the app.

Spaces wipe the disk on every restart. Pinecone keeps the vectors; `AUTO_INGEST=true` rebuilds the
local chunk store (needed for keyword search) at startup, skipping chunks already in Pinecone.
Documents uploaded through the UI are not re-created after a restart; put permanent ones in
`data/sample_kb/` or `SOURCE_URLS`.

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
| Cost control | Optional demo access code and a daily question ceiling for public deployments |
| Knowledge base | Re-ingesting a document uploads only new chunks and deletes stale ones from Pinecone; uploads are searchable immediately |
| Memory | Conversations persist in SQLite across restarts (`CHECKPOINTER=sqlite`) |
| Observability | Request id on every response and log line; JSON logs in production; audit log of every decision path; optional LangSmith tracing |
| Quality | 62 offline tests covering every route, limit, failure and endpoint; lint and tests in CI; `cli.py eval` for answer quality |

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
| `ACCESS_CODE` | empty (off) | When set, asking needs this code; the page shows an access-code field |
| `DAILY_QUESTION_LIMIT` | `0` (off) | Questions per day for the whole app: a ceiling on LLM spend |
| `MAX_UPLOAD_MB` | `10` | Upload size limit |
| `ALLOW_PRIVATE_URLS` | `false` | Let admins ingest intranet URLs |
| `SOURCE_URLS` | LangGraph agentic RAG page | Pages ingested by `cli.py ingest` (JSON list) |
| `AUTO_INGEST` | `false` | At startup, ingest `SOURCE_URLS` + `data/sample_kb` if the knowledge base is empty |
| `FRAME_ANCESTORS` | `'none'` | Sites allowed to show the UI in an iframe (e.g. `https://huggingface.co`) |
