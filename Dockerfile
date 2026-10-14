# Production image: CPU-only PyTorch, models baked in, runs as a non-root user.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    PORT=8000

WORKDIR /app

# 1) Dependencies first, so code changes don't reinstall them (Docker layer cache).
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# 2) Download the embedding and reranker models at build time, so the container starts
#    fast and never needs Hugging Face at runtime.
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBED_MODEL}'); CrossEncoder('${RERANK_MODEL}')"

# 3) App code, owned by an unprivileged user (a breach can't modify the image).
#    UID 1000 is what Hugging Face Spaces runs containers as.
RUN useradd --create-home --uid 1000 appuser \
 && mkdir -p /app/data /app/uploads && chown -R appuser:appuser /app /opt/hf-cache
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser templates ./templates
COPY --chown=appuser:appuser static ./static
COPY --chown=appuser:appuser data/sample_kb ./data/sample_kb
COPY --chown=appuser:appuser cli.py ./
USER appuser

ENV APP_ENV=production \
    LOG_JSON=true \
    HF_HUB_OFFLINE=1

EXPOSE 8000
VOLUME ["/app/data", "/app/uploads"]

# Liveness check used by Docker and orchestrators.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/api/health', timeout=4)" || exit 1

# One worker: the BM25 index, rate limiter and SQLite stores live in this process.
# Scale out with more containers behind a load balancer (see README).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*' --timeout-graceful-shutdown 20"]
