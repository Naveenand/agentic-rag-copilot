"""HTTP endpoints.

Public:   GET /api/health, GET /api/ready, POST /api/chat, POST /api/chat/stream, GET /api/sources
Admin:    POST /api/ingest/file, POST /api/ingest/url, DELETE /api/sources, GET /api/audit
          (admin endpoints need the X-Admin-Key header)
"""

import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.deps import get_app_settings, get_components, limit_chat, require_admin
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    IngestResponse,
    ReadyResponse,
    SourceInfo,
    UrlIngestRequest,
)
from app.bootstrap import Components
from app.core.config import Settings
from app.core.logging import request_id_var
from app.services.ingestion import SUPPORTED_EXTENSIONS, IngestionError, safe_filename

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health", tags=["health"])
def health():
    """Liveness: the process is up. Used by Docker / load balancers to restart a dead container."""
    return {"status": "ok"}


@router.get(
    "/ready",
    tags=["health"],
    response_model=ReadyResponse,
    responses={503: {"model": ReadyResponse, "description": "Not ready"}},
)
def ready(request: Request):
    """Readiness: models loaded and services connected. 503 until then (load balancers wait)."""
    components = getattr(request.app.state, "components", None)
    if components is None:
        reason = getattr(request.app.state, "startup_error", None) or "starting"
        body = ReadyResponse(status="not_ready", reason=reason)
        return JSONResponse(body.model_dump(), status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    return ReadyResponse(
        status="ready",
        chunks=components.kb.store.count(),
        sources=len(components.kb.store.sources()),
        web_enabled=components.agent.services.web is not None,
    )


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def _to_response(result: dict, thread_id: str, request_id: str = "") -> ChatResponse:
    check = result.get("check") or {}
    return ChatResponse(
        answer=result.get("answer", ""),
        source_used=result.get("source_used", ""),
        route=result.get("route", ""),
        standalone_question=result.get("standalone_question", ""),
        citations=result.get("citations", []),
        trace=result.get("trace", []),
        grounded=bool(check.get("grounded", True)),
        web_searched=bool(result.get("web_searched")),
        latency_s=result.get("latency_s", 0.0),
        thread_id=thread_id,
        request_id=request_id or request_id_var.get(),
    )


@router.post("/chat", response_model=ChatResponse, tags=["chat"], dependencies=[Depends(limit_chat)])
def chat(payload: ChatRequest, components: Components = Depends(get_components)):
    """Ask a question and get the final answer with citations and the agent's trace."""
    thread_id = payload.thread_id or uuid.uuid4().hex
    question = payload.question.strip()
    components.kb.refresh_if_stale()
    try:
        result = components.agent.ask(question, thread_id=thread_id)
    except Exception as exc:
        logger.exception("Chat failed")
        components.audit.record(
            question=question, request_id=request_id_var.get(), thread_id=thread_id, error=type(exc).__name__
        )
        raise  # the global handler returns a safe 500 with the request id
    components.audit.record(question=question, request_id=request_id_var.get(), thread_id=thread_id, result=result)
    return _to_response(result, thread_id)


@router.post("/chat/stream", tags=["chat"], dependencies=[Depends(limit_chat)])
def chat_stream(payload: ChatRequest, components: Components = Depends(get_components)):
    """Same as /api/chat, but streams Server-Sent Events: one `step` event per agent step, then `final`."""
    thread_id = payload.thread_id or uuid.uuid4().hex
    question = payload.question.strip()
    request_id = request_id_var.get()
    components.kb.refresh_if_stale()

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def events():
        # The server resumes this generator in a fresh worker-thread context each time, so the
        # request id is set again on every resume to keep it in the logs.
        request_id_var.set(request_id)
        try:
            for kind, node, payload_ in components.agent.stream(question, thread_id=thread_id):
                if kind == "step":
                    yield sse("step", {"node": node, "line": payload_})
                else:
                    components.audit.record(
                        question=question, request_id=request_id, thread_id=thread_id, result=payload_
                    )
                    yield sse("final", _to_response(payload_, thread_id, request_id).model_dump())
                request_id_var.set(request_id)
        except Exception as exc:
            request_id_var.set(request_id)
            logger.exception("Streaming chat failed")
            components.audit.record(
                question=question, request_id=request_id, thread_id=thread_id, error=type(exc).__name__
            )
            yield sse("error", {"detail": "The agent failed to answer. Please try again.", "request_id": request_id})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # no proxy buffering
    )


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------


@router.get("/sources", response_model=list[SourceInfo], tags=["knowledge base"])
def list_sources(components: Components = Depends(get_components)):
    """Documents currently in the knowledge base."""
    return components.kb.store.sources()


@router.post(
    "/ingest/file", response_model=IngestResponse, tags=["knowledge base"], dependencies=[Depends(require_admin)]
)
def ingest_file(
    file: UploadFile = File(...),
    components: Components = Depends(get_components),
    settings: Settings = Depends(get_app_settings),
):
    """Upload a PDF, DOCX, Markdown, TXT or HTML file. Re-uploading the same file name replaces it."""
    name = safe_filename(file.filename or "")
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"Unsupported file type. Use: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)  # read at most one byte past the limit
    if len(data) > max_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, f"File is larger than {settings.max_upload_mb} MB"
        )
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The file is empty")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.upload_dir / f"{uuid.uuid4().hex[:8]}_{name}"  # unique on disk, clean name in citations
    dest.write_bytes(data)
    try:
        return components.kb.ingest_file(dest, display_name=name).as_dict()
    except IngestionError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc


@router.post(
    "/ingest/url", response_model=IngestResponse, tags=["knowledge base"], dependencies=[Depends(require_admin)]
)
def ingest_url(payload: UrlIngestRequest, components: Components = Depends(get_components)):
    """Fetch a public web page and add it to the knowledge base. Re-ingesting a URL updates it."""
    try:
        return components.kb.ingest_url(payload.url.strip()).as_dict()
    except IngestionError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc


@router.delete("/sources", tags=["knowledge base"], dependencies=[Depends(require_admin)])
def delete_source(source: str = Query(..., min_length=1), components: Components = Depends(get_components)):
    """Remove a document (all its chunks) from Pinecone and the keyword index."""
    removed = components.kb.delete_source(source)
    if not removed:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No document with that source")
    return {"source": source, "removed_chunks": removed}


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


@router.get("/audit", tags=["audit"], dependencies=[Depends(require_admin)])
def audit(limit: int = Query(50, ge=1, le=500), components: Components = Depends(get_components)):
    """Recent questions with the path the agent took, plus totals."""
    return {"summary": components.audit.summary(), "recent": components.audit.recent(limit)}
