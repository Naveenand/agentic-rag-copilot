"""FastAPI application: API routes, web UI, middleware and startup.

Startup loads the models and connects to Pinecone/OpenAI/Tavily in a background thread, so
/api/health answers immediately while /api/ready reports "starting" until everything is loaded.
"""

import logging
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router
from app.bootstrap import Components, build_components
from app.core.config import BASE_DIR, Settings, get_settings
from app.core.logging import configure_logging, request_id_var

logger = logging.getLogger("app")
VERSION = "2.1.0"


def security_headers(frame_ancestors: str = "'none'") -> dict[str, str]:
    headers = {
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        # Only our own scripts and styles may run on the page (blocks injected scripts).
        # frame-ancestors decides which sites may show this page inside an iframe.
        "Content-Security-Policy": (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            f"connect-src 'self'; frame-ancestors {frame_ancestors}; base-uri 'self'"
        ),
    }
    if frame_ancestors.strip() == "'none'":
        headers["X-Frame-Options"] = "DENY"  # same rule for older browsers
    return headers


def create_app(
    settings: Optional[Settings] = None,
    components: Optional[Components] = None,
    builder: Callable[[Settings], Components] = build_components,
) -> FastAPI:
    """Build the app. Tests pass ready-made `components`; production builds them at startup."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    settings.check_production_safety()
    headers = security_headers(settings.frame_ancestors)

    def load_components(app: FastAPI) -> None:
        started = time.perf_counter()
        try:
            app.state.components = builder(settings)
            logger.info("Ready in %.1fs", time.perf_counter() - started)
        except Exception as exc:  # stay up so /api/ready can explain what is wrong
            app.state.startup_error = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
            logger.exception("Startup failed; the API will report not ready")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.startup_error = None
        if components is not None:
            app.state.components = components
        else:
            app.state.components = None
            threading.Thread(target=load_components, args=(app,), daemon=True, name="startup").start()
        yield

    app = FastAPI(
        title=settings.app_name,
        version=VERSION,
        lifespan=lifespan,
        docs_url=None if settings.is_production else "/docs",  # hide interactive docs in production
        redoc_url=None,
    )
    app.state.settings = settings
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", "")[:64] or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        for name, value in headers.items():
            response.headers.setdefault(name, value)
        if request.url.path.startswith("/api/") and request.url.path != "/api/health":
            logger.info(
                "%s %s -> %d (%.0f ms) [%s]",
                request.method,
                request.url.path,
                response.status_code,
                (time.perf_counter() - started) * 1000,
                request_id,
            )
        return response

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        # Never send internal error text (keys, paths, stack traces) to the browser.
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal error. Please try again.", "request_id": request_id_var.get()},
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(request: Request):
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "app_name": settings.app_name,
                "version": VERSION,
                "web_enabled": settings.enable_web,
                "access_required": bool(settings.access_code),
            },
        )

    return app


app = create_app()  # what `uvicorn app.main:app` serves
