"""Shared API dependencies: getting the components, admin authentication and rate limiting."""

import secrets
import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, Header, HTTPException, Request, status

from app.bootstrap import Components
from app.core.config import Settings


def get_app_settings(request: Request) -> Settings:
    """The settings this app instance was created with (tests create apps with their own)."""
    return request.app.state.settings


def get_components(request: Request) -> Components:
    """The agent and knowledge base, built once at startup. 503 if startup could not build them."""
    components = getattr(request.app.state, "components", None)
    if components is None:
        reason = getattr(request.app.state, "startup_error", None) or "Service is starting."
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Not ready: {reason}")
    return components


def require_admin(
    x_admin_key: str = Header(default="", description="Admin key for ingestion and audit endpoints"),
    settings: Settings = Depends(get_app_settings),
) -> None:
    # compare_digest takes the same time whether the first or last character differs,
    # so an attacker can't guess the key one character at a time from response timings.
    if not x_admin_key or not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-Admin-Key header")


class RateLimiter:
    """Sliding-window limit per client IP, kept in memory (one server process).

    With several replicas, move this to Redis or to the API gateway / load balancer.
    """

    def __init__(self) -> None:
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window_s: float = 60.0) -> None:
        if limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > window_s:
                hits.popleft()
            if len(hits) >= limit:
                retry = int(window_s - (now - hits[0])) + 1
                raise HTTPException(
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many questions; please wait a moment.",
                    headers={"Retry-After": str(retry)},
                )
            hits.append(now)


rate_limiter = RateLimiter()


def limit_chat(request: Request, settings: Settings = Depends(get_app_settings)) -> None:
    client = request.client.host if request.client else "unknown"
    rate_limiter.check(client, settings.rate_limit_per_minute)
