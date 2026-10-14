"""Shared API dependencies: getting the components, admin authentication and rate limiting."""

import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

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
    if not _same_secret(x_admin_key, settings.admin_api_key):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-Admin-Key header")


def _same_secret(given: str, expected: str) -> bool:
    # compare_digest takes the same time whether the first or last character differs, so an
    # attacker can't guess a key one character at a time from response timings.
    # Bytes, so non-ASCII input is a wrong key instead of a server error.
    return bool(given) and secrets.compare_digest(given.encode(), expected.encode())


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


class DailyQuota:
    """Counts questions per UTC day for the whole app: a hard ceiling on LLM spend for a public demo.

    Kept in memory, so it restarts at 0 with the process. The provider's monthly budget limit
    (e.g. OpenAI usage limits) is the backstop that survives restarts.
    """

    def __init__(self) -> None:
        self._day = ""
        self._count = 0
        self._lock = threading.Lock()

    def check(self, limit: int) -> None:
        if limit <= 0:
            return
        today = datetime.now(timezone.utc).date().isoformat()
        with self._lock:
            if today != self._day:
                self._day, self._count = today, 0
            if self._count >= limit:
                raise HTTPException(
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="The demo has reached its daily question limit. Please try again tomorrow.",
                )
            self._count += 1


rate_limiter = RateLimiter()
daily_quota = DailyQuota()


def limit_chat(
    request: Request,
    x_access_code: str = Header(default="", description="Demo access code, when ACCESS_CODE is set"),
    settings: Settings = Depends(get_app_settings),
) -> None:
    """Gate for every question, cheapest check first: access code, per-IP rate, daily total."""
    if settings.access_code and not _same_secret(x_access_code, settings.access_code):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="A valid access code is needed to ask questions.")
    client = request.client.host if request.client else "unknown"
    rate_limiter.check(client, settings.rate_limit_per_minute)
    daily_quota.check(settings.daily_question_limit)
