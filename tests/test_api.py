"""HTTP API tests: every endpoint, authentication, limits, streaming and error handling."""

import json

import pytest
from fastapi.testclient import TestClient

from app.api.deps import rate_limiter
from app.main import create_app

ADMIN = {"X-Admin-Key": "test-admin-key-123456"}
KB_QUESTION = "In Agentic RAG, what happens when retrieved documents are not relevant to the question?"


@pytest.fixture
def client(settings, components):
    rate_limiter._hits.clear()
    with TestClient(create_app(settings, components=components)) as c:
        yield c


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


# ---------------------------------------------------------------------------
# Health, UI, headers
# ---------------------------------------------------------------------------


def test_health_ready_and_home(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    ready = client.get("/api/ready").json()
    assert ready["status"] == "ready" and ready["chunks"] > 0 and ready["sources"] == 1 and ready["web_enabled"]
    home = client.get("/")
    assert home.status_code == 200 and "Knowledge Copilot" in home.text
    assert client.get("/static/js/app.js").status_code == 200


def test_security_headers_and_request_id(client):
    r = client.get("/api/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert "script-src 'self'" in r.headers["Content-Security-Policy"]
    assert client.get("/api/health").headers["X-Request-ID"]  # generated when absent


def test_not_ready_while_startup_fails(settings):
    def broken_builder(_settings):
        raise RuntimeError("Missing API keys: OPENAI_API_KEY")

    with TestClient(create_app(settings, builder=broken_builder)) as c:
        import time

        for _ in range(50):  # startup runs in a background thread
            r = c.get("/api/ready")
            if r.json().get("reason") != "starting":
                break
            time.sleep(0.02)
        assert r.status_code == 503 and "OPENAI_API_KEY" in r.json()["reason"]
        assert c.get("/api/health").status_code == 200  # still alive
        chat = c.post("/api/chat", json={"question": "hi"})
        assert chat.status_code == 503 and "OPENAI_API_KEY" in chat.json()["detail"]


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def test_chat_answers_with_citations_and_audits(client, components):
    r = client.post("/api/chat", json={"question": KB_QUESTION})
    assert r.status_code == 200
    body = r.json()
    assert body["source_used"] == "private_kb" and body["grounded"] and not body["web_searched"]
    assert [c["id"] for c in body["citations"]] == [1, 2] and "[7]" not in body["answer"]
    assert body["trace"][0].startswith("route=retrieve") and len(body["thread_id"]) == 32
    audit = client.get("/api/audit", headers=ADMIN).json()
    assert audit["summary"]["questions"] == 1 and audit["recent"][0]["request_id"] == body["request_id"]


def test_chat_thread_keeps_conversation(client):
    first = client.post(
        "/api/chat", json={"question": "How does the tutorial grade retrieved documents?", "thread_id": "conv-1"}
    )
    second = client.post(
        "/api/chat",
        json={"question": "And what does it do after that step if the grade is bad?", "thread_id": "conv-1"},
    )
    assert first.json()["thread_id"] == second.json()["thread_id"] == "conv-1"
    assert second.json()["standalone_question"].startswith("What happens after grading")


def test_chat_stream_sends_steps_then_final(client):
    r = client.post("/api/chat/stream", json={"question": KB_QUESTION, "thread_id": "s1"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert events[0] == ("step", {"node": "route_question", "line": events[0][1]["line"]})
    assert [e for e, _ in events].count("final") == 1 and events[-1][0] == "final"
    assert events[-1][1]["source_used"] == "private_kb" and events[-1][1]["thread_id"] == "s1"


def test_chat_stream_reports_errors_without_internals(client, fake_llm):
    fake_llm.chat_error = RuntimeError("secret-internal-path /etc/keys")
    events = parse_sse(client.post("/api/chat/stream", json={"question": KB_QUESTION}).text)
    assert events[-1][0] == "error" and "secret" not in json.dumps(events[-1][1])


def test_chat_errors_hide_internals(settings, fake_llm, components):
    fake_llm.chat_error = RuntimeError("secret-internal-path /etc/keys")
    with TestClient(create_app(settings, components=components), raise_server_exceptions=False) as c:
        r = c.post("/api/chat", json={"question": KB_QUESTION})
    assert r.status_code == 500
    assert "secret" not in r.text and r.json()["request_id"]
    assert components.audit.recent()[0]["error"] == "RuntimeError"


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "x" * 2001},
        {"question": "hi", "thread_id": "bad id with spaces"},
        {"thread_id": "abc"},
    ],
)
def test_chat_validation(client, payload):
    assert client.post("/api/chat", json=payload).status_code == 422


def test_rate_limit(client, settings):
    settings.rate_limit_per_minute = 2
    codes = [client.post("/api/chat", json={"question": "Hello there"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = client.post("/api/chat", json={"question": "Hello there"})
    assert r.status_code == 429 and int(r.headers["Retry-After"]) > 0


# ---------------------------------------------------------------------------
# Knowledge base admin
# ---------------------------------------------------------------------------


def test_admin_endpoints_need_the_key(client):
    files = {"file": ("a.md", b"# A\n\n" + b"text " * 50, "text/markdown")}
    assert client.post("/api/ingest/file", files=files).status_code == 401
    assert client.post("/api/ingest/file", files=files, headers={"X-Admin-Key": "wrong"}).status_code == 401
    assert client.get("/api/audit").status_code == 401
    assert client.delete("/api/sources", params={"source": "x"}).status_code == 401


def test_upload_then_ask_then_replace_then_delete(client, vector_index):
    doc = b"# VPN guide\n\n## Connect\n\n" + b"Employees must connect to the VPN before using internal systems. " * 5
    r = client.post("/api/ingest/file", headers=ADMIN, files={"file": ("../VPN Guide.md", doc, "text/markdown")})
    assert r.status_code == 200, r.text
    added = r.json()
    assert added["source"] == "VPN_Guide.md" and added["title"] == "VPN guide" and added["uploaded"] == added["chunks"]

    titles = {s["source"]: s["title"] for s in client.get("/api/sources").json()}
    assert titles["VPN_Guide.md"] == "VPN guide"

    answer = client.post(
        "/api/chat", json={"question": "How do employees connect to the VPN before using internal systems?"}
    ).json()
    assert answer["source_used"] == "private_kb"
    assert any(c["title"] == "VPN guide" for c in answer["citations"])

    same = client.post("/api/ingest/file", headers=ADMIN, files={"file": ("VPN Guide.md", doc, "text/markdown")}).json()
    assert same["uploaded"] == 0 and same["unchanged"] == same["chunks"]

    gone = client.delete("/api/sources", params={"source": "VPN_Guide.md"}, headers=ADMIN)
    assert gone.json()["removed_chunks"] == added["chunks"]
    assert client.delete("/api/sources", params={"source": "VPN_Guide.md"}, headers=ADMIN).status_code == 404


def test_upload_validation(client, settings):
    exe = client.post(
        "/api/ingest/file", headers=ADMIN, files={"file": ("tool.exe", b"MZ", "application/octet-stream")}
    )
    assert exe.status_code == 400
    empty = client.post("/api/ingest/file", headers=ADMIN, files={"file": ("e.md", b"", "text/markdown")})
    assert empty.status_code == 400
    settings.max_upload_mb = 0
    big = client.post("/api/ingest/file", headers=ADMIN, files={"file": ("b.md", b"# B\n\nabc", "text/markdown")})
    assert big.status_code == 413
    settings.max_upload_mb = 10
    tiny = client.post("/api/ingest/file", headers=ADMIN, files={"file": ("t.md", b"# T\n\nshort", "text/markdown")})
    assert tiny.status_code == 422 and "no chunks" in tiny.json()["detail"]
    assert not any(settings.upload_dir.glob("*t.md"))  # failed uploads are cleaned up


def test_url_ingest_blocks_private_addresses(client):
    r = client.post("/api/ingest/url", headers=ADMIN, json={"url": "http://127.0.0.1:8000/api/audit"})
    assert r.status_code == 422 and "private address" in r.json()["detail"]
