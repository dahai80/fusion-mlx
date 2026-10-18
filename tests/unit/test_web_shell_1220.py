# SPDX-License-Identifier: Apache-2.0
"""#1220: union-web shell module — web/ monitor SPA + shell registry.

Covers:
- GET / serves the built SPA index (web dist present) and falls back to
  the legacy {"status": "ok"} probe JSON when web/dist is absent.
- /assets static mount serves hashed bundles from the dist.
- web/manifest.json carries the PRD §5.1 schema fields.
- shell_registry: env gating, manifest load + POST payload shape.
- /admin/api/logs/stream SSE framing (data: {"line": ...} + rotation reset).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx import shell_registry, web_static
from fusion_mlx.routes_internal.health import probe_router

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WEB_DIR = REPO_ROOT / "web"


@pytest.fixture
def probe_app():
    application = FastAPI()
    application.include_router(probe_router)
    return application


def test_root_json_fallback_without_dist(probe_app, monkeypatch, tmp_path):
    monkeypatch.setenv("FUSION_MLX_WEB_DIST", str(tmp_path / "nope"))
    client = TestClient(probe_app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_root_serves_spa_index_with_dist(probe_app, monkeypatch, tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html><body>mlx monitor</body></html>")
    monkeypatch.setenv("FUSION_MLX_WEB_DIST", str(dist))
    client = TestClient(probe_app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "mlx monitor" in resp.text


def test_assets_mount_serves_bundle(monkeypatch, tmp_path):
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text("<html></html>")
    (assets / "index-abc.js").write_text("console.log('mlx')")
    monkeypatch.setenv("FUSION_MLX_WEB_DIST", str(dist))
    application = FastAPI()
    mounted = web_static.mount_web_static(application)
    assert mounted is True
    client = TestClient(application)
    resp = client.get("/assets/index-abc.js")
    assert resp.status_code == 200
    assert b"console.log" in resp.content


def test_assets_mount_skipped_without_dist(monkeypatch, tmp_path):
    monkeypatch.setenv("FUSION_MLX_WEB_DIST", str(tmp_path / "absent"))
    application = FastAPI()
    assert web_static.mount_web_static(application) is False


def test_manifest_schema_fields():
    manifest_path = WEB_DIR / "manifest.json"
    assert manifest_path.exists(), "web/manifest.json missing"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["id"] == "fusion-mlx"
    assert manifest["entry"] == "/shell/fusion-mlx/"
    assert manifest["path"] == "/mlx"
    assert manifest["category"] == "platform"
    assert manifest["icon"]
    assert isinstance(manifest["devices"], list) and manifest["devices"]
    for device in manifest["devices"]:
        assert device in ("desktop", "mobile-ready", "mobile-readonly", "desktop-only")
    assert manifest["title"] and manifest["title_en"]


def test_shell_registry_gating(monkeypatch):
    monkeypatch.delenv(shell_registry.GATEWAY_URL_ENV, raising=False)
    monkeypatch.delenv(shell_registry.ADMIN_KEY_ENV, raising=False)
    assert shell_registry.shell_registration_enabled() is False
    monkeypatch.setenv(shell_registry.GATEWAY_URL_ENV, "http://127.0.0.1:11432")
    assert shell_registry.shell_registration_enabled() is False
    monkeypatch.setenv(shell_registry.ADMIN_KEY_ENV, "some-admin-key")
    assert shell_registry.shell_registration_enabled() is True


class _FakeResponse:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text


class _FakeClient:
    """Captures the registration POST; returns a canned status."""

    last_request: dict = {}

    def __init__(self, *args, status_code: int = 201, text: str = "", **kwargs):
        self._status_code = status_code
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        _FakeClient.last_request = {
            "url": url,
            "json": json,
            "headers": headers,
        }
        return _FakeResponse(self._status_code, self._text)


@pytest.mark.asyncio
async def test_register_once_payload(monkeypatch):
    monkeypatch.setenv(shell_registry.GATEWAY_URL_ENV, "http://127.0.0.1:11432/")
    monkeypatch.setenv(shell_registry.ADMIN_KEY_ENV, "fg-admin-key")
    monkeypatch.setattr("httpx.AsyncClient", _FakeClient)
    ok = await shell_registry.register_once()
    assert ok is True
    sent = _FakeClient.last_request
    assert sent["url"] == "http://127.0.0.1:11432/api/v1/shell/manifests"
    assert sent["headers"]["Authorization"] == "Bearer fg-admin-key"
    assert sent["json"]["id"] == "fusion-mlx"
    assert sent["json"]["entry"] == "/shell/fusion-mlx/"


@pytest.mark.asyncio
async def test_register_once_failure_returns_false(monkeypatch):
    monkeypatch.setenv(shell_registry.GATEWAY_URL_ENV, "http://127.0.0.1:11432")
    monkeypatch.setenv(shell_registry.ADMIN_KEY_ENV, "fg-admin-key")
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda *a, **kw: _FakeClient(status_code=500, text="boom")
    )
    assert await shell_registry.register_once() is False


@pytest.mark.asyncio
async def test_log_stream_sse_framing(monkeypatch, tmp_path):
    """Drive the SSE generator directly: attach at EOF, append a line while
    the stream waits, read the follow-up event, then disconnect."""
    from fusion_mlx.admin import logs as logs_mod

    log_file = tmp_path / "server.log"
    log_file.write_text("2026-10-17 10:00:00,000 [INFO] fusion_mlx.server: hello\n")

    monkeypatch.setattr(logs_mod, "_resolve_current_log_file", lambda: log_file)
    # Fast poll so the test does not wait a full second per cycle.
    monkeypatch.setattr(logs_mod, "_STREAM_POLL_SECONDS", 0.05)

    class FakeRequest:
        disconnected = False

        async def is_disconnected(self):
            return self.disconnected

    fake_request = FakeRequest()

    resp = await logs_mod.stream_logs(fake_request, is_admin=True)
    assert resp.media_type == "text/event-stream"

    events: list[str] = []

    async def consume():
        async for chunk in resp.body_iterator:
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(line[len("data: ") :])

    consumer = asyncio.ensure_future(consume())
    await asyncio.sleep(0.2)
    log_file.write_text(
        log_file.read_text()
        + "2026-10-17 10:00:01,000 [WARN] fusion_mlx.pool: evicting\n"
    )
    for _ in range(100):
        if events:
            break
        await asyncio.sleep(0.05)
    fake_request.disconnected = True
    await asyncio.wait_for(consumer, timeout=5)

    assert events, "no SSE events captured"
    payload = json.loads(events[0])
    assert "evicting" in payload["line"]
