# SPDX-License-Identifier: Apache-2.0
# Issue #1276: web module bridge — manifest wiring, health, API proxy
# forward + auth gating, dist fallback. Static UI itself is exercised by
# the E2E in this PR's description (bun build + curl) — unit tests here
# cover the Python surface offline.
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fusion_mlx.admin import webserve

REPO_ROOT = Path(__file__).resolve().parents[2]


def _manifest() -> dict:
    return json.loads((REPO_ROOT / "web" / "manifest.json").read_text(encoding="utf-8"))


def test_manifest_declares_upstream_and_api_key_env():
    # gateway #232 dynamic proxy needs an absolute upstream + the env name
    # holding the module key (same contract cowork #125 fixed).
    m = _manifest()
    assert m["id"] == "fusion-mlx"
    assert m["upstream"].startswith("http://") or m["upstream"].startswith("https://")
    assert m["api_key_env"] == "FUSION_MLX_WEB_API_KEY"


def test_health_reports_module(monkeypatch):
    monkeypatch.setenv("FUSION_MLX_WEB_DIST", str(REPO_ROOT / "web" / "no-dist"))
    monkeypatch.delenv("FUSION_MLX_WEB_API_KEY", raising=False)
    monkeypatch.delenv("FUSION_MLX_API_KEY", raising=False)
    app = webserve.create_web_app()
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["module"] == "fusion-mlx"
    assert body["status"] == "ok"
    assert body["dist"] is False


@pytest.fixture()
def upstream(monkeypatch):
    import threading
    import time

    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import PlainTextResponse

    seen = {}

    up = FastAPI()

    @up.get("/v1/models")
    def v1_models():
        seen["path"] = "/v1/models"
        seen["auth"] = None
        return {"ok": True}

    @up.api_route("/stats/x", methods=["GET"])
    def stats_x():
        seen["path"] = "/stats/x"
        return {"ok": True}

    @up.get("/metrics")
    def metrics():
        return PlainTextResponse("metric 1\nmetric 2\n")

    @up.get("/v1/health")
    def v1_health():
        return {"status": "ok", "oom_risk": "high"}

    @up.get("/admin/api/models")
    def admin_models():
        return {
            "models": [
                {
                    "id": "mlx-community/Qwen3.5-4B-MLX-4bit",
                    "loaded": True,
                    "engine_type": "batched",
                },
                {"id": "BAAI/bge-m3", "loaded": False, "engine_type": "embedding"},
            ]
        }

    @up.middleware("http")
    async def capture(request, call_next):
        response = await call_next(request)
        if request.url.path != "/health":
            seen.setdefault("path", request.url.path)
            seen["auth"] = request.headers.get("authorization")
        return response

    cfg = uvicorn.Config(up, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(cfg)
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    monkeypatch.setenv("FUSION_MLX_WEB_ENGINE_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("FUSION_MLX_WEB_DIST", str(REPO_ROOT / "web" / "no-dist"))
    yield seen
    server.should_exit = True
    time.sleep(0.2)


def test_proxy_forwards_path_and_engine_key(upstream, monkeypatch):
    monkeypatch.setenv("FUSION_MLX_API_KEY", "engine-secret")
    monkeypatch.delenv("FUSION_MLX_WEB_API_KEY", raising=False)
    app = webserve.create_web_app()
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/v1/models")
    assert r.status_code == 200
    assert upstream["path"] == "/v1/models"
    assert upstream["auth"] == "Bearer engine-secret"


def test_proxy_client_key_gate(upstream, monkeypatch):
    monkeypatch.delenv("FUSION_MLX_API_KEY", raising=False)
    monkeypatch.setenv("FUSION_MLX_WEB_API_KEY", "module-key")
    app = webserve.create_web_app()
    c = TestClient(app, raise_server_exceptions=False)
    assert c.get("/stats/x").status_code == 401
    r = c.get("/stats/x", headers={"Authorization": "Bearer module-key"})
    assert r.status_code == 200


def test_metrics_prefix_proxied(upstream, monkeypatch):
    monkeypatch.delenv("FUSION_MLX_WEB_API_KEY", raising=False)
    monkeypatch.delenv("FUSION_MLX_API_KEY", raising=False)
    app = webserve.create_web_app()
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/metrics")
    assert r.status_code == 200
    assert "metric 1" in r.text


def test_engine_key_env_wins_over_settings(monkeypatch):
    monkeypatch.setenv("FUSION_MLX_API_KEY", "env-key")
    assert webserve.resolve_engine_api_key() == "env-key"


def test_registration_disabled_without_gateway_env(monkeypatch):
    monkeypatch.delenv("FUSION_SHELL_GATEWAY_URL", raising=False)
    monkeypatch.delenv("FUSION_GATEWAY_ADMIN_KEY", raising=False)
    assert webserve._registration_enabled() is False


def test_manifest_declares_ext_fields_1281():
    # schema v1 ext (#1281): sub-page deep links, model search provider,
    # engine notification source — gateway validates shapes at registration.
    m = _manifest()
    assert [p["route"] for p in m["pages"]] == ["/models", "/metrics", "/logs"]
    (sp,) = m["searchProviders"]
    assert sp["endpoint"] == "/shell/fusion-mlx/api/search/models"
    (n,) = m["notifications"]
    assert n["kind"] == "engine"
    assert n["endpoint"] == "/shell/fusion-mlx/api/notifications"


def test_search_models_matches_id(upstream, monkeypatch):
    monkeypatch.delenv("FUSION_MLX_WEB_API_KEY", raising=False)
    monkeypatch.delenv("FUSION_MLX_API_KEY", raising=False)
    app = webserve.create_web_app()
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/api/search/models", params={"q": "qwen"})
    assert r.status_code == 200
    items = r.json()["items"]
    assert items and items[0]["title"] == "mlx-community/Qwen3.5-4B-MLX-4bit"
    assert items[0]["route"] == "/models"


def test_search_models_no_match_empty(upstream, monkeypatch):
    monkeypatch.delenv("FUSION_MLX_WEB_API_KEY", raising=False)
    monkeypatch.delenv("FUSION_MLX_API_KEY", raising=False)
    app = webserve.create_web_app()
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/api/search/models", params={"q": "zzz-nope"})
    assert r.status_code == 200
    assert r.json()["items"] == []


def test_notifications_lists_oom_and_loaded(upstream, monkeypatch):
    monkeypatch.delenv("FUSION_MLX_WEB_API_KEY", raising=False)
    monkeypatch.delenv("FUSION_MLX_API_KEY", raising=False)
    app = webserve.create_web_app()
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/api/notifications")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == len(body["items"]) == 2
    levels = {i["level"] for i in body["items"]}
    assert levels == {"warning", "info"}
    assert any(i["id"] == "mlx-oom-high" for i in body["items"])
    assert any(i["id"].startswith("mlx-model-loaded:") for i in body["items"])
