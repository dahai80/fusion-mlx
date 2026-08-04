# SPDX-License-Identifier: Apache-2.0
# Wire + unit coverage for the #342-#346 security-hardening bundle.
#
# #342  bind default 127.0.0.1 (was 0.0.0.0)
# #343  X-Fusion-Route source-validation ASGI middleware (warn -> 403)
# #344  /metrics + /v1/status behind verify_management_access
# #345  load/unload require X-Fusion-Source: model-hub
# #346  anonymous access rejected by default (FUSION_ALLOW_ANONYMOUS override)
#
# Starlette TestClient uses host "testclient" (NOT loopback), so it stands
# in for a non-loopback LAN caller - ideal for the reject paths. Loopback
# exemptions are exercised with mocked Request objects built from a scope.

from __future__ import annotations

import asyncio

import pytest
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.testclient import TestClient


def _make_request(host: str = "testclient", headers: dict | None = None) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [
            (k.lower().encode("latin-1"), v.encode("latin-1"))
            for k, v in (headers or {}).items()
        ],
        "client": (host, 50000),
    }
    return Request(scope)


# --- #342: bind default is loopback ---------------------------------------


def test_server_config_host_defaults_to_loopback():
    from fusion_mlx.config import ServerConfig

    assert ServerConfig().host == "127.0.0.1"


# --- #343: X-Fusion-Route source-validation middleware --------------------


def _guard_app() -> FastAPI:
    from fusion_mlx.middleware.route_guard import install_route_guard_middleware

    app = FastAPI()

    @app.api_route("/{p:path}", methods=["GET", "POST", "DELETE", "OPTIONS"])
    async def _catch_all(p: str):
        return {"ok": True}

    install_route_guard_middleware(app)
    return app


def test_route_guard_warn_only_by_default():
    # No X-Fusion-Route, enforce off -> request passes through.
    client = TestClient(_guard_app())
    r = client.get("/v1/chat/completions")
    assert r.status_code == 200


def test_route_guard_enforce_rejects_missing_header(monkeypatch):
    monkeypatch.setenv("FUSION_ROUTE_ENFORCE", "true")
    client = TestClient(_guard_app())
    r = client.get("/v1/chat/completions")
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "missing_route"


def test_route_guard_enforce_allows_with_header(monkeypatch):
    monkeypatch.setenv("FUSION_ROUTE_ENFORCE", "true")
    client = TestClient(_guard_app())
    r = client.get("/v1/chat/completions", headers={"X-Fusion-Route": "gateway"})
    assert r.status_code == 200


@pytest.mark.parametrize(
    "path", ["/", "/health", "/healthz", "/readyz", "/livez", "/openapi.json"]
)
def test_route_guard_exempt_paths_pass_without_header(monkeypatch, path):
    monkeypatch.setenv("FUSION_ROUTE_ENFORCE", "true")
    client = TestClient(_guard_app())
    r = client.get(path)
    assert r.status_code != 403, f"exempt path {path} was rejected"


def test_route_guard_options_preflight_exempt(monkeypatch):
    monkeypatch.setenv("FUSION_ROUTE_ENFORCE", "true")
    client = TestClient(_guard_app())
    r = client.options("/v1/chat/completions")
    assert r.status_code != 403


# --- #344: management endpoints behind auth ------------------------------


def _mgmt_app() -> FastAPI:
    from fusion_mlx.routes_internal import health as health_mod
    from fusion_mlx.routes_internal import metrics as metrics_mod

    app = FastAPI()
    app.include_router(metrics_mod.router)
    app.include_router(health_mod.router)
    return app


def test_metrics_rejects_anonymous_non_loopback(monkeypatch):
    monkeypatch.delenv("FUSION_ALLOW_ANONYMOUS", raising=False)
    client = TestClient(_mgmt_app())
    r = client.get("/metrics")
    assert r.status_code == 401


def test_metrics_rejects_x_fusion_route_without_key(monkeypatch):
    # X-Fusion-Route is provenance, not auth: a non-loopback caller with no
    # api_key must still be rejected even when it sets the header.
    monkeypatch.delenv("FUSION_ALLOW_ANONYMOUS", raising=False)
    client = TestClient(_mgmt_app())
    r = client.get("/metrics", headers={"X-Fusion-Route": "gateway"})
    assert r.status_code == 401


def test_status_rejects_anonymous_non_loopback(monkeypatch):
    monkeypatch.delenv("FUSION_ALLOW_ANONYMOUS", raising=False)
    client = TestClient(_mgmt_app())
    r = client.get("/v1/status")
    assert r.status_code == 401


def test_status_rejects_x_fusion_route_without_key(monkeypatch):
    # X-Fusion-Route is provenance, not auth: a non-loopback caller with no
    # api_key must still be rejected even when it sets the header.
    monkeypatch.delenv("FUSION_ALLOW_ANONYMOUS", raising=False)
    client = TestClient(_mgmt_app())
    r = client.get("/v1/status", headers={"X-Fusion-Route": "gateway"})
    assert r.status_code == 401


# --- #345: load/unload require X-Fusion-Source: model-hub ----------------


def test_model_hub_source_rejects_non_loopback_no_header():
    from fusion_mlx.middleware.auth import require_model_hub_source

    req = _make_request(host="203.0.113.5")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(require_model_hub_source(req))
    assert exc.value.status_code == 403


def test_model_hub_source_allows_model_hub_header():
    from fusion_mlx.middleware.auth import require_model_hub_source

    req = _make_request(host="203.0.113.5", headers={"x-fusion-source": "model-hub"})
    assert asyncio.run(require_model_hub_source(req)) is True


def test_model_hub_source_allows_loopback_dev():
    from fusion_mlx.middleware.auth import require_model_hub_source

    req = _make_request(host="127.0.0.1")
    assert asyncio.run(require_model_hub_source(req)) is True


def test_load_endpoint_rejects_without_x_fusion_source(monkeypatch):
    # Wire-level: a route guarded by require_model_hub_source (same Depends
    # the real load/unload endpoints use) 403s a non-loopback caller with
    # no X-Fusion-Source header.
    from fusion_mlx.middleware.auth import require_model_hub_source

    monkeypatch.delenv("FUSION_ALLOW_ANONYMOUS", raising=False)
    app = FastAPI()

    @app.post("/v1/models/{model_id}/load")
    async def _load(model_id: str, _src: bool = Depends(require_model_hub_source)):
        return {"loaded": model_id}

    client = TestClient(app)
    r = client.post("/v1/models/my-model/load")
    assert r.status_code == 403


def test_load_endpoint_allows_with_x_fusion_source(monkeypatch):
    from fusion_mlx.middleware.auth import require_model_hub_source

    monkeypatch.delenv("FUSION_ALLOW_ANONYMOUS", raising=False)
    app = FastAPI()

    @app.post("/v1/models/{model_id}/load")
    async def _load(model_id: str, _src: bool = Depends(require_model_hub_source)):
        return {"loaded": model_id}

    client = TestClient(app)
    r = client.post(
        "/v1/models/my-model/load",
        headers={"X-Fusion-Source": "model-hub"},
    )
    assert r.status_code == 200


# --- #346: anonymous access rejected by default --------------------------


def test_anonymous_rejected_non_loopback_without_override(monkeypatch):
    monkeypatch.delenv("FUSION_ALLOW_ANONYMOUS", raising=False)
    from fusion_mlx.middleware.auth import _anonymous_access_allowed

    req = _make_request(host="203.0.113.5")
    assert _anonymous_access_allowed(req) is False


def test_anonymous_allowed_for_loopback_without_override(monkeypatch):
    monkeypatch.delenv("FUSION_ALLOW_ANONYMOUS", raising=False)
    from fusion_mlx.middleware.auth import _anonymous_access_allowed

    req = _make_request(host="127.0.0.1")
    assert _anonymous_access_allowed(req) is True


def test_anonymous_allowed_with_env_override():
    # FUSION_ALLOW_ANONYMOUS=true is set by the autouse conftest fixture.
    from fusion_mlx.middleware.auth import _anonymous_access_allowed

    req = _make_request(host="203.0.113.5")
    assert _anonymous_access_allowed(req) is True


def test_anonymous_rejected_with_only_x_fusion_route(monkeypatch):
    # X-Fusion-Route alone must not grant anonymous access (spoofable header).
    monkeypatch.delenv("FUSION_ALLOW_ANONYMOUS", raising=False)
    from fusion_mlx.middleware.auth import _anonymous_access_allowed

    req = _make_request(host="203.0.113.5", headers={"x-fusion-route": "gateway"})
    assert _anonymous_access_allowed(req) is False


def test_management_access_rejects_non_loopback_no_creds(monkeypatch):
    monkeypatch.delenv("FUSION_ALLOW_ANONYMOUS", raising=False)
    from fusion_mlx.middleware.auth import verify_management_access

    req = _make_request(host="203.0.113.5")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(verify_management_access(req))
    assert exc.value.status_code == 401


def test_management_access_loopback_allowed_dev(monkeypatch):
    monkeypatch.delenv("FUSION_ALLOW_ANONYMOUS", raising=False)
    from fusion_mlx.middleware.auth import verify_management_access

    req = _make_request(host="127.0.0.1")
    assert asyncio.run(verify_management_access(req)) is True


def test_management_access_rejects_x_fusion_route_when_key_configured(monkeypatch):
    # Regression for the #344 header-spoof bypass: with an api_key configured,
    # a non-loopback caller that omits the key but sets X-Fusion-Route must
    # still get 401 - the header is provenance, not a credential.
    monkeypatch.delenv("FUSION_ALLOW_ANONYMOUS", raising=False)
    from fusion_mlx.middleware import auth as auth_mod

    monkeypatch.setattr(auth_mod, "_get_configured_api_key", lambda: "server-secret")
    req = _make_request(host="203.0.113.5", headers={"x-fusion-route": "gateway"})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(auth_mod.verify_management_access(req))
    assert exc.value.status_code == 401
