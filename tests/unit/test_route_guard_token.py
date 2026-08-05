# SPDX-License-Identifier: Apache-2.0
# #352: FUSION_ROUTE_TOKEN shared-secret auth in RouteGuardMiddleware.
#
# When FUSION_ROUTE_TOKEN is set, X-Fusion-Route's VALUE must equal the secret
# (constant-time compare). Missing/mismatched -> 403 invalid_route_token. When
# unset/empty, the presence-check behavior from #343 is unchanged.

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _guard_app() -> FastAPI:
    from fusion_mlx.middleware.route_guard import install_route_guard_middleware

    app = FastAPI()

    @app.api_route("/{p:path}", methods=["GET", "POST", "DELETE", "OPTIONS"])
    async def _catch_all(p: str):
        return {"ok": True}

    install_route_guard_middleware(app)
    return app


_TOKEN = "s3cret-shared-token-7f3a"


def _client(monkeypatch, token: str | None) -> TestClient:
    monkeypatch.delenv("FUSION_ROUTE_WARN_ONLY", raising=False)
    monkeypatch.delenv("FUSION_ROUTE_ENFORCE", raising=False)
    monkeypatch.delenv("FUSION_ROUTE_TOKEN", raising=False)
    if token is not None:
        monkeypatch.setenv("FUSION_ROUTE_TOKEN", token)
    return TestClient(_guard_app())


def test_token_unset_falls_back_to_presence_check(monkeypatch):
    # No FUSION_ROUTE_TOKEN: a bare X-Fusion-Route header still passes (legacy).
    client = _client(monkeypatch, token=None)
    r = client.get("/v1/chat/completions", headers={"X-Fusion-Route": "gateway"})
    assert r.status_code == 200


def test_token_empty_string_treated_as_unset(monkeypatch):
    # Whitespace-only env value normalizes to None -> feature OFF.
    client = _client(monkeypatch, token="   ")
    r = client.get("/v1/chat/completions", headers={"X-Fusion-Route": "gateway"})
    assert r.status_code == 200


def test_token_set_header_missing_rejected(monkeypatch):
    client = _client(monkeypatch, token=_TOKEN)
    r = client.get("/v1/chat/completions")
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "invalid_route_token"


def test_token_set_header_mismatch_rejected(monkeypatch):
    client = _client(monkeypatch, token=_TOKEN)
    r = client.get("/v1/chat/completions", headers={"X-Fusion-Route": "wrong-value"})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "invalid_route_token"


def test_token_set_header_match_passes(monkeypatch):
    client = _client(monkeypatch, token=_TOKEN)
    r = client.get("/v1/chat/completions", headers={"X-Fusion-Route": _TOKEN})
    assert r.status_code == 200


def test_token_set_exempt_path_passes_without_header(monkeypatch):
    # Health probes must stay reachable even with the token enabled.
    client = _client(monkeypatch, token=_TOKEN)
    r = client.get("/health")
    assert r.status_code != 403


def test_token_set_options_preflight_exempt(monkeypatch):
    client = _client(monkeypatch, token=_TOKEN)
    r = client.options("/v1/chat/completions")
    assert r.status_code != 403


def test_token_takes_precedence_over_warn_only(monkeypatch):
    # Even with warn-only on, a configured token is enforced (stricter wins).
    monkeypatch.setenv("FUSION_ROUTE_WARN_ONLY", "true")
    client = _client(monkeypatch, token=_TOKEN)
    r = client.get("/v1/chat/completions", headers={"X-Fusion-Route": "wrong-value"})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "invalid_route_token"
