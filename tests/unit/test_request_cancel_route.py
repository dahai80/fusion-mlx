# SPDX-License-Identifier: Apache-2.0
"""Routing tests for POST /v1/requests/{id}/cancel (#886 finding 1).

The cancel route previously had zero coverage — the only test file ever
written for it was an empty stub (removed in the #0913 roster cleanup).
These tests pin the routing contract: pool-missing 503, unknown-id 404,
engine-abort success envelope, engine-error 500 (no exception echo).
"""

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.routes_internal import health


def _make_app(monkeypatch, pool):
    app = FastAPI()
    app.include_router(health.admin_router)
    monkeypatch.setattr(
        "fusion_mlx.middleware.auth.verify_api_key_or_x_api_key",
        lambda: True,
        raising=False,
    )
    import fusion_mlx.server as srv

    monkeypatch.setitem(srv._server_state, "engine_pool", pool)
    return TestClient(app, raise_server_exceptions=False)


def _engine_with_request(request_id, abort_exc=None):
    engine = MagicMock()
    scheduler = MagicMock()
    scheduler.requests = {request_id: MagicMock()}
    engine.scheduler = scheduler
    if abort_exc is not None:
        engine.abort_request = AsyncMock(side_effect=abort_exc)
    else:
        engine.abort_request = AsyncMock(return_value=True)
    return engine


def _pool_with_entries(entries):
    pool = MagicMock()
    pool._entries = entries
    return pool


def test_cancel_no_pool_returns_503(monkeypatch):
    client = _make_app(monkeypatch, None)
    resp = client.post("/v1/requests/req-1/cancel")
    assert resp.status_code == 503


def test_cancel_unknown_id_returns_404(monkeypatch):
    engine = _engine_with_request("req-other")
    pool = _pool_with_entries({"model-a": type("E", (), {"engine": engine})})
    client = _make_app(monkeypatch, pool)
    resp = client.post("/v1/requests/req-missing/cancel")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_cancel_known_id_aborts_and_returns_envelope(monkeypatch):
    engine = _engine_with_request("req-1")
    pool = _pool_with_entries({"model-a": type("E", (), {"engine": engine})})
    client = _make_app(monkeypatch, pool)
    resp = client.post("/v1/requests/req-1/cancel")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "req-1"
    assert body["cancelled"] is True
    # success envelope must NOT leak model_name (weight fingerprinting)
    assert "model_name" not in body
    engine.abort_request.assert_awaited_once_with("req-1")


def test_cancel_engine_error_returns_500_generic(monkeypatch):
    engine = _engine_with_request(
        "req-1", abort_exc=RuntimeError("/Users/x/.fusion-mlx/models/secret-repo")
    )
    pool = _pool_with_entries({"model-a": type("E", (), {"engine": engine})})
    client = _make_app(monkeypatch, pool)
    resp = client.post("/v1/requests/req-1/cancel")
    assert resp.status_code == 500
    # F-151: no exception echo — HF path must not reach the client
    assert "secret-repo" not in resp.text


def test_cancel_skips_entries_without_engine(monkeypatch):
    engine = _engine_with_request("req-1")
    pool = _pool_with_entries(
        {
            "model-none": type("E", (), {"engine": None}),
            "model-a": type("E", (), {"engine": engine}),
        }
    )
    client = _make_app(monkeypatch, pool)
    resp = client.post("/v1/requests/req-1/cancel")
    assert resp.status_code == 200
    engine.abort_request.assert_awaited_once()
