# SPDX-License-Identifier: Apache-2.0
import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel

from fusion_mlx.middleware import (
    install_exception_handlers,
    install_request_body_depth_middleware,
    install_request_body_limit_middleware,
)

logger = logging.getLogger(__name__)


def _make_app() -> FastAPI:
    app = FastAPI()

    install_request_body_limit_middleware(app)
    install_request_body_depth_middleware(app)
    install_exception_handlers(app)

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict):
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


@pytest.fixture
def client():
    app = _make_app()
    return TestClient(app, raise_server_exceptions=False)


class TestBodySizeMiddleware:
    def test_normal_request_passes(self, client):
        payload = {"model": "test", "messages": []}
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200

    def test_oversized_content_length_rejected(self, client, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_MAX_REQUEST_BYTES", "100")
        payload = {"model": "a" * 200}
        resp = client.post(
            "/v1/chat/completions",
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 413
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == "request_too_large"

    def test_unguarded_path_not_limited(self, client, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_MAX_REQUEST_BYTES", "100")
        resp = client.get("/health")
        assert resp.status_code == 200


class TestBodyDepthMiddleware:
    def test_normal_depth_passes(self, client):
        payload = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200

    def test_deeply_nested_rejected(self, client, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_MAX_BODY_DEPTH", "5")
        nested = {"a": 1}
        for _ in range(10):
            nested = {"x": nested}
        payload = {"model": "test", "messages": nested}
        resp = client.post(
            "/v1/chat/completions",
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == "request_body_too_deep"


class TestExceptionHandlers:
    def test_http_exception_envelope(self, client):
        app = _make_app()

        @app.get("/v1/not-found")
        async def not_found():
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Model not found")

        tc = TestClient(app, raise_server_exceptions=False)
        resp = tc.get("/v1/not-found")
        assert resp.status_code == 404
        body = resp.json()
        assert "error" in body
        assert "message" in body["error"]
        assert body["error"]["type"] == "not_found_error"

    def test_generic_exception_returns_500(self, client):
        app = _make_app()

        @app.get("/v1/boom")
        async def boom():
            raise RuntimeError("kaboom")

        tc = TestClient(app, raise_server_exceptions=False)
        resp = tc.get("/v1/boom")
        assert resp.status_code == 500
        body = resp.json()
        assert "error" in body
        assert body["error"]["message"] == "Internal server error"


class TestMiddlewareExports:
    def test_all_install_functions_exported(self):
        from fusion_mlx.middleware import (
            install_exception_handlers,
            install_request_body_depth_middleware,
            install_request_body_limit_middleware,
        )
        assert callable(install_exception_handlers)
        assert callable(install_request_body_depth_middleware)
        assert callable(install_request_body_limit_middleware)

    def test_auth_exports_available(self):
        from fusion_mlx.middleware import (
            verify_api_key,
            verify_api_key_or_x_api_key,
            check_rate_limit,
            check_rate_limit_or_x_api_key,
            rate_limiter,
            configure_rate_limiter,
            RateLimiter,
        )
        assert callable(verify_api_key)
        assert callable(verify_api_key_or_x_api_key)
        assert callable(check_rate_limit)
        assert callable(check_rate_limit_or_x_api_key)
        assert callable(configure_rate_limiter)
        assert isinstance(rate_limiter, RateLimiter)
