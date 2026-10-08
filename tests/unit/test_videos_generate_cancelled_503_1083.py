# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1083.

#1083: generate_video didn't catch asyncio.CancelledError (BaseException in
3.8+, not Exception). When the ProcessMemoryEnforcer aborted a video job
under memory pressure, CancelledError bypassed every except-Exception handler
and propagated to ASGI → 500 with no detail + an ASGI stack trace.
Fix: an `except asyncio.CancelledError` handler returns a structured 503.
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import fusion_mlx.api.videos_routes as vr


class _FakeVideoGenEngine:
    """Minimal stand-in for VideoGenEngine — isinstance check passes after
    patching videos_routes.VideoGenEngine to this class."""

    def __init__(self, exc: BaseException | None = None):
        self._exc = exc

    async def generate(self, **kwargs):
        if self._exc is not None:
            raise self._exc
        return [b""]


def _wire(monkeypatch, engine_exc):
    fake_engine = _FakeVideoGenEngine(engine_exc)
    fake_pool = MagicMock()
    fake_pool.get_engine = AsyncMock(return_value=fake_engine)
    monkeypatch.setattr(vr, "_pool", fake_pool)
    monkeypatch.setattr(vr, "VideoGenEngine", _FakeVideoGenEngine)
    monkeypatch.setattr(vr, "constraints_for", lambda m: MagicMock(dim_divisibility=8))
    monkeypatch.setattr(vr, "validate_params", lambda *a, **k: None)
    app = FastAPI()
    app.include_router(vr.router)
    app.dependency_overrides[vr.verify_api_key] = lambda: True
    app.dependency_overrides[vr.check_rate_limit] = lambda: True
    return TestClient(app)


class TestVideoGenerateCancelled1083:
    """CancelledError → structured 503, not ASGI 500."""

    def test_source_has_cancelled_handler(self):
        src = inspect.getsource(vr.generate_video)
        assert "except asyncio.CancelledError" in src
        # 503 + memory-guard detail + Retry-After
        assert "503" in src
        assert "memory guard" in src.lower()

    def test_cancelled_returns_503(self, monkeypatch):
        client = _wire(monkeypatch, asyncio.CancelledError())
        resp = client.post(
            "/v1/videos/generate", json={"prompt": "test", "model": "ltx2_5"}
        )
        assert resp.status_code == 503
        detail = resp.json()["detail"].lower()
        assert "memory guard" in detail or "cancelled" in detail
        assert resp.headers.get("retry-after") == "10"

    def test_generic_exception_still_500(self, monkeypatch):
        # a non-Cancelled exception must still hit the generic 500 path,
        # not be shadowed by the CancelledError handler.
        client = _wire(monkeypatch, RuntimeError("boom"))
        resp = client.post(
            "/v1/videos/generate", json={"prompt": "test", "model": "ltx2_5"}
        )
        assert resp.status_code == 500

    def test_timeout_returns_507_or_503(self, monkeypatch):
        # asyncio.TimeoutError is an Exception subclass (not CancelledError),
        # so it hits the generic except-Exception path → 500/503/507, NOT ASGI.
        # The point: it must not escape to ASGI as an unhandled CancelledError.
        client = _wire(monkeypatch, TimeoutError())
        resp = client.post(
            "/v1/videos/generate", json={"prompt": "test", "model": "ltx2_5"}
        )
        assert resp.status_code in (500, 503, 507)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
