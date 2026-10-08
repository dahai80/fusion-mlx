# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1000.

#1000: v0.7.0+ made route-guard enforce the default (#349 Phase 2), so a
standalone ``fusion-mlx serve`` (direct OpenAI-client connection, no gateway)
rejected every ``/v1/*`` request with ``403 missing_route``. The #398 fix
only set ``FUSION_ROUTE_WARN_ONLY=true`` in ``start.sh`` preflight — the CLI
``serve`` path and direct ``create_app`` callers bypassed it.

Fix (#1000): enforce is now meaningful ONLY when a gateway/tenant contract
exists to validate provenance against (``FUSION_ROUTE_TOKEN`` /
``FUSION_TENANT_ISOLATION`` / explicit ``FUSION_ROUTE_ENFORCE=true``). With
none set, the server is standalone and defaults to warn-only — direct
clients pass. The 403 body now carries a ``FUSION_ROUTE_WARN_ONLY=true``
hint, and the active mode is logged at install time.
"""

from __future__ import annotations

import logging

import pytest
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


def _clear_route_env(monkeypatch):
    monkeypatch.delenv("FUSION_ROUTE_WARN_ONLY", raising=False)
    monkeypatch.delenv("FUSION_ROUTE_ENFORCE", raising=False)
    monkeypatch.delenv("FUSION_ROUTE_TOKEN", raising=False)
    monkeypatch.delenv("FUSION_TENANT_ISOLATION", raising=False)


class TestStandaloneDefaultsToWarnOnly:
    # #1000 core: no gateway contract -> warn-only -> direct client passes.
    def test_no_env_standalone_passes(self, monkeypatch):
        _clear_route_env(monkeypatch)
        client = TestClient(_guard_app())
        r = client.get("/v1/chat/completions")
        assert r.status_code == 200, r.text

    def test_no_env_standalone_passes_images(self, monkeypatch):
        _clear_route_env(monkeypatch)
        client = TestClient(_guard_app())
        r = client.post("/v1/images/generations")
        assert r.status_code == 200, r.text

    def test_warn_only_explicit_still_passes(self, monkeypatch):
        _clear_route_env(monkeypatch)
        monkeypatch.setenv("FUSION_ROUTE_WARN_ONLY", "true")
        client = TestClient(_guard_app())
        r = client.get("/v1/chat/completions")
        assert r.status_code == 200


class TestGatewaySignalsEnforce:
    # When a gateway contract IS configured, enforce stays on (403 on
    # missing header) — the security posture is preserved for gateway/multi-
    # tenant deployments.
    def test_token_enforces_missing_header_403(self, monkeypatch):
        _clear_route_env(monkeypatch)
        monkeypatch.setenv("FUSION_ROUTE_TOKEN", "s3cr3t")
        client = TestClient(_guard_app())
        r = client.get("/v1/chat/completions")
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "invalid_route_token"

    def test_tenant_isolation_enforces_missing_header_403(self, monkeypatch):
        # Isolation on + no header: enforce fires (missing_route) — proves
        # the tenant-isolation gateway signal activates enforce mode. The
        # missing_tenant path (header present, tenant absent) is covered in
        # test_tenant_isolation.py.
        _clear_route_env(monkeypatch)
        monkeypatch.setenv("FUSION_TENANT_ISOLATION", "true")
        client = TestClient(_guard_app())
        r = client.get("/v1/chat/completions")
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "missing_route"

    def test_explicit_enforce_rejects_missing_header(self, monkeypatch):
        _clear_route_env(monkeypatch)
        monkeypatch.setenv("FUSION_ROUTE_ENFORCE", "true")
        client = TestClient(_guard_app())
        r = client.get("/v1/chat/completions")
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "missing_route"

    def test_explicit_enforce_allows_with_header(self, monkeypatch):
        _clear_route_env(monkeypatch)
        monkeypatch.setenv("FUSION_ROUTE_ENFORCE", "true")
        client = TestClient(_guard_app())
        r = client.get("/v1/chat/completions", headers={"X-Fusion-Route": "gw"})
        assert r.status_code == 200

    def test_warn_only_overrides_explicit_enforce(self, monkeypatch):
        # WARN_ONLY wins even when ENFORCE is set — explicit opt-out.
        _clear_route_env(monkeypatch)
        monkeypatch.setenv("FUSION_ROUTE_ENFORCE", "true")
        monkeypatch.setenv("FUSION_ROUTE_WARN_ONLY", "true")
        client = TestClient(_guard_app())
        r = client.get("/v1/chat/completions")
        assert r.status_code == 200


class TestMissingRouteBodyHint:
    # #1000: the 403 body must surface the FUSION_ROUTE_WARN_ONLY knob so a
    # surprised operator knows how to recover without reading docs.
    def test_403_body_includes_warn_only_hint(self, monkeypatch):
        _clear_route_env(monkeypatch)
        monkeypatch.setenv("FUSION_ROUTE_ENFORCE", "true")
        client = TestClient(_guard_app())
        r = client.get("/v1/chat/completions")
        assert r.status_code == 403
        body = r.json()["error"]
        assert body["code"] == "missing_route"
        assert "FUSION_ROUTE_WARN_ONLY=true" in body["message"], body["message"]
        assert body.get("hint") == "FUSION_ROUTE_WARN_ONLY=true", body


class TestInstallLogsMode:
    # #1000: install_route_guard_middleware logs the active mode so operators
    # see at startup whether direct clients will be rejected.
    def test_standalone_logs_warn_only(self, monkeypatch, caplog):
        _clear_route_env(monkeypatch)
        with caplog.at_level(logging.INFO, logger="fusion_mlx.middleware.route_guard"):
            _guard_app()
        assert any(
            "standalone mode" in r.message and "warn-only" in r.message
            for r in caplog.records
        ), [r.message for r in caplog.records]

    def test_enforce_logs_gateway_signals(self, monkeypatch, caplog):
        _clear_route_env(monkeypatch)
        monkeypatch.setenv("FUSION_ROUTE_TOKEN", "t")
        with caplog.at_level(logging.INFO, logger="fusion_mlx.middleware.route_guard"):
            _guard_app()
        assert any(
            "enforce mode active" in r.message and "FUSION_ROUTE_TOKEN" in r.message
            for r in caplog.records
        ), [r.message for r in caplog.records]

    def test_warn_only_env_logs_warn_only(self, monkeypatch, caplog):
        _clear_route_env(monkeypatch)
        monkeypatch.setenv("FUSION_ROUTE_WARN_ONLY", "true")
        with caplog.at_level(logging.INFO, logger="fusion_mlx.middleware.route_guard"):
            _guard_app()
        assert any("warn-only mode" in r.message for r in caplog.records), [
            r.message for r in caplog.records
        ]


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
