# SPDX-License-Identifier: Apache-2.0
# #756: multi-tenant isolation in RouteGuardMiddleware + tenant helpers.
#
# When FUSION_TENANT_ISOLATION=on, the backend half of the gateway contract:
#   - X-Fusion-Route VALUE must equal "gateway-decision" (not just present),
#     else 403 invalid_route_origin.
#   - X-Fusion-Tenant must be present + non-empty, else 403 missing_tenant.
# FUSION_ROUTE_TOKEN (shared secret) is stricter and takes precedence.
# Default OFF: single-tenant dev keeps the legacy presence-check behavior.

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.middleware.tenant import (
    GATEWAY_DECISION_ROUTE,
    scoped_principal,
    tenant_from_request,
    tenant_isolation_enabled,
)


def _guard_app() -> FastAPI:
    from fusion_mlx.middleware.route_guard import install_route_guard_middleware

    app = FastAPI()

    @app.api_route("/{p:path}", methods=["GET", "POST", "DELETE", "OPTIONS"])
    async def _catch_all(p: str):
        return {"ok": True}

    install_route_guard_middleware(app)
    return app


def _client(monkeypatch, isolation: bool, token: str | None = None) -> TestClient:
    monkeypatch.delenv("FUSION_ROUTE_WARN_ONLY", raising=False)
    monkeypatch.delenv("FUSION_ROUTE_ENFORCE", raising=False)
    monkeypatch.delenv("FUSION_ROUTE_TOKEN", raising=False)
    monkeypatch.delenv("FUSION_TENANT_ISOLATION", raising=False)
    if isolation:
        monkeypatch.setenv("FUSION_TENANT_ISOLATION", "true")
    if token is not None:
        monkeypatch.setenv("FUSION_ROUTE_TOKEN", token)
    return TestClient(_guard_app())


class TestRouteGuardTenantIsolation:
    def test_isolation_off_legacy_presence_check(self, monkeypatch):
        client = _client(monkeypatch, isolation=False)
        r = client.get("/v1/chat/completions", headers={"X-Fusion-Route": "anything"})
        assert r.status_code == 200

    def test_isolation_on_wrong_route_value_rejected(self, monkeypatch):
        client = _client(monkeypatch, isolation=True)
        r = client.get(
            "/v1/chat/completions",
            headers={"X-Fusion-Route": "not-gateway"},
        )
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "invalid_route_origin"

    def test_isolation_on_gateway_route_without_tenant_rejected(self, monkeypatch):
        client = _client(monkeypatch, isolation=True)
        r = client.get(
            "/v1/chat/completions",
            headers={"X-Fusion-Route": GATEWAY_DECISION_ROUTE},
        )
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "missing_tenant"

    def test_isolation_on_empty_tenant_rejected(self, monkeypatch):
        client = _client(monkeypatch, isolation=True)
        r = client.get(
            "/v1/chat/completions",
            headers={
                "X-Fusion-Route": GATEWAY_DECISION_ROUTE,
                "X-Fusion-Tenant": "   ",
            },
        )
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "missing_tenant"

    def test_isolation_on_gateway_route_with_tenant_passes(self, monkeypatch):
        client = _client(monkeypatch, isolation=True)
        r = client.get(
            "/v1/chat/completions",
            headers={
                "X-Fusion-Route": GATEWAY_DECISION_ROUTE,
                "X-Fusion-Tenant": "team-alpha",
            },
        )
        assert r.status_code == 200

    def test_token_takes_precedence_over_isolation(self, monkeypatch):
        # FUSION_ROUTE_TOKEN is stricter; its branch runs first and does not
        # enforce the gateway-decision value or tenant presence.
        client = _client(monkeypatch, isolation=True, token="s3cret")
        # wrong route value but matching token -> 200 (token wins)
        r = client.get("/v1/chat/completions", headers={"X-Fusion-Route": "s3cret"})
        assert r.status_code == 200

    def test_token_missing_still_rejected_under_isolation(self, monkeypatch):
        client = _client(monkeypatch, isolation=True, token="s3cret")
        r = client.get("/v1/chat/completions")
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "invalid_route_token"

    def test_exempt_path_passes_without_tenant(self, monkeypatch):
        client = _client(monkeypatch, isolation=True)
        r = client.get("/health")
        assert r.status_code != 403

    def test_options_preflight_exempt(self, monkeypatch):
        client = _client(monkeypatch, isolation=True)
        r = client.options("/v1/chat/completions")
        assert r.status_code != 403


class _CIHeaders(dict):
    # Starlette lowercases header keys on lookup; mirror that so plain-dict
    # fakes in these tests match real Request.headers.get() semantics.
    def get(self, key, default=None):
        return super().get(
            key, super().get(key.lower(), super().get(key.title(), default))
        )


class _Req:
    def __init__(self, headers, client_host="10.0.0.1"):
        self.headers = _CIHeaders(headers)

        class _C:
            host = client_host

        self.client = _C()


class TestTenantHelpers:
    def test_isolation_enabled_env_gate(self, monkeypatch):
        monkeypatch.delenv("FUSION_TENANT_ISOLATION", raising=False)
        assert tenant_isolation_enabled() is False
        monkeypatch.setenv("FUSION_TENANT_ISOLATION", "true")
        assert tenant_isolation_enabled() is True
        monkeypatch.setenv("FUSION_TENANT_ISOLATION", "0")
        assert tenant_isolation_enabled() is False

    def test_tenant_from_request_reads_header(self):
        assert tenant_from_request(_Req({"X-Fusion-Tenant": "team-a"})) == "team-a"
        assert tenant_from_request(_Req({"X-Fusion-Tenant": "  team-a  "})) == "team-a"
        assert tenant_from_request(_Req({"X-Fusion-Tenant": ""})) is None
        assert tenant_from_request(_Req({})) is None

    def test_tenant_from_request_ignores_space_id(self):
        # X-Space-Id is non-authoritative passthrough; must not influence tenant.
        req = _Req({"X-Space-Id": "spoof-tenant", "X-Fusion-Tenant": "real-tenant"})
        assert tenant_from_request(req) == "real-tenant"
        # no authoritative header at all -> None even if space-id present
        assert tenant_from_request(_Req({"X-Space-Id": "spoof"})) is None

    def test_scoped_principal_namespaces_when_isolation_on(self, monkeypatch):
        monkeypatch.setenv("FUSION_TENANT_ISOLATION", "true")
        req = _Req({"X-Fusion-Tenant": "team-a"})
        assert scoped_principal(req, "bucket-abc") == "t:team-a:bucket-abc"

    def test_scoped_principal_passthrough_when_isolation_off(self, monkeypatch):
        monkeypatch.delenv("FUSION_TENANT_ISOLATION", raising=False)
        req = _Req({"X-Fusion-Tenant": "team-a"})
        assert scoped_principal(req, "bucket-abc") == "bucket-abc"

    def test_scoped_principal_passthrough_when_no_tenant(self, monkeypatch):
        monkeypatch.setenv("FUSION_TENANT_ISOLATION", "true")
        assert scoped_principal(_Req({}), "bucket-abc") == "bucket-abc"

    def test_scoped_principal_different_tenants_isolated(self, monkeypatch):
        monkeypatch.setenv("FUSION_TENANT_ISOLATION", "true")
        a = scoped_principal(_Req({"X-Fusion-Tenant": "alpha"}), "bucket-x")
        b = scoped_principal(_Req({"X-Fusion-Tenant": "beta"}), "bucket-x")
        assert a != b
        assert a == "t:alpha:bucket-x"
        assert b == "t:beta:bucket-x"


class TestSessionPrincipalScoping:
    # #756: session_routes._session_principal composes tenant into the
    # per-caller principal so two tenants sharing a bearer-key shape cannot
    # reach each other's sessions/stats/context caps.

    def test_session_principal_namespaces_under_isolation(self, monkeypatch):
        monkeypatch.setenv("FUSION_TENANT_ISOLATION", "true")
        from fusion_mlx.api.session_routes import _session_principal

        req = _Req({"X-Fusion-Tenant": "team-a"})
        principal = _session_principal(req)
        assert principal.startswith("t:team-a:")


class TestSessionEnforcedViaRealRouter:
    # End-to-end through the actual /v1/sessions router: under isolation a
    # foreign-tenant caller gets a different scoped principal and a 404 for a
    # session owned by another tenant (IDOR stays enforced cross-tenant).

    def test_foreign_tenant_session_returns_404(self, monkeypatch, tmp_path):
        import importlib
        import os

        monkeypatch.setenv("FUSION_TENANT_ISOLATION", "true")
        monkeypatch.delenv("FUSION_ROUTE_WARN_ONLY", raising=False)
        monkeypatch.delenv("FUSION_ROUTE_ENFORCE", raising=False)
        monkeypatch.delenv("FUSION_ROUTE_TOKEN", raising=False)

        # Use the in-memory session tracker so no real model is needed.
        from fusion_mlx.api import session_routes
        from fusion_mlx.sessions import (
            get_session_tracker,
            reset_session_tracker_for_tests,
        )

        importlib.reload(session_routes)

        reset_session_tracker_for_tests()
        tracker = get_session_tracker()

        app = FastAPI()
        app.include_router(session_routes.router)
        from fusion_mlx.middleware.route_guard import install_route_guard_middleware

        install_route_guard_middleware(app)

        # Seed a session for tenant "alpha" with a known principal.
        # _session_principal under isolation = "t:alpha:<base>". Use the
        # tracker API directly with the composed principal.
        base = "bucket-seed"
        alpha_principal = f"t:alpha:{base}"
        tracker.record(
            "sess-shared",
            prompt_tokens=10,
            completion_tokens=5,
            principal=alpha_principal,
        )

        client = TestClient(app)
        # Tenant "beta" reading the alpha-owned session -> different principal
        # -> tracker.get returns None -> 404 (not a cross-tenant leak).
        r = client.get(
            "/v1/sessions/sess-shared/stats",
            headers={
                "X-Fusion-Route": GATEWAY_DECISION_ROUTE,
                "X-Fusion-Tenant": "beta",
                "X-Fusion-Api-Key": os.environ.get("FUSION_API_KEY", "test-key"),
            },
        )
        assert r.status_code == 404
