# SPDX-License-Identifier: Apache-2.0
"""ASGI source-validation middleware for the fusion-mlx link endpoint.

Issue #343: the MLX link endpoint must verify that requests arrive through
the trusted gateway chain (App -> Gateway -> MLX) by checking the
``X-Fusion-Route`` header injected by the gateway.

Two phases:
- Phase 1 (``FUSION_ROUTE_WARN_ONLY=true``): missing header -> WARN + allow
  (dev/standalone fallback for deployments without a gateway).
- Phase 2 (default): missing header -> 403 reject.

#349: enforce is the default since v0.7.0. Deployments running fusion-mlx
standalone (no gateway injecting X-Fusion-Route) set
``FUSION_ROUTE_WARN_ONLY=true`` to keep the previous warn-only behavior.
``FUSION_ROUTE_ENFORCE=true`` remains accepted as an explicit opt-in
(redundant with the new default, kept for backward compatibility).

Probe paths (/health, /healthz, /readyz, /livez, /) and CORS preflight
(OPTIONS) are exempt so k8s/load-balancer health checks and browser
preflights keep working without the header.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# #756: tenant-isolation contract. Imported lazily inside helpers to keep
# the module importable standalone (no FastAPI Request dep at import time).
_GATEWAY_DECISION_ROUTE = "gateway-decision"

_EXEMPT_PATHS: frozenset[bytes] = frozenset(
    {
        b"/",
        b"/health",
        b"/healthz",
        b"/readyz",
        b"/livez",
        b"/openapi.json",
        b"/docs",
        b"/redoc",
        b"/favicon.ico",
    }
)

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _warn_only_enabled() -> bool:
    return os.environ.get("FUSION_ROUTE_WARN_ONLY", "").strip().lower() in _TRUTHY


def _route_enforce_enabled() -> bool:
    # #349: Phase 2 enforce is now the DEFAULT. Standalone/dev deployments
    # without a gateway set FUSION_ROUTE_WARN_ONLY=true to restore the prior
    # warn-only (Phase 1) behavior. FUSION_ROUTE_ENFORCE=true remains accepted
    # as an explicit opt-in (redundant with the new default).
    if _warn_only_enabled():
        return False
    return True


def _configured_route_token() -> str | None:
    # #352: optional shared secret for cross-host gateway->MLX auth. When set,
    # X-Fusion-Route is upgraded from spoofable provenance to a credential
    # validated with hmac.compare_digest. Empty/unset = feature OFF (current
    # presence-check behavior). Env-only, like the other FUSION_ROUTE_* vars.
    token = os.environ.get("FUSION_ROUTE_TOKEN", "").strip()
    return token or None


def _tenant_isolation_enabled() -> bool:
    # #756: multi-tenant mode. Default OFF (single-tenant dev unaffected).
    # When on, a bare X-Fusion-Route presence check is insufficient — the
    # header VALUE must equal the gateway's contract constant
    # (gateway-decision) so a direct-port caller cannot inject an
    # arbitrary value. FUSION_ROUTE_TOKEN (stricter secret) takes
    # precedence when set.
    return os.environ.get("FUSION_TENANT_ISOLATION", "").strip().lower() in _TRUTHY


def _raw_path(scope: dict[str, Any]) -> bytes:
    raw_path = scope.get("raw_path")
    if raw_path is None:
        path_str = scope.get("path") or ""
        raw_path = path_str.encode("ascii", "replace")
    qmark = raw_path.find(b"?")
    if qmark != -1:
        raw_path = raw_path[:qmark]
    return raw_path


def _header_value(scope: dict[str, Any], name: str) -> str | None:
    target = name.encode("ascii").lower()
    for key, value in scope.get("headers", ()):
        if key.lower() == target:
            try:
                return value.decode("latin-1")
            except Exception:
                return None
    return None


def _client_host(scope: dict[str, Any]) -> str:
    client = scope.get("client")
    if client and isinstance(client, (list, tuple)) and client:
        return str(client[0])
    return "unknown"


class RouteGuardMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        if scope.get("method") == "OPTIONS":
            return await self.app(scope, receive, send)

        path = _raw_path(scope)
        if path in _EXEMPT_PATHS:
            return await self.app(scope, receive, send)

        route = _header_value(scope, "x-fusion-route")
        token = _configured_route_token()
        if token:
            # #352: X-Fusion-Route carries the shared secret. Constant-time
            # compare; missing/mismatched -> 403 invalid_route_token. This is
            # stricter than the presence check below, so it takes precedence.
            if not route or not hmac.compare_digest(route, token):
                logger.warning(
                    "[route_guard] rejected: invalid X-Fusion-Route token "
                    "host=%s path=%s",
                    _client_host(scope),
                    path.decode("ascii", "replace"),
                )
                body = json.dumps(
                    {
                        "error": {
                            "message": "Invalid X-Fusion-Route token",
                            "code": "invalid_route_token",
                        }
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
                await send(
                    {
                        "type": "http.response.start",
                        "status": 403,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode("ascii")),
                        ],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": body,
                        "more_body": False,
                    }
                )
                return
            return await self.app(scope, receive, send)

        if route:
            # #756: tenant-isolation mode requires the gateway contract
            # value (gateway-decision), not just any header presence. A
            # direct-port caller could otherwise inject an arbitrary
            # X-Fusion-Route value. Skipped when a shared-secret token is
            # configured (handled above with a stricter compare) or when
            # isolation is off (single-tenant dev).
            if _tenant_isolation_enabled() and route != _GATEWAY_DECISION_ROUTE:
                logger.warning(
                    "[route_guard] rejected: non-gateway X-Fusion-Route value "
                    "under tenant isolation host=%s path=%s value=%s",
                    _client_host(scope),
                    path.decode("ascii", "replace"),
                    route,
                )
                body = json.dumps(
                    {
                        "error": {
                            "message": "X-Fusion-Route must be gateway-decision "
                            "under tenant isolation",
                            "code": "invalid_route_origin",
                        }
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
                await send(
                    {
                        "type": "http.response.start",
                        "status": 403,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode("ascii")),
                        ],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": body,
                        "more_body": False,
                    }
                )
                return
            # #756: gateway stamps X-Fusion-Tenant on every upstream
            # request. Under isolation, a valid route without a tenant is
            # a misconfigured/bypassed gateway and must not reach the
            # handler (no tenant = unscoped state access).
            if _tenant_isolation_enabled():
                tenant = _header_value(scope, "x-fusion-tenant")
                if not tenant or not tenant.strip():
                    logger.warning(
                        "[route_guard] rejected: missing X-Fusion-Tenant "
                        "under tenant isolation host=%s path=%s",
                        _client_host(scope),
                        path.decode("ascii", "replace"),
                    )
                    body = json.dumps(
                        {
                            "error": {
                                "message": "Missing X-Fusion-Tenant header "
                                "under tenant isolation",
                                "code": "missing_tenant",
                            }
                        },
                        separators=(",", ":"),
                    ).encode("utf-8")
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 403,
                            "headers": [
                                (b"content-type", b"application/json"),
                                (
                                    b"content-length",
                                    str(len(body)).encode("ascii"),
                                ),
                            ],
                        }
                    )
                    await send(
                        {
                            "type": "http.response.body",
                            "body": body,
                            "more_body": False,
                        }
                    )
                    return
            return await self.app(scope, receive, send)

        if _route_enforce_enabled():
            logger.warning(
                "[route_guard] rejected: missing X-Fusion-Route host=%s path=%s",
                _client_host(scope),
                path.decode("ascii", "replace"),
            )
            body = json.dumps(
                {
                    "error": {
                        "message": "Missing X-Fusion-Route header",
                        "code": "missing_route",
                    }
                },
                separators=(",", ":"),
            ).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 403,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": body,
                    "more_body": False,
                }
            )
            return

        logger.warning(
            "[route_guard] missing X-Fusion-Route (warn-only) host=%s path=%s",
            _client_host(scope),
            path.decode("ascii", "replace"),
        )
        return await self.app(scope, receive, send)


def install_route_guard_middleware(app: Any) -> None:
    app.add_middleware(RouteGuardMiddleware)
