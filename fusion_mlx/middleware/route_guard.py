# SPDX-License-Identifier: Apache-2.0
"""ASGI source-validation middleware for the fusion-mlx link endpoint.

Issue #343: the MLX link endpoint must verify that requests arrive through
the trusted gateway chain (App -> Gateway -> MLX) by checking the
``X-Fusion-Route`` header injected by the gateway.

Two phases:
- Phase 1 (default): missing header -> WARN + allow (observability rollout).
- Phase 2 (``FUSION_ROUTE_ENFORCE=true``): missing header -> 403 reject.

Probe paths (/health, /healthz, /readyz, /livez, /) and CORS preflight
(OPTIONS) are exempt so k8s/load-balancer health checks and browser
preflights keep working without the header.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

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


def _route_enforce_enabled() -> bool:
    return os.environ.get("FUSION_ROUTE_ENFORCE", "").strip().lower() in _TRUTHY


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
        if route:
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
