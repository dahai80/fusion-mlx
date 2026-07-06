# SPDX-License-Identifier: Apache-2.0
"""ASGI fast-path for k8s liveness probes.

Short-circuits GET /healthz and GET /livez before Starlette routing,
dependency resolution, and response serialization run. Under streaming
load this drops probe p99 from "queued behind SSE chunk emissions" to
"one extra coroutine yield".

Constraints:
- Installed AFTER all other middleware so it runs OUTERMOST
- GET-only, path-equal-only (HEAD falls through for auto-derivation)
- Response shape matches routes/health.py byte-for-byte
- Cross-origin requests (with Origin header) fall through for CORS
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..config import get_config

logger = logging.getLogger(__name__)

_FAST_PATHS: frozenset[bytes] = frozenset({b"/healthz", b"/livez"})

_BASE_HEADERS: list[tuple[bytes, bytes]] = [
    (b"content-type", b"application/json"),
]


def _build_healthz_payload() -> tuple[int, bytes]:
    cfg = get_config()
    if getattr(cfg, "draining", False):
        payload = {
            "status": "draining",
            "ready": False,
            "model_loaded": getattr(cfg, "engine", None) is not None,
            "model_name": getattr(cfg, "model_name", ""),
        }
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return 503, body
    payload = {
        "status": "healthy",
        "ready": bool(getattr(cfg, "ready", False)),
        "model_loaded": getattr(cfg, "engine", None) is not None,
        "model_name": getattr(cfg, "model_name", ""),
    }
    return 200, json.dumps(payload, separators=(",", ":")).encode("utf-8")


_LIVEZ_BODY: bytes = b'{"status":"alive"}'


def _has_origin(scope: dict[str, Any]) -> bool:
    for name, _value in scope.get("headers", ()):
        if name.lower() == b"origin":
            return True
    return False


class ProbeFastPathMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        method = scope.get("method")
        if method != "GET":
            return await self.app(scope, receive, send)

        raw_path = scope.get("raw_path")
        if raw_path is None:
            path_str = scope.get("path") or ""
            raw_path = path_str.encode("ascii", "replace")

        qmark = raw_path.find(b"?")
        if qmark != -1:
            raw_path = raw_path[:qmark]

        if raw_path not in _FAST_PATHS:
            return await self.app(scope, receive, send)

        if _has_origin(scope):
            return await self.app(scope, receive, send)

        if raw_path == b"/livez":
            status_code = 200
            body = _LIVEZ_BODY
        else:
            try:
                status_code, body = _build_healthz_payload()
            except Exception:
                logger.debug("[probe_fastpath] payload build raised; falling through")
                return await self.app(scope, receive, send)

        headers = _BASE_HEADERS + [
            (b"content-length", str(len(body)).encode("ascii")),
        ]

        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": headers,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            }
        )


def install_probe_fastpath_middleware(app: Any) -> None:
    app.add_middleware(ProbeFastPathMiddleware)
