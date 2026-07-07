# SPDX-License-Identifier: Apache-2.0
"""ASGI middleware that stamps a per-request ID into the logging context.

Sets the ``request_id`` ContextVar (from ``fusion_mlx.logging_config``) for
the duration of each request so log records emitted by handlers/middleware
downstream carry the same correlation ID. Implemented as a pure ASGI
middleware (not Starlette ``BaseHTTPMiddleware``) so no extra anyio task is
spawned — that would break ContextVar propagation into the handler.

The request ID is taken from the inbound ``X-Request-Id`` header when present
(otherwise a short uuid4 prefix is generated) and echoed back on the response
as ``X-Request-Id`` for client-side correlation.
"""

from __future__ import annotations

import uuid
from typing import Any

from ..logging_config import _request_id

_REQUEST_ID_HEADER = b"x-request-id"


class RequestIdMiddleware:
    """Pure ASGI middleware — stamps request_id around the call chain."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = scope.get("headers") or []
        inbound_id: str | None = None
        for name, value in headers:
            if name == _REQUEST_ID_HEADER:
                try:
                    inbound_id = value.decode("ascii", errors="replace").strip() or None
                except Exception:
                    inbound_id = None
                break

        request_id = inbound_id or uuid.uuid4().hex[:12]
        token = _request_id.set(request_id)

        async def _send(message):
            if message.get("type") == "http.response.start":
                resp_headers = message.setdefault("headers", [])
                resp_headers.append((_REQUEST_ID_HEADER, request_id.encode("ascii")))
            await send(message)

        try:
            await self.app(scope, receive, _send)
        finally:
            _request_id.reset(token)


def install_request_id_middleware(app: Any) -> None:
    """Install RequestIdMiddleware on a Starlette/FastAPI app."""
    app.add_middleware(RequestIdMiddleware)
