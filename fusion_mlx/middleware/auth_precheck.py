# SPDX-License-Identifier: Apache-2.0
"""ASGI auth pre-check — rejects unauthenticated body-bearing requests
BEFORE body_size/body_depth buffers the body (ENG-08, #0909 audit).

Auth in fusion-mlx is primarily a FastAPI ``Depends()`` that runs after
route matching — i.e. after ASGI middleware has already read the entire
request body (up to the 8 MiB cap). An attacker can send POST
``/v1/chat/completions`` with a bad key + 7.9 MiB body repeatedly; the
server buffers 7.9 MiB each time before returning 401, a memory DoS
vector.

This middleware performs a **header-only** auth pre-check on guarded
paths with body-bearing methods. It validates the Authorization /
x-api-key header against the configured key (or FUSION_ALLOW_ANONYMOUS)
without touching the body. If the pre-check fails, 401 is emitted
immediately. The full ``Depends()`` auth (rate limit, scoped keys, etc.)
still runs downstream — this is a cheap early filter, not a replacement.

Installed between body_limit and request_id so:
- body is NOT read on rejection (runs outer to body_limit)
- request_id IS stamped (runs inner to request_id) so 401 carries a
  correlation ID
"""

from __future__ import annotations

import json as _json
import logging
from typing import Any

from .auth import (
    _anonymous_access_allowed,
    _extract_bearer_token,
    _get_configured_api_key,
)

logger = logging.getLogger(__name__)

_BODY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Mirror body_size.py guarded prefixes — only pre-check paths that will
# buffer a body.
_GUARDED_PREFIXES = (
    "/v1/",
    "/internal/",
    "/anthropic/",
    "/distributed/",
    "/admin/",
    "/rpc",
    "/stats",
)

# Exclude paths that have their own auth (e.g. admin login) or are
# intentionally public.
_EXCLUDED_PATHS = frozenset({"/v1/audio/transcriptions"})


def _path_is_guarded(path: str | None) -> bool:
    if not path:
        return False
    if path in _EXCLUDED_PATHS:
        return False
    return any(path.startswith(prefix) for prefix in _GUARDED_PREFIXES)


def _extract_headers(scope: dict[str, Any]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for raw_name, raw_value in scope.get("headers", ()):
        try:
            headers[raw_name.decode("latin-1").lower()] = raw_value.decode("latin-1")
        except (UnicodeDecodeError, ValueError):
            continue
    return headers


def _auth_passes_precheck(scope: dict[str, Any]) -> bool:
    """Header-only auth check. True if request should proceed to body buffering."""
    import secrets

    configured_key = _get_configured_api_key()
    if configured_key is None:
        # No key configured — anonymous gate decides.
        if _anonymous_access_allowed(None):
            return True
        return False

    headers = _extract_headers(scope)
    bearer = _extract_bearer_token(headers.get("authorization"))
    x_api_key = headers.get("x-api-key")
    provided = [k for k in (bearer, x_api_key) if k]
    if not provided:
        return False
    return all(secrets.compare_digest(k, configured_key) for k in provided)


class AuthPrecheckMiddleware:
    """Reject unauthenticated body-bearing requests before body buffering."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        if scope.get("method") not in _BODY_METHODS:
            return await self.app(scope, receive, send)

        path = scope.get("path")
        if not _path_is_guarded(path):
            return await self.app(scope, receive, send)

        if _auth_passes_precheck(scope):
            return await self.app(scope, receive, send)

        # Auth failed — emit 401 WITHOUT reading the body.
        logger.info(
            "Auth pre-check rejected (ENG-08): 401 before body buffering "
            "path=%s method=%s",
            path,
            scope.get("method"),
        )
        body = _json.dumps(
            {
                "error": {
                    "message": "Authentication required.",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                    "param": None,
                }
            }
        ).encode("utf-8")
        try:
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (
                            b"content-length",
                            str(len(body)).encode("ascii"),
                        ),
                        (b"connection", b"close"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body, "more_body": False})
        except Exception:
            logger.debug("auth pre-check 401 send failed (client already disconnected)")

        # Drain the request body so the client can receive our response
        # cleanly (some clients hang if the server responds before
        # consuming the body). Best-effort — if the client already
        # disconnected, this is a no-op.
        try:
            while True:
                msg = await receive()
                if msg.get("type") == "http.request" and not msg.get(
                    "more_body", False
                ):
                    break
        except Exception:
            pass


def install_auth_precheck_middleware(app: Any) -> None:
    logger.info("Installing AuthPrecheckMiddleware (ENG-08)")
    app.add_middleware(AuthPrecheckMiddleware)
