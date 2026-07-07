# SPDX-License-Identifier: Apache-2.0
"""Request-body JSON nesting-depth cap (D-DEEP-JSON DoS defense).

ASGI middleware that caps JSON nesting depth BEFORE Pydantic recurses
over the body. A deeply-nested payload (~10 KB, 1000 levels) would
blow the Python recursion limit inside Pydantic and surface as HTTP 500.

The cap is read from the ``FUSION_MLX_MAX_BODY_DEPTH`` env var at
request time (default 64). Only ``/v1/...``, ``/internal/...``, and
``/anthropic/...`` POST/PUT/PATCH/DELETE are gated.
"""

from __future__ import annotations

import json as _json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_GUARDED_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_GUARDED_PREFIXES = ("/v1/", "/internal/", "/anthropic/")
_EXCLUDED_PATHS = frozenset({"/v1/audio/transcriptions"})

_JSON_CONTENT_TYPE_OPTIONAL_PATHS = frozenset(
    {
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/responses",
        "/anthropic/v1/messages",
    }
)

DEFAULT_MAX_BODY_DEPTH = 64
MAX_BODY_DEPTH_ENV_PRIMARY = "FUSION_MLX_MAX_BODY_DEPTH"
MAX_BODY_DEPTH_ENV_LEGACY = "RAPID_MLX_MAX_BODY_DEPTH"

_legacy_warned: set[str] = set()


def _resolve_max_body_depth() -> int:
    raw = os.environ.get(MAX_BODY_DEPTH_ENV_PRIMARY, "").strip()
    if raw:
        return _parse_int_or(raw, DEFAULT_MAX_BODY_DEPTH)
    raw = os.environ.get(MAX_BODY_DEPTH_ENV_LEGACY, "").strip()
    if raw:
        if MAX_BODY_DEPTH_ENV_LEGACY not in _legacy_warned:
            _legacy_warned.add(MAX_BODY_DEPTH_ENV_LEGACY)
            logger.warning(
                "env var %s is deprecated, use %s instead",
                MAX_BODY_DEPTH_ENV_LEGACY,
                MAX_BODY_DEPTH_ENV_PRIMARY,
            )
        return _parse_int_or(raw, DEFAULT_MAX_BODY_DEPTH)
    return DEFAULT_MAX_BODY_DEPTH


def _parse_int_or(raw: str, default: int) -> int:
    try:
        return int(raw)
    except ValueError:
        return default


def _quick_depth_might_exceed(body: bytes, max_depth: int) -> bool:
    cap = max_depth + 1
    depth = 0
    peak = 0
    in_string = False
    escape_next = False
    for byte in body:
        if escape_next:
            escape_next = False
            continue
        if in_string:
            if byte == 0x5C:
                escape_next = True
            elif byte == 0x22:
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
            continue
        if byte == 0x7B or byte == 0x5B:
            depth += 1
            if depth > peak:
                peak = depth
                if peak >= cap:
                    return True
        elif byte == 0x7D or byte == 0x5D:
            depth -= 1
    return peak >= cap


def _path_is_guarded(path: str | None) -> bool:
    if not path:
        return False
    if path in _EXCLUDED_PATHS:
        return False
    return any(path.startswith(prefix) for prefix in _GUARDED_PREFIXES)


def _is_jsonish_content_type(headers, path: str | None) -> bool:
    ctype: str = ""
    for raw_name, raw_value in headers:
        if raw_name.lower() == b"content-type":
            try:
                ctype = raw_value.decode("latin-1").lower()
            except UnicodeDecodeError:
                ctype = ""
            break
    if not ctype:
        return path in _JSON_CONTENT_TYPE_OPTIONAL_PATHS
    primary = ctype.split(";", 1)[0].strip()
    if primary == "application/json":
        return True
    if primary.startswith("application/") and primary.endswith("+json"):
        return True
    return False


def _json_nesting_depth_exceeds(obj: Any, max_depth: int) -> bool:
    if max_depth <= 0:
        return False
    if not isinstance(obj, (dict, list, tuple)):
        return False
    stack: list[tuple[Any, int]] = [(obj, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > max_depth:
            return True
        if isinstance(node, dict):
            for v in node.values():
                if isinstance(v, (dict, list, tuple)):
                    stack.append((v, depth + 1))
        elif isinstance(node, (list, tuple)):
            for v in node:
                if isinstance(v, (dict, list, tuple)):
                    stack.append((v, depth + 1))
    return False


async def _send_400_depth(send, *, max_depth: int) -> None:
    body = _json.dumps(
        {
            "error": {
                "message": (
                    f"Request body JSON nesting depth exceeds the {max_depth}-level "
                    f"server cap (set via {MAX_BODY_DEPTH_ENV_PRIMARY})."
                ),
                "type": "invalid_request_error",
                "code": "request_body_too_deep",
                "param": None,
            }
        }
    ).encode("utf-8")
    try:
        await send(
            {
                "type": "http.response.start",
                "status": 400,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})
    except Exception:
        logger.debug("body-depth 400 send failed (client already disconnected)")


class RequestBodyDepthMiddleware:
    """ASGI middleware enforcing max body JSON nesting depth."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        if scope.get("method") not in _GUARDED_METHODS:
            return await self.app(scope, receive, send)
        path = scope.get("path")
        if not _path_is_guarded(path):
            return await self.app(scope, receive, send)
        max_depth = _resolve_max_body_depth()
        if max_depth <= 0:
            return await self.app(scope, receive, send)
        if not _is_jsonish_content_type(scope.get("headers", ()), path):
            return await self.app(scope, receive, send)

        chunks: list[bytes] = []
        while True:
            msg = await receive()
            mtype = msg.get("type")
            if mtype == "http.request":
                chunk = msg.get("body", b"") or b""
                if chunk:
                    chunks.append(chunk)
                if not msg.get("more_body", False):
                    break
            elif mtype == "http.disconnect":
                return await self.app(scope, _replay_with_disconnect(chunks), send)
            else:
                chunks.append(b"")
        body = b"".join(chunks)

        if not body.strip():
            return await self.app(scope, _replay_buffered(body, receive), send)

        if not _quick_depth_might_exceed(body, max_depth):
            return await self.app(scope, _replay_buffered(body, receive), send)

        try:
            parsed = _json.loads(body)
        except _json.JSONDecodeError:
            return await self.app(scope, _replay_buffered(body, receive), send)
        except RecursionError:
            logger.warning(
                "json.loads RecursionError — body depth cap rejection (depth>%d)",
                max_depth,
            )
            await _send_400_depth(send, max_depth=max_depth)
            return

        if _json_nesting_depth_exceeds(parsed, max_depth):
            logger.info(
                "Rejecting request: JSON depth exceeds %d (path=%s)", max_depth, path
            )
            await _send_400_depth(send, max_depth=max_depth)
            return

        return await self.app(scope, _replay_buffered(body, receive), send)


def _replay_buffered(body: bytes, original_receive):
    sent = {"value": False}

    async def receive():
        if not sent["value"]:
            sent["value"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await original_receive()

    return receive


def _replay_with_disconnect(chunks: list[bytes]):
    idx = {"value": 0}

    async def receive():
        i = idx["value"]
        if i < len(chunks):
            idx["value"] = i + 1
            return {
                "type": "http.request",
                "body": chunks[i],
                "more_body": True,
            }
        return {"type": "http.disconnect"}

    return receive


def install_request_body_depth_middleware(app: Any) -> None:
    logger.info(
        "Installing RequestBodyDepthMiddleware (default cap=%d)", DEFAULT_MAX_BODY_DEPTH
    )
    app.add_middleware(RequestBodyDepthMiddleware)
