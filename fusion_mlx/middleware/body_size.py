# SPDX-License-Identifier: Apache-2.0
"""Request-body size cap middleware for fusion-mlx.

ASGI-level guard that rejects oversized bodies before FastAPI/Pydantic
parses them. Two rejection paths:

1. Content-Length fast path — advertised length over cap -> 413.
2. Streaming slow path — wrap ``receive`` and tally bytes; over cap
   raises _BodyTooLargeError which we catch at the boundary to emit
   exactly one 413.

Also guards against slow-DoS (F-072): no body bytes within the
configured timeout -> 408.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_GUARDED_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

_DEFAULT_MAX_REQUEST_BYTES = 8 * 1024 * 1024  # 8 MiB
_DEFAULT_BODY_RECEIVE_TIMEOUT = 15.0


class _BodyTooLargeError(Exception):
    def __init__(self, streamed_bytes: int) -> None:
        super().__init__(f"streamed {streamed_bytes} bytes over body cap")
        self.streamed_bytes = streamed_bytes


class _BodyReceiveTimeoutError(Exception):
    def __init__(self, streamed_bytes: int, timeout: float) -> None:
        super().__init__(
            f"no body bytes received for {timeout:.1f}s "
            f"(streamed_so_far={streamed_bytes})"
        )
        self.streamed_bytes = streamed_bytes
        self.timeout = timeout


_GUARDED_PREFIXES = ("/v1/", "/internal/", "/anthropic/")
_EXCLUDED_PATHS = frozenset({"/v1/audio/transcriptions"})


def _path_is_guarded(path: str | None) -> bool:
    if not path:
        return False
    if path in _EXCLUDED_PATHS:
        return False
    return any(path.startswith(prefix) for prefix in _GUARDED_PREFIXES)


def _resolve_limit() -> int:
    try:
        raw = os.environ.get("FUSION_MLX_MAX_REQUEST_BYTES", "").strip()
        if raw:
            cap = int(raw)
            if cap >= 0:
                return cap
    except (ValueError, TypeError):
        pass
    try:
        from ..config import ServerConfig

        sc = ServerConfig()
        cap = getattr(sc, "max_request_bytes", _DEFAULT_MAX_REQUEST_BYTES)
        if cap > 0:
            return cap
    except Exception:
        pass
    return _DEFAULT_MAX_REQUEST_BYTES


def _resolve_body_receive_timeout() -> float:
    try:
        raw = os.environ.get("FUSION_MLX_BODY_RECEIVE_TIMEOUT_SECONDS", "").strip()
        if raw:
            timeout = float(raw)
            if timeout >= 0:
                return timeout
    except (ValueError, TypeError):
        pass
    return _DEFAULT_BODY_RECEIVE_TIMEOUT


class RequestBodyLimitMiddleware:
    """ASGI middleware enforcing request body size limits."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        if scope.get("method") not in _GUARDED_METHODS:
            return await self.app(scope, receive, send)

        receive_timeout = _resolve_body_receive_timeout()
        path = scope.get("path")
        path_guarded = _path_is_guarded(path)
        limit = _resolve_limit() if path_guarded else 0
        if limit == 0 and receive_timeout <= 0:
            return await self.app(scope, receive, send)

        advertised: int | None = None
        for raw_name, raw_value in scope.get("headers", ()):
            if raw_name.lower() == b"content-length":
                try:
                    advertised = int(raw_value.decode("latin-1"))
                except (UnicodeDecodeError, ValueError):
                    advertised = None
                break

        if limit > 0 and advertised is not None and advertised > limit:
            logger.info(
                "Rejecting oversized request: advertised=%d limit=%d", advertised, limit
            )
            await _send_413(send, advertised=advertised, limit=limit, streaming=False)
            return

        total = {"bytes": 0}
        downstream_started_response = {"value": False}
        downstream_completed_response = {"value": False}
        body_complete = {"value": False}
        timeout_tripped = {"value": False, "streamed": 0, "timeout": 0.0}
        response_replaced = {"value": False}

        async def bounded_receive():
            if receive_timeout > 0 and not body_complete["value"]:
                try:
                    msg = await asyncio.wait_for(receive(), timeout=receive_timeout)
                except TimeoutError:
                    timeout_tripped["value"] = True
                    timeout_tripped["streamed"] = total["bytes"]
                    timeout_tripped["timeout"] = receive_timeout
                    raise _BodyReceiveTimeoutError(
                        total["bytes"], receive_timeout
                    ) from None
            else:
                msg = await receive()
            if msg.get("type") == "http.request":
                body_len = len(msg.get("body", b"") or b"")
                total["bytes"] += body_len
                if limit > 0 and total["bytes"] > limit:
                    raise _BodyTooLargeError(total["bytes"])
                if not msg.get("more_body", False):
                    body_complete["value"] = True
            return msg

        async def guarded_send(msg):
            mtype = msg.get("type")
            if response_replaced["value"]:
                return
            if timeout_tripped["value"] and not downstream_completed_response["value"]:
                if mtype == "http.response.start":
                    downstream_started_response["value"] = True
                    body = _format_408_body(
                        timeout=timeout_tripped["timeout"],
                        streamed=timeout_tripped["streamed"],
                    )
                    msg = {
                        "type": "http.response.start",
                        "status": 408,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode("ascii")),
                            (b"connection", b"close"),
                        ],
                    }
                    await send(msg)
                    await send(
                        {"type": "http.response.body", "body": body, "more_body": False}
                    )
                    downstream_completed_response["value"] = True
                    response_replaced["value"] = True
                    return
                if mtype == "http.response.body":
                    response_replaced["value"] = True
                    return
            if mtype == "http.response.start":
                downstream_started_response["value"] = True
            elif mtype == "http.response.body" and not msg.get("more_body", False):
                downstream_completed_response["value"] = True
            await send(msg)

        try:
            await self.app(scope, bounded_receive, guarded_send)
        except _BodyReceiveTimeoutError as exc:
            if downstream_completed_response["value"]:
                logger.debug(
                    "body-receive timeout after downstream completed (streamed=%d, timeout=%.1fs)",
                    exc.streamed_bytes,
                    exc.timeout,
                )
                return
            if downstream_started_response["value"]:
                logger.warning(
                    "body-receive timeout (%.1fs) after downstream started; %d bytes streamed",
                    exc.timeout,
                    exc.streamed_bytes,
                )
                try:
                    await send(
                        {"type": "http.response.body", "body": b"", "more_body": False}
                    )
                except Exception:
                    logger.debug(
                        "terminal body frame send failed after receive timeout",
                        exc_info=True,
                    )
                return
            logger.info(
                "Sending 408: no body bytes for %.1fs (streamed=%d)",
                exc.timeout,
                exc.streamed_bytes,
            )
            await _send_408(send, timeout=exc.timeout, streamed=exc.streamed_bytes)
            return
        except _BodyTooLargeError as exc:
            if downstream_completed_response["value"]:
                logger.debug(
                    "body cap tripped after downstream completed (streamed=%d, limit=%d)",
                    exc.streamed_bytes,
                    limit,
                )
                return
            if downstream_started_response["value"]:
                logger.warning(
                    "request body cap (%d) tripped after downstream started; %d bytes streamed",
                    limit,
                    exc.streamed_bytes,
                )
                try:
                    await send(
                        {"type": "http.response.body", "body": b"", "more_body": False}
                    )
                except Exception:
                    logger.debug(
                        "terminal body frame send failed after cap trip", exc_info=True
                    )
                return
            logger.info(
                "Sending 413: body cap %d exceeded (streamed=%d)",
                limit,
                exc.streamed_bytes,
            )
            await _send_413(
                send,
                advertised=None,
                limit=limit,
                streaming=True,
                streamed=exc.streamed_bytes,
            )


def _format_message(
    *,
    advertised: int | None,
    limit: int,
    streaming: bool,
    streamed: int | None = None,
) -> str:
    if streaming:
        return (
            f"Request body too large: streamed {streamed or 0} bytes "
            f"exceeded the {limit}-byte server cap "
            "(set via FUSION_MLX_MAX_REQUEST_BYTES)"
        )
    return (
        f"Request body too large: Content-Length {advertised} bytes "
        f"exceeds the {limit}-byte server cap "
        "(set via FUSION_MLX_MAX_REQUEST_BYTES)"
    )


def _format_408_body(*, timeout: float, streamed: int) -> bytes:
    message = (
        f"Request timed out: no body bytes received for {timeout:.1f}s "
        f"(streamed={streamed} bytes; set via "
        "FUSION_MLX_BODY_RECEIVE_TIMEOUT_SECONDS)"
    )
    return _json.dumps(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "request_timeout",
                "param": None,
            }
        }
    ).encode("utf-8")


async def _send_408(send, *, timeout: float, streamed: int) -> None:
    body = _format_408_body(timeout=timeout, streamed=streamed)
    try:
        await send(
            {
                "type": "http.response.start",
                "status": 408,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})
    except Exception:
        logger.debug("body-receive 408 send failed (client already disconnected)")


async def _send_413(
    send,
    *,
    advertised: int | None,
    limit: int,
    streaming: bool,
    streamed: int | None = None,
) -> None:
    message = _format_message(
        advertised=advertised, limit=limit, streaming=streaming, streamed=streamed
    )
    body = _json.dumps(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "request_too_large",
                "param": None,
            }
        }
    ).encode("utf-8")
    try:
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})
    except Exception:
        logger.debug("body-size 413 send failed (client already disconnected)")


def install_request_body_limit_middleware(app: Any) -> None:
    logger.info("Installing RequestBodyLimitMiddleware")
    app.add_middleware(RequestBodyLimitMiddleware)
