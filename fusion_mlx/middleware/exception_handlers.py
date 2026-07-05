# SPDX-License-Identifier: Apache-2.0
"""Unified FastAPI exception handlers for fusion-mlx.

Provides consistent OpenAI-shaped error envelopes across all error
types: HTTPException, JSONDecodeError, RequestValidationError,
Pydantic ValidationError, RecursionError, and generic Exception.

Anthropic-compat routes (/v1/messages) get wrapped in the Anthropic
envelope shape: ``{"type":"error","error":{...}}``.
"""

from __future__ import annotations

import json as _json
import logging
import typing as _t
from typing import get_args, get_origin

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

logger = logging.getLogger("fusion_mlx.exception_handlers")

_REQUEST_MODEL_REGISTRY: dict[str, type[BaseModel]] = {}


def register_request_model(model_cls: type[BaseModel]) -> None:
    _REQUEST_MODEL_REGISTRY[model_cls.__name__] = model_cls


def _unwrap_optional(tp: _t.Any) -> _t.Any:
    origin = get_origin(tp)
    if origin is None:
        return tp
    args = get_args(tp)
    if not args:
        return tp
    non_none = [a for a in args if a is not type(None)]
    if len(non_none) == len(args):
        return tp
    if len(non_none) == 1:
        return non_none[0]
    return non_none[0] if non_none else tp


def _union_arms(tp: _t.Any) -> tuple[_t.Any, ...]:
    import types as _types

    origin = get_origin(tp)
    if origin is _t.Union or origin is getattr(_types, "UnionType", None):
        return tuple(a for a in get_args(tp) if a is not type(None))
    return (tp,)


def _descend_field(tp: _t.Any, hint: str | None = None) -> _t.Any:
    inner = _unwrap_optional(tp)
    arms = _union_arms(inner)
    if hint is not None and len(arms) > 1:
        for arm in arms:
            candidate = _descend_field(arm, hint=None)
            if (
                isinstance(candidate, type)
                and issubclass(candidate, BaseModel)
                and hint in candidate.model_fields
            ):
                return candidate
            if (
                isinstance(arm, type)
                and issubclass(arm, BaseModel)
                and hint in arm.model_fields
            ):
                return arm
    target = arms[0] if arms else inner
    origin = get_origin(target)
    if origin is None:
        return target
    args = get_args(target)
    if not args:
        return None
    if origin in (list, tuple, set, frozenset):
        return args[0]
    if origin is dict:
        return args[1] if len(args) >= 2 else None
    return None


_REQUEST_PATH_TO_ROOT: list[tuple[str, type[BaseModel]]] = []


def register_request_path(path_prefix: str, model_cls: type[BaseModel]) -> None:
    for i, (existing_prefix, _) in enumerate(_REQUEST_PATH_TO_ROOT):
        if existing_prefix == path_prefix:
            _REQUEST_PATH_TO_ROOT[i] = (path_prefix, model_cls)
            return
    _REQUEST_PATH_TO_ROOT.append((path_prefix, model_cls))


def _path_matches_canonical_prefix(path: str, prefix: str) -> bool:
    if path == prefix or path.startswith(prefix + "/"):
        return True
    needle = prefix
    idx = path.find(needle, 1)
    while idx != -1:
        end = idx + len(needle)
        right_ok = end == len(path) or path[end] == "/"
        if right_ok:
            return True
        idx = path.find(needle, idx + 1)
    return False


def _resolve_root_model(
    exc: object,
    loc: tuple,
    request: Request | None = None,
) -> type[BaseModel] | None:
    title = getattr(exc, "title", None)
    if isinstance(title, str):
        root = _REQUEST_MODEL_REGISTRY.get(title)
        if root is not None:
            return root
    if request is not None:
        try:
            path = request.url.path
        except Exception:
            path = None
        if path:
            for prefix, cls in _REQUEST_PATH_TO_ROOT:
                if _path_matches_canonical_prefix(path, prefix):
                    return cls
    first_str: str | None = None
    for raw in loc:
        if raw == "body":
            continue
        if isinstance(raw, str):
            first_str = raw
            break
        break
    if first_str is None:
        return None
    for cls in _REQUEST_MODEL_REGISTRY.values():
        if first_str in cls.model_fields:
            return cls
    return None


def _walk_loc_with_root(
    loc: tuple,
    root_cls: type[BaseModel] | None,
) -> tuple[list[str], str | None]:
    parts: list[str] = []
    last_field: str | None = None
    current: _t.Any = root_cls
    loc_list = list(loc)
    for idx, raw in enumerate(loc_list):
        if raw == "body":
            continue
        if isinstance(raw, int):
            parts.append(str(raw))
            if current is not None:
                hint = _peek_next_field_hint(loc_list, idx)
                current = _descend_field(current, hint=hint)
            continue
        if isinstance(raw, str) and _is_union_arm_discriminator(raw):
            continue
        if current is not None and not (
            isinstance(current, type) and issubclass(current, BaseModel)
        ):
            current = _descend_field(current, hint=raw)
        is_schema_owned = (
            isinstance(current, type)
            and issubclass(current, BaseModel)
            and raw in current.model_fields
        )
        if is_schema_owned:
            parts.append(raw)
            last_field = raw
            field_info = current.model_fields[raw]
            current = _unwrap_optional(field_info.annotation)
        else:
            parts.append("<field>")
            current = None
    return parts, last_field


def _peek_next_field_hint(loc_list: list, idx: int) -> str | None:
    for nxt in loc_list[idx + 1 :]:
        if (
            isinstance(nxt, str)
            and nxt != "body"
            and not _is_union_arm_discriminator(nxt)
        ):
            return nxt
    return None


def _extract_field_from_value_error_msg(
    msg: str,
    root_cls: type[BaseModel] | None,
) -> str | None:
    if not msg:
        return None
    stripped = msg
    prefix = "Value error, "
    if stripped.startswith(prefix):
        stripped = stripped[len(prefix) :]
    first = stripped.split(None, 1)[0] if stripped else ""
    while first and not (first[-1].isalnum() or first[-1] == "_"):
        first = first[:-1]
    if not first:
        return None
    if root_cls is not None:
        return first if first in root_cls.model_fields else None
    for cls in _REQUEST_MODEL_REGISTRY.values():
        if first in cls.model_fields:
            return first
    return first if first in _SCHEMA_OWNED_FIELD_NAMES else None


_SCHEMA_OWNED_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "max_tokens",
        "messages",
        "metadata",
        "model",
        "output_config",
        "stop_sequences",
        "stream",
        "system",
        "temperature",
        "thinking",
        "tool_choice",
        "tools",
        "top_k",
        "top_p",
        "chat_template_kwargs",
        "enable_thinking",
        "frequency_penalty",
        "function_call",
        "functions",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "min_p",
        "n",
        "parallel_tool_calls",
        "presence_penalty",
        "reasoning_max_tokens",
        "repetition_penalty",
        "response_format",
        "seed",
        "stop",
        "stream_options",
        "timeout",
        "top_logprobs",
        "video_fps",
        "video_max_frames",
        "best_of",
        "echo",
        "prompt",
        "suffix",
        "include",
        "input",
        "instructions",
        "max_output_tokens",
        "previous_response_id",
        "prompt_cache_key",
        "reasoning",
        "service_tier",
        "store",
        "text",
        "role",
        "content",
        "type",
        "source",
        "id",
        "name",
        "tool_use_id",
        "is_error",
        "tool_call_id",
        "tool_calls",
        "audio_url",
        "image_url",
        "video",
        "video_url",
        "format",
        "effort",
        "schema",
        "description",
        "strict",
        "budget_tokens",
        "function",
        "media_type",
        "data",
        "url",
        "input_schema",
        "parameters",
    }
)


def _is_union_arm_discriminator(raw: str) -> bool:
    if raw in {"str", "int", "float", "bool", "dict", "list", "bytes", "tuple"}:
        return True
    if "[" in raw and raw.endswith("]"):
        return True
    return False


def _sanitize_loc(loc: tuple) -> str:
    parts: list[str] = []
    for raw in loc:
        if raw == "body":
            continue
        if isinstance(raw, int):
            parts.append(str(raw))
            continue
        if isinstance(raw, str) and _is_union_arm_discriminator(raw):
            continue
        if isinstance(raw, str) and raw in _SCHEMA_OWNED_FIELD_NAMES:
            parts.append(raw)
        else:
            parts.append("<field>")
    return ".".join(parts)


def _render_loc_for_envelope(
    exc: object,
    loc: tuple,
    request: Request | None = None,
) -> tuple[str, str | None]:
    root_cls = _resolve_root_model(exc, loc, request)
    if root_cls is not None:
        parts, last_field = _walk_loc_with_root(loc, root_cls)
        return ".".join(parts), last_field
    rendered = _sanitize_loc(loc)
    param: str | None = None
    for part in rendered.split("."):
        if part and part in _SCHEMA_OWNED_FIELD_NAMES:
            param = part
    return rendered, param


_ANTHROPIC_ROOT_PATHS: tuple[str, ...] = ("/v1/messages",)


def _is_anthropic_path(request: Request | None) -> bool:
    if request is None:
        return False
    try:
        path = request.url.path
    except Exception:
        return False
    for root in _ANTHROPIC_ROOT_PATHS:
        if path == root or path.startswith(root + "/"):
            return True
    return False


def _wrap_for_anthropic(response: JSONResponse) -> JSONResponse:
    raw = getattr(response, "body", None)
    if raw is None:
        return response
    try:
        body = _json.loads(raw)
    except (_json.JSONDecodeError, TypeError):
        return response
    if not isinstance(body, dict):
        return response
    if body.get("type") == "error" and isinstance(body.get("error"), dict):
        return response
    if "error" not in body:
        return response
    wrapped = {"type": "error", "error": body["error"]}
    for k, v in body.items():
        if k == "error":
            continue
        wrapped[k] = v
    preserved_headers: dict[str, str] | None = None
    if response.headers:
        preserved_headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower() not in ("content-length", "content-type")
        }
        if not preserved_headers:
            preserved_headers = None
    return JSONResponse(
        status_code=response.status_code,
        content=wrapped,
        headers=preserved_headers,
    )


def _decode_error_response(exc: _json.JSONDecodeError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": f"Invalid JSON in request body: {exc.msg}",
                "type": "invalid_request_error",
                "code": "invalid_json",
                "param": None,
            }
        },
    )


def _validation_error_response(
    exc: RequestValidationError | PydanticValidationError,
    request: Request | None = None,
) -> JSONResponse:
    details: list[str] = []
    param: str | None = None
    for err in exc.errors():
        raw_loc = tuple(err.get("loc", ()))
        loc, last_field = _render_loc_for_envelope(exc, raw_loc, request)
        msg = err.get("msg", "validation error")
        if last_field is None and not loc and err.get("type") == "value_error":
            root_cls = _resolve_root_model(exc, raw_loc, request)
            recovered = _extract_field_from_value_error_msg(msg, root_cls)
            if recovered is not None:
                last_field = recovered
        details.append(f"{loc}: {msg}" if loc else msg)
        if param is None and last_field is not None:
            param = last_field
    summary = "; ".join(details) or "Invalid request body"
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": f"Invalid request body: {summary}",
                "type": "invalid_request_error",
                "code": "invalid_request",
                "param": param,
            }
        },
    )


_HTTP_ERROR_TYPE_MAP = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    405: "invalid_request_error",
    409: "conflict_error",
    429: "rate_limit_error",
}


def _http_error_response(exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
        return JSONResponse(
            status_code=exc.status_code,
            content=detail,
            headers=getattr(exc, "headers", None),
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": _HTTP_ERROR_TYPE_MAP.get(exc.status_code, "api_error"),
                "code": None,
                "param": None,
            }
        },
        headers=getattr(exc, "headers", None),
    )


def _generic_error_response() -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal server error"}},
    )


def _recursion_error_response() -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal server error"}},
    )


def _register_canonical_request_models() -> None:
    try:
        from ..api.anthropic_models import MessagesRequest
        from ..api.models import (
            ChatCompletionRequest,
            CompletionRequest,
            EmbeddingRequest,
        )
        from ..api.responses_models import ResponsesRequest
    except Exception:
        logger.debug(
            "Failed to import canonical request models for the envelope "
            "registry — falling back to closed allowlist only.",
            exc_info=True,
        )
        return
    for cls in (
        ChatCompletionRequest,
        CompletionRequest,
        EmbeddingRequest,
        MessagesRequest,
        ResponsesRequest,
    ):
        register_request_model(cls)
    register_request_path("/v1/chat/completions", ChatCompletionRequest)
    register_request_path("/v1/completions", CompletionRequest)
    register_request_path("/v1/embeddings", EmbeddingRequest)
    register_request_path("/v1/messages", MessagesRequest)
    register_request_path("/v1/responses", ResponsesRequest)


def install_exception_handlers(app: FastAPI) -> None:
    logger.info("Installing fusion-mlx exception handlers")
    _register_canonical_request_models()

    @app.exception_handler(StarletteHTTPException)
    async def _http_handler(
        request: Request,
        exc: StarletteHTTPException,
    ):
        response = _http_error_response(exc)
        if _is_anthropic_path(request):
            response = _wrap_for_anthropic(response)
        return response

    @app.exception_handler(_json.JSONDecodeError)
    async def _decode_handler(
        request: Request,
        exc: _json.JSONDecodeError,
    ):
        response = _decode_error_response(exc)
        if _is_anthropic_path(request):
            response = _wrap_for_anthropic(response)
        return response

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(
        request: Request,
        exc: RequestValidationError,
    ):
        response = _validation_error_response(exc, request)
        if _is_anthropic_path(request):
            response = _wrap_for_anthropic(response)
        return response

    @app.exception_handler(PydanticValidationError)
    async def _pydantic_validation_handler(
        request: Request,
        exc: PydanticValidationError,
    ):
        sanitized = [
            {
                "type": err.get("type", "validation_error"),
                "loc": _sanitize_loc(tuple(err.get("loc", ()))),
            }
            for err in exc.errors()
        ]
        logger.warning(
            "pydantic.ValidationError on %s %s — %d sanitized error(s): %s",
            request.method,
            request.url.path,
            len(sanitized),
            sanitized,
        )
        response = _validation_error_response(exc, request)
        if _is_anthropic_path(request):
            response = _wrap_for_anthropic(response)
        return response

    @app.exception_handler(RecursionError)
    async def _recursion_handler(
        request: Request,
        exc: RecursionError,
    ):
        logger.warning(
            "RecursionError on %s %s — caught at framework boundary, "
            "returning sanitized 500.",
            request.method,
            request.url.path,
            exc_info=True,
        )
        response = _recursion_error_response()
        if _is_anthropic_path(request):
            response = _wrap_for_anthropic(response)
        return response

    @app.exception_handler(Exception)
    async def _generic_handler(request: Request, exc: Exception):
        anthropic = _is_anthropic_path(request)
        if isinstance(exc, _json.JSONDecodeError):
            response = _decode_error_response(exc)
            return _wrap_for_anthropic(response) if anthropic else response
        if isinstance(exc, RequestValidationError):
            response = _validation_error_response(exc, request)
            return _wrap_for_anthropic(response) if anthropic else response
        if isinstance(exc, PydanticValidationError):
            response = _validation_error_response(exc, request)
            return _wrap_for_anthropic(response) if anthropic else response
        if isinstance(exc, StarletteHTTPException):
            response = _http_error_response(exc)
            return _wrap_for_anthropic(response) if anthropic else response
        if isinstance(exc, RecursionError):
            logger.warning(
                "RecursionError on %s %s (via generic handler) — "
                "returning sanitized 500.",
                request.method,
                request.url.path,
                exc_info=True,
            )
            response = _recursion_error_response()
            return _wrap_for_anthropic(response) if anthropic else response
        logger.error(
            "Unhandled exception on %s %s: %s",
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        response = _generic_error_response()
        return _wrap_for_anthropic(response) if anthropic else response
