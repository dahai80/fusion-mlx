"""Shared route-level guard helpers for OpenAI and Anthropic API routes."""

import logging
from typing import Any

from fastapi import HTTPException

from ..exceptions import InsufficientMemoryError, ModelTooLargeError

logger = logging.getLogger(__name__)

MULTIMODAL_CONTENT_TYPES = (
    "image_url",
    "image",
    "video",
    "video_url",
    "audio_url",
    "audio",
    "input_audio",
)


def _build_insufficient_memory_detail(exc: InsufficientMemoryError) -> dict[str, Any]:
    # R-P0-1 / R-P1-2 (#0908 audit) + fix-0911 §2.5: single source for the
    # memory-error detail dict. Delegates field assembly to
    # ``exc.to_error_detail()`` (exceptions.py — the canonical schema) and
    # layers the actionable "unload victim" suggestion on top. Previously
    # openai_routes (4 sites) and _guards (anthropic path) each assembled
    # the dict independently, diverging on which fields were included.
    detail = exc.to_error_detail()
    detail["status"] = 503
    if exc.loaded_models:
        unloadable = [m for m in exc.loaded_models if not m.get("pinned", False)]
        if unloadable:
            victim = unloadable[0]
            detail["suggestion"] = (
                f"Unload model {victim['model_id']} "
                f"(free ~{victim.get('memory_mb', '?')}MB) then retry"
            )
    return detail


def build_model_error_response(
    exc: Exception, *, adapter: str = "openai"
) -> HTTPException:
    # R-P0-1 (#0908 audit): unified model-error → HTTPException mapping.
    # adapter="openai" → 503 model_unavailable with structured detail.
    # adapter="anthropic" → 503 resource_exhausted with Retry-After header.
    if isinstance(exc, InsufficientMemoryError):
        detail = _build_insufficient_memory_detail(exc)
        if adapter == "anthropic":
            return HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "message": detail["message"],
                        "type": "resource_exhausted",
                        "suggestion": detail.get("suggestion"),
                    }
                },
                headers={"Retry-After": "10"},
            )
        return HTTPException(status_code=503, detail={"error": detail})
    if isinstance(exc, ModelTooLargeError):
        if adapter == "anthropic":
            return HTTPException(
                status_code=413,
                detail={
                    "error": {
                        "message": str(exc),
                        "type": "model_too_large",
                    }
                },
            )
        return HTTPException(
            status_code=413,
            detail={
                "error": {
                    "type": "model_too_large",
                    "message": str(exc),
                }
            },
        )
    raise exc


def check_chat_capability(engine, method_name: str, model_name: str) -> None:
    """Raise HTTPException(400) if the engine lacks *method_name* (e.g. ``chat`` or ``stream_chat``)."""
    if not hasattr(engine, method_name) or not callable(
        getattr(engine, method_name, None)
    ):
        label = (
            "chat completions"
            if method_name == "chat"
            else "streaming chat completions"
        )
        raise HTTPException(
            400,
            f"Model '{model_name}' does not support {label} "
            f"(engine_type={getattr(engine, 'engine_type', 'unknown')})",
        )


def check_tool_choice_support(engine, request, model_name: str) -> None:
    """Raise HTTPException(422) if engine opted out of tool calls but
    the request demands forced tool_choice (``required`` or named function).

    DiffusionEngine sets ``supports_tool_calls=False`` when its chat template
    lacks tool-call markers. Allowing ``tool_choice=required`` through would
    run a full generation, return plain text, and confuse callers expecting a
    structured tool-call response. Reject early with a clear error.
    """
    if getattr(engine, "supports_tool_calls", True):
        return
    tool_choice = getattr(request, "tool_choice", None)
    if tool_choice is None or tool_choice == "auto" or tool_choice == "none":
        return
    if isinstance(tool_choice, dict):
        logger.info(
            "Rejecting named tool_choice on engine with supports_tool_calls=False: %s",
            model_name,
        )
        raise HTTPException(
            status_code=422,
            detail=(
                f"Model '{model_name}' does not support forced tool_choice "
                f"(engine supports_tool_calls=False, tool_choice={tool_choice!r})"
            ),
        )
    logger.info(
        "Rejecting tool_choice=%s on engine with supports_tool_calls=False: %s",
        tool_choice,
        model_name,
    )
    raise HTTPException(
        status_code=422,
        detail=(
            f"Model '{model_name}' does not support forced tool_choice "
            f"(engine supports_tool_calls=False, tool_choice={tool_choice!r})"
        ),
    )


def check_multimodal_content(engine, messages, model_name: str) -> None:
    """Raise HTTPException(400) if a text-only engine receives multimodal message parts."""
    if getattr(engine, "is_mllm", False):
        return
    for msg in messages:
        content = getattr(msg, "content", "") if msg else None
        if isinstance(content, list):
            for part in content:
                pt = (
                    part.get("type", "")
                    if isinstance(part, dict)
                    else getattr(part, "type", "")
                )
                if pt in MULTIMODAL_CONTENT_TYPES:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Model '{model_name}' does not support "
                            "image, video, or audio inputs."
                        ),
                    )
