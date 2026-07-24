"""Shared route-level guard helpers for OpenAI and Anthropic API routes."""

import logging

from fastapi import HTTPException

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


def check_chat_capability(engine, method_name: str, model_name: str) -> None:
    """Raise HTTPException(400) if the engine lacks *method_name* (e.g. ``chat`` or ``stream_chat``)."""
    if not hasattr(engine, method_name) or not callable(
        getattr(engine, method_name, None)
    ):
        label = "chat completions" if method_name == "chat" else "streaming chat completions"
        raise HTTPException(
            400,
            f"Model '{model_name}' does not support {label} "
            f"(engine_type={getattr(engine, 'engine_type', 'unknown')})",
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
