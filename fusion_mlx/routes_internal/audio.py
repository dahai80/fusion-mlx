"""Compatibility shim: re-exports from api.audio_routes."""

from ..api.audio_routes import *  # noqa: F401,F403
from ..api.audio_routes import router  # noqa: F401

STT_MODEL_ALIASES = {
    "whisper-turbo": "mlx-community/whisper-large-v3-turbo",
    "whisper-large": "mlx-community/whisper-large-v3-mlx",
    "whisper-medium": "mlx-community/whisper-medium-mlx",
    "whisper-small": "mlx-community/whisper-small-mlx",
    "whisper-base": "mlx-community/whisper-base-mlx",
    "whisper-tiny": "mlx-community/whisper-tiny-mlx",
}

DEFAULT_STT_ALIAS = "whisper-turbo"

TTS_MODEL_ALIASES = {
    "kokoro": "mlx-community/kokoro-82m-v1.0-mlx",
}

_TTS_CONTENT_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "pcm": "audio/pcm;rate=24000",
}

_allowed_voices_for = {}

from fastapi import HTTPException


def _resolve_stt_model(model_id: str) -> str:
    if not model_id:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "invalid_request_error",
                    "message": "model is required",
                }
            },
        )
    if model_id == "default":
        model_id = DEFAULT_STT_ALIAS
    if model_id in STT_MODEL_ALIASES:
        return STT_MODEL_ALIASES[model_id]
    if "/" in model_id:
        return model_id
    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "code": "model_not_found",
                "message": f"Unknown STT model: {model_id}",
                "param": "model",
            }
        },
    )


def _resolve_tts_model(model_id: str) -> str:
    if not model_id:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "invalid_request_error",
                    "message": "model is required",
                }
            },
        )
    if model_id in TTS_MODEL_ALIASES:
        return TTS_MODEL_ALIASES[model_id]
    if "/" in model_id:
        return model_id
    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "code": "model_not_found",
                "message": f"Unknown TTS model: {model_id}",
                "param": "model",
            }
        },
    )


def _resolve_default_voice_literal() -> str | None:
    return "af_heart"


def _sanitize_decode_reason(text: str) -> str:
    import re

    text = re.sub(r"<\|[^|]*\|>", "", text)
    return text.strip()


def register_audio_routes(app):
    app.include_router(router)


def audio_routes_should_register() -> bool:
    return True
