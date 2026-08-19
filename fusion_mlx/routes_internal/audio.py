"""Compatibility shim: re-exports from api.audio_routes."""

import logging

from ..api.audio_routes import *  # noqa: F401,F403
from ..api.audio_routes import router  # noqa: F401
from ..audio.registry import resolve_audio_alias, stt_aliases, tts_aliases

logger = logging.getLogger(__name__)

# Single source of truth: derive the route alias tables from the audio
# registry so a single JSON edit reaches every consumer (test contract:
# test_route_alias_tables_built_from_registry). Previously these were
# hardcoded and drifted from the registry (parakeet / kokoro-82m /
# chatterbox / vibevoice / voxcpm / dia unresolvable at the route layer).
STT_MODEL_ALIASES = dict(stt_aliases())
TTS_MODEL_ALIASES = dict(tts_aliases())

# whisper-turbo was dropped from the registry; "whisper" resolves to
# mlx-community/whisper-large-v3-mlx (same family the old default pointed at).
DEFAULT_STT_ALIAS = "whisper"

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


def _resolve_default_voice_literal(model: str, voice: str) -> str:
    # R11-B-F3 (#505): resolve the literal ``"default"`` voice to the
    # registry default_voice for the model (short alias OR HF id, both
    # via resolve_audio_alias's reverse index). Non-"default" voices
    # pass through unchanged. Unknown models pass "default" through so
    # the route's family detector still owns the decision.
    if voice != "default":
        return voice
    entry = resolve_audio_alias(model)
    if entry is None or not entry.default_voice:
        return "default"
    return entry.default_voice


def _sanitize_decode_reason(text: str) -> str:
    import re

    text = re.sub(r"<\|[^|]*\|>", "", text)
    return text.strip()


def register_audio_routes(app):
    app.include_router(router)


def audio_routes_should_register() -> bool:
    return True
