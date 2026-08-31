"""Compatibility shim: re-exports from api.audio_routes."""

import logging
import re as _re

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

# R11-B-F4 (#727): case-insensitive TTS alias index. The HF repos are
# mixed-case (``Kokoro-82M-bf16``) but clients / docs frequently send the
# mixed-case form; the live aliases are lowercase. Exact match wins, then
# this lowercased fallback resolves ``KOKORO-82M-8BIT`` to the same repo.
_TTS_MODEL_ALIASES_LOWER = {k.lower(): v for k, v in TTS_MODEL_ALIASES.items()}
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

# #508 Gap A: HuggingFace repo-id structural validation. The bare
# ``"/" in model_id`` passthrough accepted path-shaped (3+ segment),
# leading/trailing slash, hidden-dir, ``.git`` suffix, ``+`` char and
# over-length ids — forwarding them to the engine loader which surfaced
# as an opaque 500 instead of a clean 404. Implemented locally (not via
# huggingface_hub.utils.validate_repo_id) so the check holds even when
# the test harness stubs ``huggingface_hub`` out of sys.modules, and to
# enforce a total-length cap the hub validator does not. Only applied to
# slash-bearing ids; short aliases resolve via the alias tables above.
_HF_REPO_ID_MAX_TOTAL = 96
# ``<org>/<repo>`` or bare ``<repo>``; each component is 1-96 chars of
# [A-Za-z0-9._-], must not start/end with ``.`` or ``-``. Mirrors the
# HuggingFace moon-landing name rule. Slash-free ids never reach here.
_HF_REPO_ID_RE = _re.compile(
    r"^(?!\.)(?!-)[A-Za-z0-9._-]{1,96}(?<!\.)(?<!-)"
    r"(?:/(?!\.)(?!-)[A-Za-z0-9._-]{1,96}(?<!\.)(?<!-))?$"
)


def _validate_hf_repo_id_or_404(model_id: str, kind: str) -> str:
    # Slash-free ids are short aliases handled by the caller; only gate
    # the HuggingFace ``<org>/<repo>`` pass-through shape.
    if "/" not in model_id:
        return model_id
    reason = None
    if model_id.count("/") > 1:
        reason = "must be in the form 'namespace/repo_name'"
    elif "--" in model_id or ".." in model_id:
        reason = "cannot contain '--' or '..'"
    elif model_id.endswith(".git"):
        reason = "cannot end with '.git'"
    elif not _HF_REPO_ID_RE.match(model_id):
        reason = (
            "must use alphanumeric chars, '-', '_' or '.' "
            "and not start/end with '-' or '.'"
        )
    elif len(model_id) > _HF_REPO_ID_MAX_TOTAL:
        reason = f"exceeds {_HF_REPO_ID_MAX_TOTAL} chars"
    if reason is not None:
        logger.info(
            "#508: rejecting malformed %s model id %r: %s", kind, model_id, reason
        )
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                    "message": f"Unknown {kind} model: {model_id} ({reason})",
                    "param": "model",
                }
            },
        )
    return model_id


def _resolve_stt_model(model_id: str) -> str:
    if not model_id:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "type": "invalid_request_error",
                    "code": "invalid_request_error",
                    "message": "model is required",
                }
            },
        )
    if model_id == "default":
        model_id = DEFAULT_STT_ALIAS
    if model_id in STT_MODEL_ALIASES:
        return STT_MODEL_ALIASES[model_id]
    if "/" not in model_id:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                    "message": f"Unknown STT model: {model_id}",
                    "param": "model",
                }
            },
        )
    return _validate_hf_repo_id_or_404(model_id, "STT")


def _resolve_tts_model(model_id: str) -> str:
    if not model_id:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "type": "invalid_request_error",
                    "code": "invalid_request_error",
                    "message": "model is required",
                }
            },
        )
    if model_id in TTS_MODEL_ALIASES:
        return TTS_MODEL_ALIASES[model_id]
    # R11-B-F4 (#727): case-insensitive fallback for mixed-case HF
    # repo names (``Kokoro-82M-bf16`` / ``KOKORO-82M-8BIT``). Exact
    # match above already handled the canonical lowercase aliases.
    lower_hit = _TTS_MODEL_ALIASES_LOWER.get(model_id.lower())
    if lower_hit is not None:
        return lower_hit
    if "/" not in model_id:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                    "message": f"Unknown TTS model: {model_id}",
                    "param": "model",
                }
            },
        )
    return _validate_hf_repo_id_or_404(model_id, "TTS")


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
