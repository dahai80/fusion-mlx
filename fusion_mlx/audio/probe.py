# SPDX-License-Identifier: Apache-2.0
import importlib.util
import logging
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Verdict:
    ok: bool
    reason: str | None = None


_cached_verdict: dict[str, _Verdict] = {}
_LANE_STATUS: dict[str, str] = {}
_LANE_REASON: dict[str, str] = {}

_LANE_SUBMODULES: dict[str, str] = {
    "tts": "mlx_audio.tts.generate",
    "stt": "mlx_audio.stt.utils",
}

_KOKORO_EXTRA_DEP = "misaki"
_KOKORO_EXTRA_HINT = (
    "Kokoro TTS requires the optional `misaki` G2P package, which is "
    "not installed. Reinstall with `pip install 'fusion-mlx[audio]'` "
    "to pull every audio dep, or `pip install misaki` for a "
    "minimal Kokoro-only install."
)

AUDIO_EXTRA_INSTALL_HINT = "Install with: pip install 'fusion-mlx[audio]'"

_AUDIO_ALIAS_TOKENS: tuple[str, ...] = (
    "whisper",
    "parakeet",
    "kokoro",
    "chatterbox",
    "vibevoice",
    "voxcpm",
)


def _reset_probe_cache() -> None:
    _cached_verdict.clear()
    _LANE_STATUS.clear()
    _LANE_REASON.clear()


def _reset_lane_status() -> None:
    _LANE_STATUS.clear()
    _LANE_REASON.clear()


def audio_lane_status(lane: str) -> dict[str, str | None]:
    status = _LANE_STATUS.get(lane, "unknown")
    reason = _LANE_REASON.get(lane)
    return {"status": status, "reason": reason}


def _record_lane_status(lane: str, status: str, reason: str | None) -> None:
    _LANE_STATUS[lane] = status
    if reason is None:
        _LANE_REASON.pop(lane, None)
    else:
        _LANE_REASON[lane] = reason


def deep_probe_audio_lane(
    lane: str, model_name: str | None = None
) -> dict[str, str | None]:
    verdict = _probe_lane(lane)
    if not verdict.ok:
        _record_lane_status(lane, "missing", verdict.reason)
        return audio_lane_status(lane)

    if lane == "stt":
        ok, reason = _dry_run_stt(model_name)
    elif lane == "tts":
        ok, reason = _dry_run_tts(model_name)
    else:
        _record_lane_status(lane, "unknown", f"unknown audio lane {lane!r}")
        return audio_lane_status(lane)

    if ok:
        _record_lane_status(lane, "ok", None)
    else:
        _record_lane_status(lane, "degraded", reason)
    return audio_lane_status(lane)


def _dry_run_stt(model_name: str | None) -> tuple[bool, str | None]:
    try:
        import wave

        from .._tempfile_safe import managed_tempfile_path
        from .stt import DEFAULT_WHISPER_MODEL, STTEngine

        with managed_tempfile_path(suffix=".wav") as wav_handle:
            wav_path = wav_handle.path
            with wave.open(wav_path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(b"\x00\x00" * 16000)
            engine = STTEngine(model_name or DEFAULT_WHISPER_MODEL)
            engine.load()
            result = engine.transcribe(wav_path)
            if not hasattr(result, "text"):
                return False, "STT result missing `text` attribute"
        return True, None
    except Exception as e:
        return False, f"STT dry-run failed: {type(e).__name__}: {e}"


def _dry_run_tts(model_name: str | None) -> tuple[bool, str | None]:
    try:
        from .tts import DEFAULT_TTS_MODEL, TTSEngine

        engine = TTSEngine(model_name or DEFAULT_TTS_MODEL)
        engine.load()
        result = engine.generate("a", voice="af_heart")
        if not hasattr(result, "audio") or len(result.audio) == 0:
            return False, "TTS result is empty (no audio produced)"
        return True, None
    except Exception as e:
        return False, f"TTS dry-run failed: {type(e).__name__}: {e}"


def _probe_lane(lane: str) -> _Verdict:
    if lane in _cached_verdict:
        return _cached_verdict[lane]

    if "" not in _cached_verdict:
        if importlib.util.find_spec("mlx_audio") is None:
            _cached_verdict[""] = _Verdict(
                ok=False, reason="mlx-audio is not installed"
            )
        else:
            _cached_verdict[""] = _Verdict(ok=True, reason=None)

    presence = _cached_verdict[""]
    if not presence.ok:
        _cached_verdict[lane] = presence
        return presence

    submod = _LANE_SUBMODULES.get(lane)
    if submod is None:
        _cached_verdict[lane] = _Verdict(
            ok=False, reason=f"unknown audio lane {lane!r}"
        )
        return _cached_verdict[lane]
    try:
        __import__(submod)
    except Exception as e:
        _cached_verdict[lane] = _Verdict(
            ok=False,
            reason=(
                f"mlx-audio {lane} import failed at runtime: "
                f"{type(e).__name__}: {e} (probing {submod})"
            ),
        )
        return _cached_verdict[lane]

    _cached_verdict[lane] = _Verdict(ok=True, reason=None)
    return _cached_verdict[lane]


def mlx_audio_available(lane: str = "tts") -> _Verdict:
    return _probe_lane(lane)


def _raise_503(verdict: _Verdict) -> None:
    from fastapi import HTTPException

    detail = verdict.reason or "mlx-audio is not available"
    raise HTTPException(
        status_code=503,
        detail=(f"{detail}. {AUDIO_EXTRA_INSTALL_HINT}"),
    )


def require_mlx_audio_tts() -> None:
    verdict = _probe_lane("tts")
    if verdict.ok:
        return
    _raise_503(verdict)


def require_mlx_audio_stt() -> None:
    verdict = _probe_lane("stt")
    if verdict.ok:
        return
    _raise_503(verdict)


require_mlx_audio = require_mlx_audio_tts


def require_kokoro_runtime() -> None:
    from fastapi import HTTPException

    if importlib.util.find_spec(_KOKORO_EXTRA_DEP) is not None:
        return
    raise HTTPException(
        status_code=503,
        detail=_KOKORO_EXTRA_HINT,
    )


def is_audio_model_alias(model_name: str | None) -> bool:
    if not isinstance(model_name, str) or not model_name:
        return False
    try:
        from .registry import is_audio_name

        if is_audio_name(model_name):
            return True
    except Exception:
        pass
    lc = model_name.lower()
    return any(tok in lc for tok in _AUDIO_ALIAS_TOKENS)


def require_audio_or_exit(model_name: str) -> None:
    if importlib.util.find_spec("mlx_audio") is not None:
        return
    print(
        f"error: model {model_name!r} is an audio alias and requires the "
        f"optional `mlx-audio` dependency (shipped with the [audio] "
        f"extra).\n" + AUDIO_EXTRA_INSTALL_HINT,
        file=sys.stderr,
    )
    sys.exit(2)
