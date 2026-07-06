# SPDX-License-Identifier: Apache-2.0
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"
DEFAULT_PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"

_VAD_MODEL_REPO = "mlx-community/silero-vad"
_VAD_TRIM_PAD_SECONDS = 0.2
_VAD_SAMPLE_RATE = 16_000
_VAD_SPEECH_THRESHOLD = 0.3
_VAD_NO_SPEECH_RMS_FLOOR = 3.16e-3

_VAD_MODEL_CACHE: Any | None = None
_VAD_IMPORT_UNAVAILABLE = False
_VAD_LOAD_FAILURE_LOGGED = False
_VAD_LOAD_LOCK = threading.Lock()

_WHISPER_PROCESSOR_SOURCE_MAP: dict[str, str] = {
    "mlx-community/whisper-large-v3-mlx": "openai/whisper-large-v3",
    "mlx-community/whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
    "mlx-community/whisper-medium-mlx": "openai/whisper-medium",
    "mlx-community/whisper-small-mlx": "openai/whisper-small",
    "mlx-community/whisper-base-mlx": "openai/whisper-base",
    "mlx-community/whisper-tiny-mlx": "openai/whisper-tiny",
}
_DEFAULT_WHISPER_PROCESSOR_FALLBACK = "openai/whisper-large-v3"


@dataclass
class TranscriptionResult:
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list | None = None


@dataclass
class _VADTrimResult:
    skipped: bool = False
    has_speech: bool = False
    waveform: Any = None
    offset_seconds: float = 0.0
    sample_rate: int = _VAD_SAMPLE_RATE


def _vad_pretrim_disabled_by_env() -> bool:
    val = os.environ.get("FUSION_MLX_STT_VAD_PRETRIM", "").strip().lower()
    return val in {"0", "false", "no", "off"}


def _get_vad_model() -> Any | None:
    global _VAD_MODEL_CACHE, _VAD_IMPORT_UNAVAILABLE, _VAD_LOAD_FAILURE_LOGGED
    if _VAD_MODEL_CACHE is not None:
        return _VAD_MODEL_CACHE
    if _VAD_IMPORT_UNAVAILABLE:
        return None
    with _VAD_LOAD_LOCK:
        if _VAD_MODEL_CACHE is not None:
            return _VAD_MODEL_CACHE
        if _VAD_IMPORT_UNAVAILABLE:
            return None
        try:
            from mlx_audio.vad import load as vad_load
        except ImportError as e:
            _VAD_IMPORT_UNAVAILABLE = True
            logger.warning(
                "VAD pre-trim disabled: mlx_audio.vad not importable (%s). "
                "Install fusion-mlx[audio] to enable the anti-hallucination "
                "guard for pure-silence clips.",
                e,
            )
            return None
        try:
            _VAD_MODEL_CACHE = vad_load(_VAD_MODEL_REPO)
        except Exception as e:
            if not _VAD_LOAD_FAILURE_LOGGED:
                logger.warning(
                    "VAD pre-trim: could not load %r (%s). Retrying on "
                    "next request. Guard falls back to unmodified "
                    "transcription until load succeeds.",
                    _VAD_MODEL_REPO,
                    e,
                )
                _VAD_LOAD_FAILURE_LOGGED = True
            return None
        _VAD_LOAD_FAILURE_LOGGED = False
        return _VAD_MODEL_CACHE


def _maybe_vad_trim(audio_path: str) -> _VADTrimResult:
    if _vad_pretrim_disabled_by_env():
        return _VADTrimResult(skipped=True)

    vad = _get_vad_model()
    if vad is None:
        return _VADTrimResult(skipped=True)

    try:
        from mlx_audio.stt.utils import load_audio
    except ImportError:
        return _VADTrimResult(skipped=True)

    try:
        waveform = load_audio(audio_path)
    except Exception as e:
        logger.debug(
            "VAD pre-trim: audio load failed for %r, deferring to Whisper: %s",
            audio_path,
            e,
        )
        return _VADTrimResult(skipped=True)

    if getattr(waveform, "shape", (0,))[-1] == 0:
        return _VADTrimResult(skipped=False, has_speech=False)

    try:
        speech_ts = vad.get_speech_timestamps(
            waveform,
            sample_rate=_VAD_SAMPLE_RATE,
            threshold=_VAD_SPEECH_THRESHOLD,
            return_seconds=True,
        )
    except Exception as e:
        logger.warning(
            "VAD pre-trim: get_speech_timestamps failed on %r: %s — "
            "falling back to unmodified transcription.",
            audio_path,
            e,
        )
        return _VADTrimResult(skipped=True)

    if not speech_ts:
        if _rms_above_floor(waveform, _VAD_NO_SPEECH_RMS_FLOOR):
            logger.info(
                "VAD reported no speech but audio RMS exceeds silence "
                "floor (%.4g) — falling back to Whisper.",
                _VAD_NO_SPEECH_RMS_FLOOR,
            )
            return _VADTrimResult(skipped=True)
        return _VADTrimResult(skipped=False, has_speech=False)

    total_seconds = waveform.shape[-1] / _VAD_SAMPLE_RATE
    try:
        first_start = float(speech_ts[0]["start"])
        last_end = float(speech_ts[-1]["end"])
    except (KeyError, TypeError, ValueError, IndexError) as e:
        logger.warning(
            "VAD pre-trim: malformed timestamp entry %r: %s — falling "
            "back to unmodified transcription.",
            speech_ts,
            e,
        )
        return _VADTrimResult(skipped=True)

    start_s = max(0.0, first_start - _VAD_TRIM_PAD_SECONDS)
    end_s = min(total_seconds, last_end + _VAD_TRIM_PAD_SECONDS)
    if end_s <= start_s:
        return _VADTrimResult(skipped=False, has_speech=False)

    start_sample = int(round(start_s * _VAD_SAMPLE_RATE))
    end_sample = int(round(end_s * _VAD_SAMPLE_RATE))
    trimmed = waveform[start_sample:end_sample]

    return _VADTrimResult(
        skipped=False,
        has_speech=True,
        waveform=trimmed,
        offset_seconds=start_s,
        sample_rate=_VAD_SAMPLE_RATE,
    )


def _rms_above_floor(waveform: Any, floor: float) -> bool:
    try:
        import mlx.core as _mx

        arr = (
            waveform
            if isinstance(waveform, _mx.array)
            else _mx.array(waveform, dtype=_mx.float32)
        )
        arr = arr.astype(_mx.float32)
        rms = float(_mx.sqrt(_mx.mean(arr * arr)))
    except Exception:
        try:
            import numpy as _np

            arr = _np.asarray(waveform, dtype=_np.float32)
            rms = float(_np.sqrt(_np.mean(arr * arr)))
        except Exception:
            return True
    return rms > floor


def _shift_segment_time(seg: Any, offset: float) -> None:
    if isinstance(seg, dict):
        for key in ("start", "end"):
            v = seg.get(key)
            if v is not None:
                seg[key] = float(v) + offset
        words = seg.get("words")
        if isinstance(words, list):
            for w in words:
                if not isinstance(w, dict):
                    continue
                for k in ("start", "end"):
                    v = w.get(k)
                    if v is not None:
                        w[k] = float(v) + offset
        return

    for k in ("start", "end"):
        if hasattr(seg, k):
            v = getattr(seg, k)
            if v is not None:
                setattr(seg, k, float(v) + offset)


class STTEngine:

    def __init__(
        self,
        model_name: str = DEFAULT_WHISPER_MODEL,
        enable_vad_pretrim: bool = True,
    ):
        self.model_name = model_name
        self.model = None
        self._loaded = False
        self._is_parakeet = "parakeet" in model_name.lower()
        self._is_whisper = "whisper" in model_name.lower()
        self._enable_vad_pretrim = enable_vad_pretrim

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from mlx_audio.stt.utils import load_model

            self.model = load_model(self.model_name)
            if self._is_whisper:
                self._ensure_whisper_processor()
            self._loaded = True
            logger.info("STT model loaded: %s", self.model_name)
        except ImportError as e:
            logger.error("mlx-audio not installed: %s", e)
            raise ImportError(
                "mlx-audio is required for STT. Install with: pip install mlx-audio"
            ) from e

    def _ensure_whisper_processor(self) -> None:
        if self.model is None:
            return
        if not hasattr(self.model, "_processor"):
            return
        if getattr(self.model, "_processor", None) is not None:
            return

        processor_source = _WHISPER_PROCESSOR_SOURCE_MAP.get(
            self.model_name, _DEFAULT_WHISPER_PROCESSOR_FALLBACK
        )
        try:
            from transformers import WhisperProcessor
        except ImportError:
            logger.warning(
                "transformers not installed; Whisper processor patch skipped"
            )
            return

        try:
            processor = WhisperProcessor.from_pretrained(processor_source)
        except Exception as e:
            logger.warning(
                "WhisperProcessor.from_pretrained(%r) failed: %s",
                processor_source,
                e,
            )
            return

        self.model._processor = processor
        logger.info(
            "Attached WhisperProcessor from %r to %r",
            processor_source,
            self.model_name,
        )

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        task: str = "transcribe",
    ) -> TranscriptionResult:
        if not self._loaded:
            self.load()

        audio_path = str(audio_path)

        trim: _VADTrimResult | None = None
        if self._is_whisper and self._enable_vad_pretrim:
            trim = _maybe_vad_trim(audio_path)
            if not trim.skipped and not trim.has_speech:
                logger.debug(
                    "VAD pre-trim: no speech detected in %r; returning "
                    "empty TranscriptionResult.",
                    audio_path,
                )
                return TranscriptionResult(
                    text="",
                    language=None,
                    duration=0.0,
                    segments=[],
                )

        try:
            kwargs = {"verbose": False}
            if language and not self._is_parakeet:
                kwargs["language"] = language
            if task and not self._is_parakeet:
                kwargs["task"] = task

            if trim is not None and not trim.skipped and trim.has_speech:
                audio_input: Any = trim.waveform
            else:
                audio_input = audio_path

            result = self.model.generate(audio_input, **kwargs)

            text = getattr(result, "text", str(result)) if result else ""
            segments = getattr(result, "segments", None)
            detected_lang = getattr(result, "language", None)

            if (
                segments
                and trim is not None
                and not trim.skipped
                and trim.has_speech
                and trim.offset_seconds > 0.0
            ):
                for seg in segments:
                    _shift_segment_time(seg, trim.offset_seconds)

            duration = None
            if segments:
                last_seg = segments[-1] if segments else None
                if last_seg and hasattr(last_seg, "end"):
                    duration = last_seg.end

            return TranscriptionResult(
                text=text.strip() if isinstance(text, str) else str(text),
                language=detected_lang,
                duration=duration,
                segments=segments,
            )
        except Exception as e:
            logger.error("Transcription failed: %s", e)
            raise

    def unload(self) -> None:
        self.model = None
        self._loaded = False
        logger.info("STT model unloaded")


def transcribe_audio(
    audio_path: str | Path,
    model_name: str = DEFAULT_WHISPER_MODEL,
    language: str | None = None,
) -> TranscriptionResult:
    engine = STTEngine(model_name)
    engine.load()
    return engine.transcribe(audio_path, language=language)
