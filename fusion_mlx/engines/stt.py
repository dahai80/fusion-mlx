# SPDX-License-Identifier: Apache-2.0
"""STT (Speech-to-Text) engine for fusion-mlx."""

import asyncio
import gc
import logging
import time
from typing import Any

import mlx.core as mx

from ..engine_core import get_executor
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)

_ISO_TO_STT_LANG: dict[str, str] = {
    "zh": "chinese",
    "yue": "cantonese",
    "en": "english",
    "de": "german",
    "es": "spanish",
    "fr": "french",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "ko": "korean",
    "ja": "japanese",
}


def _normalize_stt_generate_language(language: str | None) -> str | None:
    if not language:
        return None
    n = language.strip()
    if not n:
        return None
    return _ISO_TO_STT_LANG.get(n.lower(), n)


def _missing_processor_hint(model_name: str) -> str:
    return (
        f"STT model '{model_name}' is missing the HuggingFace processor / "
        "feature-extractor configuration. Use an HF-compatible variant or copy "
        "preprocessor_config.json from the upstream repo."
    )


def _wrap_stt_load_error(model_name: str, exc: Exception) -> Exception:
    msg = str(exc).lower()
    if any(
        h in msg
        for h in ("preprocessor_config.json", "feature extractor", "featureextractor")
    ):
        return RuntimeError(
            f"{_missing_processor_hint(model_name)} Original error: {exc}"
        )
    return exc


def _validate_stt_processor(model_name: str, model: Any) -> None:
    module_name = type(model).__module__ or ""
    if "whisper" not in module_name.lower():
        return
    if not hasattr(model, "_processor") or model._processor is not None:
        return
    raise RuntimeError(_missing_processor_hint(model_name))


class STTEngine(BaseNonStreamingEngine):
    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self._model_name = model_name
        self._model = None
        self._kwargs = kwargs

    @property
    def model_name(self) -> str:
        return self._model_name

    async def start(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Starting STT engine: {self._model_name}")
        try:
            from mlx_audio.stt.utils import load_model as _load_model
        except ImportError as exc:
            raise ImportError(
                'mlx-audio required for STT. Install with: pip install "fusion-mlx[audio]"'
            ) from exc

        model_name = self._model_name

        def _load_sync():
            return _load_model(model_name)

        loop = asyncio.get_running_loop()
        try:
            model = await asyncio.wait_for(
                loop.run_in_executor(get_executor("audio"), _load_sync), timeout=120.0
            )
        except Exception as exc:
            raise _wrap_stt_load_error(model_name, exc) from exc

        _validate_stt_processor(model_name, model)
        self._model = model

    async def stop(self) -> None:
        if self._model is None:
            return
        self._model = None
        gc.collect()
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                get_executor("audio"), lambda: (mx.synchronize(), mx.clear_cache())
            ),
            timeout=5.0,
        )

    async def transcribe(
        self, audio_path: str, language: str | None = None, **kwargs
    ) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("Engine not started.")
        model = self._model
        t0 = time.monotonic()

        def _normalize_segment(s) -> dict:
            if isinstance(s, dict):
                return s
            import dataclasses

            if dataclasses.is_dataclass(s) and not isinstance(s, type):
                return dataclasses.asdict(s)
            if hasattr(s, "__dict__"):
                return vars(s)
            return {"text": str(s)}

        def _normalize_language(raw_lang):
            if isinstance(raw_lang, list):
                raw_lang = raw_lang[0] if raw_lang else None
            if isinstance(raw_lang, str) and raw_lang.lower() == "none":
                return None
            return raw_lang

        def _transcribe_sync():
            gen_kwargs = dict(kwargs)
            gl = _normalize_stt_generate_language(language)
            if gl is not None:
                gen_kwargs["language"] = gl
            result = model.generate(audio_path, **gen_kwargs)
            if hasattr(result, "text"):
                raw_lang = (
                    _normalize_language(getattr(result, "language", None)) or language
                )
                raw_segs = getattr(result, "segments", None)
                segments = [_normalize_segment(s) for s in raw_segs] if raw_segs else []
                return {
                    "text": result.text or "",
                    "language": raw_lang,
                    "segments": segments,
                    "duration": getattr(result, "total_time", 0.0),
                }
            return {
                "text": str(result),
                "language": language,
                "segments": [],
                "duration": 0.0,
            }

        activity_id = self._begin_activity("transcribing", detail="Transcribing")
        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(get_executor("audio"), _transcribe_sync),
                timeout=60.0,
            )
            return result
        finally:
            await self._finish_activity(activity_id)

    def get_stats(self) -> dict[str, Any]:
        return {"model_name": self._model_name, "loaded": self._model is not None}

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<STTEngine model={self._model_name} status={status}>"
