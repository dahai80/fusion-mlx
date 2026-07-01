# SPDX-License-Identifier: Apache-2.0
"""TTS (Text-to-Speech) engine for fusion-mlx."""

import asyncio
import gc
import inspect
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import mlx.core as mx
import numpy as np

from ..engine_core import get_executor
from .audio_utils import DEFAULT_SAMPLE_RATE as _DEFAULT_SAMPLE_RATE
from .audio_utils import audio_to_wav_bytes as _audio_to_wav_bytes
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


class TTSEngine(BaseNonStreamingEngine):
    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self._model_name = model_name
        self._model = None
        self._kwargs = kwargs

    @staticmethod
    def _audio_array_to_pcm_bytes(audio: Any) -> bytes:
        audio_array = np.array(audio).flatten()
        audio_array = np.clip(audio_array, -1.0, 1.0)
        return (audio_array * 32767).astype(np.int16).tobytes()

    @property
    def model_name(self) -> str:
        return self._model_name

    def supports_native_tts_streaming(self) -> bool:
        if self._model is None:
            return False
        try:
            gen_params = inspect.signature(self._model.generate).parameters
        except (TypeError, ValueError):
            return False
        return "stream" in gen_params and "streaming_interval" in gen_params

    async def start(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Starting TTS engine: {self._model_name}")
        try:
            from mlx_audio.tts.utils import load_model as _load_model
        except ImportError as exc:
            raise ImportError('mlx-audio required for TTS. Install with: pip install "fusion-mlx[audio]"') from exc

        model_name = self._model_name
        def _load_sync():
            try:
                return _load_model(model_name, strict=True)
            except ValueError as exc:
                if "Expected shape" not in str(exc):
                    raise
                logger.warning(f"Strict loading failed for {model_name}, retrying strict=False")
                return _load_model(model_name, strict=False)

        loop = asyncio.get_running_loop()
        self._model = await asyncio.wait_for(
            loop.run_in_executor(get_executor("audio"), _load_sync), timeout=120.0)

    async def stop(self) -> None:
        if self._model is None:
            return
        self._model = None
        gc.collect()
        loop = asyncio.get_running_loop()
        from ..scheduler.helpers import _safe_clear_cache_for_non_llm
        await asyncio.wait_for(
            loop.run_in_executor(get_executor("audio"), _safe_clear_cache_for_non_llm), timeout=5.0)

    async def synthesize(
        self, text: str, voice: str | None = None, speed: float = 1.0,
        instructions: str | None = None, ref_audio: str | None = None,
        ref_text: str | None = None, temperature: float | None = None,
        top_k: int | None = None, top_p: float | None = None,
        repetition_penalty: float | None = None, max_tokens: int | None = None, **kwargs,
        ) -> bytes:
        if self._model is None:
            raise RuntimeError("Engine not started.")
        model = self._model
        t0 = time.monotonic()

        def _build_kwargs() -> dict[str, Any]:
            gk: dict[str, Any] = {"text": text, "verbose": False}
            gp = inspect.signature(model.generate).parameters
            if voice is not None:
                gk["voice"] = voice if "voice" in gp else gk.setdefault("instruct", voice) if "instruct" in gp else None
            if instructions is not None and "instruct" in gp:
                gk["instruct"] = instructions
            if speed != 1.0:
                gk["speed"] = speed
            if ref_audio is not None and "ref_audio" in gp:
                gk["ref_audio"] = ref_audio
                gk["ref_text"] = ref_text
            for p, v in [("temperature", temperature), ("top_k", top_k), ("top_p", top_p), ("repetition_penalty", repetition_penalty), ("max_tokens", max_tokens)]:
                if v is not None:
                    gk[p] = v
            gk.update(kwargs)
            return gk

        def _synthesize_sync():
            results = model.generate(**_build_kwargs())
            sr = getattr(model, "sample_rate", _DEFAULT_SAMPLE_RATE)
            chunks = []
            for r in results:
                a = r.audio
                if isinstance(a, mx.array) and a.dtype == mx.bfloat16:
                    a = a.astype(mx.float32)
                chunks.append(np.array(a))
            if not chunks:
                raise RuntimeError("TTS model produced no audio output")
            return _audio_to_wav_bytes(np.concatenate(chunks, axis=0), int(sr))

        activity_id = self._begin_activity("synthesizing speech", metadata={"text_length": len(text)})
        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(get_executor("audio"), _synthesize_sync), timeout=60.0)
            return result
        finally:
            await self._finish_activity(activity_id)

    async def stream_synthesize_pcm(
        self, text: str, voice: str | None = None, speed: float = 1.0,
        instructions: str | None = None, ref_audio: str | None = None,
        ref_text: str | None = None, temperature: float | None = None,
        top_k: int | None = None, top_p: float | None = None,
        repetition_penalty: float | None = None, max_tokens: int | None = None,
        streaming_interval: float = 0.4, **kwargs,
        ) -> AsyncIterator[tuple[int, int, int, bytes]]:
        if self._model is None:
            raise RuntimeError("Engine not started.")
        if not self.supports_native_tts_streaming():
            raise NotImplementedError("Loaded TTS model does not expose native streaming")
        model = self._model
        t0 = time.monotonic()

        def _build_kwargs() -> dict[str, Any]:
            gk: dict[str, Any] = {"text": text, "verbose": False, "stream": True}
            gp = inspect.signature(model.generate).parameters
            if "streaming_interval" in gp:
                gk["streaming_interval"] = streaming_interval
            if voice is not None:
                gk["voice"] = voice if "voice" in gp else gk.setdefault("instruct", voice) if "instruct" in gp else None
            if instructions is not None and "instruct" in gp:
                gk["instruct"] = instructions
            if speed != 1.0:
                gk["speed"] = speed
            if ref_audio is not None and "ref_audio" in gp:
                gk["ref_audio"] = ref_audio
                gk["ref_text"] = ref_text
            for p, v in [("temperature", temperature), ("top_k", top_k), ("top_p", top_p), ("repetition_penalty", repetition_penalty), ("max_tokens", max_tokens)]:
                if v is not None:
                    gk[p] = v
            gk.update(kwargs)
            return gk

        iterator: Any = None
        sentinel = object()
        chunk_count = 0
        total_bytes = 0

        def _next_chunk():
            nonlocal iterator
            if iterator is None:
                iterator = iter(model.generate(**_build_kwargs()))
            try:
                result = next(iterator)
            except StopIteration:
                return sentinel
            audio = getattr(result, "audio", None)
            if audio is None:
                return None
            sr = int(getattr(result, "sample_rate", getattr(model, "sample_rate", _DEFAULT_SAMPLE_RATE)))
            return sr, 1, 2, self._audio_array_to_pcm_bytes(audio)

        activity_id = self._begin_activity("streaming speech", metadata={"text_length": len(text)})
        try:
            loop = asyncio.get_running_loop()
            while True:
                chunk = await asyncio.wait_for(
                    loop.run_in_executor(get_executor("audio"), _next_chunk), timeout=30.0)
                if chunk is sentinel:
                    break
                if chunk is None:
                    continue
                sr, ch, sw, pcm = chunk
                if not pcm:
                    continue
                chunk_count += 1
                total_bytes += len(pcm)
                self._update_activity(activity_id, chunk_count=chunk_count, output_bytes=total_bytes)
                yield sr, ch, sw, pcm
        finally:
            await self._finish_activity(activity_id)

    def get_stats(self) -> dict[str, Any]:
        return {"model_name": self._model_name, "loaded": self._model is not None}

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<TTSEngine model={self._model_name} status={status}>"
