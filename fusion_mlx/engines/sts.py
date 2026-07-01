# SPDX-License-Identifier: Apache-2.0
"""STS (Speech-to-Speech) engine for fusion-mlx."""

import asyncio
import gc
import logging
import os
import tempfile
from typing import Any

import mlx.core as mx
import numpy as np

from ..engine_core import get_executor
from .audio_utils import DEFAULT_SAMPLE_RATE as _DEFAULT_SAMPLE_RATE
from .audio_utils import audio_to_wav_bytes as _audio_to_wav_bytes
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)

_CONFIG_TYPE_TO_FAMILY: dict[str, str] = {
    "deepfilternet": "deepfilternet", "mossformer2_se": "mossformer2",
    "sam_audio": "sam_audio", "lfm_audio": "lfm2", "lfm2_audio": "lfm2", "lfm2": "lfm2",
    "DeepFilterNetModel": "deepfilternet", "MossFormer2SEModel": "mossformer2",
    "SAMAudio": "sam_audio", "LFM2AudioModel": "lfm2",
}


def _detect_sts_family(model_name: str, config_model_type: str = "") -> str:
    if config_model_type:
        f = _CONFIG_TYPE_TO_FAMILY.get(config_model_type.lower())
        if f:
            return f
    config_path = os.path.join(model_name, "config.json")
    if os.path.isfile(config_path):
        try:
            import json
            with open(config_path) as f:
                cfg = json.load(f)
            for arch in cfg.get("architectures", []):
                fam = _CONFIG_TYPE_TO_FAMILY.get(arch)
                if fam:
                    return fam
            mt = cfg.get("model_type", "")
            fam = _CONFIG_TYPE_TO_FAMILY.get(mt.lower())
            if fam:
                return fam
        except (OSError, ValueError):
            pass
    nl = model_name.lower()
    if "deepfilter" in nl:
        return "deepfilternet"
    if "mossformer" in nl:
        return "mossformer2"
    if "sam-audio" in nl or "sam_audio" in nl:
        return "sam_audio"
    if "lfm2" in nl or "lfm-audio" in nl or "lfm_audio" in nl:
        return "lfm2"
    return "generic"


# Family loaders
def _load_deepfilternet(model_name: str):
    from mlx_audio.sts.models.deepfilternet import DeepFilterNetModel
    return DeepFilterNetModel.from_pretrained(model_name_or_path=model_name)

def _load_mossformer2(model_name: str):
    from mlx_audio.sts.models.mossformer2_se import MossFormer2SEModel
    return MossFormer2SEModel.from_pretrained(model_name)

def _load_sam_audio(model_name: str):
    from mlx_audio.sts.models.sam_audio import SAMAudio
    return SAMAudio.from_pretrained(model_name)

def _load_lfm2(model_name: str):
    from mlx_audio.sts.models.lfm_audio import LFM2AudioModel, LFM2AudioProcessor
    return LFM2AudioModel.from_pretrained(model_name), LFM2AudioProcessor.from_pretrained(model_name)

_FAMILY_LOADERS = {
    "deepfilternet": _load_deepfilternet, "mossformer2": _load_mossformer2,
    "sam_audio": _load_sam_audio, "lfm2": _load_lfm2,
}


def _process_deepfilternet(model, audio_path: str, **kwargs) -> bytes:
    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        model.enhance_file(str(audio_path), out_path)
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def _process_mossformer2(model, audio_path: str, **kwargs) -> bytes:
    enhanced = model.enhance(str(audio_path))
    sr = getattr(model.config, "sample_rate", 48000)
    return _audio_to_wav_bytes(enhanced, int(sr))


def _process_sam_audio(model, audio_path: str, **kwargs) -> bytes:
    descriptions = kwargs.get("descriptions", ["speech"])
    result = model.separate(audios=[str(audio_path)], descriptions=descriptions)
    target = result.target[0] if isinstance(result.target, list) else result.target
    sr = getattr(getattr(model, "config", None), "sample_rate", _DEFAULT_SAMPLE_RATE)
    return _audio_to_wav_bytes(target, int(sr))


def _process_lfm2(model_and_processor, audio_path: str, **kwargs) -> bytes:
    from mlx_audio.sts.models.lfm_audio import ChatState, LFMModality
    model, processor = model_and_processor
    from mlx_audio import audio_io
    audio_np, sr = audio_io.read(str(audio_path))
    audio_mx = mx.array(audio_np.flatten(), dtype=mx.float32)
    chat_state = ChatState(processor)
    chat_state.new_turn("user")
    chat_state.add_audio(audio_mx, sample_rate=sr)
    chat_state.end_turn()
    chat_state.new_turn("assistant")
    max_new_tokens = kwargs.get("max_new_tokens", 512)
    temperature = kwargs.get("temperature", 0.7)
    audio_temperature = kwargs.get("audio_temperature", 0.8)
    audio_frames = []
    for token, modality in model.generate_from_chat_state(
        chat_state, max_new_tokens=max_new_tokens, temperature=temperature, audio_temperature=audio_temperature,
    ):
        if modality == LFMModality.AUDIO_OUT:
            audio_frames.append(token)
    if not audio_frames:
        return _audio_to_wav_bytes(np.zeros(1600, dtype=np.float32), _DEFAULT_SAMPLE_RATE)
    codes = mx.stack(audio_frames, axis=0)
    if codes.ndim == 3:
        codes = codes.squeeze(1)
    codes = codes.transpose(1, 0)
    codes = codes[None, :, :]
    waveform = processor.decode_audio(codes)
    sr = getattr(getattr(model, "config", None), "sample_rate", _DEFAULT_SAMPLE_RATE)
    return _audio_to_wav_bytes(waveform, int(sr))


_FAMILY_PROCESSORS = {
    "deepfilternet": _process_deepfilternet, "mossformer2": _process_mossformer2,
    "sam_audio": _process_sam_audio, "lfm2": _process_lfm2,
}


class STSEngine(BaseNonStreamingEngine):
    def __init__(self, model_name: str, config_model_type: str = "", **kwargs):
        super().__init__()
        self._model_name = model_name
        self._model = None
        self._family = _detect_sts_family(model_name, config_model_type)
        self._kwargs = kwargs

    @property
    def model_name(self) -> str:
        return self._model_name

    async def start(self) -> None:
        if self._model is not None:
            return
        family = self._family
        logger.info(f"Starting STS engine: {self._model_name} (family={family})")
        loader = _FAMILY_LOADERS.get(family)
        if loader is None:
            raise ValueError(f"Unsupported STS family: {family!r}. Supported: {sorted(_FAMILY_LOADERS)}")
        model_name = self._model_name
        def _load_sync():
            try:
                return loader(model_name)
            except ImportError as exc:
                raise ImportError('mlx-audio required for STS. Install with: pip install "fusion-mlx[audio]"') from exc
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

    async def process(self, audio_path: str, **kwargs) -> bytes:
        if self._model is None:
            raise RuntimeError("Engine not started.")
        family = self._family
        processor_fn = _FAMILY_PROCESSORS.get(family)
        if processor_fn is None:
            raise ValueError(f"Unsupported STS family: {family!r}")
        model = self._model
        def _process_sync():
            return processor_fn(model, str(audio_path), **kwargs)
        activity_id = self._begin_activity("processing audio", metadata={"family": family})
        try:
            loop = asyncio.get_running_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(get_executor("audio"), _process_sync), timeout=60.0)
        finally:
            await self._finish_activity(activity_id)

    def get_stats(self) -> dict[str, Any]:
        return {"model_name": self._model_name, "loaded": self._model is not None, "family": self._family}

    def __repr__(self) -> str:
        s = "running" if self._model is not None else "stopped"
        return f"<STSEngine model={self._model_name} family={self._family} {s}>"
