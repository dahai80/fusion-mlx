# SPDX-License-Identifier: Apache-2.0
"""Music generation engine for ACE-Step1.5 (issue #988). Loads the MLX
orchestrator (DiT + VAE + Qwen3-Embedding) and exposes generate_music()."""

import asyncio
import gc
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..engine_core import get_executor
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)

ACESTEP_DEFAULT_REPO = "ACE-Step/Ace-Step1.5"
TEXT_ENCODER_DEFAULT = "Qwen/Qwen3-Embedding-0.6B"


class MusicGenEngine(BaseNonStreamingEngine):
    # ACE-Step1.5 turbo music generation engine. model_name is the ACE-Step
    # checkpoint alias (resolved to ~/.fusion-mlx/models snapshot dir).
    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self._model_name = model_name
        self._orchestrator = None
        self._kwargs = kwargs
        self._text_encoder_repo = kwargs.get(
            "text_encoder", os.environ.get("ACESTEP_TEXT_ENCODER", TEXT_ENCODER_DEFAULT)
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self) -> Any:
        return None

    async def start(self) -> None:
        if self._orchestrator is not None:
            return
        logger.info("starting MusicGenEngine: %s", self._model_name)
        checkpoint_dir = self._resolve_checkpoint_dir()

        def _load_sync():
            from ..audio.acestep.weights import load_acestep_orchestrator

            return load_acestep_orchestrator(
                checkpoint_dir,
                text_encoder_repo=self._text_encoder_repo,
                load_text_encoder=True,
            )

        loop = asyncio.get_running_loop()
        self._orchestrator = await asyncio.wait_for(
            loop.run_in_executor(get_executor("audio"), _load_sync), timeout=300.0
        )
        logger.info("MusicGenEngine ready: %s", self._model_name)

    def _resolve_checkpoint_dir(self) -> str:
        # Resolve ACE-Step alias to local snapshot dir under ~/.fusion-mlx/models.
        # A snapshot dir qualifies if it contains acestep-v15-turbo/model.safetensors.
        # The HF cache may contain stray sibling dirs (e.g. a converted
        # silence_latent snapshot) — check ALL snapshots, not just the first.
        name = self._model_name
        candidates: list[Path] = []
        if name.startswith("/") or name.startswith("~"):
            candidates.append(Path(os.path.expanduser(name)))
        else:
            base = Path(os.path.expanduser("~/.fusion-mlx/models"))
            short = name.split("/")[-1]
            for pat in [
                f"models--ACE-Step--{short}",
                f"models--{short.replace('/', '--')}",
            ]:
                d = base / pat / "snapshots"
                if d.exists():
                    candidates.extend(d.iterdir())
        for c in candidates:
            if (c / "acestep-v15-turbo" / "model.safetensors").exists():
                logger.info("acestep checkpoint dir: %s", c)
                return str(c)
        raise FileNotFoundError(
            f"ACE-Step checkpoint not found for {name!r}. "
            f"Pull with: fusion-mlx pull ACE-Step/Ace-Step1.5"
        )

    async def stop(self) -> None:
        self._orchestrator = None
        gc.collect()
        loop = asyncio.get_running_loop()
        from ..scheduler.helpers import _safe_clear_cache_for_non_llm

        await asyncio.wait_for(
            loop.run_in_executor(get_executor("audio"), _safe_clear_cache_for_non_llm),
            timeout=5.0,
        )

    async def generate_music(
        self,
        caption: str,
        lyrics: str = "",
        duration: float = 30.0,
        language: str = "en",
        bpm: str = "N/A",
        timesignature: str = "N/A",
        keyscale: str = "N/A",
        seed: int | None = 42,
        shift: float = 1.0,
        infer_method: str = "ode",
    ) -> tuple[np.ndarray, int]:
        # Returns (wav float32 [samples, 2], sample_rate).
        if self._orchestrator is None:
            raise RuntimeError("MusicGenEngine not started")

        orch = self._orchestrator
        t0 = time.time()
        loop = asyncio.get_running_loop()

        def _gen():
            return orch.generate_music(
                caption=caption,
                lyrics=lyrics,
                duration=duration,
                language=language,
                bpm=bpm,
                timesignature=timesignature,
                keyscale=keyscale,
                seed=seed,
                shift=shift,
                infer_method=infer_method,
            )

        wav = await loop.run_in_executor(get_executor("audio"), _gen)
        wav_np = np.array(wav[0], copy=False).astype(np.float32).T  # (samples, 2)
        sr = 48000
        logger.info(
            "music gen done: %s caption=%r dur=%.1fs took %.2fs",
            wav_np.shape,
            caption[:60],
            duration,
            time.time() - t0,
        )
        return wav_np, sr

    def get_stats(self) -> dict[str, Any]:
        return {
            "model_name": self._model_name,
            "loaded": self._orchestrator is not None,
        }


__all__ = ["MusicGenEngine"]
