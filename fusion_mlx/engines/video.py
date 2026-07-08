# SPDX-License-Identifier: Apache-2.0
# Video generation engine (LTX-2) for fusion-mlx.
# Wraps mlx-video's generate_video (Blaizzy/mlx-video). The verified API is
# mlx_video.models.ltx_2.generate.generate_video(model_repo, text_encoder_repo,
# prompt, ...). generate_video loads weights internally on every call (there is
# no module-level model cache), so start() front-loads the network download via
# get_model_path; generation then reads weights from the local HF cache.
import asyncio
import gc
import logging
import random
import time
from typing import Any

import mlx.core as mx

from .._tempfile_safe import managed_tempfile_path
from ..engine_core import get_executor
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)

_DEFAULT_PIPELINE = "distilled"


class VideoGenEngine(BaseNonStreamingEngine):
    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self._model_name = model_name
        self._text_encoder_repo = kwargs.get("text_encoder_repo")
        self._pipeline = kwargs.get("pipeline", _DEFAULT_PIPELINE)
        self._loaded = False
        self._kwargs = kwargs

    @property
    def model_name(self) -> str:
        return self._model_name

    async def start(self) -> None:
        if self._loaded:
            return
        logger.info("Starting VideoGen engine (mlx-video LTX-2): %s", self._model_name)

        def _resolve():
            try:
                from mlx_video import get_model_path
            except ImportError as exc:
                raise ImportError(
                    "Video generation requires mlx-video (Blaizzy/mlx-video, which "
                    "ships generate_video). NOTE: the PyPI 'mlx-video' 0.1.0 is a "
                    "different video-IO library and will NOT work. Install the "
                    "correct package: pip install git+https://github.com/Blaizzy/mlx-video.git"
                ) from exc
            return get_model_path(self._model_name)

        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(get_executor("io"), _resolve), timeout=180.0
        )
        self._loaded = True
        logger.info("VideoGen engine ready: %s", self._model_name)

    async def stop(self) -> None:
        if not self._loaded:
            return
        self._loaded = False
        gc.collect()
        loop = asyncio.get_running_loop()
        # Use the io executor (max_workers=2) rather than the single-worker
        # video executor so a stop() issued while a 600s generation is running
        # does not queue behind that generation and trip the 5s timeout.
        await asyncio.wait_for(
            loop.run_in_executor(
                get_executor("io"), lambda: (mx.synchronize(), mx.clear_cache())
            ),
            timeout=5.0,
        )

    async def generate(
        self,
        prompt: str,
        num_frames: int = 97,
        width: int = 768,
        height: int = 512,
        fps: int = 24,
        seed: int | None = None,
        n: int = 1,
        **kwargs,
    ) -> list[bytes]:
        if not self._loaded:
            raise RuntimeError("VideoGen engine not started.")

        base_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        t0 = time.monotonic()
        activity_id = self._begin_activity(
            "generating video",
            metadata={"prompt_len": len(prompt), "num_frames": num_frames, "n": n},
        )

        def _generate():
            results: list[bytes] = []
            for i in range(max(1, n)):
                mp4_bytes = _generate_one(
                    self._model_name,
                    self._text_encoder_repo,
                    self._pipeline,
                    prompt=prompt,
                    num_frames=num_frames,
                    width=width,
                    height=height,
                    fps=fps,
                    seed=base_seed + i,
                )
                results.append(mp4_bytes)
            return results

        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(get_executor("video"), _generate), timeout=600.0
            )
            elapsed = time.monotonic() - t0
            self._update_activity(activity_id, elapsed_seconds=elapsed)
            logger.info("VideoGen generated %d video(s) in %.2fs", len(result), elapsed)
            return result
        finally:
            await self._finish_activity(activity_id)

    def get_stats(self) -> dict[str, Any]:
        return {"model_name": self._model_name, "loaded": self._loaded}

    def __repr__(self) -> str:
        status = "running" if self._loaded else "stopped"
        return f"<VideoGenEngine model={self._model_name} status={status}>"


def _generate_one(
    model_repo,
    text_encoder_repo,
    pipeline,
    *,
    prompt: str,
    num_frames: int,
    width: int,
    height: int,
    fps: int,
    seed: int,
) -> bytes:
    from mlx_video.models.ltx_2.generate import PipelineType, generate_video

    pipe = PipelineType(pipeline) if isinstance(pipeline, str) else pipeline
    with managed_tempfile_path(prefix="fusion_video_", suffix=".mp4") as handle:
        temp_path = handle.path
        logger.info(
            "VideoGen generate: prompt_len=%d frames=%d %dx%d@%dfps seed=%d",
            len(prompt),
            num_frames,
            width,
            height,
            fps,
            seed,
        )
        generate_video(
            model_repo,
            text_encoder_repo,
            prompt,
            pipeline=pipe,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            fps=fps,
            output_path=temp_path,
            verbose=False,
        )
        with open(temp_path, "rb") as f:
            return f.read()
