# SPDX-License-Identifier: Apache-2.0
# Video generation engine (LTX-2) for fusion-mlx.
# Loads mlx-video LTX-2 models via the mlx-video library (Blaizzy/mlx-video).
#
# NOTE: mlx-video's load()/generate() call shape is researched from the
# upstream README (github.com was unreachable at impl time, so the API could
# not be runtime-verified). The two mlx-video calls are isolated in _load()
# and _generate_one() so a fix is localized if the shape differs. See
# docs/plans/video-generation-engine.md (Risk: mlx-video API drift).
import asyncio
import gc
import logging
import os
import tempfile
import time
from typing import Any

import mlx.core as mx

from ..engine_core import get_executor
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


class VideoGenEngine(BaseNonStreamingEngine):
    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self._model_name = model_name
        self._model_path = model_name
        self._model = None
        self._processor = None
        self._kwargs = kwargs

    @property
    def model_name(self) -> str:
        return self._model_name

    async def start(self) -> None:
        if self._model is not None:
            return
        logger.info("Starting VideoGen engine (mlx-video): %s", self._model_path)

        def _load():
            try:
                from mlx_video import load
            except ImportError as exc:
                raise ImportError(
                    "Video generation requires mlx-video. "
                    "Install it: pip install git+https://github.com/Blaizzy/mlx-video.git"
                ) from exc
            # mlx-video load() mirrors mlx-vlm (same author): returns (model, processor).
            model, processor = load(self._model_path)
            return model, processor

        loop = asyncio.get_running_loop()
        self._model, self._processor = await asyncio.wait_for(
            loop.run_in_executor(get_executor("io"), _load), timeout=180.0
        )
        logger.info("VideoGen engine loaded: %s", self._model_name)

    async def stop(self) -> None:
        if self._model is None:
            return
        self._model = None
        self._processor = None
        gc.collect()
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                get_executor("video"), lambda: (mx.synchronize(), mx.clear_cache())
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
        if self._model is None:
            raise RuntimeError("VideoGen engine not started.")

        model = self._model
        processor = self._processor
        base_seed = seed if seed is not None else 0
        t0 = time.monotonic()
        activity_id = self._begin_activity(
            "generating video",
            metadata={"prompt_len": len(prompt), "num_frames": num_frames, "n": n},
        )

        def _generate():
            results: list[bytes] = []
            for i in range(max(1, n)):
                mp4_bytes = _generate_one(
                    model,
                    processor,
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
            logger.info(
                "VideoGen generated %d video(s) in %.2fs",
                len(result),
                elapsed,
            )
            return result
        finally:
            await self._finish_activity(activity_id)

    def get_stats(self) -> dict[str, Any]:
        return {"model_name": self._model_name, "loaded": self._model is not None}

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<VideoGenEngine model={self._model_name} status={status}>"


def _generate_one(
    model,
    processor,
    *,
    prompt: str,
    num_frames: int,
    width: int,
    height: int,
    fps: int,
    seed: int,
) -> bytes:
    # Isolated mlx-video generate call (researched API; single point to fix).
    # mlx-video generate() writes the mp4 to output_path and returns the video array.
    from mlx_video import generate

    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    try:
        logger.info(
            "VideoGen generate: prompt_len=%d frames=%d %dx%d@%dfps seed=%d",
            len(prompt),
            num_frames,
            width,
            height,
            fps,
            seed,
        )
        generate(
            model,
            processor,
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            seed=seed,
            output_path=temp_path,
        )
        with open(temp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            logger.warning("VideoGen: failed to unlink temp %s", temp_path)
