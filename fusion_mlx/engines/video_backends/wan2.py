# SPDX-License-Identifier: Apache-2.0
# Wan2.2 video backend (mlx-video Blaizzy/mlx-video).
# Verified API: mlx_video.models.wan_2.generate.generate_video(model_dir,
# prompt, negative_prompt=, image=, width=, height=, num_frames=, steps=,
# guide_scale=, shift=, seed=, output_path=, scheduler=, no_compile=).
# Wan supports image-to-video (I2V) via image=. Wan's generate_video takes no
# fps argument (it controls container fps internally), so request fps is
# ignored for this backend. Signature confirmed against installed mlx-video.

import asyncio
import gc
import logging
import random
from typing import Any

import mlx.core as mx

from ..._tempfile_safe import managed_tempfile_path
from ...engine_core import get_executor
from .base import VideoBackend, VideoConstraints, VideoGenParams

logger = logging.getLogger(__name__)

# Compile is OFF by default (no_compile=True): the mx.compile path is faster
# but historically unverified for this backend. A caller opts in by passing
# no_compile=False. See VideoGenParams.no_compile.
_DEFAULT_SCHEDULER = "unipc"


class Wan2Backend(VideoBackend):
    name = "wan2"
    supports_i2v = True

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self._model_name = model_name
        self._loaded = False

    @classmethod
    def detect(cls, model_path: str) -> bool:
        return "wan" in model_path.lower()

    async def start(self, model_path: str, **kwargs: Any) -> None:
        if self._loaded:
            return
        logger.info("Starting Wan2 backend (mlx-video): %s", model_path)

        def _resolve():
            try:
                from mlx_video import get_model_path
            except ImportError as exc:
                raise ImportError(
                    "Video generation requires mlx-video (Blaizzy/mlx-video). "
                    "Install: pip install git+https://github.com/Blaizzy/mlx-video.git"
                ) from exc
            return get_model_path(model_path)

        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(get_executor("io"), _resolve), timeout=180.0
        )
        self._loaded = True
        logger.info("Wan2 backend ready: %s", model_path)

    async def stop(self) -> None:
        if not self._loaded:
            return
        self._loaded = False
        gc.collect()
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                get_executor("io"), lambda: (mx.synchronize(), mx.clear_cache())
            ),
            timeout=5.0,
        )

    async def generate(self, params: VideoGenParams) -> list[bytes]:
        base_seed = (
            params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
        )
        scheduler = params.scheduler or _DEFAULT_SCHEDULER
        # no_compile defaults to True (compile OFF = safe/verified path). A
        # caller opts into the faster compiled path by passing no_compile=False.
        no_compile = True if params.no_compile is None else params.no_compile

        def _generate():
            results: list[bytes] = []
            for i in range(max(1, params.n)):
                mp4_bytes = _generate_one(
                    self._model_name,
                    prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    image=params.image,
                    width=params.width,
                    height=params.height,
                    num_frames=params.num_frames,
                    steps=params.num_inference_steps,
                    guide_scale=params.guide_scale,
                    shift=params.shift,
                    seed=base_seed + i,
                    scheduler=scheduler,
                    no_compile=no_compile,
                    tiling=params.tiling,
                )
                results.append(mp4_bytes)
            return results

        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(get_executor("video"), _generate), timeout=600.0
        )

    def constraints(self) -> VideoConstraints:
        # Wan2.2 5B: num_frames = 4k+1 (VAE temporal compression), spatial dims
        # divisible by 16. Supports I2V.
        return VideoConstraints(
            supports_i2v=True,
            max_n=4,
            dim_divisibility=16,
            num_frames_validator=lambda nf: (nf - 1) % 4 == 0,
            num_frames_hint="num_frames must satisfy (num_frames - 1) % 4 == 0 "
            "(e.g. 41, 81, 121)",
            dim_hint="width and height must be divisible by 16",
        )


def _generate_one(
    model_dir,
    *,
    prompt: str,
    negative_prompt: str | None = None,
    image: str | None = None,
    width: int,
    height: int,
    num_frames: int,
    steps: int | None,
    guide_scale: float | None,
    shift: float | None,
    seed: int,
    scheduler: str,
    no_compile: bool = True,
    tiling: str | None = None,
) -> bytes:
    from mlx_video.models.wan_2.generate import generate_video

    with managed_tempfile_path(prefix="fusion_video_", suffix=".mp4") as handle:
        temp_path = handle.path
        logger.info(
            "Wan2 generate: prompt_len=%d frames=%d %dx%d seed=%d i2v=%s "
            "steps=%s compile=%s tiling=%s",
            len(prompt),
            num_frames,
            width,
            height,
            seed,
            bool(image),
            steps,
            not no_compile,
            tiling,
        )
        gen_kwargs: dict[str, Any] = dict(
            negative_prompt=negative_prompt,
            image=image,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            guide_scale=guide_scale,
            shift=shift,
            seed=seed,
            output_path=temp_path,
            scheduler=scheduler,
            no_compile=no_compile,
        )
        if tiling is not None:
            gen_kwargs["tiling"] = tiling
        generate_video(model_dir, prompt, **gen_kwargs)
        with open(temp_path, "rb") as f:
            return f.read()
