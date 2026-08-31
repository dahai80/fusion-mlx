# SPDX-License-Identifier: Apache-2.0
# LTX-2.5 video backend (pure-MLX port). Mirrors LTX2Backend but targets the
# LTX-2.5 22B model (Gemma4-12b text encoder, duration-head, two-stage
# spatial+temporal upsampler). generate() delegates to
# fusion_mlx.video.ltx2_5.generate.generate_video (two-stage distilled T2V,
# real-model verified). Detection runs BEFORE LTX2Backend so an "ltx-2.5"
# path does not fall through the "ltx-2" substring match (AR doc §3.3 / §7).
import asyncio
import gc
import logging
import random
from typing import Any

import mlx.core as mx
import numpy as np

from ..._tempfile_safe import managed_tempfile_path
from ...engine_core import get_executor, get_video_gen_timeout
from .base import VideoBackend, VideoConstraints, VideoGenParams

logger = logging.getLogger(__name__)

_DEFAULT_PIPELINE = "distilled"
_LTX2_5_DIM_DIV = 32


class LTX2_5Backend(VideoBackend):
    name = "ltx2_5"
    supports_i2v = True

    def __init__(
        self,
        model_name: str,
        *,
        text_encoder_repo: str | None = None,
        pipeline: str = _DEFAULT_PIPELINE,
        two_stage: bool = True,
        **kwargs: Any,
    ) -> None:
        self._model_name = model_name
        self._text_encoder_repo = text_encoder_repo
        self._pipeline = pipeline
        self._two_stage = two_stage
        self._loaded = False

    @classmethod
    def detect(cls, model_path: str) -> bool:
        p = model_path.lower()
        # 显式匹配 2.5；不依赖 "ltx-2" 子串（避免与 LTX2Backend 冲突）。
        return "ltx-2.5" in p or "ltx_2.5" in p or "ltx2.5" in p

    async def start(self, model_path: str, **kwargs: Any) -> None:
        if self._loaded:
            return
        logger.info("Starting LTX-2.5 backend (pure-MLX): %s", model_path)

        def _resolve():
            from fusion_mlx.video.ltx2_5.utils import get_model_path

            return get_model_path(model_path)

        loop = asyncio.get_running_loop()
        self._model_path = await asyncio.wait_for(
            loop.run_in_executor(get_executor("io"), _resolve), timeout=180.0
        )
        self._loaded = True
        logger.info("LTX-2.5 backend ready: %s", model_path)

    async def stop(self) -> None:
        if not self._loaded:
            return
        self._loaded = False
        gc.collect()
        loop = asyncio.get_running_loop()
        # io executor (max_workers=2) 而非单 worker video executor，避免 stop()
        # 排在长生成任务后触发 5s 超时（mirror ltx2 backend）。
        await asyncio.wait_for(
            loop.run_in_executor(
                get_executor("io"), lambda: (mx.synchronize(), mx.clear_cache())
            ),
            timeout=5.0,
        )

    async def generate(self, params: VideoGenParams) -> list[bytes]:
        if params.on_step is not None:
            logger.debug(
                "ltx2_5: on_step progress callback accepted but per-step "
                "streaming not yet emitted for this backend (issue #171 follow-up)"
            )
        base_seed = (
            params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
        )

        def _generate():
            results: list[bytes] = []
            for i in range(max(1, params.n)):
                mp4_bytes = _generate_one(
                    self._model_name,
                    self._pipeline,
                    prompt=params.prompt,
                    num_frames=params.num_frames,
                    width=params.width,
                    height=params.height,
                    fps=params.fps,
                    seed=base_seed + i,
                    num_inference_steps=params.num_inference_steps,
                    cfg_scale=params.cfg_scale,
                    tiling=params.tiling,
                    image=params.image,
                    image_strength=params.image_strength,
                    two_stage=self._two_stage,
                )
                results.append(mp4_bytes)
            return results

        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(get_executor("video"), _generate),
            timeout=get_video_gen_timeout(),
        )

    async def load_vae_encoder(self) -> None:
        if getattr(self, "_stage_vae_encoder", None) is not None:
            return
        from pathlib import Path

        from fusion_mlx.video.ltx2_5.utils import resolve_component
        from fusion_mlx.video.ltx2_5.video_vae import load_video_encoder

        # load_video_encoder expects a safetensors FILE (calls _split_vae_weights
        # → mx.load), not the vae/encoder dir. resolve_component returns the conv
        # variant file, matching generate.py:185.
        enc_path = resolve_component(Path(self._model_path), "video_vae_conv")

        def _load():
            return load_video_encoder(enc_path)

        loop = asyncio.get_running_loop()
        self._stage_vae_encoder = await loop.run_in_executor(get_executor("io"), _load)
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags["vae_encoder"] = True
        logger.info(
            "ltx2_5: vae_encoder load vae=%s", type(self._stage_vae_encoder).__name__
        )

    async def encode(self, pixels: mx.array) -> mx.array:
        if getattr(self, "_stage_vae_encoder", None) is None:
            await self.load_vae_encoder()
        vae_enc = self._stage_vae_encoder
        ndim = pixels.ndim
        if ndim not in (4, 5):
            raise ValueError(
                f"encode expects (T,H,W,3) or (1,T,H,W,3); got {tuple(pixels.shape)}"
            )
        src_np = np.array(pixels[0] if ndim == 5 else pixels)

        def _encode():
            x = mx.array(src_np)
            if x.ndim == 4:
                x = x[None]
            lat = vae_enc(x)
            lat = lat if lat.ndim == 5 else lat[None]
            mx.eval(lat)
            return lat

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("video"), _encode)
        logger.info("ltx2_5: vae encode out_shape=%s", tuple(result.shape))
        return result

    async def unload_vae_encoder(self) -> None:
        self._stage_vae_encoder = None
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags.pop("vae_encoder", None)
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        logger.info("ltx2_5: vae_encoder unload")

    def constraints(self) -> VideoConstraints:
        return VideoConstraints(
            supports_i2v=True,
            max_n=4,
            dim_divisibility=_LTX2_5_DIM_DIV,
            num_frames_validator=lambda nf: nf % 8 == 1,
            num_frames_hint="num_frames % 8 == 1 (or omit to let duration-head decide)",
            dim_hint="width and height must be divisible by 32",
        )


def _generate_one(
    model_repo,
    pipeline,
    *,
    prompt: str,
    num_frames: int,
    width: int,
    height: int,
    fps: int,
    seed: int,
    num_inference_steps: int | None = None,
    cfg_scale: float | None = None,
    tiling: str | None = None,
    image: str | None = None,
    image_strength: float = 1.0,
    two_stage: bool = True,
) -> bytes:
    from fusion_mlx.video.ltx2_5.config import LTX2_5Variant
    from fusion_mlx.video.ltx2_5.generate import generate_video

    variant = LTX2_5Variant.DISTILLED if pipeline == "distilled" else LTX2_5Variant.DEV
    gen_kwargs: dict[str, Any] = dict(
        variant=variant,
        height=height,
        width=width,
        num_frames=num_frames,
        seed=seed,
        fps=fps,
        output_path=None,
        verbose=False,
        two_stage=two_stage,
    )
    if num_inference_steps is not None:
        gen_kwargs["num_inference_steps"] = num_inference_steps
    if cfg_scale is not None:
        gen_kwargs["cfg_scale"] = cfg_scale
    if tiling is not None:
        gen_kwargs["tiling"] = tiling
    if image is not None:
        gen_kwargs["image"] = image
        gen_kwargs["image_strength"] = image_strength
    with managed_tempfile_path(prefix="fusion_video_", suffix=".mp4") as handle:
        temp_path = handle.path
        gen_kwargs["output_path"] = temp_path
        logger.info(
            "VideoGen generate (ltx2_5): prompt_len=%d frames=%d %dx%d@%dfps "
            "seed=%d steps=%s cfg=%s tiling=%s image=%s two_stage=%s",
            len(prompt),
            num_frames,
            width,
            height,
            fps,
            seed,
            num_inference_steps,
            cfg_scale,
            tiling,
            image is not None,
            two_stage,
        )
        generate_video(model_repo, prompt, **gen_kwargs)
        with open(temp_path, "rb") as f:
            return f.read()
