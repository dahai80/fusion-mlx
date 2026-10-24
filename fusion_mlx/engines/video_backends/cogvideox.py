# SPDX-License-Identifier: Apache-2.0
import asyncio
import gc
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from ..._tempfile_safe import managed_tempfile_path
from ...engine_core import get_executor, get_video_gen_timeout
from .._progress import make_sync_step_callback
from .base import VideoBackend, VideoConstraints, VideoGenParams

logger = logging.getLogger(__name__)


class CogVideoBackend(VideoBackend):
    name = "cogvideo"
    aliases = ["cogvideox", "cog_video", "cogvideo-x"]

    def __init__(self, model_path: str | Path, **kwargs):
        self.model_path = Path(model_path)
        self._model_dir = None
        self._loaded = False
        self._stage_vae_encoder = None
        self._stage_flags: dict[str, bool] = {}
        self._kwargs = kwargs

    @classmethod
    def detect(cls, model_path: str | Path) -> bool:
        p = Path(model_path)
        name = p.name.lower()
        if any(k in name for k in ("cogvideo", "cog_video")):
            return True
        if (p / "config.json").exists():
            try:
                import json

                with open(p / "config.json") as f:
                    cfg = json.load(f)
                if cfg.get("_class_name", "").lower().startswith("cogvideo"):
                    return True
                arch = cfg.get("architectures", [])
                if any("cogvideo" in a.lower() for a in arch):
                    return True
            except Exception:
                pass
        return False

    def constraints(self) -> VideoConstraints:
        return VideoConstraints(
            supports_i2v=True,
            max_n=1,
            dim_divisibility=16,
            num_frames_validator=lambda n: 9 <= n <= 161,
            num_frames_hint="9-161 frames (odd number recommended)",
            dim_hint="720x480 default, must be multiple of 16",
        )

    async def start(self, model_path: str, **kwargs) -> None:
        from fusion_mlx.video.cogvideox.utils import get_model_path

        if self._loaded:
            return
        loop = asyncio.get_running_loop()
        self._model_dir = await asyncio.wait_for(
            loop.run_in_executor(get_executor("io"), get_model_path, model_path),
            timeout=180.0,
        )
        self._loaded = True
        logger.info(f"CogVideoBackend started: {self._model_dir}")

    async def stop(self) -> None:
        self._loaded = False
        self._model_dir = None
        self._stage_vae_encoder = None
        self._stage_flags.pop("vae_encoder", None)
        gc.collect()
        # M9 (#1354 audit): never touch MX streams from the event-loop thread.
        # mx.synchronize()/mx.clear_cache() run on the io executor so the
        # worker's thread-local stream table survives (wan2.py:98-109).
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                get_executor("io"), lambda: (mx.synchronize(), mx.clear_cache())
            ),
            timeout=5.0,
        )
        logger.info("CogVideoBackend stopped")

    async def generate(self, params: VideoGenParams) -> list[bytes]:
        from fusion_mlx.video.cogvideox.generate import generate_video

        if params.output_format != "mp4":
            raise NotImplementedError(
                "cogvideox: only output_format='mp4' is supported "
                f"(got {params.output_format!r})"
            )
        base_seed = params.seed if params.seed is not None else -1
        no_compile = True if params.no_compile is None else params.no_compile
        loop = asyncio.get_running_loop()
        on_step_sync = make_sync_step_callback(params.on_step, loop)

        def _generate() -> list[bytes]:
            results: list[bytes] = []
            for i in range(max(1, params.n)):
                with managed_tempfile_path(
                    prefix="fusion_video_", suffix=".mp4"
                ) as handle:
                    generate_video(
                        model_dir=str(self._model_dir),
                        prompt=params.prompt,
                        negative_prompt=params.negative_prompt,
                        image=params.image,
                        width=params.width,
                        height=params.height,
                        num_frames=params.num_frames,
                        steps=params.num_inference_steps,
                        guide_scale=params.guide_scale,
                        shift=params.shift,
                        seed=base_seed + i if base_seed >= 0 else base_seed,
                        output_path=handle.path,
                        no_compile=no_compile,
                        on_step_sync=on_step_sync,
                        session_id=params.session_id,
                        # #731 Surface B+C: thread controlnet_image
                        # (fail-visible in generate_video — no backend model)
                        # and inpaint surfaces.
                        controlnet_image=params.controlnet_image,
                        inpaint_mask=params.inpaint_mask,
                        init_latent=params.init_latent,
                    )
                    with open(handle.path, "rb") as f:
                        results.append(f.read())
            return results

        return await asyncio.wait_for(
            loop.run_in_executor(get_executor("video"), _generate),
            timeout=get_video_gen_timeout(),
        )

    async def load_vae_encoder(self) -> None:
        if getattr(self, "_stage_vae_encoder", None) is not None:
            return
        from fusion_mlx.video.cogvideox.utils import load_config as _load_cog_config
        from fusion_mlx.video.cogvideox.utils import load_vae_encoder as _load_vae_enc

        vae_path = Path(self._model_dir) / "vae.safetensors"

        def _load():
            cfg, _quant = _load_cog_config(Path(self._model_dir))
            return _load_vae_enc(vae_path, cfg)

        loop = asyncio.get_running_loop()
        self._stage_vae_encoder = await loop.run_in_executor(get_executor("io"), _load)
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags["vae_encoder"] = True
        logger.info(
            "cogvideox: vae_encoder load vae=%s",
            type(self._stage_vae_encoder).__name__,
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
            lat = vae_enc.encode(x)
            lat = lat if lat.ndim == 5 else lat[None]
            mx.eval(lat)
            return lat

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("video"), _encode)
        logger.info("cogvideox: vae encode out_shape=%s", tuple(result.shape))
        return result

    async def unload_vae_encoder(self) -> None:
        self._stage_vae_encoder = None
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags.pop("vae_encoder", None)
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        logger.info("cogvideox: vae_encoder unload")

    async def encode_control(self, **kwargs) -> Any:
        # #731 Surface B: ControlNet conditioning is not available for cogvideox
        # — the shared ControlNet adapter is Wan2-arch (text_dim=4096,
        # patch_size=[1,2,2], token-residual block injection) and no per-backend
        # ControlNet model exists for cogvideox. Fail visibly (Rule 12): a
        # caller asking for ControlNet must NOT silently degrade to T2V.
        if kwargs.get("controlnet_image") is not None:
            raise RuntimeError(
                "cogvideox: ControlNet (Surface B) not available for this "
                "backend — no per-backend ControlNet model (see issue #731 "
                "follow-up). Refusing to silently degrade to T2V (#731)."
            )
        # No conditioning surfaces implemented for cogvideox beyond VAE encode
        # (Surface A, already on load_vae_encoder/encode). Pure T2V -> None.
        logger.info("cogvideox: encode_control pure-T2V (no controlnet/vace)")
        return None
