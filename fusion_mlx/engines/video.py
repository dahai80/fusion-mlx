# SPDX-License-Identifier: Apache-2.0
# Video generation engine for fusion-mlx.
# Thin delegator over a registered VideoBackend (see video_backends/). The
# engine owns engine-level lifecycle (start/stop), activity tracking, and the
# public generate() surface; the backend owns the mlx-video / direct-MLX
# generate path and per-backend constraints. Phase 0 ships LTX2Backend only;
# adding backends does not require touching this file.

import logging
import time
from typing import Any

import mlx.core as mx

from .base import BaseNonStreamingEngine
from ._progress import StepCallback
from .video_backends import VideoGenParams, resolve_backend

logger = logging.getLogger(__name__)


class VideoGenEngine(BaseNonStreamingEngine):
    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self._model_name = model_name
        backend_name = kwargs.pop("backend", None)
        self._backend = resolve_backend(model_name, explicit=backend_name, **kwargs)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def _loaded(self) -> bool:
        return self._backend._loaded

    async def start(self) -> None:
        logger.info("Starting VideoGen engine: %s", self._model_name)
        await self._backend.start(self._model_name)
        logger.info("VideoGen engine ready: %s", self._model_name)

    async def stop(self) -> None:
        await self._backend.stop()

    async def generate(
        self,
        prompt: str,
        num_frames: int = 97,
        width: int = 768,
        height: int = 512,
        fps: int = 24,
        seed: int | None = None,
        n: int = 1,
        on_step: StepCallback | None = None,
        **kwargs,
    ) -> list[bytes]:
        if not self._loaded:
            raise RuntimeError("VideoGen engine not started.")

        params = VideoGenParams(
            prompt=prompt,
            n=n,
            num_frames=num_frames,
            width=width,
            height=height,
            fps=fps,
            seed=seed,
            negative_prompt=kwargs.get("negative_prompt"),
            image=kwargs.get("image"),
            image_strength=kwargs.get("image_strength", 1.0),
            num_inference_steps=kwargs.get("num_inference_steps"),
            cfg_scale=kwargs.get("cfg_scale"),
            guide_scale=kwargs.get("guide_scale"),
            pipeline=kwargs.get("pipeline"),
            scheduler=kwargs.get("scheduler"),
            shift=kwargs.get("shift"),
            tiling=kwargs.get("tiling"),
            no_compile=kwargs.get("no_compile"),
            enhance_prompt=kwargs.get("enhance_prompt"),
            on_step=on_step,
        )

        t0 = time.monotonic()
        activity_id = self._begin_activity(
            "generating video",
            metadata={"prompt_len": len(prompt), "num_frames": num_frames, "n": n},
        )

        try:
            result = await self._backend.generate(params)
            elapsed = time.monotonic() - t0
            self._update_activity(activity_id, elapsed_seconds=elapsed)
            logger.info("VideoGen generated %d video(s) in %.2fs", len(result), elapsed)
            return result
        finally:
            await self._finish_activity(activity_id)

    # ------------------------------------------------------------------
    # Pipeline stage API (issue #170). Thin delegation to the backend.
    # Backends override VideoBackend stage methods to expose individual
    # stages for Fusion-ComfyUI sequential offload. Video latents are 5D
    # (batch, c, num_frames, h, w); denoise carries num_frames. Backends
    # that have not implemented a stage raise NotImplementedError.
    # ------------------------------------------------------------------
    async def load_text_encoder(self) -> None:
        await self._backend.load_text_encoder()

    async def encode_text(self, prompt: str) -> dict:
        return await self._backend.encode_text(prompt)

    async def unload_text_encoder(self) -> None:
        await self._backend.unload_text_encoder()

    async def load_dit(self) -> None:
        await self._backend.load_dit()

    async def denoise(
        self,
        latent: mx.array,
        pos_embed: mx.array,
        neg_embed: mx.array | None,
        steps: int,
        cfg: float,
        seed: int,
        num_frames: int,
    ) -> mx.array:
        return await self._backend.denoise(
            latent, pos_embed, neg_embed, steps, cfg, seed, num_frames
        )

    async def unload_dit(self) -> None:
        await self._backend.unload_dit()

    async def load_vae(self) -> None:
        await self._backend.load_vae()

    async def decode(self, latent: mx.array) -> mx.array:
        return await self._backend.decode(latent)

    async def decode_tiled(self, latent: mx.array, tile_size: int = 256) -> mx.array:
        return await self._backend.decode_tiled(latent, tile_size)

    async def unload_vae(self) -> None:
        await self._backend.unload_vae()

    def get_stats(self) -> dict[str, Any]:
        return {"model_name": self._model_name, "loaded": self._loaded}

    def __repr__(self) -> str:
        status = "running" if self._loaded else "stopped"
        return f"<VideoGenEngine model={self._model_name} status={status}>"
