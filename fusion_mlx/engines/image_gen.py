# SPDX-License-Identifier: Apache-2.0
# Image generation engine (Flux 2) for fusion-mlx.
# Loads FLUX.2-klein via mflux (Flux2Klein uses mx.compile -> faster denoise).
import asyncio
import gc
import io
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import mlx.core as mx

from ..engine_core import get_executor
from ._progress import StepCallback, make_sync_step_callback
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


def _to_latent_size(image_size: tuple[int, int]) -> tuple[int, int]:
    h, w = image_size
    h = ((h + 15) // 16) * 16
    w = ((w + 15) // 16) * 16
    return (h // 8, w // 8)


def _infer_flux2_config(model_path: str) -> str:
    name = (model_path or "").lower()
    if "base" in name and "4b" in name:
        return "flux2_klein_base_4b"
    if "base" in name and "9b" in name:
        return "flux2_klein_base_9b"
    if "4b" in name:
        return "flux2_klein_4b"
    if "kv" in name and "9b" in name:
        return "flux2_klein_9b_kv"
    return "flux2_klein_9b"


def _flux_quantize_from_env() -> int | None:
    import os

    env = os.environ.get("FUSION_FLUX_QUANT", "").strip().lower()
    if env in ("", "0", "off", "none", "bf16"):
        return None
    if env in ("w8a16", "w8", "int8", "8"):
        logger.info("Flux2Klein 量化: w8a16 (quantize=8)")
        return 8
    if env in ("w4", "nf4", "int4", "4"):
        logger.info("Flux2Klein 量化: w4 (quantize=4)")
        return 4
    logger.warning(
        "FUSION_FLUX_QUANT=%s 未知, 支持 w8a16/w4/off, 跳过量化",
        env,
    )
    return None


class _StepProgressInLoop:
    # mflux CallbackRegistry subscriber (InLoopCallback protocol). Registered
    # on flux.callbacks before generate_image; call_in_loop fires once per
    # denoise step AFTER the step completes. Bridges to the async on_step via
    # the sync callback built by make_sync_step_callback. Count resets per
    # image (registered/unregistered around each generate_image call).
    def __init__(self, sync_cb: Callable[[int, int], None] | None, total: int):
        self._sync_cb = sync_cb
        self._total = total
        self._count = 0

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps) -> None:
        self._count += 1
        if self._sync_cb is not None:
            self._sync_cb(self._count, self._total)


class ImageGenEngine(BaseNonStreamingEngine):
    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self._model_name = model_name
        self._model_path = model_name
        self._flux = None
        self._mflux_missing = False
        self._kwargs = kwargs
        # Load-time quantization (4/8-bit) reduces memory and often speeds up
        # Flux2 klein. None = load at the model's native precision. Passed
        # through to mflux Flux2Klein(quantize=...).
        self._quantize = kwargs.get("quantize")
        if self._quantize is None:
            self._quantize = _flux_quantize_from_env()

    @property
    def model_name(self) -> str:
        return self._model_name

    async def start(self) -> None:
        if self._flux is not None:
            return
        try:
            from mflux.models.common.config.model_config import ModelConfig
            from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
        except ImportError as exc:
            logger.warning(
                "ImageGen engine disabled: mflux-fusion not installed. "
                "Install with: pip install mflux-fusion  (%s)",
                exc,
            )
            self._mflux_missing = True
            return
        logger.info("Starting ImageGen engine (mflux Flux2Klein): %s", self._model_path)

        def _load():
            label = _infer_flux2_config(self._model_path)
            model_config = getattr(ModelConfig, label)()
            logger.info(
                "ImageGen loading flux2 config=%s path=%s",
                label,
                self._model_path,
            )
            flux = Flux2Klein(
                model_config=model_config,
                model_path=self._model_path,
                quantize=self._quantize,
            )
            return flux

        loop = asyncio.get_running_loop()
        self._flux = await asyncio.wait_for(
            loop.run_in_executor(get_executor("image"), _load), timeout=600.0
        )
        logger.info("ImageGen engine loaded: %s", self._model_name)

    async def stop(self) -> None:
        if self._flux is None:
            return
        self._flux = None
        gc.collect()
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                get_executor("image"), lambda: (mx.synchronize(), mx.clear_cache())
            ),
            timeout=5.0,
        )

    async def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
        seed: int | None = None,
        guidance: float = 4.0,
        n_images: int = 1,
        output_format: str = "PNG",
        scheduler: str | None = None,
        negative_prompt: str | None = None,
        on_step: StepCallback | None = None,
        **kwargs,
    ) -> list[bytes]:
        if self._flux is None:
            if self._mflux_missing:
                raise RuntimeError(
                    "Image generation unavailable: mflux-fusion not installed. "
                    "Install with: pip install mflux-fusion"
                )
            raise RuntimeError("ImageGen engine not started.")

        flux = self._flux
        base_seed = seed if seed is not None else 0
        t0 = time.monotonic()
        activity_id = self._begin_activity(
            "generating images",
            metadata={"prompt_len": len(prompt), "n_images": n_images},
        )

        loop = asyncio.get_running_loop()
        sync_cb = make_sync_step_callback(on_step, loop)

        def _generate():
            images: list[bytes] = []
            for i in range(max(1, n_images)):
                gen_kwargs: dict[str, Any] = dict(
                    seed=base_seed + i,
                    prompt=prompt,
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    guidance=guidance,
                )
                if scheduler is not None:
                    gen_kwargs["scheduler"] = scheduler
                if negative_prompt is not None:
                    logger.warning(
                        "Flux2Klein does not support negative_prompt; "
                        "ignoring (got %d chars)",
                        len(negative_prompt),
                    )
                subscriber = None
                if sync_cb is not None and getattr(flux, "callbacks", None) is not None:
                    subscriber = _StepProgressInLoop(sync_cb, steps)
                    try:
                        flux.callbacks.register(subscriber)
                    except Exception:
                        logger.debug(
                            "on_step: flux.callbacks.register failed", exc_info=True
                        )
                        subscriber = None
                try:
                    gen = flux.generate_image(**gen_kwargs)
                finally:
                    if subscriber is not None:
                        try:
                            flux.callbacks.in_loop.remove(subscriber)
                        except (ValueError, AttributeError):
                            pass
                buf = io.BytesIO()
                gen.image.save(buf, format=output_format)
                images.append(buf.getvalue())
            return images

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(get_executor("image"), _generate), timeout=600.0
            )
            elapsed = time.monotonic() - t0
            self._update_activity(activity_id, elapsed_seconds=elapsed)
            logger.info(
                "ImageGen generated %d image(s) in %.2fs",
                len(result),
                elapsed,
            )
            return result
        finally:
            await self._finish_activity(activity_id)

    # ------------------------------------------------------------------
    # Pipeline stage API (issue #170). Exposes individual pipeline stages
    # (text encoder / DiT / VAE) so Fusion-ComfyUI can map each stage to a
    # separate node and drive sequential offload. Latents flow between
    # stages as unpacked (batch, c, h, w) mx.array - the same shape mflux
    # uses for prepare_latents output and decode_packed_latents input.
    # mflux loads all stages in Flux2Klein.__init__, so load_* is
    # idempotent-when-present and unload_* drops the submodule ref to free
    # memory; reloading a single unloaded stage requires re-instantiation.
    # ------------------------------------------------------------------
    def _require_flux(self):
        if self._flux is None:
            if self._mflux_missing:
                raise RuntimeError(
                    "Image generation unavailable: mflux-fusion not installed. "
                    "Install with: pip install mflux-fusion"
                )
            raise RuntimeError("ImageGen engine not started.")
        return self._flux

    @staticmethod
    def _active_mem() -> int:
        try:
            return int(mx.metal.get_active_memory())
        except Exception:
            return -1

    @staticmethod
    def _gc_clear_cache():
        gc.collect()
        try:
            mx.metal.clear_cache()
        except Exception:
            try:
                mx.clear_cache()
            except Exception:
                logger.debug("mx clear_cache unavailable", exc_info=True)

    async def load_text_encoder(self) -> None:
        flux = self._require_flux()
        if flux.text_encoder is None:
            raise RuntimeError(
                "text_encoder was unloaded; re-instantiate ImageGenEngine to "
                "reload a stage (mflux loads all stages in __init__)."
            )
        gc.collect()
        logger.info("stage:text_encoder load active_mem=%s", self._active_mem())

    async def encode_text(self, prompt: str) -> dict:
        flux = self._require_flux()
        if flux.text_encoder is None:
            raise RuntimeError("text_encoder is unloaded; call load_text_encoder().")
        from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import (
            Flux2PromptEncoder,
        )

        def _enc():
            embed, text_ids = Flux2PromptEncoder.encode_prompt(
                prompt=prompt,
                tokenizer=flux.tokenizers["qwen3"],
                text_encoder=flux.text_encoder,
                num_images_per_prompt=1,
                max_sequence_length=512,
                text_encoder_out_layers=(9, 18, 27),
            )
            return embed, text_ids

        loop = asyncio.get_running_loop()
        embed, text_ids = await loop.run_in_executor(get_executor("image"), _enc)
        logger.info(
            "stage:text_encoder encode prompt_len=%d embed_shape=%s",
            len(prompt),
            tuple(embed.shape),
        )
        return {"embed": embed, "text_ids": text_ids}

    async def unload_text_encoder(self) -> None:
        flux = self._require_flux()
        flux.text_encoder = None
        self._gc_clear_cache()
        logger.info("stage:text_encoder unload active_mem=%s", self._active_mem())

    async def load_dit(self) -> None:
        flux = self._require_flux()
        if flux.transformer is None:
            raise RuntimeError(
                "transformer (DiT) was unloaded; re-instantiate ImageGenEngine "
                "to reload a stage (mflux loads all stages in __init__)."
            )
        gc.collect()
        logger.info("stage:dit load active_mem=%s", self._active_mem())

    async def denoise(
        self,
        latent: mx.array,
        pos_embed: mx.array,
        neg_embed: mx.array | None,
        steps: int,
        cfg: float,
        seed: int,
    ) -> mx.array:
        # Latents/embeds must be engine-native: created by encode_text or another
        # stage running in the single image-executor thread (max_workers=1,
        # _init_mlx_thread). Caller-cross-thread arrays hit MLX "no Stream(gpu,0)
        # in current thread" on the per-step mx.eval below (issue #170 constraint).
        flux = self._require_flux()
        if flux.transformer is None:
            raise RuntimeError("transformer (DiT) is unloaded; call load_dit().")
        from mflux.models.common.config.config import Config
        from mflux.models.flux2.latent_creator.flux2_latent_creator import (
            Flux2LatentCreator,
        )
        from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import (
            Flux2PromptEncoder,
        )

        if latent.ndim != 4:
            raise ValueError(
                f"denoise expects unpacked latent (batch,c,h,w); got {tuple(latent.shape)}"
            )
        batch, _c, h, w = latent.shape
        pixel_h = h * 16
        pixel_w = w * 16
        use_cfg = cfg is not None and cfg > 1.0 and neg_embed is not None

        def _denoise():
            config = Config(
                model_config=flux.model_config,
                num_inference_steps=steps,
                height=pixel_h,
                width=pixel_w,
                guidance=cfg,
                scheduler="flow_match_euler_discrete",
            )
            predict = flux._predict(flux.transformer)
            latent_ids = Flux2LatentCreator.prepare_grid_ids(latent, t_coord=0)
            text_ids = Flux2PromptEncoder.prepare_text_ids(pos_embed)
            neg_text_ids = (
                Flux2PromptEncoder.prepare_text_ids(neg_embed) if use_cfg else None
            )
            latents = Flux2LatentCreator.pack_latents(latent)
            for t in config.time_steps:
                noise = predict(
                    latents=latents,
                    latent_ids=latent_ids,
                    prompt_embeds=pos_embed,
                    text_ids=text_ids,
                    negative_prompt_embeds=neg_embed if use_cfg else None,
                    negative_text_ids=neg_text_ids,
                    guidance=cfg,
                    timestep=config.scheduler.timesteps[t],
                )
                latents = config.scheduler.step(
                    noise=noise,
                    timestep=t,
                    latents=latents,
                    sigmas=config.scheduler.sigmas,
                )
                mx.eval(latents)
            return latents.reshape(batch, h, w, latents.shape[-1]).transpose(0, 3, 1, 2)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("image"), _denoise)
        logger.info(
            "stage:dit denoise steps=%d cfg=%.2f seed=%d out_shape=%s",
            steps,
            cfg,
            seed,
            tuple(result.shape),
        )
        return result

    async def unload_dit(self) -> None:
        flux = self._require_flux()
        flux.transformer = None
        self._gc_clear_cache()
        logger.info("stage:dit unload active_mem=%s", self._active_mem())

    async def load_vae(self) -> None:
        flux = self._require_flux()
        if flux.vae is None:
            raise RuntimeError(
                "vae was unloaded; re-instantiate ImageGenEngine to reload a "
                "stage (mflux loads all stages in __init__)."
            )
        gc.collect()
        logger.info("stage:vae load active_mem=%s", self._active_mem())

    async def decode(self, latent: mx.array) -> mx.array:
        flux = self._require_flux()
        if flux.vae is None:
            raise RuntimeError("vae is unloaded; call load_vae().")
        if latent.ndim != 4:
            raise ValueError(
                f"decode expects unpacked latent (batch,c,h,w); got {tuple(latent.shape)}"
            )

        def _decode():
            return flux.vae.decode_packed_latents(latent)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("image"), _decode)
        logger.info("stage:vae decode out_shape=%s", tuple(result.shape))
        return result

    async def decode_tiled(self, latent: mx.array, tile_size: int = 256) -> mx.array:
        flux = self._require_flux()
        if flux.vae is None:
            raise RuntimeError("vae is unloaded; call load_vae().")
        if latent.ndim != 4:
            raise ValueError(
                f"decode_tiled expects unpacked latent (batch,c,h,w); got {tuple(latent.shape)}"
            )
        tiling_config = getattr(flux, "tiling_config", None)

        def _decode():
            return flux.vae.decode_packed_latents(latent, tiling_config=tiling_config)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("image"), _decode)
        logger.info(
            "stage:vae decode_tiled tile_size=%s out_shape=%s",
            tile_size,
            tuple(result.shape),
        )
        return result

    async def unload_vae(self) -> None:
        flux = self._require_flux()
        flux.vae = None
        self._gc_clear_cache()
        logger.info("stage:vae unload active_mem=%s", self._active_mem())

    def get_stats(self) -> dict[str, Any]:
        return {
            "model_name": self._model_name,
            "loaded": self._flux is not None,
            "mflux_missing": self._mflux_missing,
        }

    def __repr__(self) -> str:
        if self._mflux_missing:
            status = "disabled(mflux-missing)"
        elif self._flux is not None:
            status = "running"
        else:
            status = "stopped"
        return f"<ImageGenEngine model={self._model_name} status={status}>"
