# SPDX-License-Identifier: Apache-2.0
# Image generation engine for fusion-mlx.
# Supports Flux1 variants (ControlNet/Depth/Fill/Kontext/Redux) and Flux2 Klein.
import asyncio
import gc
import io
import logging
import time
from collections.abc import Callable
from typing import Any

import mlx.core as mx
import numpy as np

from ..cache.radix_diffusion_cache import DiffusionRadixCache
from ..engine_core import get_executor
from ._progress import StepCallback, make_sync_step_callback
from .base import BaseNonStreamingEngine


def _text_cache_enabled() -> bool:
    import os

    return os.environ.get("FUSION_DIFFUSION_TEXT_CACHE", "1").strip() not in (
        "0",
        "off",
        "false",
    )


def _prompt_hash(prompt: str) -> str:
    import hashlib

    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


logger = logging.getLogger(__name__)


def _to_latent_size(image_size: tuple[int, int]) -> tuple[int, int]:
    h, w = image_size
    h = ((h + 15) // 16) * 16
    w = ((w + 15) // 16) * 16
    return (h // 8, w // 8)


# variant -> (mflux module path, class name, ModelConfig label, default guidance)
# Flux1 variants all default to guidance=4.0; Flux2 Klein uses 1.0.
VARIANT_MAP: dict[str, tuple[str, str, str, float]] = {
    "txt2img": (
        "mflux.models.flux2.variants.txt2img.flux2_klein",
        "Flux2Klein",
        "flux2_klein_9b",
        1.0,
    ),
    # FLUX.1 base txt2img (dev / schnell). mflux ships Flux1 in
    # flux.cli.flux_generate; ModelConfig.dev()/schnell() are classmethods.
    "flux1_dev": (
        "mflux.models.flux.cli.flux_generate",
        "Flux1",
        "dev",
        4.0,
    ),
    "flux1_schnell": (
        "mflux.models.flux.cli.flux_generate",
        "Flux1",
        "schnell",
        0.0,
    ),
    "controlnet_canny": (
        "mflux.models.flux.variants.controlnet.flux_controlnet",
        "Flux1Controlnet",
        "dev_controlnet_canny",
        4.0,
    ),
    "controlnet_upscaler": (
        "mflux.models.flux.variants.controlnet.flux_controlnet",
        "Flux1Controlnet",
        "dev_controlnet_upscaler",
        4.0,
    ),
    "depth": (
        "mflux.models.flux.variants.depth.flux_depth",
        "Flux1Depth",
        "dev_depth",
        4.0,
    ),
    "fill": (
        "mflux.models.flux.variants.fill.flux_fill",
        "Flux1Fill",
        "dev_fill",
        4.0,
    ),
    "kontext": (
        "mflux.models.flux.variants.kontext.flux_kontext",
        "Flux1Kontext",
        "dev_kontext",
        4.0,
    ),
    "redux": (
        "mflux.models.flux.variants.redux.flux_redux",
        "Flux1Redux",
        "dev_redux",
        4.0,
    ),
    # SD3-Medium native MLX pipeline (no mflux ModelConfig; SD3Config inside).
    "sd3": (
        "fusion_mlx.image.sd3.generate",
        "SD3Pipeline",
        "sd3_medium",
        4.0,
    ),
    # SDXL native MLX pipeline (dual text encoders CLIP-L + OpenCLIP-G,
    # 2048 cross-attn dim). Covers sdxl_base / cosxl_edit / sdxs variants.
    "sdxl": (
        "fusion_mlx.image.sdxl.generate",
        "SDXLPipeline",
        "sdxl_base",
        7.5,
    ),
    "cosxl": (
        "fusion_mlx.image.sdxl.generate",
        "SDXLPipeline",
        "cosxl_edit",
        7.5,
    ),
    "sdxs": (
        "fusion_mlx.image.sdxl.generate",
        "SDXLPipeline",
        "sdxs",
        4.0,
    ),
    # SD1.5 native MLX pipeline (UNet4 + kl-f8 VAE + CLIP-L, 512 native res).
    # Single text encoder, no time_ids/pooled — distinct from SDXL.
    "sd15": (
        "fusion_mlx.image.sd15.generate",
        "SD15Pipeline",
        "sd15_base",
        7.5,
    ),
    # SD2.1 native MLX pipeline (UNet4 + kl-f8 VAE + ViT-H/14 CLIP, 768
    # native res, v_prediction DDIM). Cross-attention dim 1024, heads per
    # block [5,10,20,20] (head_dim 64). Distinct from SD1.5/SDXL.
    "sd2": (
        "fusion_mlx.image.sd2.generate",
        "SD2Pipeline",
        "sd2_base",
        7.5,
    ),
    # Stable Cascade (Wuerstchen 3-stage: prior -> decoder -> vqgan) native
    # MLX pipeline. Prior default guidance=4.0; decoder guidance=0.0.
    "stable_cascade": (
        "fusion_mlx.image.cascade.generate",
        "CascadePipeline",
        "stable_cascade",
        4.0,
    ),
}

# Per-call executor timeout for image generation / model load (#481). The
# hard-coded 600s killed legitimate SD1.5 1024x1024 img2img (hires-fix 2nd
# pass). Override via FUSION_IMAGE_TIMEOUT (seconds). Invalid/non-positive
# values fall back to the default with a warning.
_IMAGE_GEN_TIMEOUT_DEFAULT_S = 600.0


def get_image_gen_timeout() -> float:
    import os

    raw = os.environ.get("FUSION_IMAGE_TIMEOUT")
    if not raw:
        return _IMAGE_GEN_TIMEOUT_DEFAULT_S
    try:
        val = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "FUSION_IMAGE_TIMEOUT=%r is not a number, using default %.0fs",
            raw,
            _IMAGE_GEN_TIMEOUT_DEFAULT_S,
        )
        return _IMAGE_GEN_TIMEOUT_DEFAULT_S
    if val <= 0:
        logger.warning(
            "FUSION_IMAGE_TIMEOUT=%r <= 0, using default %.0fs",
            raw,
            _IMAGE_GEN_TIMEOUT_DEFAULT_S,
        )
        return _IMAGE_GEN_TIMEOUT_DEFAULT_S
    return val


def _infer_variant(model_path: str) -> str:
    name = (model_path or "").lower()
    if "sd3" in name or "stable-diffusion-3" in name:
        return "sd3"
    if "sdxs" in name:
        return "sdxs"
    # SD1.5: "stable-diffusion-v1-5" / "sd1.5" / "sd15". Check BEFORE sdxl
    # and BEFORE the flux1_dev "dev" fallback ("v1-5" has no "dev" but is
    # also not sdxl/sd3, so it would otherwise fall through to flux2 txt2img).
    if (
        "sd1.5" in name
        or "sd15" in name
        or "stable-diffusion-v1-5" in name
        or "stable-diffusion-v1-4" in name
    ):
        return "sd15"
    # SD2.1: "stable-diffusion-2-1" / "sd2.1" / "sd21" / "v2-1" / "768-v".
    # Check BEFORE cascade/sdxl — "stable-diffusion-2" matches neither.
    if (
        "stable-diffusion-2" in name
        or "sd2.1" in name
        or "sd2-1" in name
        or "sd21" in name
        or "v2-1" in name
        or "768-v" in name
        or "768v" in name
    ):
        return "sd2"
    # Stable Cascade / Wuerstchen: check BEFORE sdxl/sd3 since
    # "stable-cascade" contains neither substring.
    if "cascade" in name or "wuerstchen" in name:
        return "stable_cascade"
    if "cosxl" in name:
        return "cosxl"
    if "sdxl" in name or "stable-diffusion-xl" in name or "stable_diffusion-xl" in name:
        return "sdxl"
    if "controlnet" in name and "upscaler" in name:
        return "controlnet_upscaler"
    if "controlnet" in name and "canny" in name:
        return "controlnet_canny"
    if "controlnet" in name:
        return "controlnet_canny"
    if "depth" in name:
        return "depth"
    if "fill" in name:
        return "fill"
    if "kontext" in name:
        return "kontext"
    if "redux" in name:
        return "redux"
    # FLUX.1 base txt2img: distinguish from FLUX.2 klein.
    # "schnell" is unique to FLUX.1; "dev" without klein/flux2 is FLUX.1-dev.
    if "schnell" in name:
        return "flux1_schnell"
    if "dev" in name and "klein" not in name and "flux2" not in name:
        return "flux1_dev"
    return "txt2img"


def _infer_flux2_config(model_path: str) -> str:
    name = (model_path or "").lower()
    # NOTE: check "9b"/"kv" before "4b" — quantized model ids like
    # "flux2-klein-9b-4bit" contain the substring "4b" (from "4bit"),
    # which would otherwise misclassify the 9b model as the 4b config
    # (heads=24 vs 32) and break the transformer reshape (#449).
    if "kv" in name and "9b" in name:
        return "flux2_klein_9b_kv"
    if "base" in name and "9b" in name:
        return "flux2_klein_base_9b"
    if "base" in name and "4b" in name:
        return "flux2_klein_base_4b"
    if "9b" in name:
        return "flux2_klein_9b"
    if "4b" in name:
        return "flux2_klein_4b"
    return "flux2_klein_9b"


def _flux_quantize_from_env() -> int | None:
    import os

    env = os.environ.get("FUSION_FLUX_QUANT", "").strip().lower()
    if env in ("", "0", "off", "none", "bf16"):
        return None
    if env in ("w8a16", "w8", "int8", "8"):
        logger.info("Flux 量化: w8a16 (quantize=8)")
        return 8
    if env in ("w4", "nf4", "int4", "4"):
        logger.info("Flux 量化: w4 (quantize=4)")
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
    engine_type = "image_gen"

    def __init__(self, model_name: str, variant: str | None = None, **kwargs):
        super().__init__()
        self._model_name = model_name
        self._model_path = model_name
        self._flux = None
        self._mflux_missing = False
        self._kwargs = kwargs
        self._variant = variant or _infer_variant(model_name)
        if self._variant not in VARIANT_MAP:
            logger.warning(
                "Unknown variant '%s', falling back to txt2img. Available: %s",
                self._variant,
                list(VARIANT_MAP.keys()),
            )
            self._variant = "txt2img"
        self._quantize = kwargs.get("quantize")
        if self._quantize is None:
            self._quantize = _flux_quantize_from_env()
        # SD2.1 (v_prediction DDIM) + int8/int4 量化在 >768 分辨率数值
        # 不稳定: UNet 输出逐步累积量化误差, ~step 5-6 溢出为 NaN (已验证
        # 8bit/4bit @1152 均 NaN, fp16 @1152 正常; SD1.5 8bit @1152 正常,
        # 故为 SD2 v_prediction 特有). 降级为 fp16 不量化以保证稳定.
        if self._variant == "sd2" and self._quantize is not None:
            logger.warning(
                "SD2 v_prediction + quantize=%s 在高分辨率下数值不稳定, "
                "降级为 fp16 不量化 (variant=sd2)",
                self._quantize,
            )
            self._quantize = None
        # #178 UMA radix text-embedding cache (mirrors UMT5/CLIP pattern)
        self._text_cache = (
            DiffusionRadixCache(max_mb=512, name="flux_img")
            if _text_cache_enabled()
            else None
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def variant(self) -> str:
        return self._variant

    async def start(self) -> None:
        if self._flux is not None:
            return
        try:
            from mflux.models.common.config.model_config import ModelConfig
        except ImportError as exc:
            logger.warning(
                "ImageGen engine disabled: mflux-fusion not installed. "
                "Install with: pip install mflux-fusion  (%s)",
                exc,
            )
            self._mflux_missing = True
            return
        module_path, cls_name, config_label, default_guidance = VARIANT_MAP[
            self._variant
        ]
        # For Flux2 txt2img, refine config_label from model path
        if self._variant == "txt2img":
            config_label = _infer_flux2_config(self._model_path)
        logger.info(
            "Starting ImageGen engine variant=%s class=%s path=%s",
            self._variant,
            cls_name,
            self._model_path,
        )

        def _load():
            import importlib

            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
            # Native (fusion_mlx) variants carry their own config object and
            # do not use mflux's ModelConfig factory.
            if module_path.startswith("fusion_mlx."):
                logger.info(
                    "ImageGen loading native variant=%s class=%s path=%s",
                    self._variant,
                    cls_name,
                    self._model_path,
                )
                flux = cls(
                    model_config=None,
                    model_path=self._model_path,
                    quantize=self._quantize,
                )
                return flux
            model_config = getattr(ModelConfig, config_label)()
            logger.info(
                "ImageGen loading variant=%s config=%s path=%s",
                self._variant,
                config_label,
                self._model_path,
            )
            flux = cls(
                model_config=model_config,
                model_path=self._model_path,
                quantize=self._quantize,
            )
            return flux

        loop = asyncio.get_running_loop()
        self._flux = await asyncio.wait_for(
            loop.run_in_executor(get_executor("image"), _load),
            timeout=get_image_gen_timeout(),
        )
        logger.info(
            "ImageGen engine loaded: %s variant=%s", self._model_name, self._variant
        )

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
        guidance: float | None = None,
        n_images: int = 1,
        output_format: str = "PNG",
        scheduler: str | None = None,
        negative_prompt: str | None = None,
        on_step: StepCallback | None = None,
        # Variant-specific image inputs
        control_image: str | None = None,
        controlnet_strength: float | None = None,
        reference_images: list[str] | None = None,
        reference_strengths: list[float] | None = None,
        edit_image: str | None = None,
        mask_image: str | None = None,
        depth_image: str | None = None,
        image_strength: float | None = None,
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
        # Use variant default guidance when caller doesn't specify
        if guidance is None:
            _, _, _, default_guidance = VARIANT_MAP[self._variant]
            guidance = default_guidance
        t0 = time.monotonic()
        activity_id = self._begin_activity(
            "generating images",
            metadata={
                "prompt_len": len(prompt),
                "n_images": n_images,
                "variant": self._variant,
            },
        )

        loop = asyncio.get_running_loop()
        sync_cb = make_sync_step_callback(on_step, loop)

        def _generate():
            mx.default_stream(mx.default_device())
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
                # Variant-specific generate_image kwargs
                variant = self._variant
                if variant == "controlnet_canny" or variant == "controlnet_upscaler":
                    if control_image is None:
                        raise ValueError(f"variant '{variant}' requires control_image")
                    gen_kwargs["controlnet_image_path"] = control_image
                    if controlnet_strength is not None:
                        gen_kwargs["controlnet_strength"] = controlnet_strength
                elif variant == "depth":
                    if depth_image is not None:
                        gen_kwargs["depth_image_path"] = depth_image
                    elif control_image is not None:
                        gen_kwargs["image_path"] = control_image
                    if image_strength is not None:
                        gen_kwargs["image_strength"] = image_strength
                elif variant == "fill":
                    if edit_image is None or mask_image is None:
                        raise ValueError(
                            "variant 'fill' requires edit_image and mask_image"
                        )
                    gen_kwargs["image_path"] = edit_image
                    gen_kwargs["masked_image_path"] = mask_image
                    if image_strength is not None:
                        gen_kwargs["image_strength"] = image_strength
                elif variant == "kontext":
                    if edit_image is not None:
                        gen_kwargs["image_path"] = edit_image
                    elif control_image is not None:
                        gen_kwargs["image_path"] = control_image
                    if image_strength is not None:
                        gen_kwargs["image_strength"] = image_strength
                elif variant == "redux":
                    if not reference_images:
                        raise ValueError("variant 'redux' requires reference_images")
                    gen_kwargs["redux_image_paths"] = reference_images
                    if reference_strengths is not None:
                        gen_kwargs["redux_image_strengths"] = reference_strengths
                    if image_strength is not None:
                        gen_kwargs["image_strength"] = image_strength
                elif variant in ("txt2img", "flux1_dev", "flux1_schnell"):
                    if edit_image is not None or control_image is not None:
                        img = edit_image or control_image
                        gen_kwargs["image_path"] = img
                        if image_strength is not None:
                            gen_kwargs["image_strength"] = image_strength
                elif variant == "sd3":
                    if negative_prompt is not None:
                        gen_kwargs["negative_prompt"] = negative_prompt
                    shift = kwargs.get("shift")
                    if shift is not None:
                        gen_kwargs["shift"] = shift
                elif variant in ("sdxl", "cosxl", "sdxs", "sd15", "sd2"):
                    if negative_prompt is not None:
                        gen_kwargs["negative_prompt"] = negative_prompt
                if variant in ("sd3", "sdxl", "cosxl", "sdxs", "sd15", "sd2") and (
                    edit_image is not None or control_image is not None
                ):
                    # img2img / partial-denoise (#480): each pipeline encodes
                    # the init image to a latent, noises to t_start, and runs a
                    # partial denoise at image_strength (denoise fraction).
                    gen_kwargs["image_path"] = edit_image or control_image
                    if image_strength is not None:
                        gen_kwargs["image_strength"] = image_strength
                elif variant == "stable_cascade":
                    if negative_prompt is not None:
                        gen_kwargs["negative_prompt"] = negative_prompt
                    # decoder_steps / decoder_guidance optional overrides
                    d_steps = kwargs.get("decoder_steps")
                    if d_steps is not None:
                        gen_kwargs["decoder_steps"] = d_steps
                    d_guidance = kwargs.get("decoder_guidance")
                    if d_guidance is not None:
                        gen_kwargs["decoder_guidance"] = d_guidance
                if negative_prompt is not None and variant not in (
                    "sd3",
                    "sdxl",
                    "cosxl",
                    "sdxs",
                    "sd15",
                    "sd2",
                    "stable_cascade",
                ):
                    logger.warning(
                        "Flux does not support negative_prompt; "
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
                img_w, img_h = gen.image.size
                min_w = max(8, width // 2)
                min_h = max(8, height // 2)
                if img_w < min_w or img_h < min_h:
                    logger.error(
                        "ImageGen raw output collapsed: got %dx%d requested %dx%d variant=%s seed=%d",
                        img_w,
                        img_h,
                        width,
                        height,
                        self._variant,
                        base_seed + i,
                    )
                    raise RuntimeError(
                        f"ImageGen raw output width/height collapsed: got {img_w}x{img_h} "
                        f"requested {width}x{height} (cross-thread lazy eval race suspected; "
                        f"retry with a fresh seed)"
                    )
                if output_format == "raw":
                    import numpy as np

                    images.append(np.array(gen.image))
                else:
                    buf = io.BytesIO()
                    gen.image.save(buf, format=output_format)
                    images.append(buf.getvalue())
            return images

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(get_executor("image"), _generate),
                timeout=get_image_gen_timeout(),
            )
            elapsed = time.monotonic() - t0
            self._update_activity(activity_id, elapsed_seconds=elapsed)
            logger.info(
                "ImageGen generated %d image(s) in %.2fs variant=%s",
                len(result),
                elapsed,
                self._variant,
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
        # #178: radix cache lookup before encoding
        cache = self._text_cache
        key = None
        if cache is not None:
            key = f"flux_img:512:{_prompt_hash(prompt)}"
            cached = cache.get(key)
            if cached is not None:
                logger.debug("flux_img text cache hit: %s", key)
                return cached

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
        result = {"embed": embed, "text_ids": text_ids}
        if key is not None:
            cache.put(key, result)
            logger.debug("flux_img text cache miss+insert: %s", key)
        return result

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

        # Materialize the latent on the caller's (event-loop) stream before
        # dispatching to the worker. A latent built on the main thread (test
        # fakes, or a caller that constructed it off the worker) is bound to
        # this thread's GPU stream; the image worker has its own stream and
        # cannot touch a lazy graph referencing the caller's stream
        # (RuntimeError "no Stream(gpu, N) in current thread"). Same
        # caller-eval-then-dispatch pattern as encode()'s numpy bridge.
        mx.eval(latent)

        def _decode():
            result = flux.vae.decode_packed_latents(latent)
            # Materialize the output on the worker's own stream before
            # returning: a lazy decode graph stays bound to this thread's
            # GPU stream, and a caller on another thread (event loop /
            # fusion-comfyui) touching it aborts with the same stream error.
            mx.eval(result)
            return result

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

        mx.eval(latent)

        def _decode():
            result = flux.vae.decode_packed_latents(latent, tiling_config=tiling_config)
            mx.eval(result)
            return result

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("image"), _decode)
        logger.info(
            "stage:vae decode_tiled tile_size=%s out_shape=%s",
            tile_size,
            tuple(result.shape),
        )
        return result

    async def encode(self, pixels: mx.array) -> mx.array:
        flux = self._require_flux()
        if flux.vae is None:
            raise RuntimeError("vae is unloaded; call load_vae().")
        if pixels.ndim != 4:
            raise ValueError(f"encode expects (1,H,W,3); got {tuple(pixels.shape)}")
        if pixels.shape[1] % 16 != 0 or pixels.shape[2] % 16 != 0:
            raise ValueError(
                f"encode expects H,W divisible by 16 (vae_scale*patch); got {tuple(pixels.shape)}"
            )

        # pixels is built on the event-loop main thread; the image worker has
        # its own GPU stream and cannot mx.eval a lazy graph referencing the
        # main thread's stream (RuntimeError "no Stream(gpu, N) in current
        # thread"). Bridge through numpy on the caller thread (owns the source
        # stream) and rebuild an mx.array inside the worker. Same pattern as
        # Wan2Backend.encode; decode() avoids this because its latent input is
        # worker-owned (encode/denoise output already eval'd on the worker).
        pixels_np = np.array(pixels)

        def _encode():
            from mflux.models.common.vae.vae_util import VAEUtil
            from mflux.models.flux2.latent_creator.flux2_latent_creator import (
                Flux2LatentCreator,
            )
            from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import (
                _Flux2KleinEditHelpers,
            )

            # Public contract is NHWC (1,H,W,3) [0,1] (spec line 42); mflux's
            # Flux2VAE.encode / conv_in expects NCHW (1,3,H,W) (image_util.py
            # to_array transposes (0,3,1,2)). Convert here so the surface
            # stays NHWC for callers while the encoder gets NCHW.
            img_nchw = mx.array(pixels_np).transpose(0, 3, 1, 2)
            encoded = VAEUtil.encode(flux.vae, img_nchw)
            encoded = _Flux2KleinEditHelpers.ensure_4d_latents(encoded)
            encoded = _Flux2KleinEditHelpers.crop_to_even_spatial(encoded)
            encoded = Flux2LatentCreator.patchify_latents(encoded)
            encoded = _Flux2KleinEditHelpers.bn_normalize_vae_encoded_latents(
                encoded, vae=flux.vae
            )
            mx.eval(encoded)
            return encoded

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("image"), _encode)
        logger.info("stage:vae encode img out_shape=%s", tuple(result.shape))
        return result

    async def unload_vae(self) -> None:
        flux = self._require_flux()
        flux.vae = None
        self._gc_clear_cache()
        logger.info("stage:vae unload active_mem=%s", self._active_mem())

    def get_stats(self) -> dict[str, Any]:
        stats = {
            "model_name": self._model_name,
            "variant": self._variant,
            "loaded": self._flux is not None,
            "mflux_missing": self._mflux_missing,
        }
        if self._text_cache is not None:
            stats["text_cache"] = self._text_cache.stats()
        return stats

    def __repr__(self) -> str:
        if self._mflux_missing:
            status = "disabled(mflux-missing)"
        elif self._flux is not None:
            status = "running"
        else:
            status = "stopped"
        return f"<ImageGenEngine model={self._model_name} variant={self._variant} status={status}>"
