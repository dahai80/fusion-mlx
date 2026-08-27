# SPDX-License-Identifier: Apache-2.0
# Wan2.2 video backend (pure-MLX port, vendored from mlx-video).
# Phase 5: imports fusion_mlx.video.wan2 (direct MLX port) instead of
# mlx_video.models.wan_2. Verified API: generate_video(model_dir, prompt,
# negative_prompt=, image=, width=, height=, num_frames=, steps=, guide_scale=,
# shift=, seed=, output_path=, scheduler=, no_compile=, tiling=). Wan supports
# image-to-video (I2V) via image=. Wan's generate_video takes no fps argument
# (it controls container fps internally), so request fps is ignored for this
# backend. Signature confirmed against the pure-MLX port.

import asyncio
import gc
import logging
import random
import threading
from collections.abc import Callable
from typing import Any

import mlx.core as mx
import numpy as np

from ..._tempfile_safe import managed_tempfile_path
from ...engine_core import get_executor, get_video_gen_timeout
from .._progress import make_sync_step_callback
from .base import VideoBackend, VideoConstraints, VideoGenParams

logger = logging.getLogger(__name__)

_DEFAULT_SCHEDULER = "unipc"


def _active_mem() -> int:
    try:
        return int(mx.metal.get_active_memory())
    except Exception:
        return -1


def _pixels_thwc_to_ncthw(src):
    # encode() accepts 5D (1,T,H,W,3) or 4D (T,H,W,3); WanVAE.encode needs
    # NCTHW (1,3,T,H,W) float32. Symmetric with decode() output layout.
    x = mx.array(src).astype(mx.float32)
    return x.transpose(3, 0, 1, 2)[None]


async def _clear_mlx_cache() -> None:
    # Run mx.synchronize()/mx.clear_cache() on the video executor thread, NOT
    # the event-loop main thread. MLX Metal streams are thread-local: a
    # clear_cache() issued from the main thread invalidates the worker's
    # thread-local stream table, so the next run_in_executor call (e.g. VAE
    # decode after unload_dit) raises "There is no Stream(gpu, N) in current
    # thread". Keeping the sync+clear on the same worker thread that owns the
    # streams preserves the table across staged unload/load boundaries (#410).
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        get_executor("video"), lambda: (mx.synchronize(), mx.clear_cache())
    )


def _load_t5_tokenizer(model_dir):
    from pathlib import Path

    from transformers import AutoTokenizer

    # Prefer a tokenizer/ subdir shipped beside the model weights. Several
    # Wan checkpoints bundle umt5-xxl tokenizer files locally
    # (spiece.model / tokenizer.json / tokenizer_config.json). Loading from
    # the generic HF repo id "google/umt5-xxl" hits the network on every cold
    # load to validate chat_templates; under hf-mirror that round-trip can
    # exceed the 120s T5 encode timeout (issue #553) and abort generation
    # before a single denoise step. Local path = no network.
    if model_dir:
        local_tok = Path(model_dir) / "tokenizer"
        if (local_tok / "tokenizer.json").exists() or (
            local_tok / "spiece.model"
        ).exists():
            logger.info("Wan2: loading T5 tokenizer from local %s", local_tok)
            return AutoTokenizer.from_pretrained(str(local_tok))
    # Fallback to the HF hub cache (offline if already cached).
    logger.info("Wan2: loading T5 tokenizer from HF hub google/umt5-xxl")
    return AutoTokenizer.from_pretrained("google/umt5-xxl")


# Max T5 text-embedding cache entries (LRU eviction when exceeded).
_T5_EMBED_CACHE_MAX = 16
# Timeout for T5 encoder preload during start() — large model may take minutes.
_T5_PRELOAD_TIMEOUT = 300.0


def _infer_config_from_path(model_dir: str) -> "WanModelConfig":  # noqa: F821
    from fusion_mlx.video.wan2.config import WanModelConfig

    p = model_dir.lower()
    if "vace" in p:
        return WanModelConfig.wan_vace_14b()
    if "camera" in p:
        if "14b" in p:
            return WanModelConfig.wan21_fun_camera_14b()
        return WanModelConfig.wan21_fun_camera_1_3b()
    if "14b" in p:
        if "i2v" in p:
            return WanModelConfig.wan22_i2v_14b()
        return WanModelConfig.wan22_t2v_14b()
    if "5b" in p or "ti2v" in p:
        return WanModelConfig.wan22_ti2v_5b()
    logger.warning(
        "Wan2: cannot infer variant from path '%s', defaulting to 1.3B", model_dir
    )
    return WanModelConfig.wan21_t2v_1_3b()


class Wan2Backend(VideoBackend):
    name = "wan2"
    supports_i2v = True
    supports_vace = True
    supports_camera = True

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self._model_name = model_name
        self._loaded = False
        self._model_dir = None
        self._t5_encoder = None
        self._t5_tokenizer = None
        self._t5_config = None
        self._embed_cache = {}
        self._embed_cache_lock = threading.Lock()
        # Stage API (#410): per-component loaded state for ComfyUI sequential
        # offload. Mirrors skyreels._stage_flags. DiT/VAE load lazily in
        # load_dit()/load_vae(); T5 is preloaded in start() but
        # load_text_encoder() is the stage entry point.
        self._stage_flags = {
            "text_encoder": False,
            "dit": False,
            "vae": False,
        }
        self._stage_config = None
        self._stage_quant = None
        self._stage_dit_models = None
        self._stage_vae = None
        self._stage_vae_encoder = None
        self._stage_on_step = None

    @classmethod
    def detect(cls, model_path: str) -> bool:
        return "wan" in model_path.lower() or "vace" in model_path.lower()

    async def start(self, model_path: str, **kwargs: Any) -> None:
        if self._loaded:
            return
        logger.info("Starting Wan2 backend (pure-MLX): %s", model_path)

        def _resolve():
            try:
                from fusion_mlx.video.wan2.utils import get_model_path
            except ImportError as exc:
                raise ImportError(
                    "Wan2 pure-MLX port (fusion_mlx.video.wan2) is unavailable."
                ) from exc
            return get_model_path(model_path)

        loop = asyncio.get_running_loop()
        model_dir = await asyncio.wait_for(
            loop.run_in_executor(get_executor("io"), _resolve), timeout=180.0
        )
        self._model_dir = model_dir

        def _preload_t5():
            try:
                import json
                from pathlib import Path

                from fusion_mlx.video.wan2.config import WanModelConfig
                from fusion_mlx.video.wan2.utils import load_t5_encoder

                md = Path(model_dir)
                config_path = md / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config_dict = json.load(f)
                    config_dict.pop("quantization", None)
                    for key in (
                        "patch_size",
                        "vae_stride",
                        "window_size",
                        "sample_guide_scale",
                        "vace_layers",
                    ):
                        if key in config_dict and isinstance(config_dict[key], list):
                            config_dict[key] = tuple(config_dict[key])
                    config = WanModelConfig(
                        **{
                            k: v
                            for k, v in config_dict.items()
                            if k in WanModelConfig.__dataclass_fields__
                        }
                    )
                else:
                    config = _infer_config_from_path(model_dir)

                from fusion_mlx.video.wan2.utils import correct_in_dim

                config = correct_in_dim(config, md)

                t5_path = md / "t5_encoder.safetensors"
                if t5_path.exists():
                    self._t5_encoder = load_t5_encoder(t5_path, config)
                    self._t5_config = config
                    logger.info("Wan2: T5 encoder preloaded")
                else:
                    logger.info(
                        "Wan2: no t5_encoder.safetensors found, will load per-call"
                    )
            except Exception as e:
                logger.warning("Wan2: T5 preload failed, will load per-call: %s", e)

        await asyncio.wait_for(
            loop.run_in_executor(get_executor("io"), _preload_t5),
            timeout=_T5_PRELOAD_TIMEOUT,
        )

        self._loaded = True
        logger.info("Wan2 backend ready: %s", model_path)

    async def stop(self) -> None:
        if not self._loaded:
            return
        self._loaded = False
        self._t5_encoder = None
        self._t5_tokenizer = None
        self._t5_config = None
        self._stage_dit_models = None
        self._stage_vae = None
        self._stage_config = None
        self._stage_quant = None
        self._stage_on_step = None
        self._stage_flags = {
            "text_encoder": False,
            "dit": False,
            "vae": False,
        }
        with self._embed_cache_lock:
            self._embed_cache.clear()
        gc.collect()
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                get_executor("io"), lambda: (mx.synchronize(), mx.clear_cache())
            ),
            timeout=5.0,
        )

    async def _get_cached_embeds(self, prompt, neg_prompt, text_len):
        if self._t5_encoder is None:
            return None
        if neg_prompt is None and self._t5_config is not None:
            neg_prompt = self._t5_config.sample_neg_prompt
        cache_key = (prompt or "", neg_prompt or "", text_len)
        with self._embed_cache_lock:
            if cache_key in self._embed_cache:
                logger.info(
                    "Wan2: T5 embed cache hit for prompt_len=%d", len(prompt or "")
                )
                result = self._embed_cache.pop(cache_key)
                self._embed_cache[cache_key] = result
                return result

        def _encode_sync():
            from fusion_mlx.video.wan2.utils import encode_text

            if self._t5_tokenizer is None:
                self._t5_tokenizer = _load_t5_tokenizer(self._model_dir)

            logger.info(
                "Wan2: computing T5 embeds for prompt_len=%d", len(prompt or "")
            )
            context = encode_text(
                self._t5_encoder, self._t5_tokenizer, prompt, text_len
            )
            mx.eval(context)

            if neg_prompt and neg_prompt.strip():
                context_null = encode_text(
                    self._t5_encoder, self._t5_tokenizer, neg_prompt, text_len
                )
                mx.eval(context, context_null)
            else:
                context_null = None
                mx.eval(context)

            with self._embed_cache_lock:
                if len(self._embed_cache) >= _T5_EMBED_CACHE_MAX:
                    oldest = next(iter(self._embed_cache))
                    del self._embed_cache[oldest]
                self._embed_cache[cache_key] = (context, context_null)
            return (context, context_null)

        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(get_executor("io"), _encode_sync),
            timeout=120.0,
        )

    async def generate(self, params: VideoGenParams) -> list[bytes] | list[Any]:
        base_seed = (
            params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
        )
        scheduler = params.scheduler or _DEFAULT_SCHEDULER
        no_compile = True if params.no_compile is None else params.no_compile

        precomputed = None
        if self._t5_encoder is not None and self._t5_config is not None:
            text_len = self._t5_config.text_len
            precomputed = await self._get_cached_embeds(
                params.prompt, params.negative_prompt, text_len
            )

        def _generate():
            results = []
            for i in range(max(1, params.n)):
                result = _generate_one(
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
                    on_step_sync=sync_cb,
                    session_id=params.session_id,
                    output_format=params.output_format,
                    precomputed_context=precomputed,
                    control_video=params.control_video,
                    control_mask=params.control_mask,
                    reference_images=params.reference_images,
                    camera_conditions=params.camera_conditions,
                )
                results.append(result)
            return results

        loop = asyncio.get_running_loop()
        sync_cb = make_sync_step_callback(params.on_step, loop)
        return await asyncio.wait_for(
            loop.run_in_executor(get_executor("video"), _generate),
            timeout=get_video_gen_timeout(),
        )

    # ------------------------------------------------------------------
    # Pipeline stage API (issue #410). Override VideoBackend NotImplementedError
    # defaults to expose individual pipeline stages for Fusion-ComfyUI
    # sequential offload. Video latents are 5D (batch, c, num_frames, h, w).
    # Scope: T2V. I2V/VACE/camera stay on the monolithic generate() path.
    # ------------------------------------------------------------------
    def _ensure_stage_config(self):
        if self._stage_config is None:
            from pathlib import Path

            from fusion_mlx.video.wan2.stage import load_wan_config

            config, quant = load_wan_config(Path(self._model_dir))
            self._stage_config = config
            self._stage_quant = quant
            # Refresh the T5 config used by start() so encode_text_stage uses
            # the resolved text_len (start() may have inferred from path).
            if self._t5_config is None:
                self._t5_config = config
        return self._stage_config

    async def load_text_encoder(self) -> None:
        from pathlib import Path

        from fusion_mlx.video.wan2.stage import resolve_t5_path
        from fusion_mlx.video.wan2.utils import load_t5_encoder

        config = self._ensure_stage_config()
        if self._t5_encoder is None:
            t5_path = resolve_t5_path(Path(self._model_dir))

            def _load():
                return load_t5_encoder(t5_path, config)

            loop = asyncio.get_running_loop()
            self._t5_encoder = await asyncio.wait_for(
                loop.run_in_executor(get_executor("io"), _load),
                timeout=_T5_PRELOAD_TIMEOUT,
            )
        self._stage_flags["text_encoder"] = True
        gc.collect()
        logger.info("stage:text_encoder load wan2 active_mem=%s", _active_mem())

    async def encode_text(self, prompt: str) -> dict:
        from fusion_mlx.video.wan2.stage import encode_text_stage

        if self._t5_encoder is None:
            raise RuntimeError("text_encoder is unloaded; call load_text_encoder().")
        if self._t5_tokenizer is None:
            self._t5_tokenizer = _load_t5_tokenizer(self._model_dir)
        config = self._ensure_stage_config()
        t5_encoder = self._t5_encoder
        tokenizer = self._t5_tokenizer
        text_len = config.text_len

        def _enc():
            return encode_text_stage(t5_encoder, tokenizer, prompt, text_len)

        loop = asyncio.get_running_loop()
        context = await loop.run_in_executor(get_executor("video"), _enc)
        logger.info(
            "stage:text_encoder encode wan2 prompt_len=%d context_shape=%s",
            len(prompt),
            tuple(context.shape),
        )
        return {"embed": context}

    async def unload_text_encoder(self) -> None:
        self._t5_encoder = None
        self._stage_flags["text_encoder"] = False
        gc.collect()
        await _clear_mlx_cache()
        logger.info("stage:text_encoder unload wan2")

    async def load_dit(self) -> None:
        from pathlib import Path

        from fusion_mlx.video.wan2.stage import resolve_t5_path  # noqa: F401
        from fusion_mlx.video.wan2.utils import load_wan_model

        config = self._ensure_stage_config()
        quant = self._stage_quant
        model_dir = Path(self._model_dir)
        is_dual = config.dual_model

        def _load():
            if is_dual:
                low = load_wan_model(
                    model_dir / "low_noise_model.safetensors", config, quant
                )
                high = load_wan_model(
                    model_dir / "high_noise_model.safetensors", config, quant
                )
                return [low, high]
            dit_path = model_dir / "model.safetensors"
            if not dit_path.exists() and (model_dir / "dit").is_dir():
                dit_path = model_dir / "dit"
            return [load_wan_model(dit_path, config, quant)]

        loop = asyncio.get_running_loop()
        self._stage_dit_models = await asyncio.wait_for(
            loop.run_in_executor(get_executor("video"), _load),
            timeout=_T5_PRELOAD_TIMEOUT,
        )
        self._stage_flags["dit"] = True
        gc.collect()
        logger.info("stage:dit load wan2 dual=%s active_mem=%s", is_dual, _active_mem())

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
        from fusion_mlx.video.wan2.stage import compute_target_shape, run_denoise

        if self._stage_dit_models is None:
            raise RuntimeError("dit is unloaded; call load_dit().")
        config = self._ensure_stage_config()

        # cfg: the stage contract passes a single float. neg_embed is None when
        # the caller disabled CFG (no negative FusionTextEncoder node). Derive
        # cfg_disabled to match the monolith's fast path (guide_scale<=1.0).
        cfg_disabled = neg_embed is None or cfg <= 1.0
        context_null = (
            neg_embed if (not cfg_disabled and neg_embed is not None) else None
        )

        # Infer height/width/num_frames from the caller's latent shape if it is
        # a 5D (1, c, t, h, w) FusionComfyUI latent; otherwise fall back to the
        # config frame_num and the latent's spatial dims. The denoise generates
        # its own seeded noise from target_shape (the passed latent is the
        # empty zeros latent from FusionKSampler.create_empty_latent and is not
        # used as the init — T2V starts from pure noise).
        if latent.ndim == 5:
            _, _, t_lat, h_lat, w_lat = latent.shape
            num_frames = num_frames or (t_lat * config.vae_stride[0] - 1) or 1
            height = h_lat * config.vae_stride[1]
            width = w_lat * config.vae_stride[2]
        else:
            num_frames = num_frames or config.frame_num
            height = latent.shape[-2] * config.vae_stride[1]
            width = latent.shape[-1] * config.vae_stride[2]
        target_shape, seq_len, _h, _w = compute_target_shape(
            config, num_frames, height, width
        )

        guide_scale = cfg
        shift = config.sample_shift
        scheduler = "unipc"
        no_compile = True
        models = self._stage_dit_models
        on_step = self._stage_on_step

        def _denoise():
            lat_4d = run_denoise(
                config,
                models,
                pos_embed,
                context_null,
                target_shape,
                seq_len,
                steps,
                guide_scale,
                shift,
                scheduler,
                seed,
                no_compile,
                on_step=on_step,
            )
            # 5D contract: add batch dim -> (1, z_dim, t_latent, h_lat, w_lat).
            # Build the projection AND evaluate it on THIS executor thread so
            # the returned array is concrete, not a lazy graph that references
            # this call's auto-allocated Stream(gpu, N). The staged path later
            # runs VAE decode in a *separate* executor call and round-trips this
            # array through the event-loop main thread; MLX Metal streams are
            # thread-local, so a lazy array (or a main-thread [None] projection
            # of one) built on this call's streams raises
            # "There is no Stream(gpu, N) in current thread" at the decode-side
            # mx.eval. An mx.eval'd array is portable across threads. The
            # monolith shares one executor call so this is a no-op for it.
            lat_5d = lat_4d[None]
            mx.eval(lat_5d)
            return lat_5d

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("video"), _denoise)
        logger.info(
            "stage:dit denoise wan2 steps=%d cfg=%.2f out_shape=%s",
            steps,
            cfg,
            tuple(result.shape),
        )
        return result

    async def unload_dit(self) -> None:
        self._stage_dit_models = None
        self._stage_flags["dit"] = False
        gc.collect()
        await _clear_mlx_cache()
        logger.info("stage:dit unload wan2")

    async def load_vae(self) -> None:
        from pathlib import Path

        from fusion_mlx.video.wan2.stage import resolve_vae_path
        from fusion_mlx.video.wan2.utils import load_vae_decoder

        config = self._ensure_stage_config()
        vae_path = resolve_vae_path(Path(self._model_dir))

        def _load():
            return load_vae_decoder(vae_path, config)

        loop = asyncio.get_running_loop()
        # Load VAE on the *video* executor (not "io"): MLX Metal streams are
        # thread-local, and decode() runs on get_executor("video"). Weights
        # loaded on a different (io) thread bind to that thread's streams; the
        # decode-side mx.eval then raises "There is no Stream(gpu, N) in
        # current thread". Matches load_dit() + denoise() both on "video", and
        # the monolith generate() which load+decode on one executor call.
        self._stage_vae = await asyncio.wait_for(
            loop.run_in_executor(get_executor("video"), _load),
            timeout=_T5_PRELOAD_TIMEOUT,
        )
        self._stage_flags["vae"] = True
        gc.collect()
        logger.info("stage:vae load wan2 active_mem=%s", _active_mem())

    async def decode(self, latent: mx.array) -> mx.array:
        from fusion_mlx.video.wan2.stage import decode_wan_vae

        if self._stage_vae is None:
            raise RuntimeError("vae is unloaded; call load_vae().")
        config = self._ensure_stage_config()
        # Accept 5D (1, c, t, h, w) or 4D (c, t, h, w); run_denoise returns 5D.
        # NOTE: do NOT slice latent[0] here on the main thread — that builds a
        # lazy projection referencing the source array's (possibly cross-thread)
        # stream, which raises "There is no Stream(gpu, N) in current thread"
        # at decode-side mx.eval. Pass the full latent and slice on the executor
        # thread inside _decode (decode_wan_vae then mx.eval's it locally).
        vae = self._stage_vae
        ndim = latent.ndim

        def _decode():
            lat_4d = latent[0] if ndim == 5 else latent
            return decode_wan_vae(lat_4d, config, vae, tiling_config=None)

        loop = asyncio.get_running_loop()
        frames_u8 = await loop.run_in_executor(get_executor("video"), _decode)
        # frames_u8: [T, H, W, 3] uint8 -> float [T, H, W, 3] in [0,1] -> add
        # batch dim -> (1, T, H, W, 3). Matches the IMAGE contract (N,H,W,C).
        pixels = mx.array(frames_u8.astype(np.float32) / 255.0)[None]
        logger.info("stage:vae decode wan2 out_shape=%s", tuple(pixels.shape))
        return pixels

    async def decode_tiled(self, latent: mx.array, tile_size: int = 256) -> mx.array:
        from fusion_mlx.video.ltx2.video_vae.tiling import TilingConfig
        from fusion_mlx.video.wan2.stage import decode_wan_vae

        if self._stage_vae is None:
            raise RuntimeError("vae is unloaded; call load_vae().")
        config = self._ensure_stage_config()
        # Slice on the executor thread (see decode() note): main-thread
        # latent[0] builds a lazy cross-thread projection.
        vae = self._stage_vae
        ndim = latent.ndim
        # tile_size is in pixels (ComfyUI convention); auto derives spatial+temporal.
        # t-axis index: 2 for 5D (1,c,t,h,w), 1 for 4D (c,t,h,w).
        height = latent.shape[-2] * config.vae_stride[1]
        width = latent.shape[-1] * config.vae_stride[2]
        t_idx = 2 if ndim == 5 else 1
        num_frames = latent.shape[t_idx] * config.vae_stride[0] - 1
        tiling_config = TilingConfig.auto(height, width, num_frames)

        def _decode():
            lat_4d = latent[0] if ndim == 5 else latent
            return decode_wan_vae(lat_4d, config, vae, tiling_config=tiling_config)

        loop = asyncio.get_running_loop()
        frames_u8 = await loop.run_in_executor(get_executor("video"), _decode)
        pixels = mx.array(frames_u8.astype(np.float32) / 255.0)[None]
        logger.info(
            "stage:vae decode_tiled wan2 tile=%d out_shape=%s",
            tile_size,
            tuple(pixels.shape),
        )
        return pixels

    async def unload_vae(self) -> None:
        self._stage_vae = None
        self._stage_vae_encoder = None
        self._stage_flags["vae"] = False
        self._stage_flags.pop("vae_encoder", None)
        gc.collect()
        await _clear_mlx_cache()
        logger.info("stage:vae unload wan2 (decoder+encoder)")

    async def encode(self, pixels: mx.array) -> mx.array:
        from fusion_mlx.video.wan2.stage import encode_wan_vae

        if self._stage_vae_encoder is None:
            await self._load_vae_encoder_stage()
        config = self._ensure_stage_config()

        ndim = pixels.ndim
        if ndim == 5:
            src = pixels[0]
        elif ndim == 4:
            src = pixels
        else:
            raise ValueError(
                f"encode expects (T,H,W,3) or (1,T,H,W,3); got {tuple(pixels.shape)}"
            )
        vae_enc = self._stage_vae_encoder

        def _encode():
            x = _pixels_thwc_to_ncthw(src)
            lat = encode_wan_vae(x, config, vae_enc)
            lat_5d = lat[None]
            mx.eval(lat_5d)
            return lat_5d

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("video"), _encode)
        logger.info("stage:vae encode wan2 out_shape=%s", tuple(result.shape))
        return result

    async def _load_vae_encoder_stage(self) -> None:
        from pathlib import Path

        from fusion_mlx.video.wan2.stage import resolve_vae_path
        from fusion_mlx.video.wan2.utils import load_vae_encoder

        config = self._ensure_stage_config()
        vae_path = resolve_vae_path(Path(self._model_dir))

        def _load():
            return load_vae_encoder(vae_path, config)

        loop = asyncio.get_running_loop()
        self._stage_vae_encoder = await asyncio.wait_for(
            loop.run_in_executor(get_executor("video"), _load),
            timeout=_T5_PRELOAD_TIMEOUT,
        )
        self._stage_flags["vae_encoder"] = True
        gc.collect()
        logger.info("stage:vae_encoder load wan2 active_mem=%s", _active_mem())

    def set_progress_callback(self, cb):
        # Wired by FusionEngineWrapper.set_progress_callback before load_dit().
        # Stored so denoise() can forward per-step progress to the node.
        self._stage_on_step = cb

    def constraints(self) -> VideoConstraints:
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
    on_step_sync: Callable[[int, int], None] | None = None,
    session_id: str | None = None,
    output_format: str = "mp4",
    precomputed_context: tuple | None = None,
    control_video: str | None = None,
    control_mask: str | None = None,
    reference_images: list[str] | None = None,
    camera_conditions: str | None = None,
) -> bytes | Any:
    from fusion_mlx.video.wan2.generate import generate_video

    raw_output = output_format == "raw"

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
        scheduler=scheduler,
        no_compile=no_compile,
    )
    if raw_output:
        gen_kwargs["output_format"] = "raw"
    if tiling is not None:
        gen_kwargs["tiling"] = tiling
    if on_step_sync is not None:
        gen_kwargs["on_step_sync"] = on_step_sync
    if session_id is not None:
        gen_kwargs["session_id"] = session_id
    if precomputed_context is not None:
        gen_kwargs["precomputed_context"] = precomputed_context
        gen_kwargs["keep_t5"] = True
    if control_video is not None:
        gen_kwargs["control_video"] = control_video
    if control_mask is not None:
        gen_kwargs["control_mask"] = control_mask
    if reference_images is not None:
        gen_kwargs["reference_images"] = reference_images
    if camera_conditions is not None:
        gen_kwargs["camera_conditions"] = camera_conditions

    logger.info(
        "Wan2 generate (%s): prompt_len=%d frames=%d %dx%d seed=%d i2v=%s "
        "steps=%s compile=%s tiling=%s cached=%s vace=%s camera=%s",
        "raw" if raw_output else "mp4",
        len(prompt),
        num_frames,
        width,
        height,
        seed,
        bool(image),
        steps,
        not no_compile,
        tiling,
        precomputed_context is not None,
        bool(control_video),
        camera_conditions is not None,
    )

    if raw_output:
        return generate_video(model_dir, prompt, **gen_kwargs)

    with managed_tempfile_path(prefix="fusion_video_", suffix=".mp4") as handle:
        gen_kwargs["output_path"] = handle.path
        generate_video(model_dir, prompt, **gen_kwargs)
        with open(handle.path, "rb") as f:
            return f.read()
