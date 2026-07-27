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
from collections.abc import Callable
from typing import Any

import mlx.core as mx

from ..._tempfile_safe import managed_tempfile_path
from ...engine_core import get_executor, get_video_gen_timeout
from .._progress import make_sync_step_callback
from .base import VideoBackend, VideoConstraints, VideoGenParams

logger = logging.getLogger(__name__)

_DEFAULT_SCHEDULER = "unipc"

_T5_EMBED_CACHE_MAX = 16


class Wan2Backend(VideoBackend):
    name = "wan2"
    supports_i2v = True

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self._model_name = model_name
        self._loaded = False
        self._t5_encoder = None
        self._t5_tokenizer = None
        self._t5_config = None
        self._embed_cache = {}

    @classmethod
    def detect(cls, model_path: str) -> bool:
        return "wan" in model_path.lower()

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
                from fusion_mlx.video.wan2.config import WanModelConfig
                from fusion_mlx.video.wan2.utils import load_t5_encoder
                from pathlib import Path
                import json

                md = Path(model_dir)
                config_path = md / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config_dict = json.load(f)
                    config_dict.pop("quantization", None)
                    for key in ("patch_size", "vae_stride", "window_size", "sample_guide_scale"):
                        if key in config_dict and isinstance(config_dict[key], list):
                            config_dict[key] = tuple(config_dict[key])
                    config = WanModelConfig(
                        **{k: v for k, v in config_dict.items()
                           if k in WanModelConfig.__dataclass_fields__}
                    )
                else:
                    config = WanModelConfig.wan21_t2v_1_3b()

                t5_path = md / "t5_encoder.safetensors"
                if t5_path.exists():
                    self._t5_encoder = load_t5_encoder(t5_path, config)
                    self._t5_config = config
                    logger.info("Wan2: T5 encoder preloaded")
                else:
                    logger.info("Wan2: no t5_encoder.safetensors found, will load per-call")
            except Exception as e:
                logger.warning("Wan2: T5 preload failed, will load per-call: %s", e)

        await asyncio.wait_for(
            loop.run_in_executor(get_executor("io"), _preload_t5), timeout=300.0
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
        self._embed_cache.clear()
        gc.collect()
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                get_executor("io"), lambda: (mx.synchronize(), mx.clear_cache())
            ),
            timeout=5.0,
        )

    def _get_cached_embeds(self, prompt, neg_prompt, text_len):
        if self._t5_encoder is None:
            return None
        if neg_prompt is None and self._t5_config is not None:
            neg_prompt = self._t5_config.sample_neg_prompt
        cache_key = (prompt or "", neg_prompt or "", text_len)
        if cache_key in self._embed_cache:
            logger.info("Wan2: T5 embed cache hit for prompt_len=%d", len(prompt or ""))
            return self._embed_cache[cache_key]

        from fusion_mlx.video.wan2.utils import encode_text
        from transformers import AutoTokenizer

        if self._t5_tokenizer is None:
            self._t5_tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")

        logger.info("Wan2: computing T5 embeds for prompt_len=%d", len(prompt or ""))
        context = encode_text(self._t5_encoder, self._t5_tokenizer, prompt, text_len)
        if neg_prompt and neg_prompt.strip():
            context_null = encode_text(
                self._t5_encoder, self._t5_tokenizer, neg_prompt, text_len
            )
            mx.eval(context, context_null)
        else:
            context_null = None
            mx.eval(context)

        if len(self._embed_cache) >= _T5_EMBED_CACHE_MAX:
            oldest = next(iter(self._embed_cache))
            del self._embed_cache[oldest]
        self._embed_cache[cache_key] = (context, context_null)
        return (context, context_null)

    async def generate(self, params: VideoGenParams) -> list[bytes]:
        base_seed = (
            params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
        )
        scheduler = params.scheduler or _DEFAULT_SCHEDULER
        no_compile = True if params.no_compile is None else params.no_compile

        precomputed = None
        if self._t5_encoder is not None and self._t5_config is not None:
            text_len = self._t5_config.text_len
            precomputed = self._get_cached_embeds(
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
                )
                results.append(result)
            return results

        loop = asyncio.get_running_loop()
        sync_cb = make_sync_step_callback(params.on_step, loop)
        return await asyncio.wait_for(
            loop.run_in_executor(get_executor("video"), _generate),
            timeout=get_video_gen_timeout(),
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
    on_step_sync: Callable[[int, int], None] | None = None,
    session_id: str | None = None,
    output_format: str = "mp4",
    precomputed_context: tuple | None = None,
) -> bytes | Any:
    from fusion_mlx.video.wan2.generate import generate_video

    raw_output = output_format == "raw"

    if raw_output:
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
            output_format="raw",
            scheduler=scheduler,
            no_compile=no_compile,
        )
        if tiling is not None:
            gen_kwargs["tiling"] = tiling
        if on_step_sync is not None:
            gen_kwargs["on_step_sync"] = on_step_sync
        if session_id is not None:
            gen_kwargs["session_id"] = session_id
        if precomputed_context is not None:
            gen_kwargs["precomputed_context"] = precomputed_context
            gen_kwargs["keep_t5"] = True
        logger.info(
            "Wan2 generate (raw): prompt_len=%d frames=%d %dx%d seed=%d i2v=%s "
            "steps=%s compile=%s tiling=%s cached=%s",
            len(prompt), num_frames, width, height, seed, bool(image),
            steps, not no_compile, tiling, precomputed_context is not None,
        )
        return generate_video(model_dir, prompt, **gen_kwargs)

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
        if on_step_sync is not None:
            gen_kwargs["on_step_sync"] = on_step_sync
        if session_id is not None:
            gen_kwargs["session_id"] = session_id
        if precomputed_context is not None:
            gen_kwargs["precomputed_context"] = precomputed_context
            gen_kwargs["keep_t5"] = True
        generate_video(model_dir, prompt, **gen_kwargs)
        with open(temp_path, "rb") as f:
            return f.read()
