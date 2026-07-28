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

from ..._tempfile_safe import managed_tempfile_path
from ...engine_core import get_executor, get_video_gen_timeout
from .._progress import make_sync_step_callback
from .base import VideoBackend, VideoConstraints, VideoGenParams

logger = logging.getLogger(__name__)

_DEFAULT_SCHEDULER = "unipc"

# Max T5 text-embedding cache entries (LRU eviction when exceeded).
_T5_EMBED_CACHE_MAX = 16
# Timeout for T5 encoder preload during start() — large model may take minutes.
_T5_PRELOAD_TIMEOUT = 300.0


def _infer_config_from_path(model_dir: str) -> "WanModelConfig":
    from fusion_mlx.video.wan2.config import WanModelConfig

    p = model_dir.lower()
    if "vace" in p:
        return WanModelConfig(
            model_type="vace",
            in_dim=16,
            out_dim=16,
        )
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

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self._model_name = model_name
        self._loaded = False
        self._model_dir = None
        self._t5_encoder = None
        self._t5_tokenizer = None
        self._t5_config = None
        self._embed_cache = {}
        self._embed_cache_lock = threading.Lock()

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
            from transformers import AutoTokenizer

            if self._t5_tokenizer is None:
                self._t5_tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")

            logger.info(
                "Wan2: computing T5 embeds for prompt_len=%d", len(prompt or "")
            )
            context = encode_text(
                self._t5_encoder, self._t5_tokenizer, prompt, text_len
            )
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

    logger.info(
        "Wan2 generate (%s): prompt_len=%d frames=%d %dx%d seed=%d i2v=%s "
        "steps=%s compile=%s tiling=%s cached=%s",
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
    )

    if raw_output:
        return generate_video(model_dir, prompt, **gen_kwargs)

    with managed_tempfile_path(prefix="fusion_video_", suffix=".mp4") as handle:
        gen_kwargs["output_path"] = handle.path
        generate_video(model_dir, prompt, **gen_kwargs)
        with open(handle.path, "rb") as f:
            return f.read()
