# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 video backend (R2V 14B / V2V 14B / A2V 19B).

SkyReels-V3 is a reference-to-video / video-to-video / audio-to-video model
family.  The backend delegates to the vendored fusion_mlx.video.skyreels_v3
pipeline.  SkyReels supports image-to-video (I2V / R2V), so
``supports_i2v=True``.
"""

import asyncio
import gc
import logging
import random
from typing import Any

import mlx.core as mx

from ..._tempfile_safe import managed_tempfile_path
from ...engine_core import get_executor, get_video_gen_timeout
from .base import VideoBackend, VideoConstraints, VideoGenParams

logger = logging.getLogger(__name__)


def _active_mem() -> int:
    try:
        return int(mx.metal.get_active_memory())
    except Exception:
        return -1


class SkyReelsBackend(VideoBackend):
    name = "skyreels"
    supports_i2v = True

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self._model_name = model_name
        self._loaded = False
        # AtomCode fix #130: 缓存 pipeline 实例避内存泄漏 (2026-07-19)
        # 每次 generate() 新建 pipeline 会加载 DiT~14B+VAE+T5 三个大模型到 Metal 显存,
        # 不释放则多次调用累积显存撑爆 M5 Max 统一内存. 缓存 pipeline 实例,
        # stop() 时显式释放大模型引用 + gc.collect + mx.clear_cache.
        self._pipeline: Any = None
        self._pipeline_class: type | None = None
        # Stage API (#170): track which components are "loaded" for ComfyUI sequential offload.
        # Pipeline loads all 3 models in __init__, but unload_* sets them to None.
        self._stage_flags: dict[str, bool] = {
            "text_encoder": False,
            "dit": False,
            "vae": False,
        }

    @classmethod
    def detect(cls, model_path: str) -> bool:
        return "skyreels" in model_path.lower()

    async def start(self, model_path: str, **kwargs: Any) -> None:
        if self._loaded:
            return
        logger.info("Starting SkyReels backend: %s", model_path)
        self._loaded = True
        logger.info("SkyReels backend ready: %s", model_path)

    async def stop(self) -> None:
        if not self._loaded:
            return
        self._loaded = False
        # AtomCode fix #130: 显式释放 pipeline 大模型引用 (2026-07-19)
        # 不清除 DiT/VAE/T5 引用则 Metal 显存累积, 多次调用撑爆统一内存
        if self._pipeline is not None:
            try:
                # 释放 pipeline 内部大模型引用
                for attr in (
                    "dit",
                    "vae",
                    "text_encoder",
                    "clip_encoder",
                    "m5_optimizer",
                    "step_strategy",
                ):
                    if hasattr(self._pipeline, attr):
                        setattr(self._pipeline, attr, None)
                self._pipeline = None
            except Exception:
                pass
        self._pipeline_class = None
        self._stage_flags = {"text_encoder": False, "dit": False, "vae": False}
        gc.collect()
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                get_executor("io"), lambda: (mx.synchronize(), mx.clear_cache())
            ),
            timeout=5.0,
        )
        logger.info("SkyReels backend stopped: %s", self._model_name)

    async def generate(self, params: VideoGenParams) -> list[bytes]:
        if params.on_step is not None:
            logger.debug(
                "skyreels: on_step progress callback accepted but per-step "
                "streaming not yet emitted for this backend (issue #171 follow-up)"
            )
        # Detect branch from model name
        model_lower = self._model_name.lower()
        if "r2v" in model_lower or "r2v_14b" in model_lower:
            return await self._generate_r2v(params)
        elif "v2v" in model_lower or "v2v_14b" in model_lower:
            return await self._generate_v2v(params)
        elif "a2v" in model_lower or "a2v_19b" in model_lower:
            return await self._generate_a2v(params)
        else:
            # Default to R2V for SkyReels models
            logger.warning(
                "Unknown SkyReels branch, defaulting to R2V: %s", self._model_name
            )
            return await self._generate_r2v(params)

    # ------------------------------------------------------------------
    # Pipeline stage API (issue #170). Override VideoBackend NotImplementedError
    # defaults to expose individual pipeline stages for Fusion-ComfyUI
    # sequential offload. Video latents are 5D (batch, c, num_frames, h, w).
    # ------------------------------------------------------------------
    def _detect_pipeline_class(self) -> type:
        model_lower = self._model_name.lower()
        if "r2v" in model_lower or "r2v_14b" in model_lower:
            from fusion_mlx.video.skyreels_v3.pipelines import SkyReelsR2VPipeline

            return SkyReelsR2VPipeline
        elif "v2v" in model_lower or "v2v_14b" in model_lower:
            from fusion_mlx.video.skyreels_v3.pipelines import SkyReelsV2VPipeline

            return SkyReelsV2VPipeline
        elif "a2v" in model_lower or "a2v_19b" in model_lower:
            from fusion_mlx.video.skyreels_v3.pipelines import SkyReelsA2VPipeline

            return SkyReelsA2VPipeline
        else:
            from fusion_mlx.video.skyreels_v3.pipelines import SkyReelsR2VPipeline

            return SkyReelsR2VPipeline

    async def _ensure_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        pipeline_class = self._detect_pipeline_class()
        await self._get_or_create_pipeline(pipeline_class)
        # Mark all components loaded (pipeline.__init__ loads all 3)
        self._stage_flags["text_encoder"] = True
        self._stage_flags["dit"] = True
        self._stage_flags["vae"] = True
        return self._pipeline

    async def load_text_encoder(self) -> None:
        pipeline = await self._ensure_pipeline()
        if pipeline.text_encoder is None:
            raise RuntimeError(
                "text_encoder was unloaded; call _ensure_pipeline() to reload"
            )
        self._stage_flags["text_encoder"] = True
        gc.collect()
        logger.info("stage:text_encoder load active_mem=%s", _active_mem())

    async def encode_text(self, prompt: str) -> dict:
        pipeline = await self._ensure_pipeline()
        if pipeline.text_encoder is None:
            raise RuntimeError("text_encoder is unloaded; call load_text_encoder().")

        def _enc():
            context = pipeline._encode_context(prompt)
            mx.eval(context)
            return context

        loop = asyncio.get_running_loop()
        context = await loop.run_in_executor(get_executor("video"), _enc)
        logger.info(
            "stage:text_encoder encode prompt_len=%d context_shape=%s",
            len(prompt),
            tuple(context.shape),
        )
        return {"context": context}

    async def unload_text_encoder(self) -> None:
        pipeline = await self._ensure_pipeline()
        pipeline.text_encoder = None
        if hasattr(pipeline, "clip_encoder"):
            pipeline.clip_encoder = None
        self._stage_flags["text_encoder"] = False
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        logger.info("stage:text_encoder unload")

    async def load_dit(self) -> None:
        pipeline = await self._ensure_pipeline()
        if pipeline.dit is None:
            raise RuntimeError("dit was unloaded; call _ensure_pipeline() to reload")
        self._stage_flags["dit"] = True
        gc.collect()
        logger.info("stage:dit load active_mem=%s", _active_mem())

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
        pipeline = await self._ensure_pipeline()
        if pipeline.dit is None:
            raise RuntimeError("dit is unloaded; call load_dit().")

        # Build context: pos_embed is the full context from encode_text.
        # If neg_embed provided, concatenate for CFG; otherwise use pos_embed only.
        if neg_embed is not None:
            context = mx.concatenate([neg_embed, pos_embed])
        else:
            context = pos_embed

        # Derive grid_sizes / seq_lens from latent shape + DiT patch_size
        cfg_pipeline = pipeline.config
        b, c, latent_t, latent_h, latent_w = latent.shape
        pt, ph, pw = pipeline.dit.patch_size
        grid_t = latent_t // pt
        grid_h = latent_h // ph
        grid_w = latent_w // pw
        seq_lens = [grid_t * grid_h * grid_w]
        grid_sizes = [(grid_t, grid_h, grid_w)]

        # Override steps temporarily for this denoise call
        saved_steps = cfg_pipeline.num_inference_steps
        saved_cfg = cfg_pipeline.guidance_scale
        cfg_pipeline.num_inference_steps = steps
        cfg_pipeline.guidance_scale = cfg
        try:

            def _denoise():
                result = pipeline._denoise_sample(
                    latent, context, seq_lens=seq_lens, grid_sizes=grid_sizes
                )
                mx.eval(result)
                return result

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(get_executor("video"), _denoise)
            logger.info(
                "stage:dit denoise steps=%d cfg=%.2f latent_shape=%s result_shape=%s",
                steps,
                cfg,
                tuple(latent.shape),
                tuple(result.shape),
            )
            return result
        finally:
            cfg_pipeline.num_inference_steps = saved_steps
            cfg_pipeline.guidance_scale = saved_cfg

    async def unload_dit(self) -> None:
        pipeline = await self._ensure_pipeline()
        pipeline.dit = None
        pipeline.step_strategy = None
        self._stage_flags["dit"] = False
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        logger.info("stage:dit unload")

    async def load_vae(self) -> None:
        pipeline = await self._ensure_pipeline()
        if pipeline.vae is None:
            raise RuntimeError("vae was unloaded; call _ensure_pipeline() to reload")
        self._stage_flags["vae"] = True
        gc.collect()
        logger.info("stage:vae load active_mem=%s", _active_mem())

    async def decode(self, latent: mx.array) -> mx.array:
        pipeline = await self._ensure_pipeline()
        if pipeline.vae is None:
            raise RuntimeError("vae is unloaded; call load_vae().")

        def _decode():
            result = pipeline.vae.decode(latent)
            mx.eval(result)
            return result

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("video"), _decode)
        logger.info(
            "stage:vae decode latent_shape=%s result_shape=%s",
            tuple(latent.shape),
            tuple(result.shape),
        )
        return result

    async def decode_tiled(self, latent: mx.array, tile_size: int = 256) -> mx.array:
        pipeline = await self._ensure_pipeline()
        if pipeline.vae is None:
            raise RuntimeError("vae is unloaded; call load_vae().")

        def _decode_tiled():
            # tile_size is in pixels; convert to latent-space tile tuple
            latent_tile = max(1, tile_size // 8)
            result = pipeline.vae.decode(
                latent, tiling=True, tile_size=(1, latent_tile, latent_tile)
            )
            mx.eval(result)
            return result

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("video"), _decode_tiled)
        logger.info(
            "stage:vae decode_tiled tile_size=%d latent_shape=%s result_shape=%s",
            tile_size,
            tuple(latent.shape),
            tuple(result.shape),
        )
        return result

    async def unload_vae(self) -> None:
        pipeline = await self._ensure_pipeline()
        pipeline.vae = None
        self._stage_flags["vae"] = False
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        logger.info("stage:vae unload")

    async def _get_or_create_pipeline(self, pipeline_class: type) -> Any:
        """AtomCode fix #130: 获取或创建缓存 pipeline 实例.

        复用已加载的 pipeline 避每次 generate() 重建 DiT/VAE/T5 大模型,
        显存泄漏防护: 同一 pipeline_class 复用, 不同 class 时释放旧 pipeline.
        """
        if self._pipeline is not None and self._pipeline_class is pipeline_class:
            logger.debug("Reusing cached pipeline: %s", pipeline_class.__name__)
            return self._pipeline
        # 不同 class 时释放旧 pipeline 再建新
        if self._pipeline is not None:
            logger.info("Pipeline class changed, releasing old pipeline")
            try:
                for attr in (
                    "dit",
                    "vae",
                    "text_encoder",
                    "clip_encoder",
                    "m5_optimizer",
                    "step_strategy",
                ):
                    if hasattr(self._pipeline, attr):
                        setattr(self._pipeline, attr, None)
                self._pipeline = None
            except Exception:
                pass
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
        logger.info(
            "Loading pipeline %s (DiT/VAE/T5 weights, first call may take minutes)...",
            pipeline_class.__name__,
        )
        self._pipeline = pipeline_class(self._model_name)
        self._pipeline_class = pipeline_class
        logger.info("Created pipeline: %s", pipeline_class.__name__)
        try:
            self._pipeline.warmup()
        except Exception as exc:
            logger.warning("pipeline warmup raised (non-fatal): %s", exc)
        return self._pipeline

    async def _generate_r2v(self, params: VideoGenParams) -> list[bytes]:
        """R2V: 参考图 + Prompt -> 视频."""
        from fusion_mlx.video.skyreels_v3.pipelines import SkyReelsR2VPipeline

        base_seed = (
            params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
        )
        duration = max(1, params.num_frames // 24)  # fps=24 → duration in seconds

        logger.info(
            "video gen: start branch=r2v model=%s prompt_len=%d duration=%ds seed=%d",
            self._model_name,
            len(params.prompt or ""),
            duration,
            base_seed,
        )

        def _gen_one() -> bytes:
            pipeline = self._pipeline
            ref_images = [params.image] if params.image else None
            gen_kwargs: dict = {
                "prompt": params.prompt,
                "ref_images": ref_images,
                "duration": duration,
                "seed": base_seed,
            }
            # Phase-2: session tail cache for multi-shot latent reuse
            if params.session_id is not None:
                gen_kwargs["session_id"] = params.session_id
            # IP-Adapter: subject-driven image-to-video
            if params.ip_adapter_image is not None:
                gen_kwargs["ip_adapter_image"] = params.ip_adapter_image
                gen_kwargs["ip_adapter_scale"] = params.ip_adapter_scale
            # ControlNet: structural guidance via control image
            if params.controlnet_image is not None:
                gen_kwargs["controlnet_image"] = params.controlnet_image
                gen_kwargs["controlnet_strength"] = params.controlnet_strength
                gen_kwargs["control_type"] = params.control_type
            if getattr(params, "animatediff_scale", 0.0) > 0:
                gen_kwargs["animatediff_scale"] = params.animatediff_scale
            video = pipeline.generate(**gen_kwargs)
            with managed_tempfile_path(
                prefix="fusion_skyreels_", suffix=".mp4"
            ) as handle:
                pipeline.save(video, handle.path)
                with open(handle.path, "rb") as f:
                    return f.read()

        await self._get_or_create_pipeline(SkyReelsR2VPipeline)
        loop = asyncio.get_running_loop()
        results = await asyncio.wait_for(
            loop.run_in_executor(get_executor("video"), _gen_one),
            timeout=get_video_gen_timeout(),
        )
        logger.info("video gen: done branch=r2v seed=%d", base_seed)
        return [results]

    async def _generate_v2v(self, params: VideoGenParams) -> list[bytes]:
        """V2V: 输入视频 -> 续写视频."""
        from fusion_mlx.video.skyreels_v3.pipelines import SkyReelsV2VPipeline

        base_seed = (
            params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
        )
        duration = max(1, params.num_frames // 24)
        logger.info(
            "video gen: start branch=v2v model=%s prompt_len=%d duration=%ds seed=%d",
            self._model_name,
            len(params.prompt or ""),
            duration,
            base_seed,
        )

        def _gen_one() -> bytes:
            pipeline = self._pipeline
            gen_kwargs: dict = {
                "prompt": params.prompt,
                "input_video": params.image,  # V2V 可接受视频路径作为 image
                "duration": duration,
                "seed": base_seed,
            }
            # Phase-2: session tail cache for multi-shot latent reuse
            if params.session_id is not None:
                gen_kwargs["session_id"] = params.session_id
            # ControlNet: structural guidance via control image
            if params.controlnet_image is not None:
                gen_kwargs["controlnet_image"] = params.controlnet_image
                gen_kwargs["controlnet_strength"] = params.controlnet_strength
                gen_kwargs["control_type"] = params.control_type
            if getattr(params, "animatediff_scale", 0.0) > 0:
                gen_kwargs["animatediff_scale"] = params.animatediff_scale
            video = pipeline.generate(**gen_kwargs)
            with managed_tempfile_path(
                prefix="fusion_skyreels_", suffix=".mp4"
            ) as handle:
                pipeline.save(video, handle.path)
                with open(handle.path, "rb") as f:
                    return f.read()

        await self._get_or_create_pipeline(SkyReelsV2VPipeline)
        loop = asyncio.get_running_loop()
        results = await asyncio.wait_for(
            loop.run_in_executor(get_executor("video"), _gen_one),
            timeout=get_video_gen_timeout(),
        )
        logger.info("video gen: done branch=v2v seed=%d", base_seed)
        return [results]

    async def _generate_a2v(self, params: VideoGenParams) -> list[bytes]:
        """A2V: 音频 + 参考图 -> 数字人视频."""
        from fusion_mlx.video.skyreels_v3.pipelines import SkyReelsA2VPipeline

        base_seed = (
            params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
        )
        duration = max(1, params.num_frames // 24)
        logger.info(
            "video gen: start branch=a2v model=%s prompt_len=%d duration=%ds seed=%d",
            self._model_name,
            len(params.prompt or ""),
            duration,
            base_seed,
        )

        def _gen_one() -> bytes:
            pipeline = self._pipeline
            # A2V pipeline 需要 audio 和 ref_image 参数
            gen_kwargs: dict = {
                "prompt": params.prompt,
                "audio": params.extra.get("audio", ""),
                "ref_image": params.image,
                "duration": duration,
                "seed": base_seed,
            }
            # Phase-2: session tail cache for multi-shot latent reuse
            if params.session_id is not None:
                gen_kwargs["session_id"] = params.session_id
            # ControlNet: structural guidance via control image
            if params.controlnet_image is not None:
                gen_kwargs["controlnet_image"] = params.controlnet_image
                gen_kwargs["controlnet_strength"] = params.controlnet_strength
                gen_kwargs["control_type"] = params.control_type
            if getattr(params, "animatediff_scale", 0.0) > 0:
                gen_kwargs["animatediff_scale"] = params.animatediff_scale
            video = pipeline.generate(**gen_kwargs)
            with managed_tempfile_path(
                prefix="fusion_skyreels_", suffix=".mp4"
            ) as handle:
                pipeline.save(video, handle.path)
                with open(handle.path, "rb") as f:
                    return f.read()

        await self._get_or_create_pipeline(SkyReelsA2VPipeline)
        loop = asyncio.get_running_loop()
        results = await asyncio.wait_for(
            loop.run_in_executor(get_executor("video"), _gen_one),
            timeout=get_video_gen_timeout(),
        )
        logger.info("video gen: done branch=a2v seed=%d", base_seed)
        return [results]

    def constraints(self) -> VideoConstraints:
        # SkyReels-V3: VAE stride = (4, 16, 16) → spatial dims divisible by 16,
        # temporal dims: (num_frames - 1) % 4 == 0.  Supports I2V (R2V branch).
        return VideoConstraints(
            supports_i2v=True,
            max_n=1,
            dim_divisibility=16,
            num_frames_validator=lambda nf: (nf - 1) % 4 == 0,
            num_frames_hint="num_frames must satisfy (num_frames - 1) % 4 == 0 "
            "(e.g. 9, 41, 81, 121)",
            dim_hint="width and height must be divisible by 16",
        )

    def last_denoise_stats(self) -> dict[str, Any]:
        # issue #177 Phase-3: serialize the last _denoise_sample_speculative
        # run's SpecStats (set on the pipeline as _last_spec_stats) plus the
        # current env config, for the GET /v1/videos/denoise-stats surface.
        # Default-off path: when spec is off or no run happened, returns
        # available=False with zeroed counters (honest "feature surface").
        from fusion_mlx.video.skyreels_v3.speculative_denoise import (
            SpeculativeConfig,
            speculative_enabled,
        )

        config = SpeculativeConfig.from_env()
        stats = (
            getattr(self._pipeline, "_last_spec_stats", None)
            if self._pipeline is not None
            else None
        )
        if stats is not None:
            data = stats.to_dict()
            data["available"] = True
        else:
            data = {
                "macro_steps": 0,
                "accepted": [],
                "avg_accept": 0.0,
                "full_forwards": 0,
                "draft_forwards": 0,
                "baseline_steps": 0,
                "speedup": 0.0,
                "draft_strategy": config.draft_strategy,
                "available": False,
            }
        data["enabled"] = speculative_enabled()
        data["config"] = {
            "K": config.K,
            "epsilon": config.epsilon,
            "relative": config.relative,
            "eval_steps": config.eval_steps,
            "draft_strategy": config.draft_strategy,
        }
        logger.debug(
            "skyreels last_denoise_stats: enabled=%s available=%s avg_accept=%.2f",
            data["enabled"],
            data["available"],
            data["avg_accept"],
        )
        return data
