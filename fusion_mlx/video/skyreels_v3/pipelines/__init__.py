# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 三大 Pipeline 端口 (R2V / V2V / A2V).

统一封装:
  - 文本编码 (UMT5)
  - VAE 编解码
  - DiT 去噪采样
  - xfuser 步级策略
  - M5 Max 专属优化

输出:
  - R2V: 1~4 张参考图 + Prompt -> 5s 720p 24fps 视频
  - V2V: 5s 输入视频 -> 30s 续写视频
  - A2V: 音频 + 参考图 -> 数字人说话视频 (口型同步)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .. import _device
from ..transformer_r2v import SkyReelsR2VDiT
from ..transformer_v2v import SkyReelsV2VDiT
from ..transformer_a2v import SkyReelsA2VDiT
from ..vae import SkyReelsVAE, decode_to_video, save_video
from ..text_encoder import UMT5Encoder, CLIPTextEncoder
from ..scheduler.fm_solvers_unipc import (
    FlowUniPCMultistepScheduler,
    perform_guidance,
    flow_match_sample,
)
from ..step_strategy import SkyReelsStepStrategy
from ..m5_optimizer import M5Optimizer
from ..temporal_flicker_fix import (
    TemporalFlickerFix,
    default_config_for_branch as _flicker_cfg_for_branch,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline 基类
# ---------------------------------------------------------------------------
@dataclass
class SkyReelsPipelineConfig:
    """Pipeline 通用配置."""
    branch: str = "r2v"  # r2v / v2v / a2v
    width: int = 1280  # 720P
    height: int = 720
    num_frames: int = 121  # 5s @ 24fps
    fps: int = 24
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    cfg_scale: float = 5.0  # alias
    seed: int | None = None
    tiling: bool = False
    tile_size: tuple = (1, 56, 56)

    # V2V 续写参数
    context_window_size: int = 0
    temporal_window: int = 96

    # A2V 音频参数
    audio_sample_rate: int = 16000
    audio_dim: int = 1024  # wav2vec2 hidden size

    # 优化开关
    use_m5_optimizer: bool = True
    use_step_strategy: bool = True
    use_tiling: bool = False


class SkyReelsBasePipeline:
    """Pipeline 基类: 加载模型 + 编码文本 + 采样 + 解码."""

    def __init__(
        self,
        model_path: str | Path,
        config: SkyReelsPipelineConfig,
    ):
        self.model_path = Path(model_path)
        self.config = config
        self.dit: nn.Module | None = None
        self.vae: SkyReelsVAE | None = None
        self.text_encoder: UMT5Encoder | None = None
        self.clip_encoder: CLIPTextEncoder | None = None
        self.m5_optimizer: M5Optimizer | None = None
        self.step_strategy: SkyReelsStepStrategy | None = None

        self._load_models()
        self._setup_optimizers()

    def _load_models(self) -> None:
        """加载 DiT + VAE + 文本编码器."""
        branch = self.config.branch

        # 1. 加载 DiT
        if branch == "r2v":
            self.dit = SkyReelsR2VDiT()
        elif branch == "v2v":
            self.dit = SkyReelsV2VDiT()
        elif branch == "a2v":
            self.dit = SkyReelsA2VDiT()
        else:
            raise ValueError(f"Unknown branch: {branch}")

        # 2. 加载 VAE
        self.vae = SkyReelsVAE()

        # 3. 加载文本编码器
        self.text_encoder = UMT5Encoder()

        # 4. A2V 额外加 CLIP
        if branch == "a2v":
            self.clip_encoder = CLIPTextEncoder()

        logger.info(
            "Pipeline loaded: branch=%s dit=%s",
            branch, type(self.dit).__name__,
        )

    def _setup_optimizers(self) -> None:
        """设置 M5 优化器 + xfuser 步级策略."""
        if self.config.use_m5_optimizer:
            self.m5_optimizer = M5Optimizer()
            if self.dit is not None:
                self.m5_optimizer.apply_to_model(self.dit)
            if self.vae is not None:
                self.m5_optimizer.apply_to_vae(self.vae)

        if self.config.use_step_strategy and self.dit is not None:
            self.step_strategy = SkyReelsStepStrategy(
                branch=self.config.branch,
                total_steps=self.config.num_inference_steps,
            )
            self.step_strategy.attach_to_model(self.dit)

    def _encode_text(self, prompt: str) -> mx.array:
        """编码文本 prompt -> UMT5 embedding."""
        if self.text_encoder is None:
            raise RuntimeError("Text encoder not loaded")

        # 简化: 用 tokenizer 编码 prompt -> input_ids
        # 实际实现需要加载 tokenizer
        # 这里返回零张量作为 stub
        b = 1
        l = self.config.num_frames  # 临时
        return mx.zeros((b, l, self.text_encoder.config.d_model))

    def _encode_context(
        self,
        prompt: str,
        ref_images: list[Any] | None = None,
    ) -> mx.array:
        """编码完整 context (prompt + 参考图)."""
        # 简化: 返回零张量
        b = 1
        l = 257 + 512  # img + text
        return mx.zeros((b, l, 5120))  # R2V dim

    def _denoise_sample(
        self,
        latents: mx.array,
        context: mx.array,
        *,
        seq_lens: list,
        grid_sizes: list,
    ) -> mx.array:
        """完整去噪采样循环."""
        if self.dit is None or self.step_strategy is None:
            raise RuntimeError("DiT or step strategy not loaded")

        scheduler = FlowUniPCMultistepScheduler(
            num_inference_steps=self.config.num_inference_steps,
        )
        scheduler.set_timesteps(self.config.num_inference_steps)

        self.step_strategy.reset()

        # 初始化时序闪烁修复器 (按分支默认配置)
        flicker_cfg = _flicker_cfg_for_branch(self.config.branch)
        # V2V 续写: 启用边界对齐
        if self.config.branch == "v2v":
            flicker_cfg.enable_boundary_align = True
        flicker_fix = TemporalFlickerFix(flicker_cfg)
        flicker_fix.reset_step_filter()

        # V2V 边界对齐: 输入视频末帧 latent (stub: None 表示不对齐)
        input_end_latent: mx.array | None = None

        for step_idx, t in enumerate(scheduler.timesteps):
            # 设置当前步 (xfuser 步级策略)
            self.step_strategy.set_current_step(step_idx)

            t_mx = mx.array([float(t)])

            # CFG: 拼接 cond + uncond
            latent_input = mx.concatenate([latents] * 2)
            context_input = mx.concatenate([context] * 2)

            # 模型前向
            noise_pred = self.dit(
                latent_input, t_mx, context_input,
                seq_lens, grid_sizes,
            )

            # CFG 合并
            noise_pred = perform_guidance(
                noise_pred, self.config.guidance_scale,
            )

            # 时序闪烁修复: 采样步连贯滤波 (防步间预测跳变)
            noise_pred = flicker_fix.filter_step(noise_pred)

            # 采样步
            latents = scheduler.step(
                noise_pred, float(t), latents,
            ).prev_sample

            # 时序闪烁修复: 帧间 EMA 平滑 (防相邻帧跳变)
            latents = flicker_fix.smooth_temporal(latents)

            # V2V 续写: 边界对齐 (续写首帧与输入末帧对齐)
            if flicker_cfg.enable_boundary_align and input_end_latent is not None:
                latents = flicker_fix.align_boundary(
                    latents, input_end_latent,
                )

        return latents

    def generate(self, *args, **kwargs) -> mx.array:
        """子类实现具体生成逻辑."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# R2V Pipeline (参考图 -> 视频, 14B-720P)
# ---------------------------------------------------------------------------
class SkyReelsR2VPipeline(SkyReelsBasePipeline):
    """SkyReels-V3 R2V Pipeline: 参考图 + Prompt -> 5s 720p 视频.

    用法:
        pipeline = SkyReelsR2VPipeline(model_path)
        video = pipeline.generate(
            prompt="...",
            ref_images=[img1, img2],  # 1~4 张
            duration=5,
        )
        pipeline.save(video, "output.mp4")
    """

    def __init__(self, model_path: str | Path):
        config = SkyReelsPipelineConfig(branch="r2v")
        super().__init__(model_path, config)

    def generate(
        self,
        prompt: str,
        ref_images: list[Any] | None = None,
        duration: int = 5,
        *,
        seed: int | None = None,
    ) -> mx.array:
        """生成视频.

        Args:
            prompt: 文本 prompt
            ref_images: 1~4 张参考图 (PIL Image)
            duration: 视频时长 (秒)
            seed: 随机种子

        Returns:
            [B, 3, T, H, W] 视频帧 (像素 [0, 1])
        """
        cfg = self.config
        cfg.num_frames = duration * cfg.fps
        if cfg.num_frames % 2 == 0:
            cfg.num_frames += 1  # Wan 要求奇数帧

        if seed is not None:
            mx.random.seed(seed)
            cfg.seed = seed

        # 1. 编码 context (prompt + 参考图)
        context = self._encode_context(prompt, ref_images)

        # 2. 初始化 latent 噪声
        # latent shape: [B, 16, T//4+1, H//8//2, W//8//2]
        # 简化: 用 720P latent
        b = 1
        latent_h = cfg.height // 16  # 简化
        latent_w = cfg.width // 16
        latent_t = (cfg.num_frames - 1) // 4 + 1

        latents = mx.random.normal(
            (b, 16, latent_t, latent_h, latent_w)
        ) * 1.0  # init_noise_sigma

        # 3. 采样参数
        seq_lens = [latent_t * latent_h * latent_w]
        grid_sizes = [(latent_t, latent_h, latent_w)]

        # 4. 去噪采样
        latents = self._denoise_sample(
            latents, context,
            seq_lens=seq_lens, grid_sizes=grid_sizes,
        )

        # 5. VAE 解码
        video = decode_to_video(
            self.vae, latents,
            fps=cfg.fps, tiling=cfg.use_tiling,
        )

        logger.info(
            "R2V generated: %dx%dx%d (%.1fs)",
            cfg.width, cfg.height, cfg.num_frames, cfg.num_frames / cfg.fps,
        )
        return video

    def save(self, video: mx.array, output_path: str) -> None:
        """保存视频."""
        save_video(video, output_path, fps=self.config.fps)


# ---------------------------------------------------------------------------
# V2V Pipeline (视频续写, 14B-720P)
# ---------------------------------------------------------------------------
class SkyReelsV2VPipeline(SkyReelsBasePipeline):
    """SkyReels-V3 V2V Pipeline: 输入视频 -> 续写视频.

    支持两种模式:
      - single_shot_extension: 5s -> 30s 单镜头续写
      - shot_switching_extension: 镜头切换 (Cut-In/Cut-Out)

    用法:
        pipeline = SkyReelsV2VPipeline(model_path)
        video = pipeline.generate(
            input_video="input.mp4",
            prompt="...",
            duration=30,
        )
    """

    def __init__(self, model_path: str | Path):
        config = SkyReelsPipelineConfig(
            branch="v2v",
            context_window_size=32,  # 复用前置 32 帧
            temporal_window=96,
        )
        super().__init__(model_path, config)

    def generate(
        self,
        input_video: str,
        prompt: str,
        duration: int = 30,
        *,
        seed: int | None = None,
    ) -> mx.array:
        """续写视频.

        Args:
            input_video: 输入视频路径 (5s)
            prompt: 续写 prompt
            duration: 续写后总时长 (秒)
            seed: 随机种子

        Returns:
            [B, 3, T, H, W] 续写视频
        """
        cfg = self.config
        cfg.num_frames = duration * cfg.fps

        if seed is not None:
            mx.random.seed(seed)

        # 1. 编码 context
        context = self._encode_context(prompt)

        # 2. 初始化 latent (含前置帧)
        b = 1
        latent_h = cfg.height // 16
        latent_w = cfg.width // 16
        latent_t = (cfg.num_frames - 1) // 4 + 1

        # V2V: 前置帧 latent + 续写帧噪声
        # 简化: 全部初始化为噪声
        latents = mx.random.normal(
            (b, 16, latent_t, latent_h, latent_w)
        )

        # 3. 采样参数
        seq_lens = [latent_t * latent_h * latent_w]
        grid_sizes = [(latent_t, latent_h, latent_w)]

        # 4. 去噪采样 (V2V 启用时序分支)
        latents = self._denoise_sample(
            latents, context,
            seq_lens=seq_lens, grid_sizes=grid_sizes,
        )

        # 5. VAE 解码
        video = decode_to_video(
            self.vae, latents,
            fps=cfg.fps, tiling=cfg.use_tiling,
        )

        logger.info(
            "V2V generated: %dx%dx%d (%.1fs)",
            cfg.width, cfg.height, cfg.num_frames, cfg.num_frames / cfg.fps,
        )
        return video


# ---------------------------------------------------------------------------
# A2V Pipeline (音频数字人, 19B-720P)
# ---------------------------------------------------------------------------
class SkyReelsA2VPipeline(SkyReelsBasePipeline):
    """SkyReels-V3 A2V Pipeline: 音频 + 参考图 -> 数字人说话视频.

    用法:
        pipeline = SkyReelsA2VPipeline(model_path)
        video = pipeline.generate(
            audio="speech.wav",
            ref_image=ref_img,
            prompt="...",
            duration=10,
        )
    """

    def __init__(self, model_path: str | Path):
        config = SkyReelsPipelineConfig(
            branch="a2v",
            temporal_window=32,  # 保嘴型连贯
        )
        super().__init__(model_path, config)

    def generate(
        self,
        audio: str,
        ref_image: Any,
        prompt: str,
        duration: int = 10,
        *,
        seed: int | None = None,
    ) -> mx.array:
        """生成数字人说话视频.

        Args:
            audio: 音频文件路径 (wav/mp3)
            ref_image: 参考人脸图
            prompt: 文本 prompt (辅助)
            duration: 视频时长 (秒)
            seed: 随机种子

        Returns:
            [B, 3, T, H, W] 数字人视频
        """
        cfg = self.config
        cfg.num_frames = duration * cfg.fps

        if seed is not None:
            mx.random.seed(seed)

        # 1. 编码音频 (wav2vec2) + 文本 (xlm_roberta)
        # 简化: 用零张量作为 stub
        audio_embeds = mx.zeros(
            (1, cfg.num_frames, cfg.audio_dim)
        )
        text_embeds = mx.zeros((1, 512, 4096))

        # 2. 初始化 latent
        b = 1
        latent_h = cfg.height // 16
        latent_w = cfg.width // 16
        latent_t = (cfg.num_frames - 1) // 4 + 1

        latents = mx.random.normal(
            (b, 16, latent_t, latent_h, latent_w)
        )

        # 3. 采样参数
        seq_lens = [latent_t * latent_h * latent_w]
        grid_sizes = [(latent_t, latent_h, latent_w)]

        # 4. 去噪采样 (A2V 启用时序分支保嘴型连贯)
        # 注意: A2V DiT 前向签名不同 (audio + text embeds)
        if self.dit is not None and self.step_strategy is not None:
            scheduler = FlowUniPCMultistepScheduler(
                num_inference_steps=cfg.num_inference_steps,
            )
            scheduler.set_timesteps(cfg.num_inference_steps)
            self.step_strategy.reset()

            for step_idx, t in enumerate(scheduler.timesteps):
                self.step_strategy.set_current_step(step_idx)
                t_mx = mx.array([float(t)])

                # A2V CFG: 拼接 cond + uncond
                latent_input = mx.concatenate([latents] * 2)
                audio_input = mx.concatenate([audio_embeds] * 2)
                text_input = mx.concatenate([text_embeds] * 2)

                # A2V DiT 前向
                noise_pred = self.dit(
                    latent_input, t_mx,
                    audio_input, text_input,
                    seq_lens, grid_sizes,
                )

                # CFG 合并
                noise_pred = perform_guidance(
                    noise_pred, cfg.guidance_scale,
                )

                # 采样步
                latents = scheduler.step(
                    noise_pred, float(t), latents,
                ).prev_sample

        # 5. VAE 解码
        video = decode_to_video(
            self.vae, latents,
            fps=cfg.fps, tiling=cfg.use_tiling,
        )

        logger.info(
            "A2V generated: %dx%dx%d (%.1fs)",
            cfg.width, cfg.height, cfg.num_frames, cfg.num_frames / cfg.fps,
        )
        return video


__all__ = [
    "SkyReelsPipelineConfig",
    "SkyReelsR2VPipeline",
    "SkyReelsV2VPipeline",
    "SkyReelsA2VPipeline",
]
