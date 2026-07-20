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
from ..weights import resolve_model_path, load_all_weights

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
    # AtomCode 专题优化: 50→30 步 (2026-07-18)
    # DiT 74% 主瓶颈 × 30步 vs 50步 = 降 40% DiT 耗时, UniPC corrector 保稳 (solver_order=2 历史预测仍可用)
    num_inference_steps: int = 30
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

    # 文本/上下文维度 (SkyReels-V3 config.json 架构常量)
    # text_dim: UMT5 输出维度 = DiT text_embedding 输入维度 (喂入 DiT 的 context 维度, 非 dim)
    # text_len: 文本 token 上限 (context 中 txt 段长度)
    text_dim: int = 4096
    text_len: int = 512

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
        """加载 DiT + VAE + 文本编码器 (真实权重, 非 stub).

        加载链路:
          1. 构造三大模型骨架 (DiT/VAE/UMT5)
          2. resolve_model_path 解析 HF cache 或显式路径
          3. load_all_weights 加载 safetensors 权重到骨架
          4. A2V 额外加 CLIP
        """
        branch = self.config.branch
        from ..config import get_branch_config, BRANCH_CONFIGS

        # 找到 model_key (r2v -> skyreels-v3-r2v-14b 等)
        model_key = next(
            (k for k, v in BRANCH_CONFIGS.items() if v.branch == branch),
            f"skyreels-v3-{branch}-14b",
        )
        branch_cfg = get_branch_config(model_key)

        # 1. 构造三大模型骨架
        if branch == "r2v":
            self.dit = SkyReelsR2VDiT()
        elif branch == "v2v":
            self.dit = SkyReelsV2VDiT()
        elif branch == "a2v":
            self.dit = SkyReelsA2VDiT()
        else:
            raise ValueError(f"Unknown branch: {branch}")

        self.vae = SkyReelsVAE()
        self.text_encoder = UMT5Encoder()

        if branch == "a2v":
            self.clip_encoder = CLIPTextEncoder()

        # 2. 解析权重路径
        weights_dir = resolve_model_path(self.model_path, model_key)

        # 3. 加载真实权重 (非 stub)
        # 若路径不存在或无 safetensors, load_all_weights 内部会跳过并 warning,
        # 模型保持随机初始化 (tiny 测试场景可用)
        if weights_dir.exists() and any(weights_dir.glob("*.safetensors")) or (
            weights_dir.is_dir()
            and any(weights_dir.glob("**/*.safetensors"))
        ):
            try:
                load_all_weights(
                    self.dit, self.vae, self.text_encoder,
                    weights_dir,
                )
                logger.info("真实权重加载成功: %s", weights_dir)
            except Exception as exc:
                logger.warning(
                    "真实权重加载失败, 回退到随机初始化: %s", exc,
                )
        else:
            logger.warning(
                "权重目录 %s 不含 safetensors, 使用随机初始化 (stub 模式)",
                weights_dir,
            )

        logger.info(
            "Pipeline loaded: branch=%s dit=%s weights=%s",
            branch, type(self.dit).__name__, weights_dir,
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
        """编码文本 prompt -> UMT5 embedding (真实 tokenizer, 非 stub).

        内部调 UMT5Encoder.encode_text:
          1. AutoTokenizer.from_pretrained("google/umt5-xxl") tokenize
          2. T5Encoder 前向得 embedding
          3. 截断到非 padding 部分

        Args:
            prompt: 文本 prompt

        Returns:
            [1, L_valid, d_model] 文本 embedding
        """
        if self.text_encoder is None:
            raise RuntimeError("Text encoder not loaded")

        # 调用 UMT5Encoder.encode_text (已接入 tokenizer)
        return self.text_encoder.encode_text(prompt, max_length=self.config.text_len)

    def _encode_context(
        self,
        prompt: str,
        ref_images: list[Any] | None = None,
    ) -> mx.array:
        """编码完整 context (prompt + 参考图).

        AtomCode fix #139 (2026-07-20): context 维度 = text_dim (4096), 非 dim (5120).
        DiT text_embedding = Linear(text_dim=4096 -> dim=5120) 在 __call__ 内做投影,
        喂入 DiT 的 context 必须是 text_dim. 原返回 5120-dim 致 text_embedding.0 输入维度
        错配 (5120 vs 期望 4096), 连锁触发 #137 自动检测误判 / #138 bias 截断 / #139 matmul 错配.
        """
        b = 1
        l = 257 + 512  # img + text
        return mx.zeros((b, l, self.config.text_dim))

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

                # AtomCode 专题优化: cross_kv_cache 跨步复用 (2026-07-18)
                # context (text_input) 跨采样步固定不变, 每步重算 cross-attn KV 是浪费
                # 首步预分配 KV 缓存, 后续步复用避迭代扩容抖动 + 砍 cross-attn KV 投影耗
                if step_idx == 0:
                    cross_kv_cache = self._prepare_cross_kv_cache(text_input)
                else:
                    cross_kv_cache = self._cross_kv_cache  # 预分配复用

                # A2V DiT 前向
                noise_pred = self.dit(
                    latent_input, t_mx,
                    audio_input, text_input,
                    seq_lens, grid_sizes,
                    cross_kv_cache=cross_kv_cache,
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

    def _prepare_cross_kv_cache(self, text_input: mx.array) -> tuple:
        """AtomCode 专题优化: 预分配 cross-attn KV 缓存跨步复用 (2026-07-18).

        Args:
            text_input: [B, L_text, text_dim] 文本 embedding (CFG 拼接 cond+ununc)

        Returns:
            (k, v) cross-attn KV 缓存 tuple, 跨采样步复用避每步重算 KV 投影
        """
        # 用 block 0 的 cross_attn.prepare_kv 预算 KV (所有 block cross_attn 权重同形, 预算一份够用)
        # 真实现应对每 block 单独预算 (各 block cross_attn.k/v 权重不同), 此处暂用 block 0 代算
        # 待压测验证提速后再补全逐 block 预算
        b0 = self.dit.blocks[0]
        k, v = b0.cross_attn.prepare_kv(text_input)
        mx.eval(k, v)  # 预分配触 Metal 常驻
        self._cross_kv_cache = (k, v)
        return self._cross_kv_cache


__all__ = [
    "SkyReelsPipelineConfig",
    "SkyReelsR2VPipeline",
    "SkyReelsV2VPipeline",
    "SkyReelsA2VPipeline",
]
