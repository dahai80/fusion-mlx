# SPDX-License-Identifier: Apache-2.0
# MiniMax H3 配置层。所有超参逐字段对齐开源包 config.json / metadata.json，
# 不做任何臆测改值。partition 字段区分两个 checkpoint：
#   FL2VA   主生成模型（t2va/i2va/l2va/fl2va）
#   REF2VA  多参考素材风格锁定（ref2va）
import logging
from dataclasses import dataclass, field
from enum import Enum

from ..ltx2.config import BaseModelConfig

logger = logging.getLogger(__name__)


class H3Partition(Enum):
    FL2VA = "fl2va"
    REF2VA = "ref2va"


@dataclass
class H3VAEConfig(BaseModelConfig):
    # VisualVAE (AutoencoderKLLegacy) — 源自 FL2VA/video_vae/source/config.json
    ch: int = 128
    ch_mult: tuple[int, ...] = (1, 2, 2, 4, 4, 8)
    embed_dim: int = 24
    z_channels: int = 24
    in_channels: int = 3
    num_res_blocks: int = 2
    space_down: tuple[int, ...] = (2, 2, 2, 2, 1, 1)
    time_down: tuple[int, ...] = (1, 2, 2, 1, 1, 1)
    use_3d_conv: bool = True
    use_t_isolated_gn: bool = True
    use_vit_decoder: bool = True
    causal_encoder: bool = True
    causal_decoder: bool = False
    vae_ratio: int = 16
    vae_ratio_t: int = 4
    # ViT3D 解码器 — vit_decoder_kwargs
    vit_num_layers: int = 36
    vit_heads: int = 32
    vit_dim_head: int = 64
    vit_rope_dim_ratio: float = 0.75
    vit_rope_theta: float = 100.0
    vit_norm_type: str = "rms_norm"
    vit_ffn_use_gated: bool = True
    # 官方 source/config.json: qk_norm_type="rms_norm", qk_norm_affine=false
    vit_qk_norm_type: str = "rms_norm"
    vit_qk_norm_affine: bool = False
    # 分块推理默认（源自 minimax_h3_video_vae.py 包装层）
    vae_clip_length: int = 17
    vae_token_drop: int = 3
    vae_tile_size: int = 256
    vae_tile_overlap_min: int = 64


@dataclass
class H3AudioVAEConfig(BaseModelConfig):
    # AudioVAE (DAC + BigVGAN) — 源自 FL2VA/audio_vae/metadata.json
    encoder_dim: int = 64
    decoder_dim: int = 1024
    latent_dim: int = 2048
    vae_latent_channels: int = 32
    sample_rate: int = 32000
    encoder_rates: tuple[int, ...] = (2, 4, 4, 5, 5)
    decoder_rates: tuple[int, ...] = (5, 5, 2, 2, 2, 2, 2)
    decoder_type: str = "bigvgan"
    attn_proj: bool = True
    attn_heads: int = 8
    # BigVGAN 解码器核（源自 dac_audio_vae.py 32000Hz 配置）
    upsample_kernel_sizes: tuple[int, ...] = (9, 9, 4, 4, 4, 4, 4)
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)


@dataclass
class H3Config(BaseModelConfig):
    # ---- Transformer (MiniMaxH3DiTModel) — 源自 FL2VA/transformer/config.json ----
    dim: int = 5376  # hidden_size
    num_layers: int = 50
    token_refiner_layers: int = 2  # token_refiner_num_layers
    num_heads: int = 56
    head_dim: int = 128  # attention_head_dim
    ffn_dim: int = 14336  # ffn_hidden_size
    latents_dim: int = 24  # 视频 latent 通道
    audio_latents_dim: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden: int = 5376  # time_embed_hidden_size
    time_embed_dim: int = 2688
    adaln_out: int = 96768  # adaln_out_features
    final_adaln_out: int = 10752  # final_adaln_out_features
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    # ---- 模型划分 ----
    partition: H3Partition = H3Partition.FL2VA
    tasks: tuple[str, ...] = ("t2va", "i2va", "l2va", "fl2va")

    # ---- 调度器（双 shift：视频 12.0 / 音频 3.0）----
    video_shift: float = 12.0
    audio_shift: float = 3.0
    sample_steps: int = 40
    num_train_timesteps: int = 1000
    guide_scale: float = 5.0

    # ---- VAE ----
    vae: H3VAEConfig = field(default_factory=H3VAEConfig)
    audio_vae: H3AudioVAEConfig = field(default_factory=H3AudioVAEConfig)

    # ---- 文本/图像编码器 ----
    # Qwen3-VL-32B，mlx-vlm 0.5.0 已原生支持 qwen3_vl，复用其 50 层隐状态。
    encoder_model: str = "minimax/qwen3-vl-32b"
    encoder_hidden_layer: int = 50

    # ---- 输出 ----
    sample_fps: int = 24
    audio_sample_rate: int = 32000
    audio_channels: int = 2  # 立体声

    @property
    def head_dim_real(self) -> int:
        return self.dim // self.num_heads

    @classmethod
    def fl2va(cls) -> "H3Config":
        logger.info("H3Config: 构建 FL2VA partition（t2va/i2va/l2va/fl2va）")
        return cls(
            partition=H3Partition.FL2VA,
            tasks=("t2va", "i2va", "l2va", "fl2va"),
            video_shift=12.0,
            audio_shift=3.0,
        )

    @classmethod
    def ref2va(cls) -> "H3Config":
        logger.info("H3Config: 构建 REF2VA partition（ref2va）")
        return cls(
            partition=H3Partition.REF2VA,
            tasks=("ref2va",),
            video_shift=12.0,
            audio_shift=3.0,
        )
