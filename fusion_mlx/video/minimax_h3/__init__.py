# SPDX-License-Identifier: Apache-2.0
# MiniMax H3 纯 MLX 推理移植：Omni-Transformer + VisualVAE + AudioVAE + Qwen3-VL 编码器。
# 参照源码：/Users/dahai/minimax/MiniMax-H3（用户已下载开源包）。
# P6：condition 组装 + t2va video-only 去噪（UNVERIFIED，需真实模型校正）。
from .condition import build_t2va_packed
from .config import H3AudioVAEConfig, H3Config, H3Partition, H3VAEConfig
from .generate import generate_t2va_av, generate_t2va_video, generate_video
from .scheduler import MiniMaxH3Scheduler
from .text_encoder import H3_TEXT_ENCODER_LAYER, MiniMaxH3TextEncoder, load_text_encoder
from .transformer import MiniMaxH3DiTModel, load_dit_from_pretrained
from .vae import MiniMaxH3VideoVAE

__all__ = [
    "H3Config",
    "H3VAEConfig",
    "H3AudioVAEConfig",
    "H3Partition",
    "MiniMaxH3VideoVAE",
    "MiniMaxH3DiTModel",
    "load_dit_from_pretrained",
    "MiniMaxH3Scheduler",
    "MiniMaxH3TextEncoder",
    "H3_TEXT_ENCODER_LAYER",
    "load_text_encoder",
    "build_t2va_packed",
    "generate_t2va_video",
    "generate_t2va_av",
    "generate_video",
]
