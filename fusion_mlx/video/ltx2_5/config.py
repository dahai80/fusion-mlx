# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2.5 model config.
# Reuses the LTX-2/2.3 config skeleton (fusion_mlx.video.ltx2.config);
# LTX-2.5 differs in: num_layers=48, caption_channels=3840, Gemma4-12b text
# encoder, duration-head, two-stage spatial+temporal upsampler. The base
# LTXModelConfig already carries 48/3840, so this is a thin typed wrapper that
# pins the 2.5 constants explicitly and adds the distilled/dev variant enum.
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from ..ltx2.config import (
    LTXModelConfig,
    LTXModelType,
    LTXRopeType,
    TransformerConfig,
)

logger = logging.getLogger(__name__)


class LTX2_5Variant(Enum):
    # 少步蒸馏变体（首期 P0-P8），权重 ltx-2.5-22b-distilled-*-bf16。
    DISTILLED = "distilled"
    # 高质量多步 CFG 变体（P9 后续），权重 ltx-2.5-22b-dev-*-bf16。
    DEV = "dev"

    @classmethod
    def from_str(cls, value: str | LTX2_5Variant) -> LTX2_5Variant:
        if isinstance(value, cls):
            return value
        v = str(value).strip().lower()
        for member in cls:
            if member.value == v:
                return member
        raise ValueError(f"unknown LTX-2.5 variant: {value!r}")


@dataclass
class LTX2_5ModelConfig(LTXModelConfig):
    # LTX-2.5 显式常量（与 LTXModelConfig 默认值一致，此处置写以锁死 2.5 语义，
    # 防止上游 ltx2 config 默认值漂移影响 2.5 移植）。
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    in_channels: int = 128
    out_channels: int = 128
    num_layers: int = 48
    cross_attention_dim: int = 4096
    caption_channels: int = 3840
    # VAE scale factors (temporal, height, width) for LTX-2.5。
    vae_scale_factors: tuple[int, int, int] = (8, 32, 32)
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED
    use_prompt_embeddings: bool = True
    # 2.5 新增：是否启用 duration-head（省略 num_frames 时由 prompt 推断时长）。
    has_duration_head: bool = True
    # 2.5 新增：是否启用两阶段 upsampler（spatial + temporal x2）。
    has_two_stage_upsampler: bool = True

    def __post_init__(self):
        super().__post_init__()
        # LTX-2.5 固定为 AudioVideo（音视频联合）。
        self.model_type = LTXModelType.AudioVideo
        if isinstance(self.rope_type, str):
            self.rope_type = LTXRopeType(self.rope_type)
        logger.debug(
            "LTX2_5ModelConfig: layers=%d caption=%d inner=%d rope=%s",
            self.num_layers,
            self.caption_channels,
            self.inner_dim,
            self.rope_type.value,
        )

    def get_video_config(self) -> TransformerConfig | None:
        return super().get_video_config()

    @property
    def vae_temporal_factor(self) -> int:
        return self.vae_scale_factors[0]

    @property
    def vae_spatial_factor(self) -> int:
        return self.vae_scale_factors[1]


def default_ltx2_5_config(variant: LTX2_5Variant | str = LTX2_5Variant.DISTILLED) -> LTX2_5ModelConfig:
    variant = LTX2_5Variant.from_str(variant)
    cfg = LTX2_5ModelConfig()
    logger.info(
        "default_ltx2_5_config: variant=%s layers=%d caption=%d",
        variant.value,
        cfg.num_layers,
        cfg.caption_channels,
    )
    return cfg
