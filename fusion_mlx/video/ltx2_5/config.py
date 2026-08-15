# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2.5 model config (self-contained, no ..ltx2 import).
# LTX-2.5 22B: 48-layer AV DiT, Gemma4-12b text encoder, duration-head,
# two-stage spatial+temporal upsampler, embeddings connectors, keyframes abs
# pos embedding. Deltas over LTX-2 documented inline (code-verified against the
# real 22b-distilled checkpoint key tree, 4349 keys).
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class LTXModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class LTXRopeType(Enum):
    INTERLEAVED = "interleaved"
    SPLIT = "split"
    TWO_D = "2d"


class AttentionType(Enum):
    DEFAULT = "default"


class LTX2_5Variant(Enum):
    DISTILLED = "distilled"
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
class BaseModelConfig:
    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> BaseModelConfig:
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for k, v in self.__dict__.items():
            if v is not None:
                if isinstance(v, Enum):
                    result[k] = v.value
                elif hasattr(v, "to_dict"):
                    result[k] = v.to_dict()
                else:
                    result[k] = v
        return result


@dataclass
class TransformerConfig(BaseModelConfig):
    dim: int
    heads: int
    d_head: int
    context_dim: int


@dataclass
class LTX2_5ModelConfig(BaseModelConfig):
    # ---- AV DiT skeleton (identical to LTX-2, 48 layers) ----
    model_type: LTXModelType = LTXModelType.AudioVideo
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    in_channels: int = 128
    out_channels: int = 128
    num_layers: int = 48
    cross_attention_dim: int = 4096
    caption_channels: int = 3840
    audio_num_attention_heads: int = 32
    audio_attention_head_dim: int = 64
    audio_in_channels: int = 128
    audio_out_channels: int = 128
    audio_cross_attention_dim: int = 2048
    audio_caption_channels: int = 3840
    positional_embedding_theta: float = 10000.0
    positional_embedding_max_pos: list[int] | None = None
    audio_positional_embedding_max_pos: list[int] | None = None
    use_middle_indices_grid: bool = True
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED
    double_precision_rope: bool = False
    timestep_scale_multiplier: int = 1000
    av_ca_timestep_scale_multiplier: int = 1000
    norm_eps: float = 1e-6
    attention_type: AttentionType = AttentionType.DEFAULT

    # ---- LTX-2.5 delta 1: prompt-adaln ON ----
    # 真实 checkpoint 含 prompt_adaln_single(coeff=2) + 全部 attn 的 to_gate_logits，
    # 故 has_prompt_adaln 必须为 True（ltx2 默认 False）。
    has_prompt_adaln: bool = True

    # ---- LTX-2.5 delta 2: FF bias 不对称 ----
    # 真实 checkpoint：video FF 无 bias，audio FF 有 bias（Gemma4 ff_bias=false 仅作用于 video）。
    ff_bias: bool = False
    audio_ff_bias: bool = True

    # ---- LTX-2.5 delta 3: keyframes absolute pos embedding ----
    use_keyframes_abs_pos_embedding: bool = True

    # ---- LTX-2.5 delta 4: embeddings connectors ----
    # video connector: 4096 = 32 heads × 128 d_head；audio connector: 2048 = 32 × 64。
    connector_num_layers: int = 8
    connector_num_learnable_registers: int = 128
    connector_apply_gated_attention: bool = True
    connector_positional_embedding_max_pos: list[int] = field(
        default_factory=lambda: [1]
    )

    # ---- LTX-2.5 生成侧常量 ----
    has_duration_head: bool = True
    has_two_stage_upsampler: bool = True
    vae_scale_factors: tuple[int, int, int] = (8, 32, 32)
    use_prompt_embeddings: bool = True

    def __post_init__(self):
        if self.positional_embedding_max_pos is None:
            self.positional_embedding_max_pos = [20, 2048, 2048]
        if self.audio_positional_embedding_max_pos is None:
            self.audio_positional_embedding_max_pos = [20]
        if not self.has_prompt_adaln:
            self.double_precision_rope = False
        if isinstance(self.model_type, str):
            self.model_type = LTXModelType(self.model_type)
        if isinstance(self.rope_type, str):
            self.rope_type = LTXRopeType(self.rope_type)
        if isinstance(self.attention_type, str):
            self.attention_type = AttentionType(self.attention_type)
        # LTX-2.5 固定为 AudioVideo。
        self.model_type = LTXModelType.AudioVideo
        logger.debug(
            "LTX2_5ModelConfig: layers=%d caption=%d inner=%d rope=%s "
            "ff_bias=%s audio_ff_bias=%s has_prompt_adaln=%s",
            self.num_layers,
            self.caption_channels,
            self.inner_dim,
            self.rope_type.value,
            self.ff_bias,
            self.audio_ff_bias,
            self.has_prompt_adaln,
        )

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def audio_inner_dim(self) -> int:
        return self.audio_num_attention_heads * self.audio_attention_head_dim

    @property
    def connector_inner_dim(self) -> int:
        return self.inner_dim

    @property
    def audio_connector_inner_dim(self) -> int:
        return self.audio_inner_dim

    def get_video_config(self) -> TransformerConfig | None:
        if not self.model_type.is_video_enabled():
            return None
        return TransformerConfig(
            dim=self.inner_dim,
            heads=self.num_attention_heads,
            d_head=self.attention_head_dim,
            context_dim=self.cross_attention_dim,
        )

    def get_audio_config(self) -> TransformerConfig | None:
        if not self.model_type.is_audio_enabled():
            return None
        return TransformerConfig(
            dim=self.audio_inner_dim,
            heads=self.audio_num_attention_heads,
            d_head=self.audio_attention_head_dim,
            context_dim=self.audio_cross_attention_dim,
        )

    @property
    def vae_temporal_factor(self) -> int:
        return self.vae_scale_factors[0]

    @property
    def vae_spatial_factor(self) -> int:
        return self.vae_scale_factors[1]


def default_ltx2_5_config(
    variant: LTX2_5Variant | str = LTX2_5Variant.DISTILLED,
) -> LTX2_5ModelConfig:
    variant = LTX2_5Variant.from_str(variant)
    cfg = LTX2_5ModelConfig()
    logger.info(
        "default_ltx2_5_config: variant=%s layers=%d caption=%d inner=%d",
        variant.value,
        cfg.num_layers,
        cfg.caption_channels,
        cfg.inner_dim,
    )
    return cfg
