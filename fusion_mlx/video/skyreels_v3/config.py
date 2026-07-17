# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 三大分支配置注册.

对齐底座 model-config.json 格式,
在 fusion_mlx.video 注册 SkyReels-V3 三大分支.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkyReelsBranchConfig:
    """单分支配置 (r2v_14b / v2v_14b / a2v_19b)."""
    model_type: str
    branch: str  # r2v / v2v / a2v
    dim: int
    ffn_dim: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    patch_size: tuple
    in_dim: int
    out_dim: int
    text_dim: int
    text_len: int
    freq_dim: int
    window_size: tuple
    qk_norm: bool
    cross_attn_norm: bool
    eps: float
    cross_attn_type: str
    temporal_window: int  # -1 表示 R2V 不用时序
    has_audio: bool
    audio_dim: int  # 0 表示无音频分支
    hf_model_id: str  # HuggingFace 权重 ID


# ---------------------------------------------------------------------------
# 三大分支配置表
# ---------------------------------------------------------------------------
BRANCH_CONFIGS: dict[str, SkyReelsBranchConfig] = {
    "skyreels-v3-r2v-14b": SkyReelsBranchConfig(
        model_type="r2v_14b",
        branch="r2v",
        dim=5120, ffn_dim=13824, num_heads=40, num_kv_heads=40, num_layers=40,
        patch_size=(1, 2, 2), in_dim=16, out_dim=16,
        text_dim=4096, text_len=512, freq_dim=256,
        window_size=(-1, -1), qk_norm=True, cross_attn_norm=True, eps=1e-6,
        cross_attn_type="i2v_cross_attn",
        temporal_window=-1,  # R2V 不用时序分支
        has_audio=False, audio_dim=0,
        hf_model_id="Skywork/SkyReels-V3-R2V-14B",
    ),
    "skyreels-v3-v2v-14b": SkyReelsBranchConfig(
        model_type="v2v_14b",
        branch="v2v",
        dim=5120, ffn_dim=13824, num_heads=40, num_kv_heads=40, num_layers=40,
        patch_size=(1, 2, 2), in_dim=16, out_dim=16,
        text_dim=4096, text_len=512, freq_dim=256,
        window_size=(-1, -1), qk_norm=True, cross_attn_norm=True, eps=1e-6,
        cross_attn_type="i2v_cross_attn",
        temporal_window=96,  # V2V 保连贯
        has_audio=False, audio_dim=0,
        hf_model_id="Skywork/SkyReels-V3-V2V-14B",
    ),
    "skyreels-v3-a2v-19b": SkyReelsBranchConfig(
        model_type="a2v_19b",
        branch="a2v",
        dim=6144, ffn_dim=24576, num_heads=48, num_kv_heads=48, num_layers=60,
        patch_size=(1, 2, 2), in_dim=16, out_dim=16,
        text_dim=4096, text_len=512, freq_dim=256,
        window_size=(-1, -1), qk_norm=True, cross_attn_norm=True, eps=1e-6,
        cross_attn_type="i2v_cross_attn",
        temporal_window=32,  # A2V 保嘴型连贯
        has_audio=True, audio_dim=1024,  # wav2vec2
        hf_model_id="Skywork/SkyReels-V3-A2V-19B",
    ),
}


def get_branch_config(model_key: str) -> SkyReelsBranchConfig:
    """根据 model_key 获取分支配置.

    Args:
        model_key: e.g. "skyreels-v3-r2v-14b"

    Returns:
        SkyReelsBranchConfig
    """
    if model_key not in BRANCH_CONFIGS:
        raise ValueError(
            f"Unknown model_key: {model_key}. "
            f"Valid: {list(BRANCH_CONFIGS)}"
        )
    return BRANCH_CONFIGS[model_key]


def list_models() -> list[str]:
    """列出所有可用模型 key."""
    return list(BRANCH_CONFIGS)


__all__ = [
    "SkyReelsBranchConfig",
    "BRANCH_CONFIGS",
    "get_branch_config",
    "list_models",
]
