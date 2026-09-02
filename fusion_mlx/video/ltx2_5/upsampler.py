# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2.5 latent upsamplers.
# Reuses the LTX-2 spatial LatentUpsampler (fusion_mlx.video.ltx2.upsampler)
# for the stage1->stage2 spatial x2 step. NEW for 2.5: a temporal x2 upsampler
# that doubles the latent frame count (d dim) via 3D conv + temporal
# pixel-shuffle. Checkpoints:
#   latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
#   latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors
from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ..ltx2.upsampler import (
    Conv3d,
    GroupNorm3d,
    LatentUpsampler,
    ResBlock3D,
)

logger = logging.getLogger(__name__)


class TemporalPixelShuffle(nn.Module):
    # 时间轴 pixel-shuffle：channels -> d*r，reshape 到时间维翻倍。
    def __init__(self, upscale_factor: int = 2):
        super().__init__()
        self.r = upscale_factor

    def __call__(self, x: mx.array) -> mx.array:
        # x: (n, d, h, w, c)
        n, d, h, w, c = x.shape
        r = self.r
        out_c = c // r
        x = mx.reshape(x, (n, d, h, w, out_c, r))
        x = mx.transpose(x, (0, 5, 1, 2, 3, 4))
        x = mx.reshape(x, (n, d * r, h, w, out_c))
        return x


class TemporalUpsampler2x(nn.Module):
    # 时间维 x2：Conv3d(3^3, pad=1) 扩通道 -> TemporalPixelShuffle 时间翻倍。
    # checkpoint upsampler.0.weight 为 (Cout, Cin, 3, 3, 3) 各向同性卷积，非
    # (3,1,1) 时间轴卷积；padding=1 保住 H/W（否则 spatial 缩 2）。
    def __init__(self, mid_channels: int = 1024, upscale_factor: int = 2):
        super().__init__()
        self.r = upscale_factor
        self.conv = Conv3d(
            mid_channels,
            mid_channels * upscale_factor,
            kernel_size=3,
            padding=1,
        )
        self.pixel_shuffle = TemporalPixelShuffle(upscale_factor)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (n, d, h, w, c)
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class LatentTemporalUpsampler(nn.Module):
    # 两阶段桥接的时间上采样：spatial 已由 LatentUpsampler 处理，此处只做
    # temporal x2。结构对齐 spatial：initial conv/norm -> res blocks ->
    # temporal upsample -> post res blocks -> final conv。
    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 1024,
        num_blocks_per_stage: int = 4,
        temporal_scale: float = 2.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.temporal_scale = temporal_scale

        self.initial_conv = Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = GroupNorm3d(32, mid_channels)

        self.res_blocks = {
            i: ResBlock3D(mid_channels) for i in range(num_blocks_per_stage)
        }

        r = int(temporal_scale)
        self.upsampler = TemporalUpsampler2x(
            mid_channels=mid_channels, upscale_factor=r
        )

        self.post_upsample_res_blocks = {
            i: ResBlock3D(mid_channels) for i in range(num_blocks_per_stage)
        }

        self.final_conv = Conv3d(mid_channels, in_channels, kernel_size=3, padding=1)

    def __call__(self, latent: mx.array) -> mx.array:
        # latent: (n, c, d, h, w) channels-first（与 LatentUpsampler 一致）。
        x = mx.transpose(latent, (0, 2, 3, 4, 1))

        x = self.initial_conv(x)
        x = self.initial_norm(x)
        x = nn.silu(x)

        for i in sorted(self.res_blocks.keys()):
            x = self.res_blocks[i](x)

        x = self.upsampler(x)

        for i in sorted(self.post_upsample_res_blocks.keys()):
            x = self.post_upsample_res_blocks[i](x)

        x = self.final_conv(x)

        x = mx.transpose(x, (0, 4, 1, 2, 3))
        return x


# #762 flat diffusers (dgrauet) 根级文件名前缀；剥去后与 canon 模型树同名。
_FLAT_UPSAMPLER_PREFIXES = (
    "spatial_upscaler_x2_v1_0.",
    "temporal_upscaler_x2_v1_0.",
)


def _strip_flat_upsampler_prefix(key: str) -> str:
    for p in _FLAT_UPSAMPLER_PREFIXES:
        if key.startswith(p):
            return key[len(p) :]
    return key


def _is_mlx_conv_layout(value: mx.array) -> bool:
    # dgrauet 权重已是 MLX 卷积布局 (Cout, kd, kh, kw, Cin) / (Cout, kh, kw, Cin)：
    # shape[1]==kernel_size(3)。canon Comfy 存 PyTorch 布局 (Cout, Cin, kd,kh,kw)：
    # shape[1]==Cin(大)。MLX 布局则跳过转置，否则 PyTorch->MLX 转置。
    return value.ndim in (4, 5) and value.shape[1] == 3


def _sanitize_upsampler_weights(
    raw_weights: dict[str, mx.array],
) -> dict[str, mx.array]:
    sanitized: dict[str, mx.array] = {}
    for key, value in raw_weights.items():
        if key == "__metadata__":
            continue
        new_key = _strip_flat_upsampler_prefix(key)
        if new_key.startswith("upsampler.0."):
            new_key = new_key.replace("upsampler.0.", "upsampler.conv.")
        if "weight" in new_key and value.ndim == 5 and not _is_mlx_conv_layout(value):
            value = mx.transpose(value, (0, 2, 3, 4, 1))
        elif (
            ("weight" in new_key or "kernel" in new_key)
            and value.ndim == 4
            and not _is_mlx_conv_layout(value)
        ):
            value = mx.transpose(value, (0, 2, 3, 1))
        sanitized[new_key] = value
    return sanitized


def load_temporal_upsampler(
    weights_path: str | Path,
) -> tuple[LatentTemporalUpsampler, float]:
    # 加载时间上采样器：检测 mid_channels + temporal_scale。
    # 两类布局: canon Comfy (PyTorch 卷积布局, 无前缀) + flat diffusers (#762,
    # MLX 卷积布局, 根级文件名前缀)。统一 sanitize 处理前缀+转置。
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"temporal upsampler weights not found: {weights_path}")

    logger.info("Loading temporal upsampler from %s", weights_path)
    raw_weights = mx.load(str(weights_path))
    flat = any(
        any(k.startswith(p) for k in raw_weights) for p in _FLAT_UPSAMPLER_PREFIXES
    )

    sample_key = (
        "temporal_upscaler_x2_v1_0.res_blocks.0.conv1.weight"
        if flat
        else "res_blocks.0.conv1.weight"
    )
    mid_channels = (
        raw_weights[sample_key].shape[0] if sample_key in raw_weights else 512
    )

    conv_key = "upsampler.conv.weight"
    if conv_key not in raw_weights:
        flat_conv = (
            "temporal_upscaler_x2_v1_0.upsampler.0.weight"
            if flat
            else "upsampler.0.weight"
        )
        if flat_conv in raw_weights:
            conv_key = flat_conv
    if conv_key in raw_weights:
        out_channels = raw_weights[conv_key].shape[0]
        temporal_scale = float(out_channels // mid_channels)
    else:
        temporal_scale = 2.0

    logger.info(
        "Detected temporal upsampler: mid_channels=%d scale=%sx (flat=%s)",
        mid_channels,
        temporal_scale,
        flat,
    )

    upsampler = LatentTemporalUpsampler(
        in_channels=128,
        mid_channels=mid_channels,
        num_blocks_per_stage=4,
        temporal_scale=temporal_scale,
    )

    sanitized = _sanitize_upsampler_weights(raw_weights)
    upsampler.load_weights(list(sanitized.items()), strict=False)
    logger.info("Loaded %d temporal upsampler weights", len(sanitized))

    return upsampler, temporal_scale


def load_spatial_upsampler_2_5(
    weights_path: str | Path,
) -> tuple[LatentUpsampler, float]:
    # #762: 自包含 spatial 加载器 (不再透传 ltx2.load_upsampler)。flat diffusers
    # 权重带根级前缀 + 已是 MLX 卷积布局, ltx2 透传会漏剥前缀 + 双重转置。
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"spatial upsampler weights not found: {weights_path}")

    logger.info("Loading spatial upsampler from %s", weights_path)
    raw_weights = mx.load(str(weights_path))
    flat = any(
        any(k.startswith(p) for k in raw_weights) for p in _FLAT_UPSAMPLER_PREFIXES
    )

    sample_key = (
        "spatial_upscaler_x2_v1_0.res_blocks.0.conv1.weight"
        if flat
        else "res_blocks.0.conv1.weight"
    )
    mid_channels = (
        raw_weights[sample_key].shape[0] if sample_key in raw_weights else 1024
    )

    conv_key = "upsampler.conv.weight"
    if conv_key not in raw_weights:
        flat_conv = (
            "spatial_upscaler_x2_v1_0.upsampler.0.weight"
            if flat
            else "upsampler.0.weight"
        )
        if flat_conv in raw_weights:
            conv_key = flat_conv
    if conv_key in raw_weights:
        out_channels = raw_weights[conv_key].shape[0]
        ratio = out_channels // mid_channels
        rational_resampler = ratio == 9
        spatial_scale = 1.5 if rational_resampler else 2.0
    else:
        rational_resampler = False
        spatial_scale = 2.0

    logger.info(
        "Detected spatial upsampler: mid_channels=%d scale=%sx rational=%s (flat=%s)",
        mid_channels,
        spatial_scale,
        rational_resampler,
        flat,
    )

    upsampler = LatentUpsampler(
        in_channels=128,
        mid_channels=mid_channels,
        num_blocks_per_stage=4,
        spatial_scale=spatial_scale,
        rational_resampler=rational_resampler,
    )

    sanitized = _sanitize_upsampler_weights(raw_weights)
    upsampler.load_weights(list(sanitized.items()), strict=False)
    logger.info("Loaded %d spatial upsampler weights", len(sanitized))

    return upsampler, spatial_scale
