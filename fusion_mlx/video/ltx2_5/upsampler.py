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
from ..ltx2.upsampler import (
    load_upsampler as load_spatial_upsampler,
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


def load_temporal_upsampler(
    weights_path: str | Path,
) -> tuple[LatentTemporalUpsampler, float]:
    # 加载时间上采样器：检测 mid_channels + temporal_scale。
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"temporal upsampler weights not found: {weights_path}")

    logger.info("Loading temporal upsampler from %s", weights_path)
    raw_weights = mx.load(str(weights_path))

    sample_key = "res_blocks.0.conv1.weight"
    if sample_key in raw_weights:
        mid_channels = raw_weights[sample_key].shape[0]
    else:
        mid_channels = 1024

    # 实测 checkpoint 用 'upsampler.0.weight'（Conv3d 单层），模型树是
    # upsampler.conv（TemporalUpsampler2x.conv）。检测 + 重映射（mirror ltx2
    # spatial load_upsampler 的 upsampler.0.->upsampler.conv. 重映射）。
    conv_key = "upsampler.conv.weight"
    if conv_key not in raw_weights and "upsampler.0.weight" in raw_weights:
        conv_key = "upsampler.0.weight"
    if conv_key in raw_weights:
        out_channels = raw_weights[conv_key].shape[0]
        temporal_scale = float(out_channels // mid_channels)
    else:
        temporal_scale = 2.0

    logger.info(
        "Detected temporal upsampler: mid_channels=%d scale=%sx",
        mid_channels,
        temporal_scale,
    )

    upsampler = LatentTemporalUpsampler(
        in_channels=128,
        mid_channels=mid_channels,
        num_blocks_per_stage=4,
        temporal_scale=temporal_scale,
    )

    sanitized = {}
    for key, value in raw_weights.items():
        new_key = key
        if new_key.startswith("upsampler.0."):
            new_key = new_key.replace("upsampler.0.", "upsampler.conv.")
        if "weight" in new_key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))
        if ("weight" in new_key or "kernel" in new_key) and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))
        sanitized[new_key] = value

    upsampler.load_weights(list(sanitized.items()), strict=False)
    logger.info("Loaded %d temporal upsampler weights", len(sanitized))

    return upsampler, temporal_scale


# 复用 ltx2 的 spatial 加载器，便于 backend 单点引用。
def load_spatial_upsampler_2_5(
    weights_path: str | Path,
) -> tuple[LatentUpsampler, float]:
    return load_spatial_upsampler(str(weights_path))
