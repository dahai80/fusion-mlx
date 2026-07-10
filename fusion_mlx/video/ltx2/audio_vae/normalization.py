# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2 audio VAE normalization (vendored from mlx-video).
# Phase 4 Stage E: audio_vae port.
from enum import Enum

import mlx.core as mx
import mlx.nn as nn


class NormType(Enum):
    GROUP = "group"
    PIXEL = "pixel"


class PixelNorm(nn.Module):
    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        mean_sq = mx.mean(x**2, axis=self.dim, keepdims=True)
        rms = mx.sqrt(mean_sq + self.eps)
        return x / rms


def build_normalization_layer(
    in_channels: int, *, num_groups: int = 32, normtype: NormType = NormType.GROUP
) -> nn.Module:
    if normtype == NormType.GROUP:
        return nn.GroupNorm(
            num_groups=num_groups, dims=in_channels, eps=1e-6, affine=True
        )
    if normtype == NormType.PIXEL:
        return PixelNorm(dim=-1, eps=1e-6)
    raise ValueError(f"Invalid normalization type: {normtype}")
