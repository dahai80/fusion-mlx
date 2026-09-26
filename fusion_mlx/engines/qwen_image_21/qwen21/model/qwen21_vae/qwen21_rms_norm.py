# SPDX-License-Identifier: Apache-2.0
# Vendored from mflux 0.20.0 (https://github.com/filipstrand/mflux) PR#736
# MIT-licensed pure-MLX Qwen-Image-2.1 reference (txt2img + 4-ch RGBA VAE).
# Pinned to fusion-mlx's mlx stack. Upstream-tracked; do not modify forward
# numerics without parity-testing against mflux qwen21.

import mlx.core as mx
from mlx import nn


class Qwen21RMSNorm(nn.Module):
    # Wan-style norm: L2-normalize over the channel axis, then scale by sqrt(C) * gamma.
    # Operates on (B, C, H, W); the checkpoint stores gamma with shape (C, 1, 1[, 1]).

    def __init__(self, num_channels: int):
        super().__init__()
        self.weight = mx.ones((num_channels,))
        self.scale = float(num_channels) ** 0.5

    def __call__(self, x: mx.array) -> mx.array:
        x_float = x.astype(mx.float32)
        l2_norm = mx.sqrt(mx.sum(x_float * x_float, axis=1, keepdims=True))
        x_normalized = x_float / mx.maximum(l2_norm, 1e-12)
        return (x_normalized * self.scale * self.weight[None, :, None, None]).astype(x.dtype)
