# SPDX-License-Identifier: Apache-2.0
# Vendored from mflux 0.20.0 (https://github.com/filipstrand/mflux) PR#736
# MIT-licensed pure-MLX Qwen-Image-2.1 reference (txt2img + 4-ch RGBA VAE).
# Pinned to fusion-mlx's mlx stack. Upstream-tracked; do not modify forward
# numerics without parity-testing against mflux qwen21.

import mlx.core as mx
from mlx import nn

from .qwen21_rms_norm import Qwen21RMSNorm


class Qwen21AttentionBlock(nn.Module):
    # Single-head self-attention over the flattened spatial axis.

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = Qwen21RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        batch, channels, height, width = x.shape

        x = self.norm(x)
        x = mx.transpose(x, (0, 2, 3, 1))
        qkv = self.to_qkv(x)
        qkv = mx.reshape(qkv, (batch, height * width, 3, channels))
        q, k, v = qkv[:, :, 0, :], qkv[:, :, 1, :], qkv[:, :, 2, :]

        scale = 1.0 / mx.sqrt(mx.array(float(channels)))
        scores = mx.matmul(q, mx.transpose(k, (0, 2, 1))) * scale
        hidden = mx.matmul(mx.softmax(scores, axis=-1), v)

        hidden = mx.reshape(hidden, (batch, height, width, channels))
        hidden = self.proj(hidden)
        hidden = mx.transpose(hidden, (0, 3, 1, 2))
        return hidden + identity
