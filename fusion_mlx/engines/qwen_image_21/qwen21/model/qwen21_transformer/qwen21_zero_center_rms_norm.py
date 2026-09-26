# SPDX-License-Identifier: Apache-2.0
# Vendored from mflux 0.20.0 (https://github.com/filipstrand/mflux) PR#736
# MIT-licensed pure-MLX Qwen-Image-2.1 reference (txt2img + 4-ch RGBA VAE).
# Pinned to fusion-mlx's mlx stack. Upstream-tracked; do not modify forward
# numerics without parity-testing against mflux qwen21.

import mlx.core as mx
from mlx import nn


class Qwen21ZeroCenterRMSNorm(nn.Module):
    # RMSNorm whose checkpoint weight is stored zero-centered: the effective scale is weight + 1.

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        input_dtype = x.dtype
        x_float = x.astype(mx.float32)
        rrms = mx.rsqrt(mx.mean(x_float * x_float, axis=-1, keepdims=True) + self.eps)
        return (x_float * rrms * (self.weight.astype(mx.float32) + 1)).astype(input_dtype)
