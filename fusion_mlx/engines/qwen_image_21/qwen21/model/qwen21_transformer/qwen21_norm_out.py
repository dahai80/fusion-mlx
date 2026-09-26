# SPDX-License-Identifier: Apache-2.0
# Vendored from mflux 0.20.0 (https://github.com/filipstrand/mflux) PR#736
# MIT-licensed pure-MLX Qwen-Image-2.1 reference (txt2img + 4-ch RGBA VAE).
# Pinned to fusion-mlx's mlx stack. Upstream-tracked; do not modify forward
# numerics without parity-testing against mflux qwen21.

import mlx.core as mx
from mlx import nn


class Qwen21AdaLayerNormContinuous(nn.Module):
    # Final adaptive norm, scale-only: no shift, so the linear maps to embedding_dim.

    def __init__(self, embedding_dim: int = 4096, eps: float = 1e-6):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, affine=False)

    def __call__(self, hidden_states: mx.array, scale: mx.array) -> mx.array:
        return self.norm(hidden_states) * (1 + scale)
