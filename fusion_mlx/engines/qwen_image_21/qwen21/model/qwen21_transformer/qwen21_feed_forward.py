# SPDX-License-Identifier: Apache-2.0
# Vendored from mflux 0.20.0 (https://github.com/filipstrand/mflux) PR#736
# MIT-licensed pure-MLX Qwen-Image-2.1 reference (txt2img + 4-ch RGBA VAE).
# Pinned to fusion-mlx's mlx stack. Upstream-tracked; do not modify forward
# numerics without parity-testing against mflux qwen21.

import mlx.core as mx
from mlx import nn


class Qwen21SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.out = nn.Linear(mlp_hidden_size, hidden_size, bias=False)
        self.gate_layer = nn.Linear(hidden_size, mlp_hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.out(nn.silu(self.gate_layer(hidden_states)) * self.proj(hidden_states))
