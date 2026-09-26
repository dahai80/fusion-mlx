# SPDX-License-Identifier: Apache-2.0
# Vendored from mflux 0.20.0 (https://github.com/filipstrand/mflux) PR#736
# MIT-licensed pure-MLX Qwen-Image-2.1 reference (txt2img + 4-ch RGBA VAE).
# Pinned to fusion-mlx's mlx stack. Upstream-tracked; do not modify forward
# numerics without parity-testing against mflux qwen21.

import mlx.core as mx
from mlx import nn

from .qwen21_zero_center_rms_norm import Qwen21ZeroCenterRMSNorm


class Qwen21TextProjection(nn.Module):
    def __init__(self, context_in_dim: int = 4096, hidden_size: int = 4096, eps: float = 1e-6):
        super().__init__()
        self.text_norm = Qwen21ZeroCenterRMSNorm(context_in_dim, eps=eps)
        self.in_layer = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.text_norm(hidden_states)
        hidden_states = self.in_layer(hidden_states)
        hidden_states = nn.gelu_approx(hidden_states)
        return self.out_layer(hidden_states)
