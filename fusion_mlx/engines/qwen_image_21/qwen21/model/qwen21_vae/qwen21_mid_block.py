# SPDX-License-Identifier: Apache-2.0
# Vendored from mflux 0.20.0 (https://github.com/filipstrand/mflux) PR#736
# MIT-licensed pure-MLX Qwen-Image-2.1 reference (txt2img + 4-ch RGBA VAE).
# Pinned to fusion-mlx's mlx stack. Upstream-tracked; do not modify forward
# numerics without parity-testing against mflux qwen21.

import mlx.core as mx
from mlx import nn

from .qwen21_attention_block import Qwen21AttentionBlock
from .qwen21_res_block import Qwen21ResBlock


class Qwen21MidBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.resnets = [
            Qwen21ResBlock(dim, dim),
            Qwen21ResBlock(dim, dim),
        ]
        self.attentions = [Qwen21AttentionBlock(dim)]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x
