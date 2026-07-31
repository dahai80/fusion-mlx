# SPDX-License-Identifier: Apache-2.0
import logging

import mlx.core as mx
import mlx.nn as nn

from .layers import (
    MageFlowEmbedRope,
    MageFlowTimestepProjEmbeddings,
    MageFlowTransformerBlock,
    RMSNorm,
)

logger = logging.getLogger(__name__)


class MageFlowTransformer(nn.Module):
    def __init__(
        self,
        dim: int = 3072,
        num_heads: int = 24,
        head_dim: int = 128,
        depth: int = 24,
        ff_mult: float = 4.0,
        rope_dim: int = 128,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.depth = depth

        self.time_embed = MageFlowTimestepProjEmbeddings(dim)
        self.rope_embed = MageFlowEmbedRope(rope_dim)

        self.blocks = [
            MageFlowTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                head_dim=head_dim,
                ff_mult=ff_mult,
            )
            for _ in range(depth)
        ]

        self.norm_out = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, 16, bias=False)

        self.context_norm = RMSNorm(dim)
        self.img_in = nn.Linear(16, dim, bias=False)
        self.txt_in = nn.Linear(dim, dim, bias=False)

    def __call__(
        self,
        img: mx.array,
        img_ids: mx.array,
        txt: mx.array,
        txt_ids: mx.array | None = None,
        txt_mask: mx.array | None = None,
        timesteps: mx.array | None = None,
        vec: mx.array | None = None,
    ) -> mx.array:
        B = img.shape[0]

        if timesteps is not None and vec is not None:
            timestep_emb = self.time_embed(timesteps, img)
            conditioning = timestep_emb + vec
        elif vec is not None:
            conditioning = vec
        else:
            conditioning = mx.zeros((B, self.dim), dtype=img.dtype)

        h_img = self.img_in(img)
        h_txt = self.txt_in(txt)

        rope = self.rope_embed(
            mx.array(img.shape[1] if img.ndim == 3 else img.shape[2]),
            mx.array(img.shape[2] if img.ndim == 3 else img.shape[3]),
        )

        for block in self.blocks:
            h_img = block(
                h_img,
                conditioning=conditioning,
                context=h_txt,
                rope=rope,
            )

        out = self.proj_out(self.norm_out(h_img))
        return out
