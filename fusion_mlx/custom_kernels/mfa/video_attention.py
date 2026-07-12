# SPDX-License-Identifier: Apache-2.0
# Video attention - temporal mixing + spatial-temporal transformers.
# Based on CogVideo's VideoTransformerBlock and SpatialVideoTransformer, adapted for MLX.

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

from ..mfa_bridge import flash_attention

logger = logging.getLogger(__name__)


def _temporal_rearrange(x: mx.array, timesteps: int):
    B_T, S, C = x.shape
    B = B_T // timesteps
    x_4d = x.reshape(B, timesteps, S, C)
    x_4d = mx.transpose(x_4d, (0, 2, 1, 3))
    return x_4d.reshape(B * S, timesteps, C), S


def _temporal_rearrange_back(x: mx.array, B: int, S: int, T: int, C: int) -> mx.array:
    x_4d = x.reshape(B, S, T, C)
    x_4d = mx.transpose(x_4d, (0, 2, 1, 3))
    return x_4d.reshape(B * T, S, C)


class TemporalAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int | None = None,
        causal: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head or dim // heads
        inner_dim = self.heads * self.dim_head

        self.scale = self.dim_head**-0.5
        self.causal = causal

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def __call__(
        self,
        x: mx.array,
        timesteps: int,
        mask: mx.array | None = None,
    ) -> mx.array:
        x_time, S = _temporal_rearrange(x, timesteps)
        B_T, _, C = x.shape
        B = B_T // timesteps

        q = self.to_q(x_time)
        k = self.to_k(x_time)
        v = self.to_v(x_time)

        q = q.reshape(-1, timesteps, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = k.reshape(-1, timesteps, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(-1, timesteps, self.heads, self.dim_head).transpose(0, 2, 1, 3)

        out = flash_attention(q, k, v, scale=self.scale, mask=mask, causal=self.causal)

        out = out.transpose(0, 2, 1, 3).reshape(
            -1, timesteps, self.heads * self.dim_head
        )

        out = self.to_out(out)

        return _temporal_rearrange_back(out, B, S, timesteps, C)


class VideoTransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int | None = None,
        context_dim: int | None = None,
        ff_mult: int = 4,
        causal: bool = False,
        checkpoint: bool = False,
    ):
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head or dim // heads
        inner_dim = self.heads * self.dim_head
        self.is_res = inner_dim == dim
        self.checkpoint = checkpoint

        self.norm_in = nn.RMSNorm(dim) if inner_dim != dim else None
        self.ff_in = nn.Linear(dim, inner_dim, bias=True) if inner_dim != dim else None
        self.temporal_attn = TemporalAttention(
            inner_dim, heads, self.dim_head, causal=causal
        )

        self.norm2 = nn.RMSNorm(inner_dim)
        self.context_dim = context_dim
        if context_dim is not None:
            self.to_q = nn.Linear(inner_dim, inner_dim, bias=False)
            self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_out_ca = nn.Linear(inner_dim, inner_dim, bias=False)
        else:
            self.to_q = None

        self.norm1 = nn.RMSNorm(inner_dim)
        self.ff = nn.Sequential(
            nn.Linear(inner_dim, dim * ff_mult, bias=True),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim, bias=True),
        )
        self.norm3 = nn.RMSNorm(inner_dim) if not self.is_res else None

    def __call__(
        self,
        x: mx.array,
        context: mx.array | None = None,
        timesteps: int | None = None,
    ) -> mx.array:
        B_T, S, C = x.shape
        if timesteps is None:
            raise ValueError("timesteps is required for VideoTransformerBlock")

        x_skip = x
        if self.ff_in is not None and self.norm_in is not None:
            x = self.ff_in(self.norm_in(x))

        x = self.temporal_attn(x, timesteps=timesteps)
        if self.is_res:
            x = x + x_skip

        if self.to_q is not None and context is not None:
            x_skip = x
            x = self.norm2(x)
            q = self.to_q(x)
            k = self.to_k(context)
            v = self.to_v(context)

            q = q.reshape(-1, S, self.heads, self.dim_head).transpose(0, 2, 1, 3)
            k = k.reshape(-1, context.shape[1], self.heads, self.dim_head).transpose(
                0, 2, 1, 3
            )
            v = v.reshape(-1, context.shape[1], self.heads, self.dim_head).transpose(
                0, 2, 1, 3
            )

            out = flash_attention(q, k, v, scale=self.dim_head**-0.5)
            out = out.transpose(0, 2, 1, 3).reshape(-1, S, self.heads * self.dim_head)
            x = self.to_out_ca(out) + x_skip

        x_skip = x
        x_norm = self.norm3(x) if self.norm3 is not None else x
        x = self.ff(x_norm)
        if self.is_res:
            x = x + x_skip

        return x


class SpatialVideoTransformer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        depth: int = 1,
        context_dim: int | None = None,
        timesteps: int | None = None,
        time_depth: int = 1,
        disable_self_attn: bool = False,
    ):
        super().__init__()

        inner_dim = n_heads * d_head
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.d_head = d_head
        self.timesteps = timesteps

        self.norm = nn.RMSNorm(in_channels)
        self.proj_in = nn.Linear(in_channels, inner_dim, bias=True)

        self.transformer_blocks = []
        for _ in range(depth):
            block = VideoTransformerBlock(
                dim=inner_dim,
                heads=n_heads,
                dim_head=d_head,
                context_dim=context_dim,
                causal=False,
            )
            self.transformer_blocks.append(block)

        self.time_stack = []
        for _ in range(time_depth):
            block = VideoTransformerBlock(
                dim=inner_dim,
                heads=n_heads,
                dim_head=d_head,
                context_dim=context_dim,
                causal=False,
            )
            self.time_stack.append(block)

        self.proj_out = nn.Linear(inner_dim, in_channels, bias=True)

    def __call__(
        self,
        x: mx.array,
        context: mx.array | None = None,
        timesteps: int | None = None,
    ) -> mx.array:
        if timesteps is None:
            timesteps = self.timesteps
        if timesteps is None:
            raise ValueError("timesteps is required for SpatialVideoTransformer")

        if x.ndim == 4:
            B_T, C, H, W = x.shape
        else:
            B_T, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        x_in = x

        if x.ndim == 4:
            B_T, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(B_T, H * W, C)
        else:
            B_T, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        x = self.norm(x)
        x = self.proj_in(x)

        for block in self.transformer_blocks:
            x = block(x, context=context, timesteps=timesteps)

        for block in self.time_stack:
            x = block(x, context=context, timesteps=timesteps)

        x = self.proj_out(x)
        x = x.transpose(0, 2, 1).reshape(B_T, C, H, W)

        return x + x_in
