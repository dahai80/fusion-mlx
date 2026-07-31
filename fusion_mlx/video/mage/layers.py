# SPDX-License-Identifier: Apache-2.0
import logging
import math

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> mx.array:
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(start=0, stop=half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = mx.exp(exponent).astype(timesteps.dtype)
    emb = timesteps[:, None].astype(mx.float32) * emb[None, :]
    emb = scale * emb
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if flip_sin_to_cos:
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def __call__(self, timesteps: mx.array) -> mx.array:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, sample: mx.array) -> mx.array:
        sample = nn.silu(self.linear_1(sample))
        sample = self.linear_2(sample)
        return sample


class MageFlowTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            scale=1000,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def __call__(self, timestep: mx.array, hidden_states: mx.array) -> mx.array:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.astype(hidden_states.dtype))
        return timesteps_emb


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2)

    def __call__(self, x: mx.array, conditioning: mx.array) -> mx.array:
        emb = self.linear(nn.silu(conditioning))
        scale, shift = mx.split(emb, 2, axis=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        x = (x / rms).astype(dtype) * self.weight
        return x


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int | None = None, mult: float = 4.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.net = [
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim_out),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.net:
            x = layer(x)
        return x


class MageFlowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        inner_dim = self.num_heads * self.head_dim
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def __call__(
        self,
        x: mx.array,
        context: mx.array | None = None,
        rope: mx.array | None = None,
    ) -> mx.array:
        context = context if context is not None else x
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        B, L, _ = q.shape
        _, S, _ = k.shape

        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if rope is not None:
            q = _apply_rotary_emb(q, rope)
            k = _apply_rotary_emb(k, rope)

        scale = self.head_dim ** -0.5
        attn = (q * scale) @ k.transpose(0, 1, 3, 2)
        attn = nn.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.to_out(out)


def _apply_rotary_emb(x: mx.array, freqs_cis: mx.array) -> mx.array:
    x_complex = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
    x_complex = mx.view_as_complex(x_complex)
    freqs_cis = freqs_cis.unsqueeze(1)
    x_out = mx.view_as_real(x_complex * freqs_cis)
    x_out = x_out.reshape(*x.shape)
    return x_out.astype(x.dtype)


class MageFlowEmbedRope(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, h: mx.array, w: mx.array) -> mx.array:
        h = int(h.item()) if hasattr(h, "item") else int(h)
        w = int(w.item()) if hasattr(w, "item") else int(w)
        half_dim = self.dim // 2
        freqs_h = _freqs_1d(h, half_dim, self.theta)
        freqs_w = _freqs_1d(w, half_dim, self.theta)
        freqs = mx.concatenate([
            mx.concatenate([freqs_h for _ in range(w)], axis=0),
            mx.concatenate([freqs_w for _ in range(h)], axis=0),
        ], axis=-1)
        return mx.view_as_complex(freqs)


def _freqs_1d(length: int, dim: int, theta: float) -> mx.array:
    freqs = mx.arange(0, dim, dtype=mx.float32)
    freqs = 1.0 / (theta ** (2.0 * freqs / dim))
    t = mx.arange(length, dtype=mx.float32)
    freqs = mx.outer(t, freqs)
    freqs = mx.stack([mx.cos(freqs), mx.sin(freqs)], axis=-1)
    return mx.view_as_complex(freqs)


class MageFlowTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int | None = None,
        ff_mult: float = 4.0,
        conditioning_dim: int | None = None,
    ):
        super().__init__()
        conditioning_dim = conditioning_dim or dim
        self.norm1 = AdaLayerNormContinuous(dim, conditioning_dim)
        self.attn = MageFlowAttention(dim, num_heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=ff_mult)
        self.norm_cond = RMSNorm(conditioning_dim)

    def __call__(
        self,
        x: mx.array,
        conditioning: mx.array,
        context: mx.array | None = None,
        rope: mx.array | None = None,
    ) -> mx.array:
        normed = self.norm1(x, conditioning)
        attn_out = self.attn(normed, context=context, rope=rope)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x
