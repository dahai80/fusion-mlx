# SPDX-License-Identifier: Apache-2.0
# ACE-Step MLX building blocks (issue #988): FSQ quantizer, 4D bidirectional
# sliding/full attention masks, Qwen3-style RoPE, RMSNorm, MLP, Attention,
# DiT layer. Ported from ACE-Step/Ace-Step1.5 modeling_acestep_v15_turbo.py
# (MIT). No torch deps — pure mlx.nn.
from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

from .config import AceStepConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Finite Scalar Quantization (FSQ). Port of vector_quantize_pytorch.ResidualFSQ
# for num_quantizers=1 (the ACE-Step config). Quantize: x -> round(x * (L-1)/2)
# / ((L-1)/2) per dim, then residual-stack for N>1 quantizers. Levels define the
# codebook size per dim = prod(levels). No codebook embedding table needed —
# indices are the cartesian product of per-dim bucket indices.
# ---------------------------------------------------------------------------


class FSQ(nn.Module):
    # Finite Scalar Quantization (port of vector_quantize_pytorch.FSQ with
    # preserve_symmetry=True, bound_hard_clamp=True — the ResidualFSQ default).
    # dim defaults to len(levels); when caller passes dim != len(levels),
    # ResidualFSQ wraps with project_in/out Linear. Here we quantize on the
    # codebook_dim (= len(levels)) subspace directly.
    def __init__(self, levels: list[int]):
        super().__init__()
        self.levels = levels
        self.num_levels = len(levels)
        self.codebook_dim = self.num_levels
        self._levels = mx.array(levels, dtype=mx.float32)
        # codebook index basis (cumprod) for flat-index encoding
        self._basis = mx.array([1] + levels[:-1], dtype=mx.float32)
        _b = mx.cumprod(self._basis, axis=0)
        self._basis = _b
        self.codebook_size = 1
        for lv in levels:
            self.codebook_size *= lv

    def _symmetry_preserving_bound(
        self, z: mx.array, hard_clamp: bool = True
    ) -> mx.array:
        # QL(x) = 2/(L-1) * [ (L-1)*(tanh(x)+1)/2 + 0.5 ] - 1, floored
        mt = mx.tanh if not hard_clamp else lambda v: mx.clip(v, -1.0, 1.0)
        lv1 = self._levels - 1
        scale = 2.0 / lv1
        bracket = (lv1 * (mt(z) + 1.0) / 2.0) + 0.5
        bracket = mx.floor(bracket)
        return scale * bracket - 1.0

    def _scale_and_shift(self, zhat: mx.array) -> mx.array:
        # normalized [-1,1] -> bucket index [0, L-1]
        return (zhat + 1.0) / (2.0 / (self._levels - 1))

    def quantize(self, z: mx.array) -> mx.array:
        return self._symmetry_preserving_bound(z, hard_clamp=True)

    def codes_to_indices(self, zhat: mx.array) -> mx.array:
        zhat_s = self._scale_and_shift(zhat)
        return mx.sum(mx.round(zhat_s) * self._basis, axis=-1)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        # x: (..., codebook_dim). Quantize each dim independently.
        z = x.astype(mx.float32)
        quant = self.quantize(z)
        # straight-through
        out = z + mx.stop_gradient(quant - z)
        indices = self.codes_to_indices(quant)
        return out, indices


class ResidualFSQ(nn.Module):
    # Port of vector_quantize_pytorch.ResidualFSQ. When dim != len(levels),
    # wraps project_in (dim->codebook_dim) + project_out (codebook_dim->dim).
    # ACE-Step config: dim=2048, levels=[8,8,8,5,5,5] -> project 2048->6.
    # num_quantizers=1 (single FSQ layer, no residual stack).
    def __init__(self, dim: int, levels: list[int], num_quantizers: int = 1):
        super().__init__()
        self.dim = dim
        self.levels = levels
        self.num_quantizers = num_quantizers
        self.codebook_dim = len(levels)
        self.has_projections = self.codebook_dim != dim
        if self.has_projections:
            # project_in: dim->codebook_dim, project_out: codebook_dim->dim
            self.project_in = nn.Linear(dim, self.codebook_dim, bias=True)
            self.project_out = nn.Linear(self.codebook_dim, dim, bias=True)
        self.layers = [FSQ(levels) for _ in range(num_quantizers)]
        # per-quantizer scales: levels**-ind
        scales = [
            mx.array(levels, dtype=mx.float32) ** (-i) for i in range(num_quantizers)
        ]
        self.scales = mx.stack(scales, axis=0)
        # soft clamp (bound_hard_clamp=True path): 1 + 1/(L-1)
        self.soft_clamp_value = 1.0 + (1.0 / (mx.array(levels, dtype=mx.float32) - 1))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        z = x.astype(mx.float32)
        if self.has_projections:
            z = self.project_in(z)
        # soft clamp (tanh) before residual layers
        z = z / self.soft_clamp_value
        z = mx.tanh(z) * self.soft_clamp_value
        quantized_out = mx.zeros_like(z)
        residual = z
        all_indices = []
        for qi, (layer, scale) in enumerate(zip(self.layers, self.scales)):
            q, idx = layer(residual / scale)
            q = q * scale
            residual = residual - mx.stop_gradient(q)
            quantized_out = quantized_out + q
            all_indices.append(idx)
        if self.has_projections:
            quantized_out = self.project_out(quantized_out)
        out = x.astype(mx.float32) + mx.stop_gradient(
            quantized_out - x.astype(mx.float32)
        )
        indices = mx.stack(all_indices, axis=-1)
        return out, indices


# ---------------------------------------------------------------------------
# Attention masks. ACE-Step uses BIDIRECTIONAL attention (non-causal) for the
# music DiT — full (global) or sliding-window (local). Sliding = |i-j| <= W.
# Mask shape (B, 1, L, L), additive (-inf where masked, 0 where attend).
# ---------------------------------------------------------------------------


def create_4d_mask(
    seq_len: int,
    dtype: mx.Dtype,
    attention_mask: mx.array | None = None,
    sliding_window: int | None = None,
    is_causal: bool = False,
) -> mx.array:
    # attention_mask: (B, L) bool/0-1 padding mask (None = no padding). If the
    # caller passes a pre-patch-length mask, we assume all-attend (no padding).
    i = mx.arange(seq_len)[:, None]
    j = mx.arange(seq_len)[None, :]
    valid = mx.ones((seq_len, seq_len), dtype=dtype)
    if is_causal:
        valid = valid * mx.where(j > i, mx.zeros_like(valid), valid)
    if sliding_window is not None and sliding_window > 0:
        local = mx.where(
            mx.abs(i - j) <= sliding_window,
            mx.ones_like(valid),
            mx.zeros_like(valid),
        )
        valid = valid * local
    # additive: 0 attend, -inf masked
    neg = mx.log(mx.zeros((), dtype=dtype))  # -inf
    mask2d = mx.where(valid > 0, mx.zeros_like(valid), neg)  # (L, L)
    if attention_mask is not None and attention_mask.shape[-1] == seq_len:
        # (B, L) -> (B, 1, 1, L) broadcast over query axis
        am = attention_mask.astype(dtype)
        am = mx.reshape(am, (am.shape[0], 1, 1, am.shape[-1]))
        base = mask2d[None, None]  # (1, 1, L, L)
        pad_mask = mx.where(am > 0, mx.zeros_like(base), neg)
        mask4d = base + pad_mask
    else:
        mask4d = mask2d[None, None]
    return mask4d


# ---------------------------------------------------------------------------
# Qwen3-style RoPE (split / rotate_half). head_dim must be even.
# ---------------------------------------------------------------------------


def rope_freqs(head_dim: int, max_seq: int, theta: float) -> tuple[mx.array, mx.array]:
    inv_freq = 1.0 / (
        theta ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim)
    )
    pos = mx.arange(max_seq).astype(mx.float32)
    freqs = mx.outer(pos, inv_freq)
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    # duplicate to full head_dim (split rope: [d0,d1,...] -> cos=[c0,c0,c1,c1,...])
    cos = mx.concatenate([cos, cos], axis=-1)
    sin = mx.concatenate([sin, sin], axis=-1)
    return cos, sin


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rope(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> tuple[mx.array, mx.array]:
    # q,k: (B, H, L, D). cos,sin: (L, D).
    cos = cos[: q.shape[-2]][None, None]
    sin = sin[: q.shape[-2]][None, None]
    q_r = q * cos + rotate_half(q) * sin
    k_r = k * cos + rotate_half(k) * sin
    return q_r.astype(q.dtype), k_r.astype(k.dtype)


# ---------------------------------------------------------------------------
# Attention (GQA + RoPE + q/k RMSNorm + sliding/full mask). Bidirectional.
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    mask: mx.array | None = None,
) -> mx.array:
    # q: (B,H,Lq,D), k/v: (B,H,Lk,D). GQA: repeat k/v heads if H != kv_H.
    n_rep = q.shape[1] // k.shape[1]
    if n_rep > 1:
        k = mx.repeat(k, n_rep, axis=1)
        v = mx.repeat(v, n_rep, axis=1)
    attn = mx.matmul(q, mx.swapaxes(k, -2, -1)) * scale
    if mask is not None:
        # mask: (B,1,Lq,Lk) or broadcastable
        attn = attn + mask
    attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
    out = mx.matmul(attn, v)
    return out


class Attention(nn.Module):
    def __init__(self, config: AceStepConfig, layer_idx: int, is_cross: bool = False):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.is_cross = is_cross
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]
        self.sliding_window = (
            config.sliding_window
            if self.attention_type == "sliding_attention"
            else None
        )

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
        encoder_hidden_states: mx.array | None = None,
    ) -> mx.array:
        B, L, _ = hidden_states.shape
        q = self.q_proj(hidden_states).reshape(B, L, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        if encoder_hidden_states is not None:
            eB, eL, _ = encoder_hidden_states.shape
            k = self.k_proj(encoder_hidden_states).reshape(
                eB, eL, self.num_kv_heads, self.head_dim
            )
            v = self.v_proj(encoder_hidden_states).reshape(
                eB, eL, self.num_kv_heads, self.head_dim
            )
            k = self.k_norm(k).transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
        else:
            k = self.k_proj(hidden_states).reshape(
                B, L, self.num_kv_heads, self.head_dim
            )
            v = self.v_proj(hidden_states).reshape(
                B, L, self.num_kv_heads, self.head_dim
            )
            k = self.k_norm(k).transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            if position_embeddings is not None:
                cos, sin = position_embeddings
                q, k = apply_rope(q, k, cos, sin)
        out = scaled_dot_product_attention(q, k, v, self.scaling, mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, config: AceStepConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# DiT layer: AdaLN-modulated self-attn + cross-attn + MLP. scale_shift_table
# (1,6,D) + temb (1,D) -> 6 chunks (shift,scale,gate x2).
# ---------------------------------------------------------------------------


class DiTLayer(nn.Module):
    def __init__(self, config: AceStepConfig, layer_idx: int, use_cross: bool = True):
        super().__init__()
        self.self_attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config, layer_idx, is_cross=False)
        self.use_cross = use_cross
        if use_cross:
            self.cross_attn_norm = nn.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.cross_attn = Attention(config, layer_idx, is_cross=True)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)
        self.scale_shift_table = mx.random.normal((1, 6, config.hidden_size)) / (
            config.hidden_size**0.5
        )
        self.attention_type = config.layer_types[layer_idx]

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        temb: mx.array,
        self_mask: mx.array | None = None,
        encoder_hidden_states: mx.array | None = None,
        encoder_mask: mx.array | None = None,
    ) -> mx.array:
        # temb: (B, 6, D) from TimestepEmbedding.time_proj. scale_shift_table: (1, 6, D).
        # sum -> (B, 6, D), split into 6 (B, 1, D), squeeze to (B, D).
        mod = self.scale_shift_table + temb
        chunks = [mx.squeeze(c, axis=1) for c in mx.split(mod, 6, axis=1)]
        shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate = chunks

        # reshape (B, D) -> (B, 1, D) for broadcast over seq
        def _b(m):
            return m[:, None, :]

        n = self.self_attn_norm(hidden_states)
        n = n * (1 + _b(scale_msa)) + _b(shift_msa)
        attn_out = self.self_attn(
            n, mask=self_mask, position_embeddings=position_embeddings
        )
        hidden_states = hidden_states + attn_out * _b(gate_msa)
        if self.use_cross and encoder_hidden_states is not None:
            n2 = self.cross_attn_norm(hidden_states)
            cattn_out = self.cross_attn(
                n2, mask=encoder_mask, encoder_hidden_states=encoder_hidden_states
            )
            hidden_states = hidden_states + cattn_out
        n3 = self.mlp_norm(hidden_states) * (1 + _b(c_scale)) + _b(c_shift)
        ff = self.mlp(n3)
        hidden_states = hidden_states + ff * _b(c_gate)
        return hidden_states


__all__ = [
    "ResidualFSQ",
    "FSQ",
    "create_4d_mask",
    "rope_freqs",
    "apply_rope",
    "Attention",
    "MLP",
    "DiTLayer",
    "scaled_dot_product_attention",
]
