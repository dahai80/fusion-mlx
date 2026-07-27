# SPDX-License-Identifier: Apache-2.0
# Flux-style 3D RoPE for Open-Sora V2 MMDiT.
# axes_dim=[16,56,56] for head_dim=128: T=16, H=56, W=56

import logging
import math

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def rope(pos, dim, theta=10000):
    # Flux-style RoPE: compute rotation matrices per axis.
    # pos: (..., seq_len), dim: int, theta: int
    # Returns: (..., seq_len, dim//2, 2, 2) rotation matrices
    scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    omega = 1.0 / (theta**scale)
    out = mx.einsum("...n,d->...nd", pos.astype(mx.float32), omega)
    cos_val = mx.cos(out)
    sin_val = mx.sin(out)
    neg_sin_val = -sin_val
    out = mx.stack([cos_val, neg_sin_val, sin_val, cos_val], axis=-1)
    out = out.reshape(*out.shape[:-1], 2, 2)
    return out.astype(mx.float32)


def apply_rope(xq, xk, freqs_cis):
    # xq: (batch, heads, seq, head_dim), xk: same
    # freqs_cis: (..., seq, dim//2, 2, 2)
    xq_f = xq.astype(mx.float32)
    xk_f = xk.astype(mx.float32)
    xq_r = xq_f.reshape(*xq_f.shape[:-1], -1, 1, 2)
    xk_r = xk_f.reshape(*xk_f.shape[:-1], -1, 1, 2)
    while freqs_cis.ndim < xq_r.ndim:
        freqs_cis = freqs_cis[None, ...]
    xq_out = freqs_cis[..., 0] * xq_r[..., 0] + freqs_cis[..., 1] * xq_r[..., 1]
    xk_out = freqs_cis[..., 0] * xk_r[..., 0] + freqs_cis[..., 1] * xk_r[..., 1]
    xq_out = xq_out.reshape(*xq.shape).astype(xq.dtype)
    xk_out = xk_out.reshape(*xk.shape).astype(xk.dtype)
    return xq_out, xk_out


class EmbedND(nn.Module):
    def __init__(self, dim, theta, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids):
        # ids: (batch, seq_len, 3) — [T, H, W] positions
        # Returns: (batch, seq_len, dim//2, 2, 2) rotation matrices
        n_axes = ids.shape[-1]
        emb_parts = []
        for i in range(n_axes):
            part = rope(ids[..., i], self.axes_dim[i], self.theta)
            emb_parts.append(part)
        emb = mx.concatenate(emb_parts, axis=-3)
        return emb
