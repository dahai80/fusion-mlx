# SPDX-License-Identifier: Apache-2.0
import math

import mlx.core as mx

import logging

logger = logging.getLogger(__name__)


def _compute_1d_rope(dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    t = mx.arange(seq_len, dtype=mx.float32)
    angles = mx.outer(t, freqs)
    cos_half = mx.cos(angles)
    sin_half = mx.sin(angles)
    cos = mx.repeat(cos_half, 2, axis=-1)
    sin = mx.repeat(sin_half, 2, axis=-1)
    return cos, sin


def compute_3d_rope(
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 2,
    patch_size_t: int | None = None,
    head_dim: int = 64,
    spatial_interpolation_scale: float = 1.875,
    temporal_interpolation_scale: float = 1.0,
):
    pt = patch_size_t if patch_size_t is not None else 1
    grid_t = num_frames // pt if num_frames > 1 else 1
    grid_h = height // patch_size
    grid_w = width // patch_size

    dim_t = head_dim // 4
    dim_h = head_dim // 8 * 3
    dim_w = head_dim // 8 * 3

    t_cos, t_sin = _compute_1d_rope(dim_t, grid_t)
    h_cos, h_sin = _compute_1d_rope(dim_h, int(grid_h * spatial_interpolation_scale))
    w_cos, w_sin = _compute_1d_rope(dim_w, int(grid_w * spatial_interpolation_scale))

    h_cos = h_cos[:grid_h]
    h_sin = h_sin[:grid_h]
    w_cos = w_cos[:grid_w]
    w_sin = w_sin[:grid_w]

    t_cos = mx.broadcast_to(t_cos[:, None, None, :], (grid_t, grid_h, grid_w, dim_t))
    t_sin = mx.broadcast_to(t_sin[:, None, None, :], (grid_t, grid_h, grid_w, dim_t))
    h_cos = mx.broadcast_to(h_cos[None, :, None, :], (grid_t, grid_h, grid_w, dim_h))
    h_sin = mx.broadcast_to(h_sin[None, :, None, :], (grid_t, grid_h, grid_w, dim_h))
    w_cos = mx.broadcast_to(w_cos[None, None, :, :], (grid_t, grid_h, grid_w, dim_w))
    w_sin = mx.broadcast_to(w_sin[None, None, :, :], (grid_t, grid_h, grid_w, dim_w))

    cos = mx.concatenate([t_cos, h_cos, w_cos], axis=-1)
    sin = mx.concatenate([t_sin, h_sin, w_sin], axis=-1)

    cos = cos.reshape(-1, head_dim)
    sin = sin.reshape(-1, head_dim)
    return cos, sin


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    while cos.ndim < x.ndim:
        cos = cos[None, ...]
        sin = sin[None, ...]
    c = cos[..., :d]
    s = sin[..., :d]
    return mx.concatenate(
        [x1 * c - x2 * s, x2 * c + x1 * s],
        axis=-1,
    )
