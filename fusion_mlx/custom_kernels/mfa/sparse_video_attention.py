# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class SSTAConfig:
    canvas_thw: tuple[int, int, int]
    tile_thw: tuple[int, int, int]
    kernel_thw: tuple[int, int, int]
    text_len: int = 0
    topk: int = 8
    threshold: float = 0.5


def _compute_block_grid(
    canvas_thw: tuple[int, int, int],
    tile_thw: tuple[int, int, int],
) -> tuple[int, int, int]:
    T, H, W = canvas_thw
    T_t, H_t, W_t = tile_thw
    return (
        math.ceil(T / T_t),
        math.ceil(H / H_t),
        math.ceil(W / W_t),
    )


def _tile_id_to_range(
    tile_id: int,
    grid: tuple[int, int, int],
    tile_thw: tuple[int, int, int],
) -> tuple[int, int]:
    n_t, n_h, n_w = grid
    T_t, H_t, W_t = tile_thw
    t = tile_id // (n_h * n_w)
    h = (tile_id % (n_h * n_w)) // n_w
    w = tile_id % n_w

    start_t = t * T_t
    start_h = h * H_t
    start_w = w * W_t
    end_t = min(start_t + T_t, grid[0] * T_t)
    end_h = min(start_h + H_t, grid[1] * H_t)
    end_w = min(start_w + W_t, grid[2] * W_t)

    total_w = grid[2] * W_t
    total_h = grid[1] * H_t
    start = start_t * total_h * total_w + start_h * total_w + start_w
    end = end_t * total_h * total_w + end_h * total_w + end_w
    return start, end


def _compute_block_similarity(
    q: mx.array,
    k: mx.array,
    block_mask: mx.array,
    block_size: int = 64,
) -> mx.array:
    B, H, N_q, D = q.shape
    N_k = k.shape[-2]
    n_blocks_q = (N_q + block_size - 1) // block_size
    n_blocks_k = (N_k + block_size - 1) // block_size

    scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(D)

    block_scores = []
    for bq in range(n_blocks_q):
        for bk in range(n_blocks_k):
            if block_mask[bq, bk] > 0:
                qi = bq * block_size
                qe = min(qi + block_size, N_q)
                ki = bk * block_size
                ke = min(ki + block_size, N_k)
                block_scores.append(scores[:, :, qi:qe, ki:ke].mean())
    return mx.array(block_scores)


def make_ssta_block_mask(
    config: SSTAConfig,
    q: mx.array,
    k: mx.array,
    text_len: int = 0,
) -> mx.array:
    grid = _compute_block_grid(config.canvas_thw, config.tile_thw)
    n_tiles = grid[0] * grid[1] * grid[2]

    T, H, W = config.canvas_thw
    T_t, H_t, W_t = config.tile_thw
    T_k, H_k, W_k = config.kernel_thw
    total_tokens = T * H * W

    block_size = 64
    n_blocks_q = (text_len + total_tokens + block_size - 1) // block_size
    n_blocks_k = n_blocks_q

    mask = mx.zeros((n_blocks_q, n_blocks_k), dtype=mx.uint8)
    text_n_blocks = (text_len + block_size - 1) // block_size
    for i in range(text_n_blocks):
        for j in range(text_n_blocks):
            mask[i, j] = 1

    for tile_id in range(n_tiles):
        t = tile_id // (grid[1] * grid[2])
        h = (tile_id % (grid[1] * grid[2])) // grid[2]
        w = tile_id % grid[2]

        t_start = max(0, t - T_k // T_t)
        t_end = min(grid[0], t + T_k // T_t + 1)
        h_start = max(0, h - H_k // H_t)
        h_end = min(grid[1], h + H_k // H_t + 1)
        w_start = max(0, w - W_k // W_t)
        w_end = min(grid[2], w + W_k // W_t + 1)

        for tt in range(t_start, t_end):
            for hh in range(h_start, h_end):
                for ww in range(w_start, w_end):
                    neighbor_id = tt * grid[1] * grid[2] + hh * grid[2] + ww
                    start, end = _tile_id_to_range(neighbor_id, grid, config.tile_thw)
                    q_start = text_len + start
                    q_end = text_len + end
                    for bq in range(
                        q_start // block_size, (q_end + block_size - 1) // block_size
                    ):
                        for bk in range(
                            q_start // block_size,
                            (q_end + block_size - 1) // block_size,
                        ):
                            if bq < n_blocks_q and bk < n_blocks_k:
                                mask[bq, bk] = 1

    return mask


def sparse_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    block_mask: mx.array,
    *,
    scale: float | None = None,
    block_size: int = 64,
) -> mx.array:
    scale = scale if scale is not None else q.shape[-1] ** -0.5
    B, H, N_q, D = q.shape
    N_k = k.shape[-2]

    try:
        from ..mfa import _HAS_MFA_EXT, _MFA_EXT

        if _HAS_MFA_EXT and hasattr(_MFA_EXT, "sage_attention"):
            return _MFA_EXT.sage_attention(q, k, v, block_mask, scale=scale)
    except Exception:
        pass

    dense = mx.full((N_q, N_k), -float("inf"), dtype=q.dtype)
    n_blocks_q = block_mask.shape[0]
    n_blocks_k = block_mask.shape[1]
    for bq in range(n_blocks_q):
        for bk in range(n_blocks_k):
            if block_mask[bq, bk] > 0:
                i0 = bq * block_size
                i1 = min(i0 + block_size, N_q)
                j0 = bk * block_size
                j1 = min(j0 + block_size, N_k)
                dense[i0:i1, j0:j1] = 0.0

    dense = dense.reshape(1, 1, N_q, N_k)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=dense)


def ssta_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    config: SSTAConfig,
    *,
    scale: float | None = None,
    text_len: int = 0,
) -> mx.array:
    block_mask = make_ssta_block_mask(config, q, k, text_len=text_len)
    return sparse_attention(q, k, v, block_mask, scale=scale)


__all__ = [
    "SSTAConfig",
    "make_ssta_block_mask",
    "sparse_attention",
    "ssta_attention",
]
