# SPDX-License-Identifier: Apache-2.0
# Attention mask utilities - causal, sliding-window, block-sparse, ALiBi.

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


def make_causal_mask(
    seq_len_q: int,
    seq_len_kv: int | None = None,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    if seq_len_kv is None:
        seq_len_kv = seq_len_q
    mask = mx.full((seq_len_q, seq_len_kv), -float("inf"), dtype=dtype)
    i_indices = mx.arange(seq_len_q, dtype=mx.int32)[:, None]
    j_indices = mx.arange(seq_len_kv, dtype=mx.int32)[None, :]
    mask = mx.where(
        j_indices <= i_indices,
        mx.array(0.0, dtype=dtype),
        mx.array(-float("inf"), dtype=dtype),
    )
    return mask.reshape(1, 1, seq_len_q, seq_len_kv)


def make_sliding_window_mask(
    seq_len_q: int,
    seq_len_kv: int,
    window_size: int,
    causal: bool = True,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    i = mx.arange(seq_len_q, dtype=mx.float32)[:, None]
    j = mx.arange(seq_len_kv, dtype=mx.float32)[None, :]
    diff = i - j

    if causal:
        cond = (diff >= 0) & (diff < window_size)
    else:
        cond = mx.abs(diff) <= window_size

    mask = mx.where(
        cond, mx.array(0.0, dtype=dtype), mx.array(-float("inf"), dtype=dtype)
    )
    return mask.reshape(1, 1, seq_len_q, seq_len_kv)


def make_alibi_mask(
    num_heads: int,
    seq_len_q: int,
    seq_len_kv: int,
    slopes: mx.array | None = None,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    if slopes is None:
        exp = -8.0 * mx.arange(1, num_heads + 1, dtype=mx.float32) / num_heads
        slopes = 2.0**exp

    i = mx.arange(seq_len_q, dtype=mx.float32).reshape(-1, 1)
    j = mx.arange(seq_len_kv, dtype=mx.float32).reshape(1, -1)
    pos = j - i

    bias = pos.reshape(1, 1, seq_len_q, seq_len_kv) * slopes.reshape(1, -1, 1, 1)
    return bias.astype(dtype)


def mask_to_uint8(mask: mx.array) -> mx.array:
    return (mx.isnan(mask) | (mask < -1e30)).astype(mx.uint8)


def mask_from_attn_mask(
    attn_mask: mx.array,
    query_length: int,
    key_length: int,
    num_heads: int,
    batch_size: int = 1,
) -> mx.array | None:
    if attn_mask is None:
        return None
    mask = attn_mask
    ndim = mask.ndim

    if ndim == 2:
        mask = mask.reshape(1, 1, *mask.shape)
    elif ndim == 3:
        mask = mask[:, None, :, :]
    elif ndim == 4:
        pass
    else:
        raise ValueError(f"Expected 2D-4D mask, got {ndim}D")

    if mask.dtype in (mx.bool_, mx.int32, mx.int64):
        mask = mask.astype(mx.float32)
        mask = mx.where(mask < 0.5, -float("inf"), 0.0)

    return mask


def make_causal_block_mask(
    seq_len: int,
    block_size: int = 64,
) -> mx.array:
    n_blocks = (seq_len + block_size - 1) // block_size
    i = mx.arange(n_blocks, dtype=mx.int32)[:, None]
    j = mx.arange(n_blocks, dtype=mx.int32)[None, :]
    return mx.where(j <= i, mx.array(1, dtype=mx.uint8), mx.array(0, dtype=mx.uint8))
