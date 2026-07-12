# SPDX-License-Identifier: Apache-2.0
# Distributed attention - Ring + Ulysses parallelism for long-context attention.
# Based on xDiT's attention_backend.py and ring_flash_attn.py.

from __future__ import annotations

import logging
from typing import Optional

import mlx.core as mx

from ..mfa_bridge import flash_attention

logger = logging.getLogger(__name__)


def ring_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float | None = None,
    causal: bool = False,
    num_ranks: int = 1,
    rank: int = 0,
    comm_backend: object | None = None,
) -> mx.array:
    scale = scale if scale is not None else q.shape[-1] ** -0.5
    B, H, N_q, D = q.shape
    seq_len_k = k.shape[-2]

    if num_ranks <= 1:
        return flash_attention(q, k, v, scale=scale, causal=causal)

    logger.warning(
        "Ring attention with %d ranks: using local chunk only. "
        "Full ring communication not yet implemented.",
        num_ranks,
    )

    local_out = flash_attention(q, k, v, scale=scale, causal=causal)

    return local_out


def ulysses_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float | None = None,
    causal: bool = False,
    num_ranks: int = 1,
    rank: int = 0,
) -> mx.array:
    scale = scale if scale is not None else q.shape[-1] ** -0.5

    if num_ranks <= 1:
        return flash_attention(q, k, v, scale=scale, causal=causal)

    logger.warning(
        "Ulysses attention with %d ranks: local only. "
        "All-to-all gather not yet implemented.",
        num_ranks,
    )

    return flash_attention(q, k, v, scale=scale, causal=causal)


def split_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float | None = None,
    causal: bool = False,
    chunk_size: int = 4096,
) -> mx.array:
    scale = scale if scale is not None else q.shape[-1] ** -0.5
    B, H, N_q, D = q.shape
    N_k = k.shape[-2]

    if N_k <= chunk_size:
        return flash_attention(q, k, v, scale=scale, causal=causal)

    outputs = []
    lse_list = []

    for start in range(0, N_k, chunk_size):
        end = min(start + chunk_size, N_k)
        k_chunk = k[:, :, start:end, :]
        v_chunk = v[:, :, start:end, :]

        scores = mx.matmul(q, k_chunk.transpose(0, 1, 3, 2)) * scale

        if causal:
            q_pos = mx.arange(N_q, dtype=mx.float32)[None, None, :, None]
            k_pos = mx.arange(start, end, dtype=mx.float32)[None, None, None, :]
            causal_mask = (k_pos <= q_pos).astype(scores.dtype)
            scores = mx.where(causal_mask > 0, scores, -float("inf"))

        lse = mx.logsumexp(scores, axis=-1, keepdims=True)
        p = mx.exp(scores - lse)
        out_chunk = p @ v_chunk

        lse_list.append(lse)
        outputs.append(out_chunk)

    if len(outputs) == 1:
        return outputs[0]

    lse_all = mx.concatenate(lse_list, axis=-1)
    lse_max = mx.max(lse_all, axis=-1, keepdims=True)
    weights = mx.exp(lse_all - lse_max)
    weights = weights / mx.sum(weights, axis=-1, keepdims=True)

    merged = mx.zeros_like(outputs[0])
    for i, out in enumerate(outputs):
        w = weights[:, :, :, i : i + 1]
        merged = merged + out * w

    return merged


__all__ = [
    "ring_attention",
    "ulysses_attention",
    "split_attention",
]
