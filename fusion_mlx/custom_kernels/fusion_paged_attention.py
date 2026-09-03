# SPDX-License-Identifier: Apache-2.0
# Fused paged decode-attention Metal kernel (Phase 2/3).
# One thread per (batch, query_head) reads the paged physical KV pool via
# block_table indirection and runs online softmax (FlashAttention-2 scalar).
# Naive correct version: grid=(B*n_heads,1,1), threadgroup=(1,1,1).
# Task 6 adds tiling; correctness gates performance.

from __future__ import annotations

import logging
import os
from functools import cache

import mlx.core as mx

logger = logging.getLogger(__name__)


def metal_available() -> bool:
    if os.environ.get("FUSION_PAGED_FUSED_KERNEL", "off") != "on":
        return False
    return mx.metal.is_available()


@cache
def _make_paged_decode_attention_kernel():
    if not mx.metal.is_available():
        return None

    source = r"""
        const uint bh = thread_position_in_grid.x;
        const uint batch = bh / N_HEADS;
        const uint q_head = bh % N_HEADS;
        const uint kv_head = q_head / GQA_FACTOR;

        float m = -1e30f;
        float l = 0.0f;
        float o[HEAD_DIM];
        for (uint d = 0; d < HEAD_DIM; ++d) o[d] = 0.0f;

        const uint num_blocks = (NUM_KV + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (uint lb = 0; lb < num_blocks; ++lb) {
          uint pb = block_table[lb];
          uint block_len = (lb + 1 == num_blocks) ? (NUM_KV - lb * BLOCK_SIZE) : BLOCK_SIZE;
          for (uint t = 0; t < block_len; ++t) {
            float s = 0.0f;
            for (uint d = 0; d < HEAD_DIM; ++d) {
              s += q[batch * N_HEADS * HEAD_DIM + q_head * HEAD_DIM + d]
                   * keys_pool[((pb * B + batch) * N_KV_HEADS + kv_head) * BLOCK_SIZE * HEAD_DIM
                               + t * HEAD_DIM + d];
            }
            float m_new = metal::max(m, s);
            float exp_m = metal::exp(m - m_new);
            float exp_s = metal::exp(s - m_new);
            l = l * exp_m + exp_s;
            for (uint d = 0; d < HEAD_DIM; ++d) {
              o[d] = o[d] * exp_m
                   + exp_s * values_pool[((pb * B + batch) * N_KV_HEADS + kv_head) * BLOCK_SIZE * HEAD_DIM
                                         + t * HEAD_DIM + d];
            }
            m = m_new;
          }
        }
        for (uint d = 0; d < HEAD_DIM; ++d) {
          out[batch * N_HEADS * HEAD_DIM + q_head * HEAD_DIM + d] = o[d] / l;
        }
    """

    return mx.fast.metal_kernel(
        name="fusion_paged_decode_attention",
        input_names=["q", "keys_pool", "values_pool", "block_table"],
        output_names=["out"],
        source=source,
    )


_logged_compile = False


def paged_decode_attention(
    q,
    keys_pool,
    values_pool,
    block_table,
    num_kv,
    scale,
    gqa_factor,
    stream=None,
):
    if not metal_available():
        logger.debug(
            "paged fused decode kernel unavailable (FUSION_PAGED_FUSED_KERNEL=%s)",
            os.environ.get("FUSION_PAGED_FUSED_KERNEL", "off"),
        )
        return None

    kernel = _make_paged_decode_attention_kernel()
    if kernel is None:
        return None

    B = q.shape[0]
    n_heads = q.shape[1]
    head_dim = q.shape[3]
    n_kv_heads = keys_pool.shape[2]
    block_size = keys_pool.shape[3]

    q_scaled = (q.astype(mx.float32) * float(scale)).astype(q.dtype)

    global _logged_compile
    if not _logged_compile:
        logger.info("paged fused decode kernel: grid=(%d) compiled", B * n_heads)
        _logged_compile = True

    out = kernel(
        inputs=[q_scaled, keys_pool, values_pool, block_table],
        template=[
            ("BLOCK_SIZE", block_size),
            ("HEAD_DIM", head_dim),
            ("GQA_FACTOR", gqa_factor),
            ("NUM_KV", num_kv),
            ("N_HEADS", n_heads),
            ("N_KV_HEADS", n_kv_heads),
            ("B", B),
        ],
        grid=(B * n_heads, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(B, n_heads, 1, head_dim)],
        output_dtypes=[q.dtype],
        init_value=0,
        stream=stream or mx.gpu,
    )
    return out[0]
