# SPDX-License-Identifier: Apache-2.0
# Fused paged decode-attention Metal kernel (Phase 2/3).
# One threadgroup per (batch, query_head) reads the paged physical KV pool
# via block_table indirection and runs online softmax (FlashAttention-2).
# Task 3 (#772): tiled rewrite — cooperative threadgroup shared-mem staging
# of K/V blocks, thread 0 runs the online-softmax accumulation over
# shared-mem reads (provably bit-equivalent to the scalar kernel numerics).
# Correctness-only; no perf claim. Fallback to scalar kernel when
# BLOCK_SIZE * HEAD_DIM exceeds a shared-mem-safe threshold.

from __future__ import annotations

import logging
import os
from functools import cache

import mlx.core as mx

logger = logging.getLogger(__name__)

# Shared-mem staging threshold (floats). If BLOCK_SIZE * HEAD_DIM exceeds this,
# fall back to the scalar kernel path to avoid oversubscribing threadgroup mem.
_TILE_SHARED_MEM_LIMIT = 8192

# Fixed threadgroup size for the tiled kernel.
_TILE_THREADS = 64


def metal_available() -> bool:
    if os.environ.get("FUSION_PAGED_FUSED_KERNEL", "off") != "on":
        return False
    return mx.metal.is_available()


@cache
def _make_paged_decode_attention_kernel_scalar():
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
        float sc = float(softcap[0]);

        const uint num_kv = kv_lens[batch];
        const uint num_blocks = (num_kv + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (uint lb = 0; lb < num_blocks; ++lb) {
          uint pb = block_table[batch * MAX_BLOCKS + lb];
          uint block_len = (lb + 1 == num_blocks) ? (num_kv - lb * BLOCK_SIZE) : BLOCK_SIZE;
          for (uint t = 0; t < block_len; ++t) {
            uint kv_pos = lb * BLOCK_SIZE + t;
            if (SLIDING_WINDOW > 0) {
              if (kv_pos + SLIDING_WINDOW < num_kv) {
                continue;
              }
            }
            float s = 0.0f;
            for (uint d = 0; d < HEAD_DIM; ++d) {
              s += float(q[batch * N_HEADS * HEAD_DIM + q_head * HEAD_DIM + d])
                   * float(keys_pool[((pb * B + batch) * N_KV_HEADS + kv_head) * BLOCK_SIZE * HEAD_DIM
                               + t * HEAD_DIM + d]);
            }
            if (sc > 0.0f) {
              s = metal::tanh(s / sc) * sc;
            }
            float m_new = metal::max(m, s);
            float exp_m = metal::exp(m - m_new);
            float exp_s = metal::exp(s - m_new);
            l = l * exp_m + exp_s;
            for (uint d = 0; d < HEAD_DIM; ++d) {
              o[d] = o[d] * exp_m
                   + exp_s * float(values_pool[((pb * B + batch) * N_KV_HEADS + kv_head) * BLOCK_SIZE * HEAD_DIM
                                         + t * HEAD_DIM + d]);
            }
            m = m_new;
          }
        }
        for (uint d = 0; d < HEAD_DIM; ++d) {
          out[batch * N_HEADS * HEAD_DIM + q_head * HEAD_DIM + d] = o[d] / l;
        }
    """

    return mx.fast.metal_kernel(
        name="fusion_paged_decode_attention_scalar",
        input_names=[
            "q",
            "keys_pool",
            "values_pool",
            "block_table",
            "kv_lens",
            "softcap",
        ],
        output_names=["out"],
        source=source,
    )


@cache
def _make_paged_decode_attention_kernel_tiled():
    if not mx.metal.is_available():
        return None

    # Tiled kernel: grid=(B*n_heads*THREADS,1,1), threadgroup=(THREADS,1,1).
    # One threadgroup owns one (batch, q_head). Threads cooperatively load
    # the current K/V block into threadgroup shared memory (strided loop),
    # barrier, then thread 0 runs the online-softmax accumulation over
    # shared-mem reads — identical float-op order to the scalar kernel,
    # so numerics are bit-equivalent. The cooperative load is where the
    # threadgroup parallelism is used; the softmax math stays single-thread.
    source = r"""
        const uint bh = threadgroup_position_in_grid.x;
        const uint tid = thread_position_in_threadgroup.x;
        const uint batch = bh / N_HEADS;
        const uint q_head = bh % N_HEADS;
        const uint kv_head = q_head / GQA_FACTOR;

        threadgroup float k_tile[BLOCK_SIZE * HEAD_DIM];
        threadgroup float v_tile[BLOCK_SIZE * HEAD_DIM];

        float m = -1e30f;
        float l = 0.0f;
        float o[HEAD_DIM];
        if (tid == 0) {
          for (uint d = 0; d < HEAD_DIM; ++d) o[d] = 0.0f;
        }
        float sc = float(softcap[0]);

        const uint num_kv = kv_lens[batch];
        const uint num_blocks = (num_kv + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (uint lb = 0; lb < num_blocks; ++lb) {
          uint pb = block_table[batch * MAX_BLOCKS + lb];
          uint block_len = (lb + 1 == num_blocks) ? (num_kv - lb * BLOCK_SIZE) : BLOCK_SIZE;

          // Cooperative load of K and V block into threadgroup shared mem.
          const uint tile_elems = BLOCK_SIZE * HEAD_DIM;
          for (uint i = tid; i < tile_elems; i += THREADS) {
            uint t = i / HEAD_DIM;
            uint d = i % HEAD_DIM;
            if (t < block_len) {
              k_tile[i] = float(keys_pool[((pb * B + batch) * N_KV_HEADS + kv_head) * BLOCK_SIZE * HEAD_DIM
                                  + t * HEAD_DIM + d]);
              v_tile[i] = float(values_pool[((pb * B + batch) * N_KV_HEADS + kv_head) * BLOCK_SIZE * HEAD_DIM
                                    + t * HEAD_DIM + d]);
            } else {
              k_tile[i] = 0.0f;
              v_tile[i] = 0.0f;
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          // Thread 0 runs the online-softmax accumulation over shared-mem
          // reads — same float-op order as the scalar kernel.
          if (tid == 0) {
            for (uint t = 0; t < block_len; ++t) {
              uint kv_pos = lb * BLOCK_SIZE + t;
              if (SLIDING_WINDOW > 0) {
                if (kv_pos + SLIDING_WINDOW < num_kv) {
                  continue;
                }
              }
              float s = 0.0f;
              for (uint d = 0; d < HEAD_DIM; ++d) {
                s += float(q[batch * N_HEADS * HEAD_DIM + q_head * HEAD_DIM + d])
                     * k_tile[t * HEAD_DIM + d];
              }
              if (sc > 0.0f) {
                s = metal::tanh(s / sc) * sc;
              }
              float m_new = metal::max(m, s);
              float exp_m = metal::exp(m - m_new);
              float exp_s = metal::exp(s - m_new);
              l = l * exp_m + exp_s;
              for (uint d = 0; d < HEAD_DIM; ++d) {
                o[d] = o[d] * exp_m + exp_s * v_tile[t * HEAD_DIM + d];
              }
              m = m_new;
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
          for (uint d = 0; d < HEAD_DIM; ++d) {
            out[batch * N_HEADS * HEAD_DIM + q_head * HEAD_DIM + d] = o[d] / l;
          }
        }
    """

    return mx.fast.metal_kernel(
        name="fusion_paged_decode_attention_tiled",
        input_names=[
            "q",
            "keys_pool",
            "values_pool",
            "block_table",
            "kv_lens",
            "softcap",
        ],
        output_names=["out"],
        source=source,
    )


import threading as _threading

_logged_compile = False
_logged_compile_lock = _threading.Lock()


def paged_decode_attention(
    q,
    keys_pool,
    values_pool,
    block_table,
    num_kv,
    scale,
    gqa_factor,
    sliding_window=0,
    softcap=0.0,
    stream=None,
):
    if not metal_available():
        logger.debug(
            "paged fused decode kernel unavailable (FUSION_PAGED_FUSED_KERNEL=%s)",
            os.environ.get("FUSION_PAGED_FUSED_KERNEL", "off"),
        )
        return None

    B = q.shape[0]
    n_heads = q.shape[1]
    head_dim = q.shape[3]
    n_kv_heads = keys_pool.shape[2]
    block_size = keys_pool.shape[3]

    q_scaled = q.astype(mx.float32) * float(scale)
    sw = int(sliding_window)
    sc = float(softcap)
    softcap_arr = mx.array([sc], dtype=mx.float32)

    if isinstance(num_kv, mx.array):
        kv_lens_arr = num_kv.astype(mx.uint32)
        if kv_lens_arr.shape[0] != B:
            logger.warning(
                "paged fused decode: kv_lens len=%d != B=%d, broadcasting",
                kv_lens_arr.shape[0],
                B,
            )
            kv_lens_arr = mx.broadcast_to(kv_lens_arr[:1], (B,)).astype(mx.uint32)
        max_kv = int(mx.max(kv_lens_arr).item())
    else:
        max_kv = int(num_kv)
        kv_lens_arr = mx.array([max_kv] * B, dtype=mx.uint32)

    if block_table.ndim == 1:
        max_blocks = block_table.shape[0]
        bt_2d = mx.broadcast_to(
            block_table.reshape(1, max_blocks), (B, max_blocks)
        ).reshape(B * max_blocks)
    else:
        max_blocks = block_table.shape[1]
        bt_2d = block_table.reshape(B * max_blocks)
    bt_mx = bt_2d.astype(mx.uint32)

    tile_elems = block_size * head_dim
    use_tiled = tile_elems <= _TILE_SHARED_MEM_LIMIT

    global _logged_compile
    # D16 (audit): guard the one-shot compile log with a lock so two
    # concurrent first-callers don't both log.
    with _logged_compile_lock:
        if _logged_compile:
            already_logged = True
        else:
            _logged_compile = True
            already_logged = False
    if not already_logged:
        if use_tiled:
            logger.info(
                "paged fused decode kernel: tiled path grid=(%d) threadgroup=(%d) "
                "block_size=%d head_dim=%d tile_elems=%d sw=%d softcap=%s "
                "B=%d max_kv=%d max_blocks=%d",
                B * n_heads,
                _TILE_THREADS,
                block_size,
                head_dim,
                tile_elems,
                sw,
                sc,
                B,
                max_kv,
                max_blocks,
            )
        else:
            logger.info(
                "paged fused decode kernel: scalar fallback grid=(%d) "
                "block_size=%d head_dim=%d tile_elems=%d > limit=%d sw=%d softcap=%s "
                "B=%d max_kv=%d max_blocks=%d",
                B * n_heads,
                block_size,
                head_dim,
                tile_elems,
                _TILE_SHARED_MEM_LIMIT,
                sw,
                sc,
                B,
                max_kv,
                max_blocks,
            )

    if use_tiled:
        kernel = _make_paged_decode_attention_kernel_tiled()
        if kernel is None:
            return None
        out = kernel(
            inputs=[q_scaled, keys_pool, values_pool, bt_mx, kv_lens_arr, softcap_arr],
            template=[
                ("BLOCK_SIZE", block_size),
                ("HEAD_DIM", head_dim),
                ("GQA_FACTOR", gqa_factor),
                ("MAX_BLOCKS", max_blocks),
                ("N_HEADS", n_heads),
                ("N_KV_HEADS", n_kv_heads),
                ("B", B),
                ("SLIDING_WINDOW", sw),
                ("THREADS", _TILE_THREADS),
            ],
            grid=(B * n_heads * _TILE_THREADS, 1, 1),
            threadgroup=(_TILE_THREADS, 1, 1),
            output_shapes=[(B, n_heads, 1, head_dim)],
            output_dtypes=[mx.float32],
            init_value=0,
            stream=stream,
        )
    else:
        kernel = _make_paged_decode_attention_kernel_scalar()
        if kernel is None:
            return None
        out = kernel(
            inputs=[q_scaled, keys_pool, values_pool, bt_mx, kv_lens_arr, softcap_arr],
            template=[
                ("BLOCK_SIZE", block_size),
                ("HEAD_DIM", head_dim),
                ("GQA_FACTOR", gqa_factor),
                ("MAX_BLOCKS", max_blocks),
                ("N_HEADS", n_heads),
                ("N_KV_HEADS", n_kv_heads),
                ("B", B),
                ("SLIDING_WINDOW", sw),
            ],
            grid=(B * n_heads, 1, 1),
            threadgroup=(1, 1, 1),
            output_shapes=[(B, n_heads, 1, head_dim)],
            output_dtypes=[mx.float32],
            init_value=0,
            stream=stream,
        )
    return out[0].astype(q.dtype)
