# SPDX-License-Identifier: Apache-2.0
"""Fused INT4 Dequant + GEMV Metal kernel (PRD stage-2 base layer).

Custom Metal kernel that reads packed uint32 int4 weights, unpacks 8
nibbles per uint32 (little-endian, low nibble first), dequantizes affine
(val = scale*nib + bias, per group), and accumulates the dot product with x
in registers — zero intermediate buffer.

Packed layout (verified parity 2026-09-21 vs mx.quantized_matmul, max diff
4.8e-7): weight (M, K//8) uint32, scales/biases (M, n_groups) float32.

PRODUCTION STATUS (honest, measured 2026-09-21, M-series, MLX 0.31.2):
  The custom kernel is SLOWER than MLX native mx.quantized_matmul across 8
  optimization variants (+6% to +246%). Native quantized_matmul is the
  production path — already default-ON via nn.QuantizedLinear. This module
  is an OPT-IN alternative (FUSION_FUSED_QUANT_GEMV=1) kept as:
    1. Base-layer Metal capability demonstration (correct int4 kernel)
    2. Foundation for future M5 NAx / heterogeneous CPU+GPU work (PRD st.3)
    3. Fusion experiments (dequant+matmul+bias in one kernel)

When FUSION_FUSED_QUANT_GEMV is unset (default), fused_dequant_gemv_int4
delegates to mx.quantized_matmul — zero behavior change, zero regression.

Degrade switch: FUSION_FUSED_QUANT_GEMV (default "0" = use native).
  "1" = use custom Metal kernel (slower today; for experimentation).
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)

_FUSED_QUANT_GEMV_KERNEL = None
_TG = 256
_TILE_M = 8


def is_fused_quant_gemv_enabled() -> bool:
    return os.environ.get("FUSION_FUSED_QUANT_GEMV", "0") == "1"


def _get_kernel():
    global _FUSED_QUANT_GEMV_KERNEL
    if _FUSED_QUANT_GEMV_KERNEL is not None:
        return _FUSED_QUANT_GEMV_KERNEL
    _FUSED_QUANT_GEMV_KERNEL = mx.fast.metal_kernel(
        name="fused_dequant_gemv_int4",
        input_names=["w", "scales", "biases", "x", "meta"],
        output_names=["out"],
        source=_METAL_SOURCE,
        header="",
    )
    logger.info("fused_dequant_gemv_int4 Metal kernel compiled")
    return _FUSED_QUANT_GEMV_KERNEL


def fused_dequant_gemv_int4(
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    x: mx.array,
    group_size: int = 128,
    bits: int = 4,
) -> mx.array:
    """Fused INT4 dequant + GEMV.

    weight: (M, K//8) uint32 packed int4. x: (K,) or (1, K). out: (M,).
    When FUSION_FUSED_QUANT_GEMV != "1", delegates to mx.quantized_matmul.
    batch>1 or bits!=4 delegates to native (compute-bound, native GEMM wins).
    K>4096 delegates to native (shared-x buffer exceeds threadgroup mem).
    """
    if not is_fused_quant_gemv_enabled() or bits != 4:
        return _native_quant_matmul(weight, scales, biases, x, group_size, bits)
    M = weight.shape[0]
    n_groups = scales.shape[1]
    K = n_groups * group_size
    if x.ndim == 2:
        if x.shape[0] != 1:
            return _native_quant_matmul(weight, scales, biases, x, group_size, bits)
        x = x[0]
    if x.shape[0] != K or K > 4096:
        return _native_quant_matmul(weight, scales, biases, x, group_size, bits)
    x = x.astype(mx.float32)
    meta = mx.array([float(M), float(K), float(group_size), float(_TILE_M)], mx.float32)
    kernel = _get_kernel()
    nb = (M + _TILE_M - 1) // _TILE_M
    out = kernel(
        inputs=[weight, scales, biases, x, meta],
        grid=(nb * _TG, 1, 1),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(M,)],
        output_dtypes=[mx.float32],
    )[0]
    return out


def _native_quant_matmul(weight, scales, biases, x, group_size, bits):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    out = mx.quantized_matmul(
        x,
        weight,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )
    return out[0] if out.shape[0] == 1 else out


_METAL_SOURCE = """
uint M = uint(meta[0]);
uint K = uint(meta[1]);
uint gs = uint(meta[2]);
uint TILE_M = uint(meta[3]);
uint n_groups = K / gs;
uint K_u32 = K / 8;
uint blk = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint nthreads = threads_per_threadgroup.x;
uint m_base = blk * TILE_M;
if (m_base >= M) return;
threadgroup float xs[4096];
for (uint k = tid; k < K; k += nthreads) xs[k] = x[k];
threadgroup_barrier(mem_flags::mem_threadgroup);
float partial[8];
for (uint t = 0; t < 8; t++) partial[t] = 0.0f;
for (uint ui = tid; ui < K_u32; ui += nthreads) {
    uint kb = ui * 8;
    float xv0 = xs[kb], xv1 = xs[kb+1], xv2 = xs[kb+2], xv3 = xs[kb+3];
    float xv4 = xs[kb+4], xv5 = xs[kb+5], xv6 = xs[kb+6], xv7 = xs[kb+7];
    for (uint t = 0; t < TILE_M; t++) {
        uint m = m_base + t;
        if (m >= M) break;
        uint pack = w[m * K_u32 + ui];
        uint g0 = kb / gs;
        uint g1 = (kb + 7) / gs;
        const device float* s_row = scales + m * n_groups;
        const device float* b_row = biases + m * n_groups;
        if (g0 == g1) {
            float sc = s_row[g0], bi = b_row[g0];
            partial[t] += (sc * float(pack & 0xF) + bi) * xv0;
            partial[t] += (sc * float((pack >> 4) & 0xF) + bi) * xv1;
            partial[t] += (sc * float((pack >> 8) & 0xF) + bi) * xv2;
            partial[t] += (sc * float((pack >> 12) & 0xF) + bi) * xv3;
            partial[t] += (sc * float((pack >> 16) & 0xF) + bi) * xv4;
            partial[t] += (sc * float((pack >> 20) & 0xF) + bi) * xv5;
            partial[t] += (sc * float((pack >> 24) & 0xF) + bi) * xv6;
            partial[t] += (sc * float((pack >> 28) & 0xF) + bi) * xv7;
        } else {
            partial[t] += (s_row[kb/gs] * float(pack & 0xF) + b_row[kb/gs]) * xv0;
            partial[t] += (s_row[(kb+1)/gs] * float((pack >> 4) & 0xF) + b_row[(kb+1)/gs]) * xv1;
            partial[t] += (s_row[(kb+2)/gs] * float((pack >> 8) & 0xF) + b_row[(kb+2)/gs]) * xv2;
            partial[t] += (s_row[(kb+3)/gs] * float((pack >> 12) & 0xF) + b_row[(kb+3)/gs]) * xv3;
            partial[t] += (s_row[(kb+4)/gs] * float((pack >> 16) & 0xF) + b_row[(kb+4)/gs]) * xv4;
            partial[t] += (s_row[(kb+5)/gs] * float((pack >> 20) & 0xF) + b_row[(kb+5)/gs]) * xv5;
            partial[t] += (s_row[(kb+6)/gs] * float((pack >> 24) & 0xF) + b_row[(kb+6)/gs]) * xv6;
            partial[t] += (s_row[(kb+7)/gs] * float((pack >> 28) & 0xF) + b_row[(kb+7)/gs]) * xv7;
        }
    }
}
threadgroup float red[8];
for (uint t = 0; t < TILE_M; t++) {
    float sg = simd_sum(partial[t]);
    ushort sgid = tid / 32, lane = tid % 32, nsimd = nthreads / 32;
    if (tid < 8) red[tid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0 && sgid < nsimd && sgid < 8) red[sgid] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0 && lane < nsimd && lane < 8) {
        float v = simd_sum(red[lane]);
        if (lane == 0) { uint m = m_base + t; if (m < M) out[m] = v; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""

__all__ = ["fused_dequant_gemv_int4", "is_fused_quant_gemv_enabled"]
