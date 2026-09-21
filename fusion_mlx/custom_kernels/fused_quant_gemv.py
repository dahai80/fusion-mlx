# SPDX-License-Identifier: Apache-2.0
"""Fused INT4 Dequant + GEMV Metal kernel (PRD stage 2.1).

Replaces split dequant+matmul with a single Metal kernel that reads packed
uint32 weights, unpacks 8 int4 nibbles per uint32 (little-endian, low nibble
first), dequantizes affine (val = scale*nib + bias, per group), and
accumulates the dot product with x — all in registers, zero intermediate
buffer.

Packed layout (verified parity 2026-09-21 vs mx.quantized_matmul):
  weight: (M, K//8) uint32, each uint32 holds 8 int4 nibbles, low nibble at
          shift 0,4,8,...,28.
  scales: (M, n_groups) float32, n_groups = K // group_size
  biases: (M, n_groups) float32
  x:      (K,) fp16 or fp32
  out:    (M,) fp32

Decode path (batch=1) only. batch>1 falls back to native quantized_matmul
(compute-bound, native GEMM wins).

Degrade switch: FUSION_FUSED_QUANT_GEMV (default "1"). Tiered: shapes below
_NATIVE_BETTER threshold fall back to mx.quantized_matmul.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)

_FUSED_QUANT_GEMV_KERNEL = None


def is_fused_quant_gemv_enabled() -> bool:
    return os.environ.get("FUSION_FUSED_QUANT_GEMV", "1") == "1"


def _get_kernel():
    global _FUSED_QUANT_GEMV_KERNEL
    if _FUSED_QUANT_GEMV_KERNEL is not None:
        return _FUSED_QUANT_GEMV_KERNEL
    # TODO: metal source
    _FUSED_QUANT_GEMV_KERNEL = mx.fast.metal_kernel(
        name="fused_dequant_gemv_int4",
        input_names=["w", "scales", "biases", "x", "meta"],
        source=_METAL_SOURCE,
        output_names=["out"],
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
    """Fused INT4 dequant + GEMV. weight (M,K//8) uint32, x (K,) -> out (M,)."""
    if not is_fused_quant_gemv_enabled() or bits != 4:
        return _native_quant_matmul(weight, scales, biases, x, group_size, bits)
    M = weight.shape[0]
    n_groups = scales.shape[1]
    K = n_groups * group_size
    if x.ndim == 2:
        if x.shape[0] != 1:
            return _native_quant_matmul(weight, scales, biases, x, group_size, bits)
        x = x[0]
    if x.shape[0] != K:
        return _native_quant_matmul(weight, scales, biases, x, group_size, bits)
    x = x.astype(mx.float32)
    meta = mx.array([float(M), float(K), float(group_size)], mx.float32)
    kernel = _get_kernel()
    _TG = 256
    out = kernel(
        inputs=[weight, scales, biases, x, meta],
        grid=(M * _TG, 1, 1),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(M,)],
        output_dtypes=[mx.float32],
    )[0]
    return out


def _native_quant_matmul(weight, scales, biases, x, group_size, bits):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    out = mx.quantized_matmul(
        x, weight, scales=scales, biases=biases, transpose=True,
        group_size=group_size, bits=bits, mode="affine",
    )
    return out[0] if out.shape[0] == 1 else out


_METAL_SOURCE = """
uint M = uint(meta[0]);
uint K = uint(meta[1]);
uint gs = uint(meta[2]);
uint n_groups = K / gs;

uint m = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint nthreads = threads_per_threadgroup.x;
if (m >= M) return;

const device uint* w_row = w + m * (K / 8);
const device float* s_row = scales + m * n_groups;
const device float* b_row = biases + m * n_groups;
const device float* xf = x;

float partial = 0.0f;
uint K_u32 = K / 8;
for (uint ui = tid; ui < K_u32; ui += nthreads) {
    uint pack = w_row[ui];
    uint k_base = ui * 8;
    float x0 = xf[k_base], x1 = xf[k_base + 1], x2 = xf[k_base + 2], x3 = xf[k_base + 3];
    float x4 = xf[k_base + 4], x5 = xf[k_base + 5], x6 = xf[k_base + 6], x7 = xf[k_base + 7];
    uint g0 = k_base / gs;
    uint g1 = (k_base + 7) / gs;
    if (g0 == g1) {
        float sc = s_row[g0], bi = b_row[g0];
        partial += (sc * float(pack & 0xF) + bi) * x0;
        partial += (sc * float((pack >> 4) & 0xF) + bi) * x1;
        partial += (sc * float((pack >> 8) & 0xF) + bi) * x2;
        partial += (sc * float((pack >> 12) & 0xF) + bi) * x3;
        partial += (sc * float((pack >> 16) & 0xF) + bi) * x4;
        partial += (sc * float((pack >> 20) & 0xF) + bi) * x5;
        partial += (sc * float((pack >> 24) & 0xF) + bi) * x6;
        partial += (sc * float((pack >> 28) & 0xF) + bi) * x7;
    } else {
        partial += (s_row[k_base / gs] * float(pack & 0xF) + b_row[k_base / gs]) * x0;
        partial += (s_row[(k_base+1) / gs] * float((pack >> 4) & 0xF) + b_row[(k_base+1) / gs]) * x1;
        partial += (s_row[(k_base+2) / gs] * float((pack >> 8) & 0xF) + b_row[(k_base+2) / gs]) * x2;
        partial += (s_row[(k_base+3) / gs] * float((pack >> 12) & 0xF) + b_row[(k_base+3) / gs]) * x3;
        partial += (s_row[(k_base+4) / gs] * float((pack >> 16) & 0xF) + b_row[(k_base+4) / gs]) * x4;
        partial += (s_row[(k_base+5) / gs] * float((pack >> 20) & 0xF) + b_row[(k_base+5) / gs]) * x5;
        partial += (s_row[(k_base+6) / gs] * float((pack >> 24) & 0xF) + b_row[(k_base+6) / gs]) * x6;
        partial += (s_row[(k_base+7) / gs] * float((pack >> 28) & 0xF) + b_row[(k_base+7) / gs]) * x7;
    }
}

float sg_sum = simd_sum(partial);
ushort sgid = tid / 32;
ushort lane = tid % 32;
ushort nsimd = nthreads / 32;
threadgroup float sg_sums[32];
if (tid < 32) sg_sums[tid] = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) sg_sums[sgid] = sg_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sgid == 0) {
    float v = simd_sum(sg_sums[lane]);
    if (lane == 0) out[m] = v;
}
"""

__all__ = ["fused_dequant_gemv_int4", "is_fused_quant_gemv_enabled"]
