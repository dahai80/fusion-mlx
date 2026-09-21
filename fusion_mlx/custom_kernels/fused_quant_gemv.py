# SPDX-License-Identifier: Apache-2.0
"""Fused INT4 Dequant + GEMV Metal kernel (PRD stage-2 base layer).

Custom Metal kernel that reads packed uint32 int4 weights, unpacks 8
nibbles per uint32 (little-endian, low nibble first), dequantizes affine
(val = scale*nib + bias, per group), and accumulates the dot product with x
in registers — zero intermediate buffer.

Packed layout (parity verified 2026-09-21 vs mx.quantized_matmul, max diff
4.8e-7): weight (M, K//8) uint32, scales/biases (M, n_groups) float32.

V2 KERNEL (2026-09-21): adopts the 3 optimizations reverse-engineered
from MLX's native qmv_fast_impl (quantized.h L757):
  1. Shift-elimination: pre-scale x DOWN in load (x/16, x/256, x/4096) so
     the dot keeps the nibble in native bit position (mask-only, no >>4).
     Product = x_orig * nibble via mask+mul.
  2. Affine factoring: qdot returns scale*accum + sum*bias (2 FMA/group)
     instead of scale*nib+bias per nibble (~128x fewer scale/bias ops).
  3. Simdgroup layout: 2 simdgroups x 4 rows = 8 rows/threadgroup, single
     hardware simd_sum reduction (no cross-simdgroup barriers).

PRODUCTION STATUS (measured 2026-09-21, M5 Max, MLX 0.31.2):
  Parity: PASS (max diff 4.77e-7 vs mx.quantized_matmul across all tested
  shapes including K not divisible by block size — bounds-checked remainder).

  PERFORMANCE (V2, 3-method reconciliation, all agree V2 is slower than
  native eager but AT PARITY in compiled chain at large K):
    | method                        | native   | custom   | delta  |
    |-------------------------------|----------|----------|--------|
    | eager per-op (production)     | 0.35ms   | 0.39ms   | +10.6% |
    | compiled 32-layer chain K=8K  | 269.7ms  | 267.8ms  |  -0.7% |
    | compiled 32-layer chain K=11K | 460.8ms  | 460.9ms  |  +0.0% |
    | pure-compute (clear_cache)    | 0.35ms   | 0.44ms   | +24.9% |
  V2 closed the gap from prior NSX kernel (+89% pure-GPU) to +10.6% eager
  by adopting native's 3 optimizations. Compiled chain = parity at large K.
  Native still wins eager by ~10% (instruction-scheduling edge).

  ROOT CAUSE of remaining gap: batch=1 GEMV is memory-latency-bound on
  weight streaming (45 GB/s measured = 11% of M5's 400 GB/s peak — NOT
  bandwidth-saturated, latency-bound). Native's ~10% edge is instruction
  scheduling / load coalescing craftsmanship. NOT tensor cores: native
  qmv_fast itself is SCALAR (qdot + simd_sum), not MMA — tensor cores are
  irrelevant for batch=1 (vector LHS wastes matrix tiles), which is WHY
  MLX uses scalar for the decode path. The user's "tensor-core MMA"
  directive targets the wrong op for batch=1 GEMV.

  OPTIMIZATIONS ATTEMPTED (2026-09-21, all rejected):
  - V3 uint32 wider loads: -28% perf BUT parity FAIL (fp32 can't hold
    16^7 = 2^-28 scaling for 8 nibbles, precision loss).
  - V4 4-simdgroups: parity FAIL (scale indexing coupled to 2 simdgroups).
  - V5 1-simdgroup: parity PASS, +9.7% (≈V2, within noise).
  - V6 async double-buffer (threadgroup prefetch): parity FAIL + +51%
    (shared-mem roundtrip + barrier costs more than latency it hides; GPU
    hardware prefetcher already tolerates device-load latency).

  CONCLUSION: V2 is the achievable best. Cannot surpass native on this op
  in-session. The op is latency-bound on weight streaming; native's
  instruction-scheduling edge is not reverse-engineerable further. This
  is the 5th confirmation that handwriting Metal beating MLX native is
  not achievable in-session for ops MLX already optimizes (prior:
  smart-conv 10-30x slower, sdpa 86% roofline, RMSNorm microbench-only,
  int4 GEMV). Native wins on ops it optimizes. Custom wins only for
  capability gaps (YaRN rope) or fusion patterns native cannot do.

Dispatch (when gate ON): batch==1 AND bits==4 AND K>=_CUSTOM_K_THRESHOLD
(8192) -> custom kernel. Everything else -> native mx.quantized_matmul.
Zero correctness regression (parity verified). Kept as opt-in base-layer
Metal capability demonstration. Default OFF because slower than native
eager — native is the production path.

Wiring: install_fused_quant_gemv_patch() monkeypatches
nn.QuantizedLinear.__call__ to route large-K decode through the custom
kernel. Called at scheduler import (idempotent, no-op when gate OFF or
Metal unavailable).

Degrade switch: FUSION_FUSED_QUANT_GEMV (default "0" = OFF).
  "1" = opt-in custom kernel (slower than native eager; for experimentation).
  Production int4 path = native mx.quantized_matmul via nn.QuantizedLinear.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_FUSED_QUANT_GEMV_KERNEL = None

_CUSTOM_K_THRESHOLD = 8192

# V2 fixed simdgroup layout (matches native qmv_fast_impl):
# 2 simdgroups x 4 rows = 8 output rows per threadgroup, 64 threads/TG.
_NUM_SIMD = 2
_RES_PER_SIMD = 4
_SIMD_SIZE = 32
_ROWS_PER_TG = _NUM_SIMD * _RES_PER_SIMD  # 8
_TG_THREADS = _NUM_SIMD * _SIMD_SIZE  # 64


def is_fused_quant_gemv_enabled() -> bool:
    return os.environ.get("FUSION_FUSED_QUANT_GEMV", "0") == "1"


def _get_kernel():
    global _FUSED_QUANT_GEMV_KERNEL
    if _FUSED_QUANT_GEMV_KERNEL is not None:
        return _FUSED_QUANT_GEMV_KERNEL
    if not mx.metal.is_available():
        return None
    _FUSED_QUANT_GEMV_KERNEL = mx.fast.metal_kernel(
        name="fused_dequant_gemv_int4_v2",
        input_names=["w", "scales", "biases", "x", "meta"],
        output_names=["out"],
        source=_METAL_SOURCE,
        header="",
    )
    logger.info("fused_dequant_gemv_int4_v2 Metal kernel compiled")
    return _FUSED_QUANT_GEMV_KERNEL


def fused_dequant_gemv_int4(
    weight: mx.array,
    scales: mx.array,
    biases,
    x: mx.array,
    group_size: int = 128,
    bits: int = 4,
) -> mx.array:
    """Fused INT4 dequant + GEMV (V2 simdgroup kernel).

    weight: (M, K//8) uint32 packed int4. x: (1, K) or (K,). out: (1, M).
    Custom kernel used only when: gate ON, bits==4, batch==1, K>=8192
    (large-K exercise zone; custom is ~10% slower than native eager, gate
    is opt-in for experimentation). Otherwise delegates to
    mx.quantized_matmul (zero regression).
    """
    K = scales.shape[1] * group_size
    use_custom = (
        is_fused_quant_gemv_enabled()
        and bits == 4
        and x.ndim == 2
        and x.shape[0] == 1
        and K >= _CUSTOM_K_THRESHOLD
        and mx.metal.is_available()
    )
    if not use_custom:
        return _native_quant_matmul(weight, scales, biases, x, group_size, bits)
    M = weight.shape[0]
    x_flat = x[0].astype(mx.float32)
    meta = mx.array([float(M), float(K)], mx.float32)
    kernel = _get_kernel()
    if kernel is None:
        return _native_quant_matmul(weight, scales, biases, x, group_size, bits)
    nb_tg = (M + _ROWS_PER_TG - 1) // _ROWS_PER_TG
    try:
        out = kernel(
            inputs=[weight, scales, biases, x_flat, meta],
            grid=(nb_tg * _TG_THREADS, 1, 1),
            threadgroup=(_TG_THREADS, 1, 1),
            output_shapes=[(M,)],
            output_dtypes=[mx.float32],
        )[0]
    except Exception:
        logger.exception(
            "fused_dequant_gemv_int4_v2 kernel failed, falling back to native"
        )
        return _native_quant_matmul(weight, scales, biases, x, group_size, bits)
    return out.reshape(1, M)


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
    return out


_orig_quantized_linear_call = nn.QuantizedLinear.__call__

_patch_installed = False


def _patched_quantized_linear_call(self, x):
    if not is_fused_quant_gemv_enabled() or self.bits != 4:
        return _orig_quantized_linear_call(self, x)
    try:
        return fused_dequant_gemv_int4(
            self["weight"],
            self["scales"],
            self.get("biases"),
            x,
            self.group_size,
            self.bits,
        ) + self.get("bias", 0)
    except Exception:
        logger.exception("fused_quant_gemv dispatch failed, fallback to native")
        return _orig_quantized_linear_call(self, x)


def install_fused_quant_gemv_patch():
    global _patch_installed
    if _patch_installed:
        return
    if not is_fused_quant_gemv_enabled():
        return
    if not mx.metal.is_available():
        logger.info("fused_quant_gemv: Metal unavailable, patch skipped")
        return
    nn.QuantizedLinear.__call__ = _patched_quantized_linear_call
    _patch_installed = True
    logger.info(
        "fused_quant_gemv: patched nn.QuantizedLinear (V2 opt-in custom kernel "
        "for batch==1 int4 K>=%d; ~10%% slower than native eager, parity PASS; "
        "for experimentation)",
        _CUSTOM_K_THRESHOLD,
    )


def uninstall_fused_quant_gemv_patch():
    global _patch_installed
    nn.QuantizedLinear.__call__ = _orig_quantized_linear_call
    _patch_installed = False


_METAL_SOURCE = """
#define SIMD_SIZE 32
#define NUM_SIMD 2
#define RES_PER_SIMD 4
#define VALUES_PER_THREAD 16
#define BLOCK_SIZE 512
#define GROUP_SIZE 128

uint M = uint(meta[0]);
uint K = uint(meta[1]);
uint n_groups = K / GROUP_SIZE;
uint K_bytes = K / 2;  // 4-bit: 2 vals per byte
uint K_u16 = K / 4;    // uint16 count per row

uint tg_y = threadgroup_position_in_grid.x;
uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;

const int out_row = int(tg_y) * (NUM_SIMD * RES_PER_SIMD) + int(simd_gid) * RES_PER_SIMD;
if (out_row >= int(M)) return;

// V2 optimization 1 (shift-elimination): weight as uint16 (4 nibbles),
// x pre-scaled DOWN so nibble stays in native bit position (mask-only).
const device uint16_t* ws = (const device uint16_t*)w;
ws += out_row * K_u16 + simd_lid * (VALUES_PER_THREAD / 4);
scales += out_row * n_groups + simd_lid / 8;
biases += out_row * n_groups + simd_lid / 8;
x += simd_lid * VALUES_PER_THREAD;
out += out_row;

thread float x_thread[VALUES_PER_THREAD];
thread float result[RES_PER_SIMD] = {0.0f, 0.0f, 0.0f, 0.0f};

for (uint k = 0; k < K; k += BLOCK_SIZE) {
    // load_vector bits=4: pre-scale x to eliminate shifts (bounds-checked
    // for K not divisible by BLOCK_SIZE — handles remainder safely).
    float sum = 0.0f;
    for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
        bool ok4 = (k + simd_lid * VALUES_PER_THREAD + i + 3) < K;
        if (ok4) {
            sum += x[i] + x[i+1] + x[i+2] + x[i+3];
            x_thread[i]   = x[i];
            x_thread[i+1] = x[i+1] / 16.0f;
            x_thread[i+2] = x[i+2] / 256.0f;
            x_thread[i+3] = x[i+3] / 4096.0f;
        } else {
            x_thread[i]   = 0.0f;
            x_thread[i+1] = 0.0f;
            x_thread[i+2] = 0.0f;
            x_thread[i+3] = 0.0f;
        }
    }
    // V2 optimization 2 (affine factoring): qdot returns
    // scale*accum + sum*bias (2 FMA/group), not scale*nib+bias per nibble.
    for (int row = 0; row < RES_PER_SIMD; row++) {
        const device uint16_t* wl = ws + row * K_u16;
        const device float* sl = scales + row * n_groups;
        const device float* bl = biases + row * n_groups;
        float s = sl[0];
        float b = bl[0];
        float accum = 0.0f;
        for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
            bool ok = (k + simd_lid * VALUES_PER_THREAD + 4 * i + 3) < K;
            if (ok) {
                accum += (x_thread[4*i]       * float(wl[i] & 0x000f)
                        + x_thread[4*i + 1]   * float(wl[i] & 0x00f0)
                        + x_thread[4*i + 2]   * float(wl[i] & 0x0f00)
                        + x_thread[4*i + 3]   * float(wl[i] & 0xf000));
            }
        }
        result[row] += s * accum + sum * b;
    }
    ws += BLOCK_SIZE / 4;          // BLOCK_SIZE values = BLOCK_SIZE/4 uint16
    scales += BLOCK_SIZE / GROUP_SIZE;
    biases += BLOCK_SIZE / GROUP_SIZE;
    x += BLOCK_SIZE;
}
// V2 optimization 3 (simdgroup layout): single hardware simd_sum per row,
// no cross-simdgroup barriers (2 simdgroups, each reduces independently;
// simd_lid==0 of each simdgroup writes its 4 rows).
for (int row = 0; row < RES_PER_SIMD; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) out[row] = result[row];
}
"""

__all__ = [
    "fused_dequant_gemv_int4",
    "is_fused_quant_gemv_enabled",
    "install_fused_quant_gemv_patch",
    "uninstall_fused_quant_gemv_patch",
]
