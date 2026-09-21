# SPDX-License-Identifier: Apache-2.0
"""Fused INT4 Dequant + GEMV Metal kernel (PRD stage-2 base layer).

Custom Metal kernel that reads packed uint32 int4 weights, unpacks 8
nibbles per uint32 (little-endian, low nibble first), dequantizes affine
(val = scale*nib + bias, per group), and accumulates the dot product with x
in registers — zero intermediate buffer.

Packed layout (parity verified 2026-09-21 vs mx.quantized_matmul, max diff
4.8e-7): weight (M, K//8) uint32, scales/biases (M, n_groups) float32.

PRODUCTION STATUS (measured 2026-09-21, M5 Max, MLX 0.31.2):
  Parity: PASS (max diff 4.8e-7 vs mx.quantized_matmul; end-to-end real
  model decode produces IDENTICAL tokens — verified on Qwen3-8B-4bit).

  PERFORMANCE: custom kernel is SLOWER than native across ALL measurement
  methods (corrected 2026-09-21 — prior "isolated-chain -16% win" was a
  measurement artifact from thermal drift + a K>4096 guard bug that
  silently ran native on both sides):
    | method                        | native   | custom   | delta  |
    |-------------------------------|----------|----------|--------|
    | eager per-op (production)     | 2.08ms   | 2.10ms   | +0.8%  |
    | compiled 32-layer chain       | 16.1ms   | 29.6ms   | +83.8% |
    | pure-GPU per-iter (compiled)  | 14.4ms   | 27.1ms   | +88.6% |
  Native wins everywhere. Production decode is EAGER (fusion-mlx
  BatchedEngine does NOT wrap the model forward in mx.compile — verified
  in engines/batched.py; only embedding/reranker/image_gen/sampler
  compile). So the eager per-op number is the production-relevant one:
  native is faster even there (+0.8% to +11% across shapes).

  ROOT CAUSE (corrected): NOT mx.compile graph opacity. The native
  quantized_matmul Metal kernel is genuinely ~1.9x faster GPU code
  (pure-GPU measurement: native 14.4ms vs custom 27.1ms per 32-layer
  iter). MLX's kernel uses tensor-core MMA / register tiling / async
  prefetch that this hand-written kernel does not. No graph pass fixes a
  slower kernel — stage-4 C++ graph-optimizer work is MOOT here.

  This is the 5th confirmation that handwriting Metal kernels beating MLX
  native is not achievable in-session for ops MLX already optimizes
  (prior: smart-conv 10-30x slower, sdpa 86% roofline, RMSNorm
  microbench-only, int4 GEMV now 1.9x slower). Native wins on ops it
  already optimizes. Custom wins only for capability gaps (YaRN rope) or
  fusion patterns native cannot do.

  Dispatch (when gate ON): batch==1 AND bits==4 AND K>=_CUSTOM_K_THRESHOLD
  (8192) -> custom kernel. Everything else -> native mx.quantized_matmul.
  Zero correctness regression (parity IDENTICAL). Kept as opt-in base-layer
  Metal capability demonstration + foundation for future work. Default OFF
  because slower than native — native is the production path.

Wiring: install_fused_quant_gemv_patch() monkeypatches
nn.QuantizedLinear.__call__ to route large-K decode through the custom
kernel. Called at scheduler import (idempotent, no-op when gate OFF or
Metal unavailable).

Degrade switch: FUSION_FUSED_QUANT_GEMV (default "0" = OFF).
  "1" = opt-in custom kernel (slower than native; for experimentation).
  Default OFF because native is faster. Production int4 path = native
  mx.quantized_matmul via nn.QuantizedLinear (default-ON).
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_FUSED_QUANT_GEMV_KERNEL = None

_CUSTOM_K_THRESHOLD = 8192

_CONFIG = {
    8192: (4, 128),
    11008: (16, 128),
    12288: (8, 128),
    14336: (8, 128),
    18944: (8, 128),
}


def _tile_for_k(k: int) -> tuple[int, int]:
    for thresh in sorted(_CONFIG.keys(), reverse=True):
        if k >= thresh:
            return _CONFIG[thresh]
    return (16, 128)


def is_fused_quant_gemv_enabled() -> bool:
    return os.environ.get("FUSION_FUSED_QUANT_GEMV", "0") == "1"


def _get_kernel():
    global _FUSED_QUANT_GEMV_KERNEL
    if _FUSED_QUANT_GEMV_KERNEL is not None:
        return _FUSED_QUANT_GEMV_KERNEL
    if not mx.metal.is_available():
        return None
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
    biases,
    x: mx.array,
    group_size: int = 128,
    bits: int = 4,
) -> mx.array:
    """Fused INT4 dequant + GEMV.

    weight: (M, K//8) uint32 packed int4. x: (1, K) or (K,). out: (1, M).
    Custom kernel used only when: gate ON, bits==4, batch==1, K>=8192
    (large-K exercise zone; custom is slower than native, gate is opt-in
    for experimentation). Otherwise delegates to mx.quantized_matmul
    (zero regression).
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
    tile_m, tg = _tile_for_k(K)
    meta = mx.array([float(M), float(K), float(group_size), float(tile_m)], mx.float32)
    kernel = _get_kernel()
    if kernel is None:
        return _native_quant_matmul(weight, scales, biases, x, group_size, bits)
    nb = (M + tile_m - 1) // tile_m
    try:
        out = kernel(
            inputs=[weight, scales, biases, x_flat, meta],
            grid=(nb * tg, 1, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[(M,)],
            output_dtypes=[mx.float32],
        )[0]
    except Exception:
        logger.exception(
            "fused_dequant_gemv_int4 kernel failed, falling back to native"
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
        "fused_quant_gemv: patched nn.QuantizedLinear (opt-in custom kernel "
        "for batch==1 int4 K>=%d; slower than native, for experimentation)",
        _CUSTOM_K_THRESHOLD,
    )


def uninstall_fused_quant_gemv_patch():
    global _patch_installed
    nn.QuantizedLinear.__call__ = _orig_quantized_linear_call
    _patch_installed = False


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
const device float* xf = x;
float partial[16];
for (uint t = 0; t < 16; t++) partial[t] = 0.0f;
for (uint ui = tid; ui < K_u32; ui += nthreads) {
    uint kb = ui * 8;
    float xv0 = xf[kb], xv1 = xf[kb+1], xv2 = xf[kb+2], xv3 = xf[kb+3];
    float xv4 = xf[kb+4], xv5 = xf[kb+5], xv6 = xf[kb+6], xv7 = xf[kb+7];
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

__all__ = [
    "fused_dequant_gemv_int4",
    "is_fused_quant_gemv_enabled",
    "install_fused_quant_gemv_patch",
    "uninstall_fused_quant_gemv_patch",
]
