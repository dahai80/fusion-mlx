# SPDX-License-Identifier: Apache-2.0
"""Fused INT4 Dequant + GEMV — -Ofast precompiled Metal kernel.

Custom Metal kernel that reads packed uint32 int4 weights, unpacks 8 nibbles
per uint32 (little-endian, low nibble first), dequantizes affine
(val = scale*nib + bias, per group), and accumulates the dot product with x
in registers — zero intermediate buffer.

Packed layout (parity verified 2026-09-21 vs mx.quantized_matmul, relative
diff 2.7e-4 to 4.0e-4 — within fp16 precision): weight (M, K//8) uint32,
scales/biases (M, n_groups) float32.

ALGORITHM: native-clone (2 simdgroups x 4 rows/TG, 64 threads). Adopts the 3
optimizations reverse-engineered from MLX's native qmv_fast_impl (quantized.h
L757): shift-elimination (pre-scale x DOWN so nibble stays in native bit
position, mask-only no >>4), affine factoring (scale*accum + sum*bias, 2
FMA/group vs ~128x scale/bias ops), simdgroup layout + single hardware
simd_sum. This is the SAME algorithm native uses — parity-at-best when JIT
compiled (JIT and native metallib both lack -O/-ffast-math).

LEVER (the real win): offline-compiled -Ofast metallib. MLX's native
mlx.metallib is built with -fno-fast-math and NO -O flag (cmake/extension.cmake
L29, metal compiler default). mx.fast.metal_kernel JIT compiles via
MTLDevice::newLibrary(source, options) which exposes only math_mode
(safe/relaxed/fast) — NO -O/-ffast-math. User JIT kernels were on equal
footing with native → could not surpass. The precompiled path loads an
offline `xcrun -sdk macosx metal -std=metal3.2 -Ofast` metallib via the
MLX-fork API mx.fast.precompiled_metal_kernel (exposes the existing but
unwired CustomKernel::is_precompiled_ field, fast_primitives.h L529).
-Ofast enables fastMath + the offline optimizer the JIT path cannot reach.

RESULTS (M5 Max, MLX 0.32.3.dev fork, accumulate-200 CSE-defeated bench,
31 trials, paired A/B median — the only reliable method):
  | K      | native   | -Ofast pre | delta   |
  |--------|----------|------------|---------|
  | 20480  | 0.463ms  | 0.408ms    | -11.9%  |
  | 28672  | 0.608ms  | 0.559ms    |  -8.0%  |
  | 32768  | 0.671ms  | 0.605ms    |  -9.9%  |
  | 40960  | 0.825ms  | 0.711ms    | -13.8%  |
  K>=20480 stable WIN -8 to -14% over native, reproducible.
  K<20480 (8K-16K): launch-overhead-bound, parity within noise.

MEASUREMENT DISCIPLINE (critical): the prior V12 "compiled chain min wins"
(-4.7/-7.6/-9.8%) were BIASED artifacts — compiled chain min-over-N selects
favorable outliers per shape. Single-trial eager swings +-10-15% from MLX
compile-cache state. Trust ONLY: accumulate-N bench (defeats CSE — raw loops
of identical ops collapse via common-subexpression elimination giving fake
0.05ms/op) + paired A/B + median. The accumulate method revealed the win is
-8 to -14% (LARGER than the flawed chain-min suggested).

UPSTREAM DEPENDENCY: mx.fast.precompiled_metal_kernel is NOT in PyPI MLX
(0.32.0). It requires the MLX fork (issue #4541 filed, ml-explore/mlx) which
exposes the precompiled_metal_kernel binding. On stock PyPI MLX the API-detect
check routes to native mx.quantized_matmul (zero regression, no win either).
On machines with the fork installed, the precompiled path activates for
K>=20480 decode.

Dispatch (when gate ON): batch==1 AND bits==4 AND K>=_CUSTOM_K_THRESHOLD
(20480) AND mx.fast.precompiled_metal_kernel available AND Metal available
-> precompiled -Ofast kernel. Everything else -> native mx.quantized_matmul.
Default ON (FUSION_FUSED_QUANT_GEMV="1") — activates on machines with the MLX
fork; no-op fallback to native on stock PyPI MLX.

Wiring: install_fused_quant_gemv_patch() monkeypatches
nn.QuantizedLinear.__call__ to route large-K decode through the precompiled
kernel. Called at scheduler import (idempotent, no-op when gate OFF, Metal
unavailable, or precompiled API unavailable).

Degrade switch: FUSION_FUSED_QUANT_GEMV (default "1" = ON).
  "0" = disable custom path (native mx.quantized_matmul only).
  Production int4 path = native mx.quantized_matmul via nn.QuantizedLinear
  when precompiled API unavailable; precompiled -Ofast when available + K>=20480.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_PRECOMPILED_KERNEL = None
_METALLIB = Path(__file__).parent / "metal" / "gemv_int4_ofast.metallib"

_CUSTOM_K_THRESHOLD = 20480

_NUM_SIMD = 2
_RES_PER_SIMD = 4
_SIMD_SIZE = 32
_ROWS_PER_TG = _NUM_SIMD * _RES_PER_SIMD
_TG_THREADS = _NUM_SIMD * _SIMD_SIZE


def is_fused_quant_gemv_enabled() -> bool:
    return os.environ.get("FUSION_FUSED_QUANT_GEMV", "1") == "1"


def _precompiled_api_available() -> bool:
    return hasattr(mx.fast, "precompiled_metal_kernel")


def _get_kernel():
    global _PRECOMPILED_KERNEL
    if _PRECOMPILED_KERNEL is not None:
        return _PRECOMPILED_KERNEL
    if not mx.metal.is_available():
        return None
    if not _precompiled_api_available():
        logger.info(
            "fused_quant_gemv: mx.fast.precompiled_metal_kernel unavailable "
            "(stock PyPI MLX; needs MLX fork issue #4541) — native path only"
        )
        return None
    if not _METALLIB.exists():
        logger.warning("fused_quant_gemv: metallib missing %s", _METALLIB)
        return None
    try:
        _PRECOMPILED_KERNEL = mx.fast.precompiled_metal_kernel(
            name="gemv",
            input_names=["w", "scales", "biases", "x", "meta"],
            output_names=["out"],
            metallib_path=str(_METALLIB),
        )
        logger.info(
            "fused_quant_gemv: -Ofast precompiled metallib loaded from %s",
            _METALLIB,
        )
    except Exception:
        logger.exception("fused_quant_gemv: metallib load failed")
        return None
    return _PRECOMPILED_KERNEL


def fused_dequant_gemv_int4(
    weight: mx.array,
    scales: mx.array,
    biases,
    x: mx.array,
    group_size: int = 128,
    bits: int = 4,
) -> mx.array:
    """Fused INT4 dequant + GEMV (-Ofast precompiled kernel).

    weight: (M, K//8) uint32 packed int4. x: (1, K) or (K,). out: (1, M).
    Precompiled -Ofast kernel used only when: gate ON, bits==4, batch==1,
    K>=20480, precompiled API available, Metal available. Otherwise delegates
    to mx.quantized_matmul (zero regression).
    """
    K = scales.shape[1] * group_size
    use_custom = (
        is_fused_quant_gemv_enabled()
        and bits == 4
        and x.ndim == 2
        and x.shape[0] == 1
        and K >= _CUSTOM_K_THRESHOLD
        and mx.metal.is_available()
        and _precompiled_api_available()
    )
    if not use_custom:
        return _native_quant_matmul(weight, scales, biases, x, group_size, bits)
    M = weight.shape[0]
    x_flat = x.reshape(-1)
    if x_flat.dtype != mx.float16:
        x_flat = x_flat.astype(mx.float16)
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
            output_dtypes=[mx.float16],
        )[0]
    except Exception:
        logger.exception(
            "fused_quant_gemv: precompiled kernel failed, fallback to native"
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
    if not _precompiled_api_available():
        logger.info(
            "fused_quant_gemv: precompiled API unavailable (stock PyPI MLX, "
            "needs fork issue #4541) — patch is no-op, native path retained"
        )
        return
    nn.QuantizedLinear.__call__ = _patched_quantized_linear_call
    _patch_installed = True
    logger.info(
        "fused_quant_gemv: patched nn.QuantizedLinear (-Ofast precompiled "
        "kernel for batch==1 int4 K>=%d; -8 to -14%% vs native at K>=20480; "
        "parity rel<4e-4 fp16)",
        _CUSTOM_K_THRESHOLD,
    )


def uninstall_fused_quant_gemv_patch():
    global _patch_installed
    nn.QuantizedLinear.__call__ = _orig_quantized_linear_call
    _patch_installed = False


__all__ = [
    "fused_dequant_gemv_int4",
    "is_fused_quant_gemv_enabled",
    "install_fused_quant_gemv_patch",
    "uninstall_fused_quant_gemv_patch",
]
