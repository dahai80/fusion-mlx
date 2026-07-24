# SPDX-License-Identifier: Apache-2.0
"""Phase C custom kernels — weight/activation fusion targets.

Scope (see PHASE_C.md for evidence + verdict):
  - W4A8 fused MatMul+quant (8-bit activation) — OPEN, viability under measurement.
  - Fused GDN megakernel — OPEN, hand-written Metal (mx.compile non-viable).
  - NVFP4 — blocked on upstream mlx#2962.
  - ANE / tensor-parallel / JetSpec — upstream/platform-blocked.

Already-covered (NOT Phase C gaps, do not re-implement):
  - Fused dequant SDPA → mlx_vlm.turboquant (fused_mse_sdpa kernel family).
  - W4 weight fused MatMul → mx.quantized_matmul / mlx.nn.QuantizedLinear.

Native kernels are built via the glm_moe_dsa CMake path; until a Phase C
extension is built, is_native_available() returns False and call-sites fall
back to the mlx primitive path.
"""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

try:
    from . import _ext
except Exception as exc:  # pragma: no cover - depends on local native build
    _ext = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


NATIVE_SYMBOLS: tuple[str, ...] = ("w4a8_fused_matmul", "moe_ffn_fused")


def is_native_available() -> bool:
    return _ext is not None


def import_error() -> Exception | None:
    return _IMPORT_ERROR


def has_symbol(name: str) -> bool:
    return hasattr(_ext, name) or hasattr(mx.fast, name)


def native_symbols() -> tuple[str, ...]:
    if _ext is None:
        return ()
    return tuple(name for name in NATIVE_SYMBOLS if hasattr(_ext, name))


def missing_symbols(required: tuple[str, ...]) -> list[str]:
    return [name for name in required if not has_symbol(name)]


def w4a8_fused_matmul(
    w_quantized: mx.array,
    w_scales: mx.array,
    w_biases: mx.array,
    activations: mx.array,
    *,
    group_size: int = 64,
    bits: int = 4,
    act_bits: int = 8,
):
    """Fused 4-bit weight + 8-bit activation quantized MatMul.

    Native path (w4a8_fused_matmul Metal kernel) is not yet built — falls
    back to mx.quantized_matmul on fp16 activations, i.e. W4-only. The A8
    quantization upside is therefore NOT realized in the fallback; the
    viability harness (scripts/bench_phase_c_w4a8_viability.py) measures
    the activation-quant overhead that a native kernel must absorb.
    """
    if is_native_available() and has_symbol("w4a8_fused_matmul"):
        return _ext.w4a8_fused_matmul(
            w_quantized,
            w_scales,
            w_biases,
            activations,
            group_size,
            bits,
            act_bits,
        )
    logger.debug("phase_c w4a8 fallback to mx.quantized_matmul (fp16 activations)")
    return mx.quantized_matmul(
        w_quantized,
        w_scales,
        w_biases,
        activations,
        group_size=group_size,
        bits=bits,
    )


from .glm_moe_ffn import moe_ffn_fused, is_native_available as moe_ffn_is_native_available

__all__ = [
    "NATIVE_SYMBOLS",
    "is_native_available",
    "import_error",
    "has_symbol",
    "native_symbols",
    "missing_symbols",
    "w4a8_fused_matmul",
    "moe_ffn_fused",
    "moe_ffn_is_native_available",
]
