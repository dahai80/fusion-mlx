# SPDX-License-Identifier: Apache-2.0
# TurboQuant - quantized attention kernels for FP8/INT8/NF4 KV cache.
# Keeps K/V quantized during computation, dequantizes on-the-fly.

from __future__ import annotations

import logging

import mlx.core as mx

from ..mfa import _HAS_MFA_EXT, _MFA_EXT
from .quantize import (
    QUANT_FP8_E4M3,
    QUANT_FP8_E5M2,
    QUANT_INT8,
    QUANT_NF4,
    dequantize,
    quantize_fp8,
    quantize_int8,
    quantize_nf4,
)

logger = logging.getLogger(__name__)


def turboquant_attention(
    q: mx.array,
    k_quant: mx.array,
    v_quant: mx.array,
    k_scale: mx.array,
    v_scale: mx.array,
    quant_type: int,
    *,
    scale: float | None = None,
    mask: mx.array | None = None,
    causal: bool = False,
    window_size: int = 0,
) -> mx.array:
    scale = scale if scale is not None else q.shape[-1] ** -0.5

    if _HAS_MFA_EXT and hasattr(_MFA_EXT, "flash_attention_quantized"):
        try:
            return _MFA_EXT.flash_attention_quantized(
                q,
                k_quant,
                v_quant,
                k_scale,
                v_scale,
                quant_type=quant_type,
                scale=scale,
                mask=mask,
                causal=causal,
                window_size=window_size,
            )
        except Exception as exc:
            logger.debug(
                "MFA TurboQuant failed (%s), falling back to dequant+SDPA", exc
            )

    k = dequantize(k_quant, k_scale, quant_type, q.dtype)
    v = dequantize(v_quant, v_scale, quant_type, q.dtype)

    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def turboquant_kv_cache_attention(
    q: mx.array,
    cache: object,
    *,
    scale: float | None = None,
    mask: mx.array | None = None,
    causal: bool = True,
) -> mx.array:
    from ...turboquant_kv import BatchTurboQuantKVCache

    real_cache = cache
    if hasattr(cache, "_cache"):
        real_cache = cache._cache

    from mlx_vlm.turboquant import TurboQuantKVCache as _TQCache

    if not isinstance(real_cache, (_TQCache, BatchTurboQuantKVCache)):
        raise TypeError(f"Expected TurboQuantKVCache, got {type(real_cache)}")

    scale = scale if scale is not None else q.shape[-1] ** -0.5

    if q.shape[-2] == 1:
        return real_cache.decode_attention(q, scale=scale, mask=mask)

    result = real_cache.prefill_attention(q, scale=scale, mask=mask)
    if result is not None:
        return result

    k, v = real_cache.dequantize()
    return mx.fast.scaled_dot_product_attention(
        q, k.astype(q.dtype), v.astype(q.dtype), scale=scale, mask=mask
    )


def quantize_and_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    quant_type: int = QUANT_INT8,
    *,
    scale: float | None = None,
    mask: mx.array | None = None,
    causal: bool = False,
) -> mx.array:
    k_quant, k_s = _quantize_by_type(k, quant_type)
    v_quant, v_s = _quantize_by_type(v, quant_type)
    return turboquant_attention(
        q,
        k_quant,
        v_quant,
        k_s,
        v_s,
        quant_type,
        scale=scale,
        mask=mask,
        causal=causal,
    )


def _quantize_by_type(tensor: mx.array, quant_type: int) -> tuple[mx.array, mx.array]:
    if quant_type == QUANT_FP8_E4M3:
        return quantize_fp8(tensor, use_e5m2=False)
    if quant_type == QUANT_FP8_E5M2:
        return quantize_fp8(tensor, use_e5m2=True)
    if quant_type == QUANT_INT8:
        return quantize_int8(tensor)
    if quant_type == QUANT_NF4:
        return quantize_nf4(tensor)
    raise ValueError(f"Unknown quant_type: {quant_type}")


__all__ = [
    "turboquant_attention",
    "turboquant_kv_cache_attention",
    "quantize_and_attention",
]
