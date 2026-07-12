# SPDX-License-Identifier: Apache-2.0
# Quantization helpers for FP8/INT8/NF4 KV cache quantization.

from __future__ import annotations

import logging

import mlx.core as mx

from ..mfa import _HAS_MFA_EXT, _MFA_EXT

logger = logging.getLogger(__name__)

QUANT_NONE = 0
QUANT_FP8_E4M3 = 3
QUANT_FP8_E5M2 = 4
QUANT_INT8 = 5
QUANT_NF4 = 6


def quantize_fp8(
    tensor: mx.array,
    use_e5m2: bool = False,
) -> tuple[mx.array, mx.array]:
    if _HAS_MFA_EXT and hasattr(_MFA_EXT, "quantize_fp8"):
        return _MFA_EXT.quantize_fp8(tensor, use_e5m2=use_e5m2)

    t = tensor.astype(mx.float32)
    orig_shape = t.shape
    if t.ndim == 4:
        b, h, n, d = orig_shape
        abs_max = mx.max(mx.abs(t), axis=(-2, -1))
    else:
        abs_max = mx.max(mx.abs(t), axis=-1, keepdims=True)

    fp8_max = 57344.0 if use_e5m2 else 448.0
    scale = abs_max / fp8_max
    scale = mx.clip(scale, 1e-12, None)

    if t.ndim == 4:
        scale_exp = scale[:, :, None, None]
    else:
        scale_exp = scale[:, None]

    scaled = t / scale_exp
    scaled = mx.clip(scaled, -fp8_max, fp8_max)

    quantized = ((scaled / fp8_max) * 127.0 + 128.0).astype(mx.uint8)
    return quantized, scale


def quantize_int8(
    tensor: mx.array,
) -> tuple[mx.array, mx.array]:
    if _HAS_MFA_EXT and hasattr(_MFA_EXT, "quantize_int8"):
        return _MFA_EXT.quantize_int8(tensor)

    t = tensor.astype(mx.float32)
    if t.ndim == 4:
        abs_max = mx.max(mx.abs(t), axis=(-2, -1))
    else:
        abs_max = mx.max(mx.abs(t), axis=-1, keepdims=True)

    scale = abs_max / 127.0
    scale = mx.clip(scale, 1e-12, None)

    if t.ndim == 4:
        scale_exp = scale[:, :, None, None]
    else:
        scale_exp = scale[:, None]

    scaled = t / scale_exp
    scaled = mx.clip(scaled, -127.0, 127.0)
    quantized = (scaled + 128.0).astype(mx.uint8)
    return quantized, scale


def quantize_nf4(
    tensor: mx.array,
) -> tuple[mx.array, mx.array]:
    if _HAS_MFA_EXT and hasattr(_MFA_EXT, "quantize_nf4"):
        return _MFA_EXT.quantize_nf4(tensor)

    nf4_lut = mx.array(
        [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ],
        dtype=mx.float32,
    )

    t = tensor.astype(mx.float32)
    if t.ndim == 4:
        b, h, n, d = t.shape
        abs_max = mx.max(mx.abs(t), axis=(-2, -1))
        scale = abs_max / mx.max(mx.abs(nf4_lut))
        scale = mx.clip(scale, 1e-12, None)
        scale_exp = scale[:, :, None, None]
    else:
        n = t.shape[0]
        d = t.shape[-1]
        abs_max = mx.max(mx.abs(t), axis=-1, keepdims=True)
        scale = abs_max / mx.max(mx.abs(nf4_lut))
        scale = mx.clip(scale, 1e-12, None)
        scale_exp = scale[:, None]

    normalized = t / scale_exp

    flat = normalized.reshape(-1, 1)
    diffs = mx.abs(flat - nf4_lut[None, :])
    indices = mx.argmin(diffs, axis=-1).astype(mx.uint8)

    indices = indices.reshape(-1, 2)
    packed = ((indices[:, 1] << 4) | indices[:, 0]).astype(mx.uint8)

    if t.ndim == 4:
        packed = packed.reshape(b, h, n, d // 2)
    else:
        packed = packed.reshape(n, d // 2)

    return packed, scale


def dequantize(
    quantized: mx.array,
    scale: mx.array,
    quant_type: int,
    orig_dtype: mx.Dtype = mx.float16,
) -> mx.array:
    if quant_type == QUANT_INT8:
        scaled = quantized.astype(mx.float32) - 128.0
        if scale.ndim == 2:
            scale_exp = scale[:, :, None, None]
        else:
            scale_exp = scale[:, None]
        return (scaled * scale_exp).astype(orig_dtype)

    if quant_type == QUANT_NF4:
        nf4_lut = mx.array(
            [
                -1.0,
                -0.6961928009986877,
                -0.5250730514526367,
                -0.39491748809814453,
                -0.28444138169288635,
                -0.18477343022823334,
                -0.09105003625154495,
                0.0,
                0.07958029955625534,
                0.16093020141124725,
                0.24611230194568634,
                0.33791524171829224,
                0.44070982933044434,
                0.5626170039176941,
                0.7229568362236023,
                1.0,
            ],
            dtype=mx.float32,
        )
        lo = (quantized & 0x0F).astype(mx.uint8)
        hi = (quantized >> 4).astype(mx.uint8)
        unpacked = mx.stack([lo, hi], axis=-1).reshape(*quantized.shape[:-1], -1)
        flat = unpacked.reshape(-1).astype(mx.int32)
        values = nf4_lut[flat].reshape(unpacked.shape)
        if scale.ndim == 2:
            scale_exp = scale[:, :, None, None]
        else:
            scale_exp = scale[:, None]
        return (values * scale_exp).astype(orig_dtype)

    if quant_type in (QUANT_FP8_E4M3, QUANT_FP8_E5M2):
        scaled = quantized.astype(mx.float32) - 128.0
        fp8_max = 57344.0 if quant_type == QUANT_FP8_E5M2 else 448.0
        scaled = scaled / 127.0 * fp8_max
        if scale.ndim == 2:
            scale_exp = scale[:, :, None, None]
        else:
            scale_exp = scale[:, None]
        return (scaled * scale_exp).astype(orig_dtype)

    raise ValueError(f"Unknown quant_type: {quant_type}")


__all__ = [
    "QUANT_NONE",
    "QUANT_FP8_E4M3",
    "QUANT_FP8_E5M2",
    "QUANT_INT8",
    "QUANT_NF4",
    "quantize_fp8",
    "quantize_int8",
    "quantize_nf4",
    "dequantize",
]
