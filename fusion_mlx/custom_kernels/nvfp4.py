# SPDX-License-Identifier: Apache-2.0
# NVFP4 (E2M1 + E4M3 block scale) weight reader.
#
# Compatibility bridge: dequantize NVIDIA NVFP4 checkpoints to bf16 at load
# time so MLX can execute them without a separate conversion step.
#
# NVFP4 layout (NVIDIA TensorRT-LLM / cutlass spec):
#   - weights: 4-bit E2M1 floats, 2 packed per uint8 byte (little-endian
#     nibble order: element[2k] in the low nibble, [2k+1] in the high nibble).
#   - block scales: E4M3 float8, one scale per NVFP4_BLOCK_SIZE=16 weights,
#     stored row-major alongside the weight (shape typically (out, in/16)).
#   - E2M1 magnitudes (3-bit code 0..7, sign in bit 3):
#       {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
#
# MLX 0.32 has no native float4 / from_nv_fp4 op, so the E2M1 decode is a
# software LUT gather and E4M3 scales reuse mx.from_fp8. Dequant happens at
# load time -> the 4-bit storage/bandwidth win is NOT retained at inference;
# this reader is a format-compatibility bridge only (not a speed path).

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

NVFP4_BLOCK_SIZE = 16

# Full 16-value E2M1 LUT indexed by the raw 4-bit code (bit 3 = sign).
# Magnitudes follow the NVIDIA NVFP4 E2M1 value set.
_NVFP4_LUT = mx.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=mx.float32,
)

# Sibling scale-tensor suffixes emitted by the common NVFP4 exporters
# (TensorRT-LLM, llm-compressor, HF nvfp4 quants). Tried in priority order.
_NVFP4_SCALE_SUFFIXES = (
    "_weight_scale_inv",
    "_weight_scale",
    "_scale_inv",
    "_scale",
    "_block_scale",
    "_weight_block_scale",
    "_scaling_factor",
)


def _scale_to_float32(scales: mx.array) -> mx.array:
    # E4M3 block scales arrive either as raw uint8 bytes (mx.from_fp8 path)
    # or already as a float dtype (some HF exports store scales as fp32/fp16).
    dtype = scales.dtype
    if dtype == mx.uint8:
        return mx.from_fp8(scales, dtype=mx.float32)
    if "float8" in str(dtype):
        return scales.astype(mx.float32)
    return scales.astype(mx.float32)


def dequant_nvfp4(
    packed: mx.array,
    scales: mx.array,
    shape: tuple[int, ...],
) -> mx.array:
    # Dequantize NVFP4 packed weights to a bf16 mx.array of `shape`.
    #
    # packed:  uint8 mx.array, 2 E2M1 nibbles per byte (little-endian).
    # scales:  uint8 (E4M3 bytes) or float block scales, one per 16 weights,
    #          row-major order matching the flattened weight.
    # shape:   final weight shape (e.g. (out_features, in_features)).
    packed = packed.astype(mx.uint8)
    low = packed & mx.array(0x0F, dtype=mx.uint8)
    high = (packed >> mx.array(4, dtype=mx.uint8)) & mx.array(0x0F, dtype=mx.uint8)
    # Interleave to element order: [low0, high0, low1, high1, ...].
    codes = mx.reshape(mx.stack([low, high], axis=-1), (-1,)).astype(mx.int32)
    values = _NVFP4_LUT[codes]

    numel = int(codes.size)
    if numel % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"nvfp4: numel {numel} not divisible by block size "
            f"{NVFP4_BLOCK_SIZE}; weight must be block-padded by the exporter"
        )

    scale_f = mx.reshape(_scale_to_float32(scales), (-1,))
    n_blocks = numel // NVFP4_BLOCK_SIZE
    if scale_f.size < n_blocks:
        raise ValueError(f"nvfp4: need {n_blocks} block scales, got {scale_f.size}")
    scale_per_elem = mx.repeat(scale_f[:n_blocks], NVFP4_BLOCK_SIZE, axis=0)
    out = values * scale_per_elem
    return mx.reshape(out, shape).astype(mx.bfloat16)


def _is_uint8(t: mx.array) -> bool:
    return getattr(t, "dtype", None) == mx.uint8


def _looks_nvfp4(weight: mx.array, scale: mx.array) -> bool:
    # A uint8 weight is only NVFP4 if its sibling scale covers exactly one
    # block per 16 weights. This size relation rules out coincidental
    # uint8+_scale pairs in non-NVFP4 checkpoints.
    if not _is_uint8(weight):
        return False
    numel = int(weight.size) * 2
    if numel % NVFP4_BLOCK_SIZE != 0:
        return False
    return int(scale.size) == numel // NVFP4_BLOCK_SIZE


def _infer_shape(weight: mx.array, scale: mx.array) -> tuple[int, ...]:
    numel = int(weight.size) * 2
    # 2D scale (out, in/16) is the common DiT/Linear layout.
    if scale.ndim == 2:
        out = int(scale.shape[0])
        if out > 0 and numel % out == 0:
            return (out, numel // out)
    return (numel,)


def dequant_nvfp4_weights(
    weights: dict,
    shape_hints: dict | None = None,
) -> dict:
    # In-place NVFP4 dequant of a {key: mx.array} weights dict (the shape
    # returned by mx.load / _load_safetensors_dir). Conservative: only fires
    # when a uint8 weight has a sibling block-scale tensor with the right
    # 1-scale-per-16-elements size relation. Non-NVFP4 dicts are untouched.
    shape_hints = shape_hints or {}
    for wkey in list(weights.keys()):
        # A scale key may have been consumed by an earlier weight in this
        # same pass; skip keys that are no longer present.
        if wkey not in weights:
            continue
        weight = weights[wkey]
        if not _is_uint8(weight):
            continue
        skey = next(
            (wkey + suf for suf in _NVFP4_SCALE_SUFFIXES if wkey + suf in weights),
            None,
        )
        if skey is None:
            continue
        scale = weights[skey]
        if not _looks_nvfp4(weight, scale):
            logger.debug("nvfp4: %s + %s size mismatch, skip", wkey, skey)
            continue
        shape = shape_hints.get(wkey) or _infer_shape(weight, scale)
        try:
            weights[wkey] = dequant_nvfp4(weight, scale, shape)
            logger.info("nvfp4 dequant %s -> bf16 %s (scale %s)", wkey, shape, skey)
            del weights[skey]
        except Exception as exc:
            logger.warning("nvfp4 dequant failed %s: %s", wkey, exc)
    return weights


__all__ = [
    "NVFP4_BLOCK_SIZE",
    "dequant_nvfp4",
    "dequant_nvfp4_weights",
]
