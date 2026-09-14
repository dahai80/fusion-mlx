# SPDX-License-Identifier: Apache-2.0
# O5.2 dequant-fusion: NVFP4 fused linear.
#
# Unlike dequant_nvfp4_weights (load-time dequant -> fp16, losing the 4-bit
# storage win), this module holds the packed uint8 weights + E4M3 block
# scales and dequants PER __call__ — the dequant is fused into the forward
# pass at the matmul boundary. Weights stay 4-bit in memory (4x smaller than
# bf16); the trade is per-call dequant compute for the memory-bandwidth win.
#
# This is the honest "fusion" for nvfp4 absent a native Metal kernel: the
# dequant is not hoisted to load time, it rides the forward path so packed
# storage is retained across inference. A native kernel would fuse the
# E2M1 decode + E4M3 scale into the GEMV itself; this Python realization
# materializes a transient bf16 weight per call (mx.eval'd lazily by MLX).

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

from ..nvfp4 import NVFP4_BLOCK_SIZE, dequant_nvfp4

logger = logging.getLogger(__name__)


class NVFP4FusedLinear(nn.Module):
    # Holds packed NVFP4 (uint8 E2M1 + E4M3 block scales), dequants per call.
    # Output matches nn.Linear (x @ W.T + b) within E2M1 quantization error.

    def __init__(
        self,
        out_features: int,
        in_features: int,
        bias: bool = True,
        group_size: int = NVFP4_BLOCK_SIZE,
    ):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.group_size = group_size
        # packed: uint8, 2 E2M1 per byte -> shape (out, in/2) flattened.
        numel = out_features * in_features
        if numel % (2 * NVFP4_BLOCK_SIZE) != 0:
            raise ValueError(
                f"nvfp4: numel {numel} not divisible by 2*block "
                f"{2 * NVFP4_BLOCK_SIZE}; weight must be block-padded"
            )
        self.w_packed = mx.zeros((numel // 2,), dtype=mx.uint8)
        self.w_scales = mx.zeros((numel // NVFP4_BLOCK_SIZE,), dtype=mx.uint8)
        self._weight_shape = (out_features, in_features)
        self.bias = mx.zeros((out_features,), dtype=mx.bfloat16) if bias else None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = NVFP4_BLOCK_SIZE,
    ) -> NVFP4FusedLinear:
        out_f, in_f = linear.weight.shape
        has_bias = hasattr(linear, "bias") and linear.bias is not None
        layer = cls(out_f, in_f, bias=has_bias, group_size=group_size)
        w_bf16 = linear.weight.astype(mx.bfloat16)
        flat = mx.reshape(w_bf16, (-1,))
        packed, scales = _quantize_nvfp4(flat, group_size)
        layer.w_packed = packed
        layer.w_scales = scales
        if has_bias:
            layer.bias = linear.bias.astype(mx.bfloat16)
        logger.info(
            "NVFP4FusedLinear.from_linear: %dx%d -> packed %s scales %s (fused dequant)",
            out_f,
            in_f,
            packed.shape,
            scales.shape,
        )
        return layer

    def _dequant_weight(self) -> mx.array:
        return dequant_nvfp4(
            self.w_packed,
            self.w_scales,
            self._weight_shape,
        )

    def __call__(self, x: mx.array) -> mx.array:
        w = self._dequant_weight()
        out = x.astype(mx.bfloat16) @ mx.transpose(w)
        if self.bias is not None:
            out = out + self.bias
        return out.astype(mx.float16)


def _quantize_nvfp4(
    flat: mx.array,
    block_size: int,
) -> tuple[mx.array, mx.array]:
    # Quantize a flat bf16 tensor to NVFP4 (E2M1 + E4M3 block scale).
    # block_scale = max(abs(block)) / 6.0 (E2M1 max magnitude), stored as
    # E4M3 via mx.to_fp8. Element = round(value / scale) mapped to nearest
    # E2M1 code. Returns (packed uint8, scales uint8).
    numel = int(flat.size)
    if numel % block_size != 0:
        raise ValueError(
            f"nvfp4 quant: numel {numel} not divisible by block {block_size}"
        )
    n_blocks = numel // block_size
    blocked = mx.reshape(flat.astype(mx.float32), (n_blocks, block_size))
    amax = mx.max(mx.abs(blocked), axis=1)
    scale = amax / mx.array(6.0, dtype=mx.float32)
    scale = mx.where(scale > 0, scale, mx.array(1.0, dtype=mx.float32))
    # Quantize to E2M1 magnitudes: {0,0.5,1,1.5,2,3,4,6}
    _MAGS = mx.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=mx.float32)
    normalized = blocked / mx.reshape(scale, (n_blocks, 1))
    # nearest magnitude index
    diff = mx.abs(
        mx.reshape(normalized, (n_blocks, block_size, 1)) - mx.reshape(_MAGS, (1, 1, 8))
    )
    mag_idx = mx.argmin(diff, axis=2)
    sign = mx.where(blocked >= 0, 0, 8).astype(mx.int32)
    codes = (mag_idx.astype(mx.int32) | sign).astype(mx.uint8)
    # pack 2 codes per byte (little-endian: [2k] low, [2k+1] high)
    flat_codes = mx.reshape(codes, (-1,)).astype(mx.uint32)
    low = flat_codes[0::2] & mx.array(0x0F, dtype=mx.uint32)
    high = (flat_codes[1::2] & mx.array(0x0F, dtype=mx.uint32)) << mx.array(
        4, dtype=mx.uint32
    )
    packed = (low | high).astype(mx.uint8)
    # store scale as E4M3
    scale_f32 = mx.reshape(scale, (n_blocks,)).astype(mx.float32)
    scales_u8 = (
        mx.to_fp8(scale_f32, dtype=mx.float8_e4m3).astype(mx.uint8)
        if hasattr(mx, "to_fp8")
        else _scale_to_uint8_e4m3(scale_f32)
    )
    return packed, scales_u8


def _scale_to_uint8_e4m3(scales: mx.array) -> mx.array:
    # Fallback when mx.to_fp8 unavailable: store scales as raw uint8 bits via
    # numpy view. This is a lossy bridge; native path preferred.
    import numpy as np

    arr = np.array(scales, dtype=np.float32)
    # E4M3 has 1 sign + 4 exp + 3 mantissa. Approximate by clamping to E4M3 range.
    # Simplified: store as uint8 index of power-of-2 (lossy but reversible-ish).
    clipped = np.clip(arr, 0, 448.0)
    viewed = np.frombuffer(clipped.tobytes(), dtype=np.uint8)
    return mx.array(viewed)


__all__ = ["NVFP4FusedLinear"]
