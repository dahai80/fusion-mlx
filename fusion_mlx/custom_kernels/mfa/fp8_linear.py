# SPDX-License-Identifier: Apache-2.0
# FP8 linear layer - W8A8 block-scaled quantization for memory-efficient inference.
# Based on xDiT's xFuserFP8BlockScaleLinear and mlx_mfa FP8 quantization.

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_FP8_MAX = 448.0
_FP8_BLOCK = 128


def _pad_to_multiple(x: mx.array, block: int, dim: int = -1) -> tuple[mx.array, int]:
    ndim = x.ndim
    size = x.shape[dim]
    pad_amount = (-size) % block
    if pad_amount == 0:
        return x, 0
    pad_widths = [(0, 0)] * ndim
    pad_widths[dim] = (0, pad_amount)
    return mx.pad(x, pad_widths), pad_amount


def _quantize_block_fp8(
    tensor: mx.array, block: int = _FP8_BLOCK
) -> tuple[mx.array, mx.array]:
    orig_shape = tensor.shape
    K = orig_shape[-1]
    tensor_f32 = tensor.astype(mx.float32)

    padded, pad_amount = _pad_to_multiple(tensor_f32, block, dim=-1)
    K_pad = padded.shape[-1]

    n_blocks = K_pad // block
    if n_blocks == 0:
        absmax = mx.max(mx.abs(padded), axis=-1, keepdims=True)
        scale = mx.clip(absmax / _FP8_MAX, 1e-12, None)
        scaled = mx.clip(padded / scale, -_FP8_MAX, _FP8_MAX)
        fp8 = ((scaled / _FP8_MAX) * 127.0 + 128.0).astype(mx.uint8)
        if pad_amount > 0:
            fp8 = fp8[..., :K]
        return fp8, scale.squeeze(-1)

    blocked = padded.reshape(*orig_shape[:-1], n_blocks, block)

    absmax = mx.max(mx.abs(blocked), axis=-1, keepdims=True)
    scale = mx.clip(absmax / _FP8_MAX, 1e-12, None)

    scaled = mx.clip(blocked / scale, -_FP8_MAX, _FP8_MAX)

    fp8 = ((scaled / _FP8_MAX) * 127.0 + 128.0).astype(mx.uint8)

    if pad_amount > 0:
        fp8_flat = fp8.reshape(*orig_shape[:-1], K_pad)
        fp8 = fp8_flat[..., :K]

    return fp8, scale.squeeze(-1)


def _dequantize_block_fp8(
    fp8: mx.array,
    scale: mx.array,
    block: int = _FP8_BLOCK,
    dtype: mx.Dtype = mx.float16,
) -> mx.array:
    scaled = fp8.astype(mx.float32) - 128.0
    scaled = scaled / 127.0 * _FP8_MAX

    if fp8.ndim == scale.ndim + 1 and fp8.shape[-1] == block:
        blocked = scaled
        scale_exp = scale[..., None, :]
        deq = blocked * mx.swapaxes(scale_exp, -2, -1)
        out = deq.reshape(*fp8.shape[:-2], fp8.shape[-2] * block)
    else:
        K = fp8.shape[-1]
        n_blocks = scale.shape[-1]
        total = n_blocks * block
        if total > K:
            padded = mx.pad(scaled, [(0, 0)] * (scaled.ndim - 1) + [(0, total - K)])
        else:
            padded = scaled
        blocked = padded.reshape(*fp8.shape[:-1], n_blocks, block)
        scale_exp = scale[..., None, :]
        deq = blocked * mx.swapaxes(scale_exp, -2, -1)
        out = deq.reshape(*fp8.shape[:-1], total)
        if total > K:
            out = out[..., :K]

    return out.astype(dtype)


class FP8Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = _FP8_BLOCK,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        w_fp16 = (
            mx.random.uniform(shape=(out_features, in_features), dtype=mx.float16)
            * 0.02
            - 0.01
        )
        w_fp8, w_scale = _quantize_block_fp8(w_fp16, block_size)
        self.weight_fp8 = w_fp8
        self.weight_scale = w_scale

        if bias:
            self.bias = mx.zeros((out_features,), dtype=mx.float16)
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, block_size: int = _FP8_BLOCK) -> FP8Linear:
        result = cls.__new__(cls)
        result.in_features = linear.weight.shape[1]
        result.out_features = linear.weight.shape[0]
        result.block_size = block_size

        w_fp8, w_scale = _quantize_block_fp8(linear.weight, block_size)
        result.weight_fp8 = w_fp8
        result.weight_scale = w_scale
        result.bias = linear.bias

        return result

    def __call__(self, x: mx.array) -> mx.array:
        w = _dequantize_block_fp8(
            self.weight_fp8,
            self.weight_scale,
            self.block_size,
            x.dtype,
        )

        out = x @ w.T

        if self.bias is not None:
            out = out + self.bias.astype(x.dtype)

        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"block_size={self.block_size}, "
            f"bias={self.bias is not None}"
        )


def quantize_fp8_linear(model: nn.Module, block_size: int = _FP8_BLOCK) -> nn.Module:
    for name, child in list(model.__dict__.items()):
        if isinstance(child, nn.Linear):
            setattr(model, name, FP8Linear.from_linear(child, block_size))
        elif hasattr(child, "children"):
            quantize_fp8_linear(child, block_size)
    return model


__all__ = [
    "FP8Linear",
    "quantize_fp8_linear",
    "_quantize_block_fp8",
    "_dequantize_block_fp8",
]
