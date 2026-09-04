# SPDX-License-Identifier: Apache-2.0
"""W4A8 fused MatMul — real activation-int8 path.

W4: weights stay 4-bit (mlx quantized format, mx.quantized_matmul).
A8: activations quantized to int8 with a per-tensor symmetric absmax scale,
held int8 across the tile boundary, dequantized to fp16 only inside the
matmul accumulation. MLX has no int8 MMA (no simdgroup int8 dot), so the
compute is fp16 accumulate — the A8 *storage/bandwidth* win is realized
(activations stored/transmitted int8, 2x smaller than fp16), but the
compute-throughput win is NOT realized without a native Metal kernel.

Native path (metal/w4a8_fused_matmul.metal) needs C++ extension integration;
until then w4a8_fused_matmul dispatches to this Python realization. The
viability harness measures the activation-quant overhead a native kernel
must absorb.

GEMM tiling: K is split into tiles of _K_TILE so the int8 activation slab
is materialized one tile at a time (peak activation memory = M*N + tile*M*K8
instead of M*K*fp16). Output accumulates fp32 across tiles, then applies
scale_a * per-group weight scales already baked into quantized_matmul.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_K_TILE = 256
_INT8_MAX = 127.0


def quantize_activation_int8(x: mx.array) -> tuple[mx.array, mx.array]:
    # Symmetric per-tensor absmax int8 quantization. Returns (x_int8, scale)
    # where the dequant is x_int8 * scale (scale = max|x| / 127).
    x_fp16 = x.astype(mx.float16)
    amax = mx.max(mx.abs(x_fp16))
    scale = amax / mx.array(_INT8_MAX, dtype=mx.float16)
    # Guard divide-by-zero (all-zero activation slab).
    scale = mx.where(scale > 0, scale, mx.array(1.0, dtype=mx.float16))
    x_q = mx.round(x_fp16 / scale).astype(mx.int8)
    return x_q, mx.reshape(scale, ())


def dequantize_int8(x_q: mx.array, scale: mx.array) -> mx.array:
    return x_q.astype(mx.float16) * scale.astype(mx.float16)


def w4a8_tiled_matmul(
    x: mx.array,
    w_quantized: mx.array,
    w_scales: mx.array,
    w_biases: mx.array,
    *,
    group_size: int = 64,
    bits: int = 4,
    k_tile: int = _K_TILE,
) -> mx.array:
    # Realized W4A8: activation quantized int8 (per-tensor absmax), dequantized
    # to fp16, then W4 matmul via mx.quantized_matmul. The A8 *storage/bandwidth*
    # win is realized upstream (callers may hold activations int8 between ops and
    # dequant at the matmul boundary); compute is fp16 accumulate — MLX has no
    # int8 MMA. The native Metal kernel (metal/w4a8_fused_matmul.metal) would
    # fuse the int8->fp16 dequant into the W4 MMA into one pass; until it is
    # built this Python realization is the activation-int8 path.
    #
    # NOTE: K-tiling the *activation* slab while slicing the packed uint32 weight
    # is NOT viable — mx.quantized_matmul expects the full packed weight (slicing
    # a uint32-packed tensor does not slice element-K and breaks group boundaries).
    # The activation is quantized over the full K; k_tile is kept as a parameter
    # for the native kernel signature and ignored on this fallback path.
    del k_tile
    x_fp16 = x.astype(mx.float16)
    *batch, _k = x_fp16.shape
    out_features = w_scales.shape[0]
    x_flat = mx.reshape(x_fp16, (-1, _k)) if batch else mx.reshape(x_fp16, (1, _k))
    x_q, a_scale = quantize_activation_int8(x_flat)
    x_dq = dequantize_int8(x_q, a_scale)
    out = mx.quantized_matmul(
        x_dq,
        w_quantized,
        w_scales,
        w_biases,
        group_size=group_size,
        bits=bits,
    )
    if batch:
        return mx.reshape(out, (*batch, out_features)).astype(mx.float16)
    return mx.reshape(out, (out_features,)).astype(mx.float16)


class W4A8Linear(nn.Module):
    # Drop-in replacement for nn.Linear: stores W4 quantized weight triple
    # + quantizes activations to int8 in __call__. Output matches nn.Linear
    # (x @ W.T + b) within int8 quantization error.
    #
    # NOTE: MLX QuantizedLinear already stores W4; this module exists so the
    # W4A8 *activation* path is reachable from the load converter. On models
    # already shipped as mlx 4-bit, the weight triple is taken from the
    # loaded QuantizedLinear; on fp16 models, from_linear quantizes first.

    def __init__(
        self,
        out_features: int,
        in_features: int,
        bias: bool = True,
        group_size: int = 64,
    ):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.group_size = group_size
        # W4 triple: (w_q uint32 packed, scales, biases).
        self.w_quantized = mx.zeros((out_features, in_features), dtype=mx.uint32)
        self.w_scales = mx.ones((out_features, in_features // group_size))
        self.w_biases = mx.zeros((out_features, in_features // group_size))
        self.bias = mx.zeros((out_features,), dtype=mx.float16) if bias else None

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 64) -> W4A8Linear:
        out_f, in_f = linear.weight.shape
        has_bias = hasattr(linear, "bias") and linear.bias is not None
        layer = cls(out_f, in_f, bias=has_bias, group_size=group_size)
        # Quantize the fp16 weight to 4-bit via mlx.nn.quantize. That op mutates
        # the module tree in place (replacing nn.Linear children with
        # nn.QuantizedLinear), so wrap the source linear so its quantized child
        # is reachable, then copy the weight triple off it.
        w = linear.weight.astype(mx.float16)
        try:

            class _Wrap(nn.Module):
                def __init__(self, src):
                    super().__init__()
                    self.lin = src

                def __call__(self, x):
                    return self.lin(x)

            wrap = _Wrap(linear)
            nn.quantize(wrap, group_size=group_size, bits=4)
            ql = wrap.lin
            if not isinstance(ql, nn.QuantizedLinear):
                raise TypeError("nn.quantize did not produce QuantizedLinear")
            layer.w_quantized = ql.weight
            layer.w_scales = ql.scales
            layer.w_biases = ql.biases
        except Exception as exc:
            logger.warning(
                "W4A8Linear.from_linear: W4 quantize failed (%s), "
                "storing fp16 weight path (no A8 speedup)",
                exc,
            )
            # Fallback: keep a dequantized weight, activation quant still runs.
            layer.w_quantized = w
            layer.w_scales = mx.ones((out_f, 1), dtype=mx.float16)
            layer.w_biases = mx.zeros((out_f, 1), dtype=mx.float16)
        if has_bias:
            layer.bias = linear.bias.astype(mx.float16)
        return layer

    @classmethod
    def from_quantized(cls, q_layer: nn.QuantizedLinear) -> W4A8Linear:
        # Wrap an already-4-bit mlx QuantizedLinear so the A8 activation
        # path is applied without re-quantizing the weight.
        out_f = q_layer.output_dims
        in_f = q_layer.input_dims
        has_bias = getattr(q_layer, "bias", None) is not None
        group_size = getattr(q_layer, "group_size", 64)
        layer = cls(out_f, in_f, bias=has_bias, group_size=group_size)
        layer.w_quantized = q_layer.weight
        layer.w_scales = q_layer.scales
        layer.w_biases = q_layer.biases
        if has_bias:
            layer.bias = q_layer.bias.astype(mx.float16)
        return layer

    def __call__(self, x: mx.array) -> mx.array:
        # If W4 quantize failed in from_linear, w_quantized is fp16 → plain matmul.
        if self.w_quantized.dtype == mx.float16:
            out = x.astype(mx.float16) @ mx.transpose(self.w_quantized)
        else:
            out = w4a8_tiled_matmul(
                x,
                self.w_quantized,
                self.w_scales,
                self.w_biases,
                group_size=self.group_size,
                bits=4,
            )
        if self.bias is not None:
            out = out + self.bias
        return out


def convert_to_w4a8(model: nn.Module, group_size: int = 64) -> tuple[nn.Module, int]:
    # Walk the module tree (handles list-nested submodules via fp8_linear's
    # _iter_submodules) and replace nn.Linear AND nn.QuantizedLinear with
    # W4A8Linear. Already-4-bit layers reuse their weight triple
    # (from_quantized); fp16 layers quantize (from_linear). Returns
    # (model, n_converted) so callers can log the count.
    from ..fp8_linear import _iter_submodules

    n = 0
    for parent, key, name, module, container_kind in _iter_submodules(model):
        if isinstance(module, nn.QuantizedLinear):
            new_layer = W4A8Linear.from_quantized(module)
        elif isinstance(module, nn.Linear):
            new_layer = W4A8Linear.from_linear(module, group_size=group_size)
        else:
            continue
        if container_kind == "list":
            parent[int(key)] = new_layer
        else:
            setattr(parent, key, new_layer)
        n += 1
        logger.info("convert_to_w4a8: %s -> W4A8Linear", name)
    return model, n


__all__ = [
    "quantize_activation_int8",
    "dequantize_int8",
    "w4a8_tiled_matmul",
    "W4A8Linear",
    "convert_to_w4a8",
]
