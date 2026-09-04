# SPDX-License-Identifier: Apache-2.0
"""Phase C W4A8 / NVFP4 / fused-GDN kernel correctness tests.

Small-tensor only — no real model load. Verifies quantize_activation_int8
round-trip, w4a8_tiled_matmul vs reference, W4A8Linear vs nn.Linear,
convert_to_w4a8 count, NVFP4 dequant, and FusedGDN forward shape.
"""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


def test_quantize_activation_int8_roundtrip():
    from fusion_mlx.custom_kernels.phase_c.w4a8_kernel import (
        dequantize_int8,
        quantize_activation_int8,
    )

    x = mx.array(
        [[0.1, -0.2, 0.3, 0.0, 0.5, -0.5, 0.9, -0.9]],
        dtype=mx.float16,
    )
    x_q, scale = quantize_activation_int8(x)
    assert x_q.dtype == mx.int8
    x_dq = dequantize_int8(x_q, scale)
    err = float(mx.max(mx.abs(x_dq - x)))
    # int8 symmetric absmax: max error ~ max|x|/127 < 0.01 for these magnitudes.
    assert err < 0.02, f"int8 round-trip error {err} too large"


def test_quantize_activation_int8_allzero():
    from fusion_mlx.custom_kernels.phase_c.w4a8_kernel import (
        quantize_activation_int8,
    )

    x = mx.zeros((1, 8), dtype=mx.float16)
    x_q, scale = quantize_activation_int8(x)
    assert int(mx.max(mx.abs(x_q))) == 0
    # divide-by-zero guarded: scale forced to 1.0, no NaN.
    assert not mx.isnan(scale).any()


def test_w4a8_tiled_matmul_a8_error_bounded():
    # Compare W4A8 (int8 activation) vs W4-only (fp16 activation) on the SAME
    # weight triple. The difference is the A8-induced quantization error only,
    # which must be small (int8 absmax over a bounded-activation slab).
    from fusion_mlx.custom_kernels.phase_c.w4a8_kernel import w4a8_tiled_matmul

    out_f, in_f = 8, 64
    w = mx.random.normal((out_f, in_f), dtype=mx.float16)
    # Bounded activations so int8 absmax scale is well-conditioned.
    x = mx.random.normal((2, in_f), dtype=mx.float16) * mx.array(0.3, dtype=mx.float16)
    lin = nn.Linear(in_f, out_f, bias=False)
    lin.weight = w

    class _Wrap(nn.Module):
        def __init__(self, src):
            super().__init__()
            self.lin = src

        def __call__(self, x):
            return self.lin(x)

    wrap = _Wrap(lin)
    nn.quantize(wrap, group_size=64, bits=4)
    q = wrap.lin
    out_a8 = w4a8_tiled_matmul(
        x,
        q.weight,
        q.scales,
        q.biases,
        group_size=64,
        bits=4,
    )
    ref_w4 = mx.quantized_matmul(
        x.astype(mx.float16),
        q.weight,
        q.scales,
        q.biases,
        group_size=64,
        bits=4,
    )
    err = float(mx.max(mx.abs(out_a8 - ref_w4)))
    # A8 absmax int8 over |x|<~1: step ~ max|x|/127 < 0.01, output error small.
    assert err < 0.1, f"A8-induced matmul error {err} too large"


def test_w4a8linear_from_linear_a8_error_bounded():
    # W4A8Linear vs a W4-only QuantizedLinear built from the same source:
    # the delta is the activation-int8 quantization, not the W4 weight error.
    from fusion_mlx.custom_kernels.phase_c import W4A8Linear

    in_f, out_f = 64, 8
    lin = nn.Linear(in_f, out_f, bias=True)
    lin.weight = mx.random.normal((out_f, in_f), dtype=mx.float16)
    lin.bias = mx.random.normal((out_f,), dtype=mx.float16)

    class _Wrap(nn.Module):
        def __init__(self, src):
            super().__init__()
            self.lin = src

        def __call__(self, x):
            return self.lin(x)

    wrap = _Wrap(lin)
    nn.quantize(wrap, group_size=64, bits=4)
    q_ref = wrap.lin
    # from_linear quantizes a fresh copy of the same weight values.
    lin2 = nn.Linear(in_f, out_f, bias=True)
    lin2.weight = lin.weight
    lin2.bias = lin.bias
    w4 = W4A8Linear.from_linear(lin2, group_size=64)
    x = mx.random.normal((3, in_f), dtype=mx.float16) * mx.array(0.3, dtype=mx.float16)
    out_w4a8 = w4(x)
    out_w4 = q_ref(x)
    err = float(mx.max(mx.abs(out_w4a8 - out_w4)))
    assert err < 0.2, f"W4A8Linear A8-induced error {err} too large"


def test_convert_to_w4a8_count():
    from fusion_mlx.custom_kernels.phase_c import convert_to_w4a8

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 8)
            self.fc2 = nn.Linear(64, 4)

        def __call__(self, x):
            return self.fc2(self.fc1(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = [Block(), Block()]
            self.head = nn.Linear(64, 2)

        def __call__(self, x):
            return self.head(self.blocks[0](x))

    model = Model()
    _, n = convert_to_w4a8(model, group_size=64)
    # 2 blocks * 2 linears + 1 head = 5.
    assert n == 5, f"convert_to_w4a8 found {n}, expected 5"


def test_nvfp4_dequant_roundtrip():
    from fusion_mlx.custom_kernels.nvfp4 import (
        NVFP4_BLOCK_SIZE,
        dequant_nvfp4,
    )

    out_f, in_f = 4, 16
    numel = out_f * in_f
    # Construct a synthetic NVFP4 weight: pick known E2M1 codes (0..7 + sign).
    codes = mx.array([0, 1, 2, 3, 4, 5, 6, 7] * (numel // 8), dtype=mx.uint8)
    # Pack two codes per byte (little-endian: low nibble = element[2k]).
    low = codes[0::2].astype(mx.uint8)
    high = codes[1::2].astype(mx.uint8) << mx.array(4, dtype=mx.uint8)
    packed = low | high
    scales = mx.ones((numel // NVFP4_BLOCK_SIZE,), dtype=mx.float32)
    w = dequant_nvfp4(packed, scales, (out_f, in_f))
    assert w.dtype == mx.bfloat16
    assert w.shape == (out_f, in_f)
    # Element 0 code=0 -> 0.0, element 1 code=1 -> 0.5, scale 1.0.
    vals = w.astype(mx.float32).flatten().tolist()
    assert abs(vals[0]) < 1e-3
    assert abs(vals[1] - 0.5) < 1e-3


def test_dequant_nvfp4_weights_noop_on_fp16():
    from fusion_mlx.custom_kernels.nvfp4 import dequant_nvfp4_weights

    weights = {
        "layer.weight": mx.random.normal((4, 4), dtype=mx.float16),
        "layer.bias": mx.zeros((4,), dtype=mx.float16),
    }
    before = {k: v for k, v in weights.items()}
    dequant_nvfp4_weights(weights)
    # Non-uint8 checkpoint: untouched (same objects).
    for k in before:
        assert weights[k] is before[k], f"nvfp4 mutated non-NVFP4 key {k}"


def test_fused_gdn_forward_shape_diagonal():
    from fusion_mlx.custom_kernels.phase_c import FusedGDN

    gdn = FusedGDN(channels=8, diagonal=True)
    x = mx.random.normal((2, 8), dtype=mx.float16)
    out = gdn(x)
    assert out.shape == (2, 8)
    assert out.dtype == mx.float16
    # Output finite (no NaN from rsqrt of zero — eps guards).
    assert not mx.isnan(out).any()


def test_fused_gdn_channel_first():
    from fusion_mlx.custom_kernels.phase_c import FusedGDN

    gdn = FusedGDN(channels=4, diagonal=True)
    x = mx.random.normal((4, 2, 2), dtype=mx.float16)
    out = gdn(x)
    assert out.shape == (4, 2, 2)


def test_apply_fused_gdn_noop_no_gdn_modules():
    from fusion_mlx.custom_kernels.phase_c import apply_fused_gdn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)

        def __call__(self, x):
            return self.fc(x)

    model = Model()
    out = apply_fused_gdn(model)
    assert out is model
