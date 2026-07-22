# SPDX-License-Identifier: Apache-2.0
# Unit tests for the NVFP4 (E2M1 + E4M3 block scale) weight reader.
#
# Known-answer vectors come from the NVIDIA NVFP4 E2M1 value set
# {0, 0.5, 1, 1.5, 2, 3, 4, 6} (sign in bit 3), NOT from the reader under
# test, so a wrong LUT fails the assertion rather than hiding behind it.

import mlx.core as mx
import pytest

from fusion_mlx.custom_kernels.nvfp4 import (
    NVFP4_BLOCK_SIZE,
    dequant_nvfp4,
    dequant_nvfp4_weights,
)

# E4M3 byte encodings (verified against mx.from_fp8) for deterministic scales.
_E4M3_1_0 = 0x38  # 1.0
_E4M3_2_0 = 0x40  # 2.0


def _pack(codes):
    # Pack a list of 4-bit codes into uint8 bytes, little-endian nibble order
    # (element[2k] in low nibble). Pads to NVFP4_BLOCK_SIZE with code 0.
    codes = list(codes)
    if len(codes) % NVFP4_BLOCK_SIZE != 0:
        pad = NVFP4_BLOCK_SIZE - (len(codes) % NVFP4_BLOCK_SIZE)
        codes += [0] * pad
    packed = [(codes[2 * k + 1] << 4) | codes[2 * k] for k in range(len(codes) // 2)]
    return mx.array(packed, dtype=mx.uint8)


def test_e2m1_lut_known_answer_all_codes():
    # All 16 E2M1 codes with a unit block scale must produce the spec values.
    codes = list(range(16))
    packed = _pack(codes)
    scale = mx.array([_E4M3_1_0], dtype=mx.uint8)
    out = dequant_nvfp4(packed, scale, (16,)).astype(mx.float32)
    expected = [
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
    ]
    for got, exp in zip(out.tolist(), expected):
        assert got == pytest.approx(exp, abs=1e-2), (got, exp)


def test_e4m3_block_scale_applied():
    # A scale of 2.0 must double every magnitude.
    codes = [2, 4, 6] + [0] * 13  # magnitudes 1, 2, 4
    packed = _pack(codes)
    scale = mx.array([_E4M3_2_0], dtype=mx.uint8)
    out = dequant_nvfp4(packed, scale, (16,)).astype(mx.float32).tolist()
    assert out[0] == pytest.approx(2.0, abs=1e-2)
    assert out[1] == pytest.approx(4.0, abs=1e-2)
    assert out[2] == pytest.approx(8.0, abs=1e-2)


def test_sign_bit_negative_codes():
    codes = [9, 11, 15] + [0] * 13  # -0.5, -1.5, -6.0
    packed = _pack(codes)
    scale = mx.array([_E4M3_1_0], dtype=mx.uint8)
    out = dequant_nvfp4(packed, scale, (16,)).astype(mx.float32).tolist()
    assert out[0] == pytest.approx(-0.5, abs=1e-2)
    assert out[1] == pytest.approx(-1.5, abs=1e-2)
    assert out[2] == pytest.approx(-6.0, abs=1e-2)


def test_multiple_blocks_per_scale_2d():
    # 2 blocks (32 weights), one scale per block, reshaped to (2, 16).
    codes = [6] * 16 + [4] * 16  # block0 all 4.0, block1 all 2.0
    packed = _pack(codes)
    scales = mx.array([_E4M3_1_0, _E4M3_2_0], dtype=mx.uint8)  # 1.0, 2.0
    out = dequant_nvfp4(packed, scales, (2, 16)).astype(mx.float32)
    # block0: 4.0 * 1.0 = 4.0 ; block1: 2.0 * 2.0 = 4.0
    assert all(v == pytest.approx(4.0, abs=1e-2) for v in out.reshape(-1).tolist())


def test_round_trip_pack_dequant_self_consistent():
    # Quantize known bf16 values to the nearest E2M1 magnitude (scale 1.0),
    # pack, dequant, and recover exactly the quantized values. This pins the
    # pack/unpack/LUT path end-to-end.
    mag = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    lut_signed = mag + [-m for m in mag]
    src = [
        0.4,
        0.6,
        1.2,
        2.9,
        -3.1,
        -5.9,
        4.1,
        0.0,
        1.4,
        2.2,
        3.0,
        5.9,
        -0.5,
        -1.5,
        -4.0,
        6.0,
    ]
    # Nearest E2M1 code (index into lut_signed).
    quant_codes = [min(range(16), key=lambda c: abs(lut_signed[c] - v)) for v in src]
    expected = [lut_signed[c] for c in quant_codes]
    packed = _pack(quant_codes)
    scale = mx.array([_E4M3_1_0], dtype=mx.uint8)
    out = dequant_nvfp4(packed, scale, (16,)).astype(mx.float32).tolist()
    for got, exp in zip(out, expected):
        assert got == pytest.approx(exp, abs=1e-2), (got, exp)


def test_non_divisible_block_size_raises():
    # 6 weights (3 bytes) is not block-aligned -> exporter must pad.
    packed = mx.array([0x00, 0x00, 0x00], dtype=mx.uint8)
    scale = mx.array([_E4M3_1_0], dtype=mx.uint8)
    with pytest.raises(ValueError):
        dequant_nvfp4(packed, scale, (6,))


def test_dequant_weights_dict_transform():
    # A uint8 weight + sibling _weight_scale_inv uint8 scale dequantizes
    # in place and consumes the scale key.
    codes = [6] * 16  # all 4.0
    packed = _pack(codes)
    scale = mx.array([_E4M3_1_0], dtype=mx.uint8)
    weights = {
        "layer.0.weight": packed,
        "layer.0.weight_scale_inv": scale,
        "layer.0.bias": mx.zeros((16,), dtype=mx.bfloat16),
    }
    dequant_nvfp4_weights(weights)
    assert "layer.0.weight_scale_inv" not in weights
    assert weights["layer.0.weight"].dtype == mx.bfloat16
    assert weights["layer.0.bias"].dtype == mx.bfloat16  # untouched
    out = weights["layer.0.weight"].astype(mx.float32).tolist()
    assert all(v == pytest.approx(4.0, abs=1e-2) for v in out)


def test_dequant_weights_2d_shape_inference():
    # 2D scale (out, in/16) drives shape inference to (out, in).
    out_f, in_f = 4, 16
    codes = [2] * (out_f * in_f)  # all 1.0
    packed = _pack(codes)
    n_blocks = (out_f * in_f) // NVFP4_BLOCK_SIZE
    scales = mx.full((out_f, in_f // NVFP4_BLOCK_SIZE), _E4M3_1_0, dtype=mx.uint8)
    assert scales.size == n_blocks
    weights = {"w.weight": packed, "w.weight_scale": scales}
    dequant_nvfp4_weights(weights)
    assert weights["w.weight"].shape == (out_f, in_f)
    assert "w.weight_scale" not in weights


def test_non_nvfp4_dict_untouched():
    # A normal bf16/fp32 checkpoint has no uint8 weights -> no-op.
    weights = {
        "a.weight": mx.zeros((4, 4), dtype=mx.bfloat16),
        "b.weight_scale": mx.ones((4,), dtype=mx.float32),
    }
    before = {k: v for k, v in weights.items()}
    dequant_nvfp4_weights(weights)
    assert set(weights.keys()) == set(before.keys())
    assert weights["a.weight"].dtype == mx.bfloat16


def test_size_mismatch_skips_dequant():
    # uint8 weight + sibling scale but WRONG size relation -> skip, no crash.
    packed = mx.zeros((4,), dtype=mx.uint8)  # 8 weights, needs 0.5 block
    scale = mx.array([_E4M3_1_0, _E4M3_1_0], dtype=mx.uint8)  # 2 scales, mismatch
    weights = {"w.weight": packed, "w.weight_scale_inv": scale}
    dequant_nvfp4_weights(weights)
    # Not dequantized: still uint8, scale still present.
    assert weights["w.weight"].dtype == mx.uint8
    assert "w.weight_scale_inv" in weights
