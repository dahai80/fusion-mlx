# SPDX-License-Identifier: Apache-2.0
"""#1208: Wan2.2 q8 fp32 containment.

The #500/#503 NaN fix cast q8 GEMM inputs to float32 above
FUSION_WAN2_Q8_FP32_SEQ — but the fp32 OUTPUT flowed back into the
residual stream, so after block 1 every op in all 30 blocks ran fp32
(~6.7x FLOP-utilization collapse, ~95 s/step at seq=18480). The fix
keeps fp32 GEMM accumulation but casts outputs straight back to the
residual dtype: fp32 never re-enters the stream.
"""

import os
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn

from fusion_mlx.video.wan2.attention import (
    WanCrossAttention,
    WanSelfAttention,
    _q8_fp32_accum,
)
from fusion_mlx.video.wan2.generate import WAN2_SAFE_SEQ, _auto_enable_attn_chunk
from fusion_mlx.video.wan2.transformer import WanFFN

BF16 = mx.bfloat16


class _RecordingLinear(nn.Module):
    """nn.Linear stand-in that records its input dtype per call."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.weight = mx.ones((dim_out, dim_in), dtype=BF16) * 0.01
        self.bias = mx.zeros((dim_out,), dtype=BF16)
        self.input_dtypes = []

    def __call__(self, x):
        self.input_dtypes.append(x.dtype)
        return (x @ self.weight.T + self.bias).astype(x.dtype)


def test_q8_fp32_accum_casts_in_and_back():
    lin = _RecordingLinear(8, 8)
    x = mx.zeros((1, 4, 8), dtype=BF16)
    out = _q8_fp32_accum(lin, x, out_dtype=BF16)
    assert lin.input_dtypes == [mx.float32]
    assert out.dtype == BF16


def test_q8_fp32_accum_no_out_dtype_keeps_fp32():
    lin = _RecordingLinear(8, 8)
    x = mx.zeros((1, 4, 8), dtype=BF16)
    out = _q8_fp32_accum(lin, x)
    assert lin.input_dtypes == [mx.float32]
    assert out.dtype == mx.float32


def _cast_norms_bf16(attn):
    # Real checkpoints store RMSNorm weights in bf16; freshly initialized
    # ones are float32 and would promote the short-seq path (test artifact).
    for name in ("norm_q", "norm_k"):
        norm = getattr(attn, name, None)
        if norm is not None:
            norm.weight = norm.weight.astype(BF16)
    return attn


def _make_self_attn():
    attn = WanSelfAttention(dim=64, num_heads=4)
    for name in ("q", "k", "v", "o"):
        setattr(attn, name, _RecordingLinear(64, 64))
    return _cast_norms_bf16(attn)


def test_self_attn_long_seq_output_stays_bf16():
    attn = _make_self_attn()
    s = 200
    x = mx.zeros((1, s, 64), dtype=BF16)
    grid_sizes = [([1, 10, 10], [10, 10, 10])]
    freqs = mx.zeros((64, 2), dtype=mx.float32)

    def fake_rope(t, gs, freqs, precomputed_cos_sin=None):
        return t.astype(mx.float32)

    with (
        patch("fusion_mlx.video.wan2.attention.rope_apply", fake_rope),
        patch.dict(os.environ, {"FUSION_WAN2_Q8_FP32_SEQ": "64"}),
    ):
        out = attn(x, seq_lens=[s], grid_sizes=grid_sizes, freqs=freqs)

    # The o-proj output that re-enters the residual stream must NOT be
    # fp32 — fp32 poisoning is the #1208 root cause.
    assert out.dtype == BF16
    # q/k/v/o GEMMs still run fp32-input (fp32 accumulation, #500/#503).
    assert attn.q.input_dtypes == [mx.float32]
    assert attn.k.input_dtypes == [mx.float32]
    assert attn.v.input_dtypes == [mx.float32]
    assert attn.o.input_dtypes == [mx.float32]


def test_self_attn_short_seq_path_unchanged():
    attn = _make_self_attn()
    s = 16
    x = mx.zeros((1, s, 64), dtype=BF16)
    grid_sizes = [([1, 4, 4], [4, 4, 4])]
    freqs = mx.zeros((16, 2), dtype=mx.float32)

    def fake_rope(t, gs, freqs, precomputed_cos_sin=None):
        return t.astype(BF16)

    with patch("fusion_mlx.video.wan2.attention.rope_apply", fake_rope):
        out = attn(x, seq_lens=[s], grid_sizes=grid_sizes, freqs=freqs)

    assert out.dtype == BF16
    # No fp32 GEMM below the threshold — legacy fast path intact.
    assert attn.q.input_dtypes == [BF16]
    assert attn.o.input_dtypes == [BF16]


def _make_cross_attn():
    attn = WanCrossAttention(dim=64, num_heads=4)
    for name in ("q", "k", "v", "o"):
        setattr(attn, name, _RecordingLinear(64, 64))
    return _cast_norms_bf16(attn)


def test_cross_attn_long_seq_output_stays_bf16():
    attn = _make_cross_attn()
    s = 200
    x = mx.zeros((1, s, 64), dtype=BF16)
    context = mx.zeros((1, 8, 64), dtype=BF16)

    with patch.dict(os.environ, {"FUSION_WAN2_Q8_FP32_SEQ": "64"}):
        out = attn(x, context)

    assert out.dtype == BF16
    # q GEMM sees the fp32 stream input; k/v see fp32 context (kept for
    # the same accumulation safety margin); o sees fp32 SDPA output.
    assert attn.q.input_dtypes == [mx.float32]
    assert attn.k.input_dtypes == [mx.float32]
    assert attn.v.input_dtypes == [mx.float32]
    assert attn.o.input_dtypes == [mx.float32]


def test_cross_attn_short_seq_path_unchanged():
    attn = _make_cross_attn()
    x = mx.zeros((1, 16, 64), dtype=BF16)
    context = mx.zeros((1, 8, 64), dtype=BF16)

    out = attn(x, context)

    assert out.dtype == BF16
    assert attn.q.input_dtypes == [BF16]
    assert attn.k.input_dtypes == [BF16]


def test_ffn_long_seq_output_matches_input_dtype():
    ffn = WanFFN(dim=64, ffn_dim=128)
    ffn.fc1 = _RecordingLinear(64, 128)
    ffn.fc2 = _RecordingLinear(128, 64)
    x = mx.zeros((1, 200, 64), dtype=BF16)

    with patch.dict(os.environ, {"FUSION_WAN2_Q8_FP32_SEQ": "64"}):
        out = ffn(x)

    # Residual stream stays bf16 — fp32 confined to the fc1->fc2 body.
    assert out.dtype == BF16
    assert ffn.fc1.input_dtypes == [mx.float32]
    assert ffn.fc2.input_dtypes == [mx.float32]


def test_ffn_short_seq_path_unchanged():
    ffn = WanFFN(dim=64, ffn_dim=128)
    ffn.fc1 = _RecordingLinear(64, 128)
    ffn.fc2 = _RecordingLinear(128, 64)
    x = mx.zeros((1, 16, 64), dtype=BF16)

    out = ffn(x)

    assert out.dtype == BF16
    assert ffn.fc1.input_dtypes == [BF16]
    assert ffn.fc2.input_dtypes == [BF16]


def test_auto_chunk_derivation_fewer_dispatches():
    # #1208: geometric chunk derivation — seq just past the threshold
    # should get 2 Q-chunks, not 3 (8192 hard-code paid 3 for seq=18480).
    cases = {
        16384: 0,  # at/below threshold -> no chunking
        18480: WAN2_SAFE_SEQ,  # 2 chunks (was 3 at 8192)
        27280: WAN2_SAFE_SEQ,  # 2 chunks
        50000: WAN2_SAFE_SEQ,  # 4 chunks (was 7)
    }
    for seq, expected in cases.items():
        os.environ.pop("FUSION_WAN2_ATTN_CHUNK", None)
        _auto_enable_attn_chunk(seq)
        got = int(os.environ.get("FUSION_WAN2_ATTN_CHUNK", "0"))
        assert got == expected, f"seq={seq}: chunk={got}, expected {expected}"
        if got:
            # User-set env must never be overridden.
            os.environ["FUSION_WAN2_ATTN_CHUNK"] = "4096"
            _auto_enable_attn_chunk(seq)
            assert os.environ["FUSION_WAN2_ATTN_CHUNK"] == "4096"
        os.environ.pop("FUSION_WAN2_ATTN_CHUNK", None)
