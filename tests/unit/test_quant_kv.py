# SPDX-License-Identifier: Apache-2.0
"""Tests for PR-K: Q4_0/Q8_0 quantized KV online decompression + FP32 softmax.

Verifies the llama.cpp block codecs against a numpy reference, the chunked
online attention path against an exact fp32 manual reference (golden KL
< 1e-6), the degraded path, the OFF-mode stock passthrough, and
ShimQuantizedKVCache storage/trim/meta semantics.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from fusion_mlx.eval.golden_reference import assert_logits_aligned
from fusion_mlx.shim.quant_kv import (
    ShimQuantizedKVCache,
    degraded_attention,
    is_quant_kv_enabled,
    online_attention,
    q40_dequantize,
    q40_quantize,
    q80_dequantize,
    q80_quantize,
)


def _enable(monkeypatch):
    monkeypatch.setenv("FUSION_SHIM_QUANT_KV", "1")


def _kv(b=2, h=4, t=10, d=64, seed=0, dtype=mx.float16):
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((b, h, t, d)).astype(np.float16)).astype(dtype)
    return x


def _q(b=2, h=4, l=3, d=64, seed=1, dtype=mx.float16):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((b, h, l, d)).astype(np.float16)).astype(dtype)


def _manual_fp32_attention(q, k, v, scale, mask=None, gqa_groups=1):
    # Exact fp32 reference: dequantized inputs, fp32 scores + softmax.
    qf = q.astype(mx.float32) * scale
    kf = k.astype(mx.float32)
    vf = v.astype(mx.float32)
    if gqa_groups > 1:
        B, Hkv, T, D = kf.shape
        H = Hkv * gqa_groups
        qf = mx.reshape(qf, (B, Hkv, gqa_groups, q.shape[-2], D))
        s = qf @ mx.transpose(kf, (0, 1, 3, 2))[:, :, None]
        p = mx.softmax(s, axis=-1)
        out = p @ vf[:, :, None]
        return mx.reshape(out, (B, H, q.shape[-2], D))
    s = qf @ mx.transpose(kf, (0, 1, 3, 2))
    if mask is not None:
        s = mx.where(mask.astype(mx.bool_), s, mx.array(float("-inf"), mx.float32))
    p = mx.softmax(s, axis=-1)
    return p @ vf


class TestQ40Codec:
    def test_shapes_and_compression(self):
        x = _kv()
        d, p = q40_quantize(x)
        assert d.shape == (2, 4, 10, 2)
        assert p.shape == (2, 4, 10, 32)
        assert d.dtype == mx.float16 and p.dtype == mx.uint8
        assert p.nbytes + d.nbytes < x.nbytes * 0.5

    def test_roundtrip_error_bound(self):
        x = _kv()
        d, p = q40_quantize(x)
        xr = q40_dequantize(d, p)
        # Q4_0 step = d, d = max|x|/8.
        dmax = mx.max(mx.abs(x), axis=-1) / 8.0
        err = mx.abs(xr - x.astype(mx.float32))
        # Bound is d (not d/2): the block max element rounds to ±8 and
        # clamps to ±7 (llama.cpp block_q4_0 convention).
        assert mx.max(err).item() <= mx.max(dmax).item() + 1e-6

    def test_numpy_reference_exact(self):
        x = _kv(b=1, h=1, t=1, d=64)
        d, p = q40_quantize(x)
        xb = np.array(x[0, 0, 0].astype(mx.float32))
        blocks = xb.reshape(2, 32)
        dn = np.abs(blocks).max(axis=-1) / 8.0
        q_ref = np.clip(np.round(blocks / dn[:, None]) + 8, 0, 15).astype(np.uint8)
        pk = np.array(p[0, 0, 0])
        got = np.stack([pk & 0xF, pk >> 4], axis=-1).reshape(2, 32)
        assert (got == q_ref).all()
        np.testing.assert_allclose(
            np.array(d[0, 0, 0].astype(mx.float32)), dn, rtol=1e-3
        )

    def test_rejects_bad_head_dim(self):
        with pytest.raises(ValueError):
            q40_quantize(mx.zeros((1, 1, 1, 50)))


class TestQ80Codec:
    def test_shapes_and_compression(self):
        x = _kv()
        d, q = q80_quantize(x)
        assert d.shape == (2, 4, 10, 2)
        assert q.shape == (2, 4, 10, 64)
        assert q.dtype == mx.int8
        assert q.nbytes + d.nbytes < x.nbytes * 0.6

    def test_roundtrip_error_bound(self):
        x = _kv()
        d, q = q80_quantize(x)
        xr = q80_dequantize(d, q)
        dmax = mx.max(mx.abs(x), axis=-1) / 8.0
        err = mx.abs(xr - x.astype(mx.float32))
        assert mx.max(err).item() <= mx.max(dmax).item() * 0.5 + 1e-6

    def test_q80_tighter_than_q40(self):
        x = _kv()
        d4, p4 = q40_quantize(x)
        d8, q8 = q80_quantize(x)
        err4 = mx.max(mx.abs(q40_dequantize(d4, p4) - x.astype(mx.float32))).item()
        err8 = mx.max(mx.abs(q80_dequantize(d8, q8) - x.astype(mx.float32))).item()
        assert err8 <= err4


class TestOnlineAttention:
    def test_matches_manual_fp32_reference(self, monkeypatch):
        _enable(monkeypatch)
        x = _kv()
        d, p = q80_quantize(x)
        xr = q80_dequantize(d, p)
        q = _q()
        out = online_attention(q, (d, p), (d, p), 1 / 8, chunk=4)
        ref = _manual_fp32_attention(q, xr, xr, 1 / 8)
        assert_logits_aligned(ref, out, tol=1e-6, label="online vs manual fp32")

    def test_chunk_invariance(self, monkeypatch):
        _enable(monkeypatch)
        x = _kv(t=32)
        d, p = q80_quantize(x)
        q = _q()
        a = online_attention(q, (d, p), (d, p), 1 / 8, chunk=4)
        b = online_attention(q, (d, p), (d, p), 1 / 8, chunk=1024)
        assert_logits_aligned(b, a, tol=1e-6, label="chunk invariance")

    def test_fully_masked_first_chunk_no_nan(self, monkeypatch):
        _enable(monkeypatch)
        # Regression: a caller-supplied bool mask that hides ALL of the
        # first chunk for every row previously NaN-poisoned the running
        # softmax (exp(-inf - -inf) = NaN) even though later chunks have
        # visible keys — full-bleed SDPA would be finite.
        x = _kv(t=8)
        d, p = q80_quantize(x)
        xr = q80_dequantize(d, p)
        q = _q()
        L = q.shape[-2]
        mask = mx.array(np.ones((L, 8), dtype=bool))
        mask[:, :4] = False
        out = online_attention(q, (d, p), (d, p), 1 / 8, mask=mask, chunk=4)
        assert not np.isnan(np.asarray(out)).any()
        ref = _manual_fp32_attention(q, xr, xr, 1 / 8, mask=mask)
        assert_logits_aligned(ref, out, tol=1e-6, label="masked-first-chunk")

    def test_causal_mask(self, monkeypatch):
        _enable(monkeypatch)
        x = _kv()
        d, p = q80_quantize(x)
        xr = q80_dequantize(d, p)
        q = _q()
        out = online_attention(q, (d, p), (d, p), 1 / 8, mask="causal")
        # Queries sit at the END of the cache: query i is at global pos
        # T - L + i and attends keys j <= that.
        T, L = 10, 3
        qi = np.arange(L)[:, None] + (T - L)
        ki = np.arange(T)[None]
        causal = mx.array(qi >= ki)
        ref = _manual_fp32_attention(q, xr, xr, 1 / 8, mask=causal)
        assert_logits_aligned(ref, out, tol=1e-6, label="causal online")

    def test_additive_mask(self, monkeypatch):
        _enable(monkeypatch)
        x = _kv()
        d, p = q80_quantize(x)
        xr = q80_dequantize(d, p)
        q = _q()
        m = mx.triu(mx.full((3, 10), float("-inf"), mx.float16), k=1)
        out = online_attention(q, (d, p), (d, p), 1 / 8, mask=m)
        ref = mx.fast.scaled_dot_product_attention(
            q, xr.astype(mx.float16), xr.astype(mx.float16), scale=1 / 8, mask=m
        )
        assert_logits_aligned(ref, out, tol=1e-6, label="additive mask online")

    def test_gqa(self, monkeypatch):
        _enable(monkeypatch)
        x = _kv()
        d, p = q80_quantize(x)
        xr = q80_dequantize(d, p)
        q = _q(h=8)
        out = online_attention(q, (d, p), (d, p), 1 / 8)
        ref = _manual_fp32_attention(q, xr, xr, 1 / 8, gqa_groups=2)
        assert_logits_aligned(ref, out, tol=1e-6, label="gqa online")

    def test_q40_path(self, monkeypatch):
        _enable(monkeypatch)
        x = _kv()
        d, p = q40_quantize(x)
        xr = q40_dequantize(d, p)
        q = _q()
        out = online_attention(q, (d, p), (d, p), 1 / 8)
        ref = _manual_fp32_attention(q, xr, xr, 1 / 8)
        assert_logits_aligned(ref, out, tol=1e-6, label="q40 online")

    def test_rejects_gqa_mismatch(self, monkeypatch):
        _enable(monkeypatch)
        x = _kv(h=4)
        d, p = q80_quantize(x)
        q = _q(h=6)
        with pytest.raises(ValueError):
            online_attention(q, (d, p), (d, p), 1 / 8)

    def test_rejects_unknown_mask_mode(self, monkeypatch):
        _enable(monkeypatch)
        x = _kv()
        d, p = q80_quantize(x)
        with pytest.raises(ValueError):
            online_attention(_q(), (d, p), (d, p), 1 / 8, mask="sliding")


class TestDegradedAttention:
    def test_matches_manual_fp32_reference(self, monkeypatch):
        _enable(monkeypatch)
        x = _kv()
        d, p = q80_quantize(x)
        xr = q80_dequantize(d, p)
        q = _q()
        out = degraded_attention(q, (d, p), (d, p), 1 / 8)
        ref = _manual_fp32_attention(q, xr, xr, 1 / 8)
        assert_logits_aligned(ref, out, tol=1e-6, label="degraded vs manual fp32")

    def test_online_equals_degraded(self, monkeypatch):
        _enable(monkeypatch)
        x = _kv()
        d, p = q80_quantize(x)
        q = _q()
        on = online_attention(q, (d, p), (d, p), 1 / 8, chunk=4)
        dg = degraded_attention(q, (d, p), (d, p), 1 / 8)
        assert_logits_aligned(dg, on, tol=1e-6, label="online vs degraded")


class TestShimQuantizedKVCache:
    def test_rejects_bad_bits(self):
        with pytest.raises(ValueError):
            ShimQuantizedKVCache(bits=5)

    def test_disabled_passthrough_matches_stock(self, monkeypatch):
        monkeypatch.setenv("FUSION_SHIM_QUANT_KV", "0")
        assert is_quant_kv_enabled() is False
        c = ShimQuantizedKVCache(bits=4)
        x = _kv()
        q = _q()
        c.update_and_fetch(x, x)
        out = c.attention(q, 1 / 8)
        ref = mx.fast.scaled_dot_product_attention(q, x, x, scale=1 / 8)
        assert mx.array_equal(out, ref)
        assert c.offset == 10
        assert c.nbytes == x.nbytes * 2

    def test_enabled_stores_quantized(self, monkeypatch):
        _enable(monkeypatch)
        c = ShimQuantizedKVCache(bits=4)
        x = _kv()
        c.update_and_fetch(x, x)
        d, p = c.keys
        assert p.dtype == mx.uint8
        assert c.nbytes < x.nbytes * 2 * 0.5
        assert c.offset == 10

    def test_multi_step_and_attention(self, monkeypatch):
        _enable(monkeypatch)
        c = ShimQuantizedKVCache(bits=8)
        x = _kv()
        q = _q()
        c.update_and_fetch(x[:, :, :6], x[:, :, :6])
        c.update_and_fetch(x[:, :, 6:], x[:, :, 6:])
        out = c.attention(q, 1 / 8)
        xr = q80_dequantize(*c.keys)
        ref = _manual_fp32_attention(q, xr, xr, 1 / 8)
        assert_logits_aligned(ref, out, tol=1e-6, label="cache multi-step")

    def test_trim_and_meta_state(self, monkeypatch):
        _enable(monkeypatch)
        c = ShimQuantizedKVCache(bits=8, chunk=256)
        x = _kv()
        c.update_and_fetch(x, x)
        assert c.is_trimmable() is True
        assert c.trim(3) == 3
        assert c.offset == 7
        assert c.meta_state == ("7", "8", "256")
        c2 = ShimQuantizedKVCache(bits=4)
        c2.meta_state = c.meta_state
        assert c2.offset == 7
        assert c2.quant_bits == 8
        assert c2.chunk == 256

    def test_empty_and_state(self, monkeypatch):
        _enable(monkeypatch)
        c = ShimQuantizedKVCache(bits=4)
        assert c.empty() is True
        assert c.nbytes == 0
        x = _kv()
        c.update_and_fetch(x, x)
        assert c.empty() is False
        kd, kq = c.keys
        prev_state = c.state
        c.state = prev_state
        assert c.keys[0] is kd
        assert c.keys[1] is kq

    def test_no_bits_attribute(self):
        # mlx-lm's sdpa wrapper routes on hasattr(cache, "bits") — the
        # shim cache must NOT expose it (its block layout is not the
        # affine quantized_matmul format).
        c = ShimQuantizedKVCache(bits=4)
        assert not hasattr(c, "bits")

    def test_disabled_trim(self, monkeypatch):
        monkeypatch.setenv("FUSION_SHIM_QUANT_KV", "0")
        c = ShimQuantizedKVCache(bits=4)
        x = _kv()
        c.update_and_fetch(x, x)
        assert c.trim(4) == 4
        assert c.offset == 6
        assert c.keys.shape[-2] == 6
        assert c.values.shape[-2] == 6
