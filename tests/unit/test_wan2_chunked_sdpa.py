# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Wan2.1 chunked SDPA (#440 large-seq Metal freeze)."""

import pytest

pytest.importorskip("mlx")  # suite needs mlx runtime; skip if absent

import mlx.core as mx

from fusion_mlx.video.wan2.attention import _attn_chunk_size, _sdpa, _sdpa_chunked


def _rand_qkv(b=1, h=4, s=64, d=8):
    q = mx.random.normal((b, h, s, d))
    k = mx.random.normal((b, h, s, d))
    v = mx.random.normal((b, h, s, d))
    return q, k, v


def test_attn_chunk_size_default_disabled(monkeypatch):
    monkeypatch.delenv("FUSION_WAN2_ATTN_CHUNK", raising=False)
    assert _attn_chunk_size() == 0


def test_attn_chunk_size_env_override(monkeypatch):
    monkeypatch.setenv("FUSION_WAN2_ATTN_CHUNK", "8192")
    assert _attn_chunk_size() == 8192


def test_attn_chunk_size_bad_value_disabled(monkeypatch):
    monkeypatch.setenv("FUSION_WAN2_ATTN_CHUNK", "not-a-number")
    assert _attn_chunk_size() == 0


def test_attn_chunk_size_negative_clamped(monkeypatch):
    monkeypatch.setenv("FUSION_WAN2_ATTN_CHUNK", "-5")
    assert _attn_chunk_size() == 0


def test_sdpa_chunked_disabled_matches_sdpa(monkeypatch):
    monkeypatch.delenv("FUSION_WAN2_ATTN_CHUNK", raising=False)
    q, k, v = _rand_qkv(s=32)
    scale = (q.shape[-1]) ** -0.5
    direct = _sdpa(q, k, v, scale)
    chunked = _sdpa_chunked(q, k, v, scale)
    mx.eval(direct, chunked)
    assert mx.allclose(direct, chunked, atol=1e-5).item() is True


def test_sdpa_chunked_single_chunk_matches_direct(monkeypatch):
    # chunk >= seq -> one chunk, must equal unchunked result
    monkeypatch.setenv("FUSION_WAN2_ATTN_CHUNK", "64")
    q, k, v = _rand_qkv(s=32)
    scale = (q.shape[-1]) ** -0.5
    direct = _sdpa(q, k, v, scale)
    chunked = _sdpa_chunked(q, k, v, scale)
    mx.eval(direct, chunked)
    assert mx.allclose(direct, chunked, atol=1e-5).item() is True


def test_sdpa_chunked_multi_chunk_matches_direct(monkeypatch):
    # split into several chunks; softmax(QK^T)V is row-independent so
    # chunked Q along seq is exact
    monkeypatch.setenv("FUSION_WAN2_ATTN_CHUNK", "16")
    q, k, v = _rand_qkv(s=64)
    scale = (q.shape[-1]) ** -0.5
    direct = _sdpa(q, k, v, scale)
    chunked = _sdpa_chunked(q, k, v, scale)
    mx.eval(direct, chunked)
    assert mx.allclose(direct, chunked, atol=1e-5).item() is True


def test_sdpa_chunked_shape_preserved(monkeypatch):
    monkeypatch.setenv("FUSION_WAN2_ATTN_CHUNK", "8")
    q, k, v = _rand_qkv(b=2, h=6, s=40, d=4)
    scale = (q.shape[-1]) ** -0.5
    out = _sdpa_chunked(q, k, v, scale)
    mx.eval(out)
    assert out.shape == (2, 6, 40, 4)


def test_sdpa_chunked_uneven_last_chunk(monkeypatch):
    # seq not divisible by chunk -> last chunk is short; must still match
    monkeypatch.setenv("FUSION_WAN2_ATTN_CHUNK", "25")
    q, k, v = _rand_qkv(s=70)
    scale = (q.shape[-1]) ** -0.5
    direct = _sdpa(q, k, v, scale)
    chunked = _sdpa_chunked(q, k, v, scale)
    mx.eval(direct, chunked)
    assert mx.allclose(direct, chunked, atol=1e-5).item() is True
