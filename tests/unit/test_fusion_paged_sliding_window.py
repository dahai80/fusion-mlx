from __future__ import annotations

import mlx.core as mx
import pytest

from fusion_mlx.custom_kernels.fusion_paged_attention import (
    metal_available,
    paged_decode_attention,
)


def _build_pool(num_kv, block_size, B, n_kv_heads, head_dim, seed=7):
    mx.random.seed(seed)
    num_blocks = (num_kv + block_size - 1) // block_size
    keys_pool = (
        mx.random.normal(shape=(num_blocks, B, n_kv_heads, block_size, head_dim)) * 0.1
    )
    values_pool = (
        mx.random.normal(shape=(num_blocks, B, n_kv_heads, block_size, head_dim)) * 0.1
    )
    block_table = mx.array(list(range(num_blocks)), dtype=mx.uint32)
    return keys_pool, values_pool, block_table, num_blocks


def _logical_kv(
    keys_pool, num_blocks_used, num_kv, B, n_kv_heads, block_size, head_dim
):
    k_parts = [keys_pool[pb] for pb in range(num_blocks_used)]
    k_all = mx.concatenate(
        [p.reshape(B, n_kv_heads, block_size, head_dim) for p in k_parts], axis=2
    )[:, :, :num_kv, :]
    return k_all


def _ref_full_causal(q, k_all, v_all, scale, gqa_factor, softcap=0.0):
    B, n_heads, _, head_dim = q.shape
    n_kv = k_all.shape[1]
    num_kv = k_all.shape[2]
    if gqa_factor > 1:
        k_view = mx.repeat(k_all, gqa_factor, axis=1)
        v_view = mx.repeat(v_all, gqa_factor, axis=1)
    else:
        k_view = k_all
        v_view = v_all
    scores = mx.matmul(q, k_view.transpose(0, 1, 3, 2)) * float(scale)
    if softcap > 0.0:
        scores = mx.tanh(scores / float(softcap)) * float(softcap)
    scores = mx.softmax(scores, axis=-1)
    out = mx.matmul(scores, v_view)
    return out


def _ref_sliding_window(q, k_all, v_all, scale, gqa_factor, window):
    B, n_heads, _, head_dim = q.shape
    n_kv = k_all.shape[1]
    num_kv = k_all.shape[2]
    if gqa_factor > 1:
        k_view = mx.repeat(k_all, gqa_factor, axis=1)
        v_view = mx.repeat(v_all, gqa_factor, axis=1)
    else:
        k_view = k_all
        v_view = v_all
    scores = mx.matmul(q, k_view.transpose(0, 1, 3, 2)) * float(scale)
    q_pos = num_kv - 1
    mask = mx.full((1, 1, 1, num_kv), -1e9)
    for t in range(num_kv):
        if t >= q_pos - window + 1 and t <= q_pos:
            mask[..., 0, 0, 0, t] = 0.0
    scores = scores + mask
    scores = mx.softmax(scores, axis=-1)
    out = mx.matmul(scores, v_view)
    return out


@pytest.mark.skipif(not metal_available(), reason="metal kernel unavailable")
def test_full_causal_unchanged():
    block_size = 16
    num_kv = 33
    n_heads = 8
    n_kv_heads = 2
    head_dim = 64
    gqa = n_heads // n_kv_heads
    B = 1
    scale = 1.0 / (head_dim**0.5)
    mx.random.seed(7)
    q = mx.random.normal(shape=(B, n_heads, 1, head_dim)) * 0.1
    keys_pool, values_pool, block_table, num_blocks = _build_pool(
        num_kv, block_size, B, n_kv_heads, head_dim
    )
    out_default = paged_decode_attention(
        q, keys_pool, values_pool, block_table, num_kv, scale, gqa
    )
    out_explicit = paged_decode_attention(
        q,
        keys_pool,
        values_pool,
        block_table,
        num_kv,
        scale,
        gqa,
        sliding_window=0,
        softcap=0.0,
    )
    rel = mx.max(mx.abs(out_default - out_explicit)) / (
        mx.max(mx.abs(out_default)) + 1e-9
    )
    assert float(rel) < 1e-6, f"default vs explicit-zero rel diff {float(rel)}"


@pytest.mark.skipif(not metal_available(), reason="metal kernel unavailable")
def test_sliding_window_matches_reference():
    block_size = 16
    num_kv = 40
    n_heads = 4
    n_kv_heads = 4
    head_dim = 32
    gqa = 1
    B = 1
    window = 16
    scale = 1.0 / (head_dim**0.5)
    mx.random.seed(11)
    q = mx.random.normal(shape=(B, n_heads, 1, head_dim)) * 0.1
    keys_pool, values_pool, block_table, num_blocks = _build_pool(
        num_kv, block_size, B, n_kv_heads, head_dim, seed=11
    )
    out = paged_decode_attention(
        q,
        keys_pool,
        values_pool,
        block_table,
        num_kv,
        scale,
        gqa,
        sliding_window=window,
    )
    k_all = _logical_kv(
        keys_pool, num_blocks, num_kv, B, n_kv_heads, block_size, head_dim
    )
    v_all = _logical_kv(
        values_pool, num_blocks, num_kv, B, n_kv_heads, block_size, head_dim
    )
    ref = _ref_sliding_window(q, k_all, v_all, scale, gqa, window)
    assert out.shape == ref.shape
    rel = mx.max(mx.abs(out - ref)) / (mx.max(mx.abs(ref)) + 1e-9)
    assert float(rel) < 1e-2, f"sliding window rel diff {float(rel)} too large"


@pytest.mark.skipif(not metal_available(), reason="metal kernel unavailable")
def test_softcap_matches_reference():
    block_size = 16
    num_kv = 33
    n_heads = 8
    n_kv_heads = 2
    head_dim = 64
    gqa = 4
    B = 1
    softcap = 1.0
    scale = 1.0 / (head_dim**0.5)
    mx.random.seed(13)
    q = mx.random.normal(shape=(B, n_heads, 1, head_dim)) * 0.5
    keys_pool, values_pool, block_table, num_blocks = _build_pool(
        num_kv, block_size, B, n_kv_heads, head_dim, seed=13
    )
    out = paged_decode_attention(
        q,
        keys_pool,
        values_pool,
        block_table,
        num_kv,
        scale,
        gqa,
        softcap=softcap,
    )
    k_all = _logical_kv(
        keys_pool, num_blocks, num_kv, B, n_kv_heads, block_size, head_dim
    )
    v_all = _logical_kv(
        values_pool, num_blocks, num_kv, B, n_kv_heads, block_size, head_dim
    )
    ref = _ref_full_causal(q, k_all, v_all, scale, gqa, softcap=softcap)
    assert out.shape == ref.shape
    rel = mx.max(mx.abs(out - ref)) / (mx.max(mx.abs(ref)) + 1e-9)
    assert float(rel) < 1e-2, f"softcap rel diff {float(rel)} too large"
