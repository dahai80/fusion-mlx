import mlx.core as mx
import pytest

from fusion_mlx.custom_kernels.fusion_paged_attention import (
    metal_available,
    paged_decode_attention,
)


def _ref_decode(q, k_all, v_all, scale, gqa_factor):
    # k_all/v_all: [B, n_kv_heads, num_kv, head_dim] materialized logical view
    # q: [B, n_heads, 1, head_dim]
    B, n_heads, _, head_dim = q.shape
    n_kv = k_all.shape[1]
    num_kv = k_all.shape[2]
    k_view = mx.reshape(k_all, (B, n_kv, num_kv, head_dim))
    v_view = mx.reshape(v_all, (B, n_kv, num_kv, head_dim))
    if gqa_factor > 1:
        k_view = mx.repeat(k_view, gqa_factor, axis=1)
        v_view = mx.repeat(v_view, gqa_factor, axis=1)
    return mx.fast.scaled_dot_product_attention(q, k_view, v_view, scale=scale)


@pytest.mark.skipif(not metal_available(), reason="metal kernel unavailable")
@pytest.mark.parametrize(
    "block_size,num_kv,n_heads,n_kv_heads,head_dim,gqa",
    [
        (16, 33, 8, 8, 64, 1),
        (16, 33, 8, 2, 64, 4),
        (16, 16, 4, 4, 32, 1),
        (16, 1, 8, 2, 64, 4),
    ],
)
def test_paged_decode_matches_sdpa(
    block_size, num_kv, n_heads, n_kv_heads, head_dim, gqa
):
    mx.random.seed(7)
    B = 1
    scale = 1.0 / (head_dim**0.5)
    q = mx.random.normal(shape=(B, n_heads, 1, head_dim)) * 0.1
    keys_pool = mx.random.normal(shape=(8, B, n_kv_heads, block_size, head_dim)) * 0.1
    values_pool = mx.random.normal(shape=(8, B, n_kv_heads, block_size, head_dim)) * 0.1
    num_blocks_used = (num_kv + block_size - 1) // block_size
    block_table = mx.array(list(range(num_blocks_used)), dtype=mx.uint32)
    out = paged_decode_attention(
        q,
        keys_pool,
        values_pool,
        block_table,
        num_kv,
        scale,
        gqa,
    )
    # materialize the SAME logical view for the reference
    k_parts = [keys_pool[pb] for pb in range(num_blocks_used)]
    v_parts = [values_pool[pb] for pb in range(num_blocks_used)]
    k_all = mx.concatenate(
        [p.reshape(B, n_kv_heads, block_size, head_dim) for p in k_parts], axis=2
    )[:, :, :num_kv, :]
    v_all = mx.concatenate(
        [p.reshape(B, n_kv_heads, block_size, head_dim) for p in v_parts], axis=2
    )[:, :, :num_kv, :]
    ref = _ref_decode(q, k_all, v_all, scale, gqa)
    assert out.shape == ref.shape
    rel = mx.max(mx.abs(out - ref)) / (mx.max(mx.abs(ref)) + 1e-9)
    assert float(rel) < 1e-2, f"rel diff {float(rel)} too large"


@pytest.mark.skipif(not metal_available(), reason="metal kernel unavailable")
def test_paged_decode_batched_per_seq_kv_lens():
    mx.random.seed(42)
    B = 2
    n_heads = 8
    n_kv_heads = 2
    head_dim = 64
    gqa = n_heads // n_kv_heads
    block_size = 16
    scale = 1.0 / (head_dim**0.5)
    kv_lens = [33, 17]
    max_kv = max(kv_lens)
    max_blocks = (max_kv + block_size - 1) // block_size
    q = mx.random.normal(shape=(B, n_heads, 1, head_dim)) * 0.1
    keys_pool = (
        mx.random.normal(shape=(max_blocks, B, n_kv_heads, block_size, head_dim)) * 0.1
    )
    values_pool = (
        mx.random.normal(shape=(max_blocks, B, n_kv_heads, block_size, head_dim)) * 0.1
    )
    bt_2d = mx.array([list(range(max_blocks)) for _ in range(B)], dtype=mx.uint32)
    kv_lens_arr = mx.array(kv_lens, dtype=mx.uint32)
    out = paged_decode_attention(
        q,
        keys_pool,
        values_pool,
        bt_2d,
        kv_lens_arr,
        scale,
        gqa,
    )
    assert out.shape == (B, n_heads, 1, head_dim)
    for bi in range(B):
        nkv = kv_lens[bi]
        nb = (nkv + block_size - 1) // block_size
        k_parts = [keys_pool[pb, bi] for pb in range(nb)]
        v_parts = [values_pool[pb, bi] for pb in range(nb)]
        k_all = mx.concatenate(
            [p.reshape(n_kv_heads, block_size, head_dim) for p in k_parts], axis=1
        )[:, :nkv, :]
        v_all = mx.concatenate(
            [p.reshape(n_kv_heads, block_size, head_dim) for p in v_parts], axis=1
        )[:, :nkv, :]
        k_4d = k_all[None]
        v_4d = v_all[None]
        if gqa > 1:
            k_4d = mx.repeat(k_4d, gqa, axis=1)
            v_4d = mx.repeat(v_4d, gqa, axis=1)
        ref = mx.fast.scaled_dot_product_attention(
            q[bi : bi + 1], k_4d, v_4d, scale=scale
        )
        rel = mx.max(mx.abs(out[bi : bi + 1] - ref)) / (mx.max(mx.abs(ref)) + 1e-9)
        assert float(rel) < 1e-2, f"batch {bi} rel diff {float(rel)} too large"
