import mlx.core as mx
import pytest

from fusion_mlx.custom_kernels.paged_kv_pool import (
    FusionPagedKVPool,
    FusionPagedRequestCache,
)


def _fill(cache, n, n_kv_heads=2, head_dim=8):
    for _ in range(n):
        k = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        v = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        cache.update_and_fetch(k, v)


def test_pool_allocates_distinct_blocks_per_request():
    pool = FusionPagedKVPool(block_size=4, num_blocks=16, n_kv_heads=2, head_dim=8)
    a = FusionPagedRequestCache(pool, request_id="a")
    b = FusionPagedRequestCache(pool, request_id="b")
    _fill(a, 5)
    _fill(b, 3)
    assert set(a.block_table).isdisjoint(set(b.block_table))
    assert pool.available() == 16 - len(a.block_table) - len(b.block_table)


def test_pool_free_request_returns_blocks():
    pool = FusionPagedKVPool(block_size=4, num_blocks=16, n_kv_heads=2, head_dim=8)
    a = FusionPagedRequestCache(pool, request_id="a")
    _fill(a, 5)
    used = len(a.block_table)
    pool.free_request("a")
    assert pool.available() == 16


def test_pool_exhausted_raises():
    pool = FusionPagedKVPool(block_size=4, num_blocks=2, n_kv_heads=2, head_dim=8)
    a = FusionPagedRequestCache(pool, request_id="a")
    _fill(a, 8)  # fills 2 blocks
    with pytest.raises(RuntimeError, match="pool exhausted"):
        _fill(a, 1)  # needs a 3rd block


def test_request_cache_matches_independent_paged_cache():
    from fusion_mlx.custom_kernels.paged_kv_cache import FusionPagedKVCache

    n_kv_heads, head_dim = 2, 8
    mx.random.seed(3)
    pool = FusionPagedKVPool(
        block_size=4,
        num_blocks=32,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dtype=mx.float32,
    )
    pooled_a = FusionPagedRequestCache(pool, request_id="a")
    pooled_b = FusionPagedRequestCache(pool, request_id="b")
    mx.random.seed(3)
    solo_a = FusionPagedKVCache(block_size=4, num_blocks=16)
    solo_b = FusionPagedKVCache(block_size=4, num_blocks=16)
    for _ in range(9):
        ka = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        va = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        kb = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        vb = mx.random.normal(shape=(1, n_kv_heads, 1, head_dim)) * 0.1
        pa_k, pa_v = pooled_a.update_and_fetch(ka, va)
        sa_k, sa_v = solo_a.update_and_fetch(ka, va)
        pb_k, pb_v = pooled_b.update_and_fetch(kb, vb)
        sb_k, sb_v = solo_b.update_and_fetch(kb, vb)
        assert mx.allclose(pa_k, sa_k).item()
        assert mx.allclose(pa_v, sa_v).item()
        assert mx.allclose(pb_k, sb_k).item()
        assert mx.allclose(pb_v, sb_v).item()


def test_request_cache_merge_stacks_batch():
    n_kv_heads, head_dim = 2, 8
    pool = FusionPagedKVPool(
        block_size=4, num_blocks=32, n_kv_heads=n_kv_heads, head_dim=head_dim
    )
    a = FusionPagedRequestCache(pool, request_id="a")
    b = FusionPagedRequestCache(pool, request_id="b")
    _fill(a, 5)
    _fill(b, 3)
    merged = FusionPagedRequestCache.merge([a, b])
    assert merged.state[0].shape[0] == 2  # B=N=2
    assert merged.offset == 5
