# SPDX-License-Identifier: Apache-2.0

import mlx.core as mx

from fusion_mlx.custom_kernels.paged_kv_pool import FusionPagedKVPool


def _make_pool(num_blocks=4):
    return FusionPagedKVPool(
        block_size=1,
        num_blocks=num_blocks,
        n_kv_heads=1,
        head_dim=4,
        dtype=mx.float32,
    )


def test_lru_evicts_idle_request():
    pool = _make_pool(num_blocks=4)
    rid_a = "req-a"
    for _ in range(4):
        pool.alloc_block(rid_a)
    assert pool.available() == 0
    rid_b = "req-b"
    pb = pool.alloc_block(rid_b, active_ids={rid_b})
    assert pb in range(4)
    assert pool.available() == 3
    assert rid_a not in pool.in_use.values()
    assert rid_b in pool.in_use.values()


def test_lru_no_evict_active_self():
    pool = _make_pool(num_blocks=4)
    rid_a = "req-a"
    for _ in range(4):
        pool.alloc_block(rid_a)
    assert pool.available() == 0
    try:
        pool.alloc_block(rid_a, active_ids={rid_a})
        raise AssertionError("expected RuntimeError for self-eviction")
    except RuntimeError as exc:
        assert "pool exhausted" in str(exc)


def test_lru_touch_flips_victim_order():
    pool = _make_pool(num_blocks=4)
    rid_a = "req-a"
    for _ in range(2):
        pool.alloc_block(rid_a)
    rid_b = "req-b"
    for _ in range(2):
        pool.alloc_block(rid_b)
    assert pool.available() == 0
    pool.touch(rid_a)
    rid_c = "req-c"
    pb = pool.alloc_block(rid_c, active_ids={rid_c})
    assert pb in range(4)
    assert rid_a in pool.in_use.values()
    assert rid_b not in pool.in_use.values()
    assert rid_c in pool.in_use.values()


def test_lru_no_evictable_raises():
    pool = _make_pool(num_blocks=2)
    rid_a = "req-a"
    rid_b = "req-b"
    pool.alloc_block(rid_a)
    pool.alloc_block(rid_b)
    assert pool.available() == 0
    try:
        pool.alloc_block("req-c", active_ids={rid_a, rid_b})
        raise AssertionError("expected RuntimeError when no evictable idle request")
    except RuntimeError as exc:
        assert "pool exhausted" in str(exc)


def test_alloc_block_backward_compat_active_ids_none():
    pool = _make_pool(num_blocks=2)
    rid_a = "req-a"
    pool.alloc_block(rid_a)
    pool.alloc_block(rid_a)
    assert pool.available() == 0
    try:
        pool.alloc_block(rid_a)
        raise AssertionError("expected RuntimeError (no evictable, self protected)")
    except RuntimeError as exc:
        assert "pool exhausted" in str(exc)
