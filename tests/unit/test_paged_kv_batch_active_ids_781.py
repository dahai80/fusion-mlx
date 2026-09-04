# SPDX-License-Identifier: Apache-2.0
# #781: batch-wide active_ids must reach FusionPagedKVPool.alloc_block so
# LRU eviction never reclaims an actively-decoding peer, and evicted
# victims' dangling block_tables are cleared fail-visible.
from __future__ import annotations

import mlx.core as mx

from fusion_mlx.custom_kernels.fusion_paged_kv import (
    _GLOBAL_CACHE_REGISTRY,
    evict_request_by_id,
    invalidate_request,
)
from fusion_mlx.custom_kernels.paged_kv_pool import (
    FusionPagedKVPool,
    FusionPagedRequestCache,
)


def _make_pool(cap: int = 4) -> FusionPagedKVPool:
    return FusionPagedKVPool(
        block_size=2,
        num_blocks=cap,
        n_kv_heads=1,
        head_dim=4,
        dtype=mx.float32,
    )


class TestBatchWideActiveIds:
    # #781: with the batch-wide active set published, a request actively
    # decoding in the batch is NOT evicted to make room for another peer.

    def test_active_peer_not_evicted(self):
        pool = _make_pool(cap=4)
        # request "a" fills all 4 blocks (block_size=2 -> 8 tokens = 4 blocks).
        cache_a = FusionPagedRequestCache(pool, "a")
        k = mx.zeros((1, 1, 8, 4), dtype=mx.float32)
        v = mx.zeros((1, 1, 8, 4), dtype=mx.float32)
        cache_a.update_and_fetch(k, v)
        assert pool.available() == 0

        # Publish batch-wide active set: both "a" and "b" are decoding.
        pool.set_active_ids({"a", "b"})
        # "b" tries to alloc -> pool exhausted, but "a" is active -> no
        # evictable idle -> must raise (not silently evict "a").
        cache_b = FusionPagedRequestCache(pool, "b")
        try:
            cache_b.update_and_fetch(
                mx.zeros((1, 1, 2, 4), dtype=mx.float32),
                mx.zeros((1, 1, 2, 4), dtype=mx.float32),
            )
            raised = False
        except RuntimeError:
            raised = True
        assert raised, "active peer 'a' must not be evicted to fit 'b'"

    def test_idle_peer_evicted_when_not_in_active_set(self):
        pool = _make_pool(cap=4)
        cache_a = FusionPagedRequestCache(pool, "a")
        k = mx.zeros((1, 1, 8, 4), dtype=mx.float32)
        v = mx.zeros((1, 1, 8, 4), dtype=mx.float32)
        cache_a.update_and_fetch(k, v)
        assert pool.available() == 0

        # Only "b" is active now; "a" is idle -> evictable.
        pool.set_active_ids({"b"})
        cache_b = FusionPagedRequestCache(pool, "b")
        # Should succeed by evicting idle "a".
        cache_b.update_and_fetch(
            mx.zeros((1, 1, 2, 4), dtype=mx.float32),
            mx.zeros((1, 1, 2, 4), dtype=mx.float32),
        )
        assert pool.available() >= 0

    def test_touch_active_refreshes_lru(self):
        # #781: a decode step that allocates no new blocks must still
        # refresh last-access so LRU does not go stale and evict a peer
        # that is actually active.
        pool = _make_pool(cap=4)
        pool._last_access["a"] = 1
        pool._last_access["b"] = 2
        pool.set_active_ids({"a", "b"})
        pool.touch_active()
        step_after = pool._step
        # both active ids refreshed to the latest step.
        assert pool._last_access["a"] == step_after
        assert pool._last_access["b"] == step_after


class TestVictimInvalidation:
    # #781: when LRU reclaims a peer's blocks, that peer's registered
    # cache handles must have their dangling block_table cleared so a
    # later fetch fails visibly instead of reading the wrong request's
    # data.

    def test_invalidate_clears_dangling_block_table(self):
        pool = _make_pool(cap=4)
        cache_a = FusionPagedRequestCache(pool, "a")
        _GLOBAL_CACHE_REGISTRY["a"] = [cache_a]
        cache_a.update_and_fetch(
            mx.zeros((1, 1, 8, 4), dtype=mx.float32),
            mx.zeros((1, 1, 8, 4), dtype=mx.float32),
        )
        assert cache_a.block_table != []

        cleared = invalidate_request("a")
        assert cleared == 1
        assert cache_a.block_table == []
        assert cache_a.offset == 0
        # registry entry preserved for eventual completion cleanup.
        assert "a" in _GLOBAL_CACHE_REGISTRY
        _GLOBAL_CACHE_REGISTRY.pop("a", None)

    def test_evict_callback_fires_on_lru_reclaim(self):
        pool = _make_pool(cap=4)
        cache_a = FusionPagedRequestCache(pool, "a")
        _GLOBAL_CACHE_REGISTRY["a"] = [cache_a]
        cache_a.update_and_fetch(
            mx.zeros((1, 1, 8, 4), dtype=mx.float32),
            mx.zeros((1, 1, 8, 4), dtype=mx.float32),
        )
        from fusion_mlx.custom_kernels.fusion_paged_kv import invalidate_request

        pool.set_evict_callback(invalidate_request)
        # "b" active, "a" idle -> eviction reclaims "a" + fires callback.
        pool.set_active_ids({"b"})
        cache_b = FusionPagedRequestCache(pool, "b")
        cache_b.update_and_fetch(
            mx.zeros((1, 1, 2, 4), dtype=mx.float32),
            mx.zeros((1, 1, 2, 4), dtype=mx.float32),
        )
        # callback cleared "a"'s dangling block_table.
        assert cache_a.block_table == []
        assert cache_a.offset == 0
        _GLOBAL_CACHE_REGISTRY.pop("a", None)
        evict_request_by_id("b")
