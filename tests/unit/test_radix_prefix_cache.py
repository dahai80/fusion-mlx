# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the radix-tree prefix KV cache."""

from __future__ import annotations

from unittest.mock import MagicMock

from fusion_mlx.cache.radix_prefix_cache import RadixPrefixCache, _RadixNode


def _make_paged(block_size: int = 4, max_blocks: int = 64):
    mgr = MagicMock()
    mgr.block_size = block_size

    class _Block:
        def __init__(self, bid, token_count=0):
            self.block_id = bid
            self.token_count = token_count

    counter = {"n": 0}
    mgr.allocated_blocks = {}

    def _create_block_table(request_id):
        from fusion_mlx.cache.paged_cache import BlockTable

        return BlockTable(request_id=request_id)

    def _allocate_block():
        bid = counter["n"]
        counter["n"] += 1
        if bid >= max_blocks:
            return None
        blk = _Block(bid, token_count=block_size)
        mgr.allocated_blocks[bid] = blk
        return blk

    def _increment_ref(bid):
        return True

    def _delete_block_table(request_id):
        return None

    mgr.create_block_table = _create_block_table
    mgr.allocate_block = _allocate_block
    mgr.increment_ref = _increment_ref
    mgr.delete_block_table = _delete_block_table
    return mgr


def _make_cache(block_size: int = 4, max_blocks: int = 64):
    model = MagicMock()
    paged = _make_paged(block_size, max_blocks)
    return RadixPrefixCache(model=model, paged_cache_manager=paged), paged


class TestInsertAndLookup:
    def test_miss_on_empty(self):
        cache, _ = _make_cache()
        bt, remaining = cache.fetch_cache("r1", [1, 2, 3, 4])
        assert bt is None
        assert remaining == [1, 2, 3, 4]
        assert cache.get_stats()["misses"] == 1

    def test_empty_tokens_returns_none(self):
        cache, _ = _make_cache()
        bt, remaining = cache.fetch_cache("r1", [])
        assert bt is None
        assert remaining == []

    def test_store_then_fetch_full_hit(self):
        cache, _ = _make_cache(block_size=4)
        tokens = [10, 11, 12, 13, 14, 15, 16, 17]
        cache.store_cache("r1", tokens, cache_data=[])
        bt, remaining = cache.fetch_cache("r2", tokens)
        assert bt is not None
        assert remaining == []
        assert bt.num_tokens == 8
        assert len(bt.block_ids) == 2
        assert cache.get_stats()["hits"] == 1

    def test_partial_prefix_hit_returns_remaining(self):
        cache, _ = _make_cache(block_size=4)
        cache.store_cache("r1", [10, 11, 12, 13, 14, 15, 16, 17], cache_data=[])
        bt, remaining = cache.fetch_cache(
            "r2", [10, 11, 12, 13, 14, 15, 16, 17, 99, 98, 97, 96]
        )
        assert bt is not None
        assert remaining == [99, 98, 97, 96]
        assert bt.num_tokens == 8

    def test_no_false_match_on_divergent_prefix(self):
        cache, _ = _make_cache(block_size=4)
        cache.store_cache("r1", [10, 11, 12, 13], cache_data=[])
        bt, remaining = cache.fetch_cache("r2", [10, 11, 12, 99])
        assert bt is None
        assert remaining == [10, 11, 12, 99]
        assert cache.get_stats()["misses"] == 1

    def test_subblock_tokens_not_cached(self):
        cache, _ = _make_cache(block_size=4)
        # 5 tokens -> only first block (4) materialized
        cache.store_cache("r1", [1, 2, 3, 4, 5], cache_data=[])
        bt, remaining = cache.fetch_cache("r2", [1, 2, 3, 4, 5, 6, 7, 8])
        assert bt is not None
        assert remaining == [5, 6, 7, 8]
        assert bt.num_tokens == 4


class TestEvictionAndRelease:
    def test_release_clears_request_table(self):
        cache, _ = _make_cache(block_size=4)
        cache.store_cache("r1", [1, 2, 3, 4], cache_data=[])
        assert "r1" in cache._request_tables
        cache.release_cache("r1")
        assert "r1" not in cache._request_tables

    def test_clear_request_entry_retains_blocks(self):
        cache, _ = _make_cache(block_size=4)
        cache.store_cache("r1", [1, 2, 3, 4], cache_data=[])
        cache.clear_request_entry("r1")
        assert "r1" not in cache._request_tables
        # block should still be reusable for a future fetch
        bt, remaining = cache.fetch_cache("r2", [1, 2, 3, 4])
        assert bt is not None
        assert remaining == []

    def test_block_exhaustion_stops_store(self):
        cache, _ = _make_cache(block_size=4, max_blocks=2)
        # 8 tokens need 2 blocks — fills the pool
        cache.store_cache("r1", [1, 2, 3, 4, 5, 6, 7, 8], cache_data=[])
        # another store with new tokens cannot allocate -> stops gracefully
        cache.store_cache("r2", [9, 10, 11, 12, 13, 14, 15, 16], cache_data=[])
        # first request's prefix still fetchable
        bt, _ = cache.fetch_cache("r3", [1, 2, 3, 4])
        assert bt is not None


class TestFork:
    def test_fork_copies_block_ids(self):
        cache, _ = _make_cache(block_size=4)
        cache.store_cache("r1", [1, 2, 3, 4, 5, 6, 7, 8], cache_data=[])
        forked = cache.fork_cache("r1", "r2")
        assert forked is not None
        assert forked.num_tokens == 8
        assert len(forked.block_ids) == 2
        assert "r2" in cache._request_tables

    def test_fork_missing_source_returns_none(self):
        cache, _ = _make_cache()
        assert cache.fork_cache("nope", "r2") is None


class TestInterfaceParity:
    def test_has_block_size_attr(self):
        cache, _ = _make_cache(block_size=8)
        assert cache.block_size == 8

    def test_has_fetch_store_release_fork(self):
        cache, _ = _make_cache()
        for meth in ("fetch_cache", "store_cache", "release_cache", "fork_cache", "clear_request_entry"):
            assert callable(getattr(cache, meth))

    def test_root_is_radix_node(self):
        cache, _ = _make_cache()
        assert isinstance(cache._root, _RadixNode)


class TestSharedPrefixAcrossRequests:
    def test_two_requests_share_prefix_blocks(self):
        cache, _ = _make_cache(block_size=4)
        # request A
        cache.store_cache("a", [1, 2, 3, 4, 5, 6, 7, 8], cache_data=[])
        # request B shares first block, diverges second
        bt, remaining = cache.fetch_cache("b", [1, 2, 3, 4, 20, 21, 22, 23])
        assert bt is not None
        assert bt.num_tokens == 4
        assert remaining == [20, 21, 22, 23]
        # then B stores its divergence
        cache.store_cache("b", [1, 2, 3, 4, 20, 21, 22, 23], cache_data=[])
        # request C with B's full prefix hits both blocks
        bt2, rem2 = cache.fetch_cache("c", [1, 2, 3, 4, 20, 21, 22, 23, 30, 31, 32, 33])
        assert bt2 is not None
        assert bt2.num_tokens == 8
        assert rem2 == [30, 31, 32, 33]
