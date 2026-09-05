# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the radix-tree prefix KV cache."""

from __future__ import annotations

from unittest.mock import MagicMock

from fusion_mlx.cache.radix_prefix_cache import RadixPrefixCache, _RadixNode


def _make_paged(block_size: int = 4, max_blocks: int = 64):
    mgr = MagicMock()
    mgr.block_size = block_size
    mgr.model_name = "test-model"

    class _Block:
        def __init__(self, bid, token_count=0):
            self.block_id = bid
            self.token_count = token_count
            # chain hash sentinel so fetch_cache's KV-backed guard matches.
            self.block_hash = f"hash-{bid}".encode()

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

    def _free_block(bid):
        mgr.allocated_blocks.pop(bid, None)
        return True

    def _get_block_table(request_id):
        return None

    mgr.create_block_table = _create_block_table
    mgr.allocate_block = _allocate_block
    mgr.increment_ref = _increment_ref
    mgr.delete_block_table = _delete_block_table
    mgr.free_block = _free_block
    mgr.get_block_table = _get_block_table
    mgr.handle_memory_pressure = lambda n: True
    mgr.find_cached_block = lambda tokens, parent_hash, **kw: None
    mgr.register_block_hash = lambda *a, **kw: None
    mgr.get_memory_usage = lambda: {"blocks": len(mgr.allocated_blocks)}
    mgr.clear = lambda: mgr.allocated_blocks.clear()
    return mgr


class _FakeKV:
    # Lightweight stand-in for the BlockAware KV delegate. The radix trie
    # under test is a pure token-id index over block_ids; KV persistence is
    # BlockAware's responsibility and is covered by its own tests. This fake
    # allocates blocks via the paged manager and returns a BlockTable so the
    # trie indexing / fetch matching logic can be exercised in isolation.

    def __init__(self, paged, block_size):
        self.paged_cache = paged
        self.block_size = block_size
        self.paged_ssd_cache = None
        self._request_tables = {}

    def store_cache(self, request_id, tokens, cache_data, **kw):
        import time

        from fusion_mlx.cache.paged_cache import BlockTable
        from fusion_mlx.cache.prefix_cache import BlockCacheEntry

        if not tokens:
            return None
        block_table = BlockTable(request_id=request_id)
        n_full = len(tokens) // self.block_size
        for i in range(n_full):
            blk = self.paged_cache.allocate_block()
            if blk is None:
                break
            blk.token_count = self.block_size
            self.paged_cache.increment_ref(blk.block_id)
            block_table.block_ids.append(blk.block_id)
            block_table.num_tokens += self.block_size
        self._request_tables[request_id] = BlockCacheEntry(
            block_table=block_table, last_access=time.time()
        )
        return block_table

    def fetch_cache(self, request_id, tokens, **kw):
        return None, tokens

    def reconstruct_cache(self, block_table, promote_to_hot_cache=True):
        return []

    def preload_blocks(self, block_table):
        return 0

    def release_cache(self, request_id):
        self._request_tables.pop(request_id, None)
        self.paged_cache.delete_block_table(request_id)

    def clear_request_entry(self, request_id):
        self._request_tables.pop(request_id, None)

    def fork_cache(self, src, new_id):
        import time

        from fusion_mlx.cache.paged_cache import BlockTable
        from fusion_mlx.cache.prefix_cache import BlockCacheEntry

        entry = self._request_tables.get(src)
        if entry is None:
            return None
        new_table = BlockTable(request_id=new_id)
        for bid in entry.block_table.block_ids:
            self.paged_cache.increment_ref(bid)
            blk = self.paged_cache.allocated_blocks.get(bid)
            if blk:
                new_table.block_ids.append(bid)
                new_table.num_tokens += blk.token_count
        self._request_tables[new_id] = BlockCacheEntry(
            block_table=new_table, last_access=time.time()
        )
        return new_table

    def clear(self):
        self._request_tables.clear()

    def set_paged_ssd_cache_manager(self, mgr):
        self.paged_ssd_cache = mgr

    def set_cold_restore_callback(self, cb):
        pass


def _make_cache(block_size: int = 4, max_blocks: int = 64):
    model = MagicMock()
    paged = _make_paged(block_size, max_blocks)
    cache = RadixPrefixCache(model=model, paged_cache_manager=paged)
    # Replace the real BlockAware delegate with the lightweight fake so the
    # trie logic is tested without a full paged-cache + SSD stack.
    cache._kv_cache = _FakeKV(paged, block_size)
    cache.paged_ssd_cache = None
    return cache, paged


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
        for meth in (
            "fetch_cache",
            "store_cache",
            "release_cache",
            "fork_cache",
            "clear_request_entry",
        ):
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
