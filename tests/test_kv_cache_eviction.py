import time
from unittest.mock import MagicMock

from fusion_mlx.cache.paged_cache import (
    BlockHash,
    PagedCacheManager,
)


class TestEvictionTiming:
    """Verify LRU eviction respects last_access timestamps."""

    def _make_manager(self, max_blocks=16, initial_blocks=16):
        return PagedCacheManager(
            block_size=4,
            max_blocks=max_blocks,
            initial_blocks=initial_blocks,
            model_name="test",
        )

    def test_lru_order_eviction(self):
        mgr = self._make_manager()
        blocks = mgr.get_new_blocks(5)
        # Touch blocks in reverse order so oldest is first allocated
        time.sleep(0.01)
        for i, b in enumerate(blocks):
            b.block_hash = BlockHash(bytes([i]))
            mgr.cached_block_hash_to_block.insert(b.block_hash, b)
            time.sleep(0.01)

        # Free all so they sit in free queue with ref_count=0
        for b in blocks:
            mgr.free_block(b.block_id)

        evictable = mgr.get_evictable_blocks(3)
        assert len(evictable) == 3
        # Oldest (smallest last_access) should be first
        assert evictable[0].last_access <= evictable[1].last_access
        assert evictable[1].last_access <= evictable[2].last_access

    def test_evict_lru_blocks_respects_order(self):
        mgr = self._make_manager(initial_blocks=5)
        blocks = mgr.get_new_blocks(4)
        for i, b in enumerate(blocks):
            b.block_hash = BlockHash(bytes([i]))
            mgr.cached_block_hash_to_block.insert(b.block_hash, b)
        for b in blocks:
            mgr.free_block(b.block_id)
            # Re-insert after free since free_block clears hash cache
            if b.block_hash is not None:
                mgr.cached_block_hash_to_block.insert(b.block_hash, b)

        evicted = mgr.evict_lru_blocks(2)
        assert evicted == 2
        stats = mgr.get_stats()
        assert stats.evictions == 2

    def test_no_eviction_when_all_referenced(self):
        mgr = self._make_manager(max_blocks=4, initial_blocks=4)
        blocks = mgr.get_new_blocks(3)
        # All blocks have ref_count=1, none are evictable
        evictable = mgr.get_evictable_blocks(10)
        assert evictable == []

    def test_touch_prevents_eviction(self):
        mgr = self._make_manager()
        blocks = mgr.get_new_blocks(3)
        # Free block 0 and 1, keep block 2 referenced
        mgr.free_block(blocks[0].block_id)
        mgr.free_block(blocks[1].block_id)
        mgr.touch([blocks[2]])

        evictable = mgr.get_evictable_blocks(10)
        ids = [b.block_id for b in evictable]
        assert blocks[2].block_id not in ids


class TestSettleBarrier:
    """Test the deferred-clear delay (settle barrier) logic."""

    def test_deferred_clear_delay_default(self):
        from fusion_mlx.scheduler.helpers import _deferred_clear_delay

        s = MagicMock()
        s.running = {}
        s._DEFERRED_CLEAR_DELAY = 4
        assert _deferred_clear_delay(s) == 4

    def test_deferred_clear_delay_scales_with_batch(self):
        from fusion_mlx.scheduler.helpers import _deferred_clear_delay

        s = MagicMock()
        s.running = {str(i): None for i in range(20)}
        s._DEFERRED_CLEAR_DELAY = 4
        # 20 running -> ceil(20/4)=5, 4+5=9
        assert _deferred_clear_delay(s) == 9

    def test_deferred_clear_delay_floor_two(self):
        from fusion_mlx.scheduler.helpers import _deferred_clear_delay

        s = MagicMock()
        s.running = {}
        s._DEFERRED_CLEAR_DELAY = 0
        assert _deferred_clear_delay(s) == 2

    def test_deferred_clear_delay_cap_sixteen(self):
        from fusion_mlx.scheduler.helpers import _deferred_clear_delay

        s = MagicMock()
        s.running = {str(i): None for i in range(200)}
        s._DEFERRED_CLEAR_DELAY = 20
        assert _deferred_clear_delay(s) == 16


class TestTwoPhaseEviction:
    """Test cold eviction followed by permanent eviction."""

    def _make_with_ssd(self):
        mgr = PagedCacheManager(
            block_size=4,
            max_blocks=16,
            initial_blocks=16,
            model_name="test",
        )
        ssd = MagicMock()
        ssd.has_block = MagicMock(return_value=True)
        mgr.set_paged_ssd_cache_manager(ssd)
        return mgr

    def test_mark_block_cold_ref_zero(self):
        mgr = self._make_with_ssd()
        blocks = mgr.get_new_blocks(2)
        b = blocks[0]
        mgr.free_block(b.block_id)
        # ref_count is now 0
        assert mgr.mark_block_cold(b.block_id) is True

    def test_mark_block_cold_ref_nonzero(self):
        mgr = self._make_with_ssd()
        blocks = mgr.get_new_blocks(2)
        # ref_count=1, should fail
        assert mgr.mark_block_cold(blocks[0].block_id) is False

    def test_mark_block_cold_null_block(self):
        mgr = self._make_with_ssd()
        assert mgr.mark_block_cold(0) is False

    def test_permanent_eviction_clears_metadata(self):
        mgr = self._make_with_ssd()
        blocks = mgr.get_new_blocks(2)
        b = blocks[0]
        b.block_hash = BlockHash(b"perm-test")
        mgr.cached_block_hash_to_block.insert(b.block_hash, b)
        mgr.free_block(b.block_id)

        assert mgr.evict_block_permanently(b.block_id) is True
        # Hash should be cleared
        assert b.block_hash is None
        assert b.token_count == 0
        assert b.block_id not in mgr.allocated_blocks

    def test_permanent_eviction_ref_nonzero_fails(self):
        mgr = self._make_with_ssd()
        blocks = mgr.get_new_blocks(2)
        # ref_count=1
        assert mgr.evict_block_permanently(blocks[0].block_id) is False

    def test_permanent_eviction_out_of_range(self):
        mgr = self._make_with_ssd()
        assert mgr.evict_block_permanently(9999) is False

    def test_evict_by_hash(self):
        mgr = self._make_with_ssd()
        blocks = mgr.get_new_blocks(2)
        b = blocks[0]
        h = BlockHash(b"evict-by-hash")
        b.block_hash = h
        mgr.cached_block_hash_to_block.insert(h, b)
        mgr.free_block(b.block_id)
        # Re-insert after free since free_block clears hash cache
        mgr.cached_block_hash_to_block.insert(h, b)

        assert mgr.evict(h) is True
        assert b.block_hash is None

    def test_evict_by_block_id(self):
        mgr = self._make_with_ssd()
        blocks = mgr.get_new_blocks(2)
        b = blocks[0]
        b.block_hash = BlockHash(b"evict-by-id")
        mgr.cached_block_hash_to_block.insert(b.block_hash, b)
        mgr.free_block(b.block_id)

        assert mgr.evict(b.block_id) is True

    def test_handle_memory_pressure_evicts(self):
        mgr = self._make_with_ssd()
        blocks = mgr.get_new_blocks(5)
        for b in blocks:
            mgr.free_block(b.block_id)
        # Request more than available free
        # After freeing 5, free queue should have 11 (16-1 null-5 allocated + 5 freed)
        # So this should pass without eviction
        assert mgr.handle_memory_pressure(3) is True

    def test_cold_blocks_list(self):
        mgr = self._make_with_ssd()
        blocks = mgr.get_new_blocks(3)
        for i, b in enumerate(blocks):
            b.block_hash = BlockHash(bytes([i]))
        mgr.free_block(blocks[0].block_id)
        mgr.free_block(blocks[1].block_id)

        cold = mgr.get_cold_blocks()
        # All 3 have hashes, none are null
        assert len(cold) == 3
        assert mgr.cold_block_count == 3
