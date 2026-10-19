# SPDX-License-Identifier: Apache-2.0
# Tests for #1269: paged-cache content-invalidation callbacks must never fire
# while the paged-cache locks are held (ABBA deadlock with the prefix cache
# store worker holding _cache_lock and acquiring _hash_map_lock).
import threading

from fusion_mlx.cache.paged_cache import PagedCacheManager


class TestDeferredInvalidation:
    def _manager(self, max_blocks=8):
        return PagedCacheManager(
            block_size=4, max_blocks=max_blocks, initial_blocks=max_blocks
        )

    def test_callback_not_fired_under_paged_lock(self):
        # The wedge: eviction/free fired the prefix callback inline while
        # _hash_map_lock was held; callback takes prefix _cache_lock -> ABBA.
        mgr = self._manager()
        lock_state = {}

        def cb(block_id):
            lock_state["hash_map_locked"] = mgr._hash_map_lock.locked()
            lock_state["block_table_locked"] = mgr._block_table_lock.locked()

        mgr.register_block_freed_callback(cb)
        block = mgr.allocate_block()
        assert block is not None
        block.block_hash = b"abc"
        assert mgr.free_block(block.block_id) is True
        # Callback ran, and ran AFTER the paged locks were released.
        assert lock_state == {"hash_map_locked": False, "block_table_locked": False}

    def test_memory_pressure_reclaim_fires_deferred_callback(self):
        # Production shape: store_cache worker retains blocks in the hash
        # index with ref_count=0; memory-pressure reclaim evicts them and
        # their hash entries, firing invalidation callbacks.
        mgr = self._manager(max_blocks=4)
        seen = []

        def cb(block_id):
            seen.append(block_id)
            assert not mgr._hash_map_lock.locked()

        mgr.register_block_freed_callback(cb)
        blocks = mgr.get_new_blocks(3)
        for b in blocks:
            b.block_hash = f"hash-{b.block_id}".encode()
            b.ref_count = 0
        assert mgr.handle_memory_pressure(3) is True
        assert sorted(seen) == sorted(b.block_id for b in blocks)

    def test_handle_memory_pressure_flushes(self):
        mgr = self._manager(max_blocks=4)
        fired = []
        mgr.register_block_freed_callback(lambda bid: fired.append(bid))
        blocks = mgr.get_new_blocks(3)
        for b in blocks:
            b.block_hash = f"hash-{b.block_id}".encode()
            b.ref_count = 0
        assert mgr.handle_memory_pressure(2) is True
        assert len(fired) >= 1
        assert mgr._pending_content_invalidated == []

    def test_refcount_free_no_callback(self):
        mgr = self._manager()
        fired = []
        mgr.register_block_freed_callback(lambda bid: fired.append(bid))
        block = mgr.allocate_block()
        block.ref_count = 2
        assert mgr.free_block(block.block_id) is False
        assert fired == []

    def test_flush_without_callbacks_is_noop(self):
        mgr = self._manager()
        block = mgr.allocate_block()
        block.block_hash = b"x"
        mgr.free_block(block.block_id)
        mgr._flush_content_invalidated()

    def test_pending_queue_drained_after_flush(self):
        mgr = self._manager()
        fired = []
        mgr.register_block_freed_callback(fired.append)
        blocks = mgr.get_new_blocks(2)
        for b in blocks:
            b.block_hash = b"h"
        mgr.free_blocks_batch(blocks)
        assert mgr._pending_content_invalidated == []
        assert sorted(fired) == sorted(b.block_id for b in blocks)


class TestNoDeadlockComposition:
    def test_evict_during_held_prefix_cache_lock(self):
        # Reproduce the production interleave: prefix store worker holds
        # _cache_lock and calls paged find_cached_block (takes
        # _hash_map_lock); step thread runs evict_lru_blocks concurrently.
        # With inline callbacks this deadlocks; with deferral it completes.
        from fusion_mlx.cache.prefix_cache import BlockAwarePrefixCache

        paged = PagedCacheManager(block_size=4, max_blocks=16, initial_blocks=16)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)
        done = threading.Event()

        def store_worker():
            # _store_cache_locked holds _cache_lock across the whole store
            # and calls paged find_cached_block (acquires _hash_map_lock).
            with cache._cache_lock:
                paged.find_cached_block(list(range(4)))
            done.set()

        t = threading.Thread(target=store_worker, daemon=True)
        t.start()
        # Concurrent eviction path: fires the block-freed callback.
        paged.evict_lru_blocks(1)
        assert done.wait(timeout=10), "store worker deadlocked against eviction"
        t.join(timeout=1)
