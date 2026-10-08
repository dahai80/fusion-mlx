# SPDX-License-Identifier: Apache-2.0
"""Tests for MLLMPrefixCacheManager — concurrency + correctness.

#1015: _cache (OrderedDict) and _current_memory were read/written from
multiple threads without a lock. Concurrent fetch/store/eviction could
mutate the OrderedDict mid-iteration, raising "OrderedDict mutated during
iteration" RuntimeError and drifting _current_memory.
"""

import threading

import pytest

from fusion_mlx.cache.mllm_cache import MLLMPrefixCacheManager


def _store_entry(mgr, key: str, token_count: int = 4) -> None:
    mgr.store(
        images=[],
        prompt=key,
        vision_embeddings=None,
        kv_cache=[],
        token_ids=list(range(token_count)),
        num_image_tokens=0,
        model_name="test",
    )


class TestMLLMCacheConcurrency:
    @pytest.fixture
    def mgr(self):
        m = MLLMPrefixCacheManager(max_entries=20, max_memory_mb=4)
        # _evict_by_memory's Layer 2 calls mx.get_cache_memory() /
        # mx.clear_cache(), which would clear the live MLX allocator if a
        # prod server shares this process. Unit tests only need the
        # logical-memory layer — stub Layer 2 out.
        original_evict = m._evict_by_memory

        def safe_evict(required_size: int) -> None:
            while m._current_memory + required_size > m.max_memory and m._cache:
                oldest_key = next(iter(m._cache))
                oldest_entry = m._cache.pop(oldest_key)
                m._current_memory -= oldest_entry.memory_size
                m.stats.evictions += 1

        m._evict_by_memory = safe_evict
        yield m
        # Restore in case the manager is reused (it won't be, but keep clean).
        m._evict_by_memory = original_evict

    def test_concurrent_store_fetch_clear_no_runtime_error(self, mgr):
        """#1015: concurrent store / fetch / clear from many threads must
        not raise "OrderedDict mutated during iteration" or corrupt
        _current_memory."""
        errors: list[Exception] = []
        iters_per_thread = 200

        def writer():
            try:
                for i in range(iters_per_thread):
                    _store_entry(mgr, f"w{i % 10}", token_count=4)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def reader():
            try:
                for i in range(iters_per_thread):
                    mgr.fetch([], f"w{i % 10}", [1, 2, 3])
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def clearer():
            try:
                for _ in range(iters_per_thread // 4):
                    mgr.clear()
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = (
            [threading.Thread(target=writer, daemon=True) for _ in range(2)]
            + [threading.Thread(target=reader, daemon=True) for _ in range(2)]
            + [threading.Thread(target=clearer, daemon=True)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
            assert not t.is_alive(), f"thread {t.name} hung"

        assert not errors, f"concurrent access raised: {errors}"

    def test_store_then_fetch_hit(self, mgr):
        """Basic correctness after locking: store then fetch returns a hit."""
        _store_entry(mgr, "hello", token_count=4)
        entry, match_len = mgr.fetch([], "hello", [0, 1, 2, 3])
        assert entry is not None
        assert match_len == 4

    def test_current_memory_consistent_after_overwrite(self, mgr):
        """#1015 + E-35: overwriting the same key must not double-count
        _current_memory, even under the lock."""
        _store_entry(mgr, "dup", token_count=4)
        mem_after_first = mgr._current_memory
        _store_entry(mgr, "dup", token_count=4)
        assert (
            mgr._current_memory == mem_after_first
        ), "overwrite double-counted _current_memory"

    def test_clear_resets_memory(self, mgr):
        _store_entry(mgr, "a", token_count=4)
        _store_entry(mgr, "b", token_count=4)
        assert len(mgr) == 2
        mgr.clear()
        assert mgr._current_memory == 0
        assert len(mgr) == 0

    def test_get_stats_under_lock(self, mgr):
        _store_entry(mgr, "s", token_count=4)
        stats = mgr.get_stats()
        assert stats["entries"] == 1
        assert stats["memory_used_mb"] >= 0
