# SPDX-License-Identifier: Apache-2.0
"""Tests for pure-memory KV reconstruct (#158).

When ``paged_ssd_cache_dir`` is not configured (default config), the prefix
cache must still reconstruct KV tensors so ``cached_tokens > 0`` on prefix hits.
This is exercised via ``PagedSSDCacheManager(hot_cache_only=True, cache_dir=None)``
- an in-memory LRU with no disk backing.
"""

import pytest

from fusion_mlx.cache.paged_ssd_cache import PagedSSDCacheManager

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def _make_cache_data(num_layers=2, seq_len=8, heads=2, head_dim=4):
    return [
        (
            mx.zeros((1, heads, seq_len, head_dim)),
            mx.zeros((1, heads, seq_len, head_dim)),
        )
        for _ in range(num_layers)
    ]


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestPureMemoryKVReconstruct:
    """#158: hot_cache_only=True with cache_dir=None is a pure-memory KV store."""

    @pytest.fixture
    def manager(self):
        mgr = PagedSSDCacheManager(
            cache_dir=None,
            max_size_bytes=100 * 1024**2,
            hot_cache_max_bytes=1 * 1024**2,
            hot_cache_only=True,
            expected_num_layers=2,
            expected_block_size_tokens=64,
        )
        yield mgr
        mgr.close()

    def test_no_disk_backing(self, manager):
        assert manager._cache_dir is None
        assert manager._writer_thread is None
        assert manager._hot_cache_only is True

    def test_save_then_load_roundtrip_tensor_equality(self, manager):
        block_hash = b"pure_mem_roundtrip_0001"
        # Non-zero tensors so equality is meaningful (not just zeros).
        cache_data = [
            (mx.ones((1, 2, 8, 4)) * 3.0, mx.ones((1, 2, 8, 4)) * 5.0) for _ in range(2)
        ]
        ok = manager.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=8,
            model_name="test-model",
            layer_cache_types=["KVCache"] * 2,
        )
        assert ok is True

        data, meta = manager.load_block_with_metadata(block_hash)
        # Rule 9: assert real equality, not truthiness.
        assert data is not None, "load returned None after save (reconstruct failed)"
        assert meta is not None
        assert len(data) == 2
        for (k_orig, v_orig), (k_rest, v_rest) in zip(cache_data, data):
            assert mx.array_equal(k_orig, k_rest), "key tensor mismatch after roundtrip"
            assert mx.array_equal(
                v_orig, v_rest
            ), "value tensor mismatch after roundtrip"
        assert meta["num_layers"] == 2
        assert meta["token_count"] == 8

    def test_pending_writes_does_not_leak(self, manager):
        """save_block must not populate _pending_writes in hot_cache_only
        (writer thread never runs; pending entries would pin tensor memory)."""
        for i in range(5):
            manager.save_block(
                block_hash=f"leak_{i}".encode(),
                cache_data=_make_cache_data(),
                token_count=8,
                model_name="test-model",
                layer_cache_types=["KVCache"] * 2,
            )
        assert len(manager._pending_writes) == 0
        assert len(manager._pending_write_hashes) == 0

    def test_evicted_block_reports_miss_and_cleans_index(self, manager):
        """When LRU evicts a block's tensor, a subsequent load must return None
        (miss) - not crash in _get_file_path on cache_dir=None - and drop the
        stale metadata from _index."""
        mgr = PagedSSDCacheManager(
            cache_dir=None,
            max_size_bytes=100 * 1024**2,
            hot_cache_max_bytes=1,  # evict aggressively
            hot_cache_only=True,
            expected_num_layers=2,
            expected_block_size_tokens=64,
        )
        try:
            first = b"evict_first_0001"
            mgr.save_block(
                block_hash=first,
                cache_data=_make_cache_data(),
                token_count=8,
                model_name="test-model",
                layer_cache_types=["KVCache"] * 2,
            )
            for i in range(4):
                mgr.save_block(
                    block_hash=f"evict_other_{i}".encode(),
                    cache_data=_make_cache_data(),
                    token_count=8,
                    model_name="test-model",
                    layer_cache_types=["KVCache"] * 2,
                )
            data, meta = mgr.load_block_with_metadata(first)
            assert data is None, "evicted block should miss, got data"
            assert meta is None
            assert mgr.get_block_metadata(first) is None
        finally:
            mgr.close()

    def test_delete_block_no_crash(self, manager):
        block_hash = b"pure_mem_delete_0001"
        manager.save_block(
            block_hash=block_hash,
            cache_data=_make_cache_data(),
            token_count=8,
            model_name="test-model",
            layer_cache_types=["KVCache"] * 2,
        )
        # Must not raise (guards disk unlink when _cache_dir is None).
        assert manager.delete_block(block_hash) is True
        assert manager.load_block_with_metadata(block_hash) == (None, None)

    def test_clear_no_crash(self, manager):
        for i in range(3):
            manager.save_block(
                block_hash=f"clear_{i}".encode(),
                cache_data=_make_cache_data(),
                token_count=8,
                model_name="test-model",
                layer_cache_types=["KVCache"] * 2,
            )
        removed = manager.clear()
        assert removed >= 3
        assert len(manager._hot_cache) == 0

    def test_close_without_writer_thread(self, manager):
        # close() must not blow up when _writer_thread is None (hot_cache_only).
        manager.close()

    def test_get_effective_max_size_no_crash(self, manager):
        # _get_effective_max_size must skip shutil.disk_usage when
        # _cache_dir is None (hot_cache_only) - guard prevents NoneType crash.
        result = manager._get_effective_max_size()
        assert result == manager._max_size

    def test_verify_and_repair_index_no_crash(self, manager):
        # verify_and_repair_index must skip the rglob loop when
        # _cache_dir is None (hot_cache_only) - guard prevents NoneType crash.
        report = manager.verify_and_repair_index()
        assert report == {"orphaned_files_removed": 0, "stale_entries_evicted": 0}

    def test_cache_dir_properties_return_none(self, manager):
        # API contract: cache_dir/cache_path are Path | None in pure-memory mode.
        assert manager.cache_dir is None
        assert manager.cache_path is None
