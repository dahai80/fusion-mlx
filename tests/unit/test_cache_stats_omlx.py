# SPDX-License-Identifier: Apache-2.0
import logging

import pytest

from fusion_mlx.cache.interface import CacheManager
from fusion_mlx.cache.stats import (
    BaseCacheStats,
    PagedCacheStats,
    PagedSSDCacheStats,
    PrefixCacheStats,
    VLMCacheStats,
)

logger = logging.getLogger(__name__)


class TestBaseCacheStats:
    def test_default_values(self):
        stats = BaseCacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0

    def test_with_values(self):
        stats = BaseCacheStats(hits=10, misses=5, evictions=2)
        assert stats.hits == 10
        assert stats.misses == 5
        assert stats.evictions == 2

    def test_total_queries(self):
        stats = BaseCacheStats(hits=10, misses=5)
        assert stats.total_queries == 15

    def test_hit_rate_with_queries(self):
        stats = BaseCacheStats(hits=75, misses=25)
        assert stats.hit_rate == pytest.approx(0.75)

    def test_hit_rate_zero_queries(self):
        stats = BaseCacheStats()
        assert stats.hit_rate == 0.0

    def test_record_hit(self):
        stats = BaseCacheStats()
        stats.record_hit()
        assert stats.hits == 1
        stats.record_hit()
        stats.record_hit()
        assert stats.hits == 3

    def test_record_miss(self):
        stats = BaseCacheStats()
        stats.record_miss()
        assert stats.misses == 1

    def test_record_eviction(self):
        stats = BaseCacheStats()
        stats.record_eviction()
        assert stats.evictions == 1

    def test_reset(self):
        stats = BaseCacheStats(hits=100, misses=50, evictions=10)
        stats.reset()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0

    def test_to_dict(self):
        stats = BaseCacheStats(hits=10, misses=5, evictions=2)
        d = stats.to_dict()
        assert d["hits"] == 10
        assert d["misses"] == 5
        assert d["evictions"] == 2


class TestPrefixCacheStats:
    def test_default_values(self):
        stats = PrefixCacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.tokens_saved == 0

    def test_tokens_saved(self):
        stats = PrefixCacheStats(tokens_saved=1000)
        assert stats.tokens_saved == 1000

    def test_total_queries_property(self):
        stats = PrefixCacheStats(hits=10, misses=5)
        assert stats.total_queries == 15


class TestPagedCacheStats:
    def test_default_values(self):
        stats = PagedCacheStats()
        assert stats.total_blocks == 0
        assert stats.allocated_blocks == 0
        assert stats.free_blocks == 0

    def test_with_values(self):
        stats = PagedCacheStats(
            total_blocks=1000,
            allocated_blocks=500,
            free_blocks=500,
            shared_blocks=50,
            total_tokens_cached=32000,
            cow_copies=10,
        )
        assert stats.total_blocks == 1000
        assert stats.allocated_blocks == 500
        assert stats.free_blocks == 500

    def test_utilization(self):
        stats = PagedCacheStats(total_blocks=100, allocated_blocks=75)
        assert stats.utilization == pytest.approx(0.75)

    def test_utilization_zero_blocks(self):
        stats = PagedCacheStats(total_blocks=0)
        assert stats.utilization == 0.0

    def test_reset(self):
        stats = PagedCacheStats(
            hits=100,
            misses=50,
            evictions=10,
            total_blocks=1000,
            allocated_blocks=500,
            cow_copies=20,
        )
        stats.reset()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.cow_copies == 0
        assert stats.total_blocks == 1000
        assert stats.allocated_blocks == 500


class TestVLMCacheStats:
    def test_default_values(self):
        stats = VLMCacheStats()
        assert stats.tokens_saved == 0
        assert stats.image_cache_hits == 0

    def test_record_image_hit(self):
        stats = VLMCacheStats()
        stats.record_image_hit()
        assert stats.image_cache_hits == 1
        stats.record_image_hit()
        stats.record_image_hit()
        assert stats.image_cache_hits == 3

    def test_reset(self):
        stats = VLMCacheStats(hits=100, tokens_saved=5000, image_cache_hits=25)
        stats.reset()
        assert stats.hits == 0
        assert stats.tokens_saved == 0
        assert stats.image_cache_hits == 0


class TestPagedSSDCacheStats:
    def test_default_values(self):
        stats = PagedSSDCacheStats()
        assert stats.saves == 0
        assert stats.loads == 0
        assert stats.errors == 0
        assert stats.total_size_bytes == 0
        assert stats.num_files == 0

    def test_with_values(self):
        stats = PagedSSDCacheStats(
            saves=100,
            loads=50,
            errors=5,
            total_size_bytes=1024 * 1024 * 100,
            num_files=150,
        )
        assert stats.saves == 100
        assert stats.loads == 50
        assert stats.errors == 5

    def test_save_rate(self):
        stats = PagedSSDCacheStats(saves=90, errors=10)
        assert stats.save_rate == pytest.approx(0.9)

    def test_save_rate_no_operations(self):
        stats = PagedSSDCacheStats()
        assert stats.save_rate == 0.0

    def test_record_save(self):
        stats = PagedSSDCacheStats()
        stats.record_save()
        assert stats.saves == 1

    def test_record_load(self):
        stats = PagedSSDCacheStats()
        stats.record_load()
        assert stats.loads == 1
        assert stats.hits == 1

    def test_record_error(self):
        stats = PagedSSDCacheStats()
        stats.record_error()
        assert stats.errors == 1

    def test_reset(self):
        stats = PagedSSDCacheStats(
            hits=100,
            misses=50,
            saves=80,
            loads=70,
            errors=5,
            total_size_bytes=1024 * 1024,
            num_files=100,
        )
        stats.reset()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.saves == 0
        assert stats.loads == 0
        assert stats.errors == 0
        assert stats.total_size_bytes == 1024 * 1024
        assert stats.num_files == 100


class TestCacheManagerInterface:
    def test_interface_methods_are_abstract(self):
        with pytest.raises(TypeError):
            CacheManager()

    def test_utilization_property(self):
        class MockCacheManager(CacheManager):
            def __init__(self, size, max_size):
                self._size = size
                self._max_size = max_size

            def fetch(self, key):
                return None, False

            def store(self, key, value):
                return True

            def evict(self, key):
                return True

            def clear(self):
                return 0

            def get_stats(self):
                return BaseCacheStats()

            @property
            def size(self):
                return self._size

            @property
            def max_size(self):
                return self._max_size

        cache = MockCacheManager(size=75, max_size=100)
        assert cache.utilization == pytest.approx(0.75)
        cache_zero = MockCacheManager(size=0, max_size=0)
        assert cache_zero.utilization == 0.0


class TestStatsIntegration:
    def test_hit_rate_tracking(self):
        stats = BaseCacheStats()
        for _ in range(70):
            stats.record_hit()
        for _ in range(30):
            stats.record_miss()
        assert stats.hits == 70
        assert stats.misses == 30
        assert stats.hit_rate == pytest.approx(0.70)

    def test_prefix_cache_tokens_efficiency(self):
        stats = PrefixCacheStats()
        stats.record_hit()
        stats.tokens_saved += 1024
        stats.record_hit()
        stats.tokens_saved += 512
        stats.record_miss()
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.tokens_saved == 1536
        assert stats.hit_rate == pytest.approx(2 / 3)

    def test_paged_cache_block_tracking(self):
        stats = PagedCacheStats(
            total_blocks=1000, allocated_blocks=100, free_blocks=900
        )
        stats.allocated_blocks += 50
        stats.free_blocks -= 50
        assert stats.allocated_blocks == 150
        assert stats.free_blocks == 850
        assert stats.utilization == pytest.approx(0.15)

    def test_paged_ssd_cache_io_tracking(self):
        stats = PagedSSDCacheStats(total_size_bytes=1024 * 1024 * 1024, num_files=0)
        for _ in range(100):
            stats.record_save()
            stats.num_files += 1
        for _ in range(5):
            stats.record_error()
        for _ in range(80):
            stats.record_load()
        assert stats.saves == 100
        assert stats.errors == 5
        assert stats.loads == 80
        assert stats.hits == 80
        assert stats.num_files == 100
