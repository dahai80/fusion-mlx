# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.cache.interface — CacheManager ABC + utilization."""

from __future__ import annotations

import pytest

from fusion_mlx.cache.interface import CacheManager


class _FixedCache(CacheManager):
    """Concrete cache for testing the ABC contract + utilization property."""

    def __init__(self, size: int, max_size: int):
        self._size = size
        self._max_size = max_size

    def fetch(self, key):
        return (None, False)

    def store(self, key, value) -> bool:
        return True

    def evict(self, key) -> bool:
        return False

    def clear(self) -> int:
        self._size = 0
        return 0

    def get_stats(self):
        from fusion_mlx.cache.stats import BaseCacheStats

        return BaseCacheStats()

    @property
    def size(self) -> int:
        return self._size

    @property
    def max_size(self) -> int:
        return self._max_size


class TestCacheManagerABC:
    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            CacheManager()

    def test_concrete_impl_ok(self):
        c = _FixedCache(size=0, max_size=10)
        assert c.size == 0
        assert c.max_size == 10


class TestUtilization:
    def test_zero_max_size_returns_zero(self):
        c = _FixedCache(size=5, max_size=0)
        assert c.utilization == 0.0

    def test_half_utilization(self):
        c = _FixedCache(size=5, max_size=10)
        assert c.utilization == 0.5

    def test_full_utilization(self):
        c = _FixedCache(size=10, max_size=10)
        assert c.utilization == 1.0

    def test_empty_cache_zero_utilization(self):
        c = _FixedCache(size=0, max_size=10)
        assert c.utilization == 0.0


class TestConcreteMethods:
    def test_fetch_returns_tuple(self):
        c = _FixedCache(size=0, max_size=10)
        val, hit = c.fetch("k")
        assert hit is False
        assert val is None

    def test_store_returns_bool(self):
        c = _FixedCache(size=0, max_size=10)
        assert c.store("k", "v") is True

    def test_evict_returns_bool(self):
        c = _FixedCache(size=0, max_size=10)
        assert c.evict("k") is False

    def test_clear_resets_size(self):
        c = _FixedCache(size=3, max_size=10)
        assert c.clear() == 0
        assert c.size == 0

    def test_get_stats_returns_base(self):
        from fusion_mlx.cache.stats import BaseCacheStats

        c = _FixedCache(size=0, max_size=10)
        stats = c.get_stats()
        assert isinstance(stats, BaseCacheStats)
