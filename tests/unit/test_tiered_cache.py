# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

from fusion_mlx.cache.paged_cache import PagedCacheManager
from fusion_mlx.cache.tiered_cache import TieredCacheManager, TieredCacheStats


class TestTieredCacheStats:
    def test_default_values(self):
        s = TieredCacheStats()
        assert s.hot_hits == 0
        assert s.cold_hits == 0
        assert s.promotions == 0
        assert s.demotions == 0
        assert s.cow_copies_during_demotion == 0

    def test_reset(self):
        s = TieredCacheStats()
        s.hot_hits = 5
        s.cold_hits = 3
        s.promotions = 2
        s.demotions = 1
        s.cow_copies_during_demotion = 1
        s.reset()
        assert s.hot_hits == 0
        assert s.cold_hits == 0
        assert s.promotions == 0
        assert s.demotions == 0
        assert s.cow_copies_during_demotion == 0

    def test_to_dict(self):
        s = TieredCacheStats()
        s.hot_hits = 10
        s.cold_hits = 5
        d = s.to_dict()
        assert d["hot_hits"] == 10
        assert d["cold_hits"] == 5
        assert "hits" in d
        assert "misses" in d


class TestTieredCacheManager:
    def _make_hot(self):
        hot = PagedCacheManager(
            block_size=64,
            max_blocks=100,
            enable_caching=True,
            model_name="test",
            initial_blocks=50,
        )
        return hot

    def _make_cold(self):
        cold = MagicMock()
        cold.fetch = MagicMock(return_value=(None, False))
        cold.save_block = MagicMock(return_value=True)
        cold.evict = MagicMock(return_value=False)
        cold.clear = MagicMock(return_value=0)
        cold.get_stats = MagicMock(
            return_value=MagicMock(to_dict=MagicMock(return_value={}))
        )
        cold.get_stats_dict = MagicMock(return_value={})
        return cold

    def test_init_no_cold(self):
        hot = self._make_hot()
        tm = TieredCacheManager(hot=hot, cold=None)
        assert tm.hot is hot
        assert tm.cold is None

    def test_init_with_cold(self):
        hot = self._make_hot()
        cold = self._make_cold()
        tm = TieredCacheManager(hot=hot, cold=cold)
        assert tm.cold is cold

    def test_fetch_hot_hit(self):
        hot = self._make_hot()
        cold = self._make_cold()
        tm = TieredCacheManager(hot=hot, cold=cold)

        from fusion_mlx.cache.paged_cache import CacheBlock

        block = CacheBlock(block_id=1)
        block.block_hash = b"\x01" * 32
        hot.cached_block_hash_to_block.insert(block.block_hash, block)

        result, found = tm.fetch(block.block_hash)
        assert found is True
        assert result is block
        assert tm.get_stats().hot_hits == 1

    def test_fetch_cold_hit_with_promotion(self):
        hot = self._make_hot()
        cold = self._make_cold()
        tm = TieredCacheManager(hot=hot, cold=cold, promotion_on_cold_hit=True)

        fake_data = ["layer0_data", "layer1_data"]
        cold.fetch = MagicMock(return_value=(fake_data, True))

        block_hash = b"\x02" * 32
        result, found = tm.fetch(block_hash)
        assert found is True
        assert result == fake_data
        assert tm.get_stats().cold_hits == 1
        assert tm.get_stats().promotions == 1

    def test_fetch_cold_hit_no_promotion(self):
        hot = self._make_hot()
        cold = self._make_cold()
        tm = TieredCacheManager(hot=hot, cold=cold, promotion_on_cold_hit=False)

        fake_data = ["layer0_data"]
        cold.fetch = MagicMock(return_value=(fake_data, True))

        result, found = tm.fetch(b"\x03" * 32)
        assert found is True
        assert tm.get_stats().cold_hits == 1
        assert tm.get_stats().promotions == 0

    def test_fetch_miss(self):
        hot = self._make_hot()
        cold = self._make_cold()
        tm = TieredCacheManager(hot=hot, cold=cold)

        result, found = tm.fetch(b"\xff" * 32)
        assert found is False
        assert result is None
        assert tm.get_stats().misses == 1

    def test_store_to_cold(self):
        hot = self._make_hot()
        cold = self._make_cold()
        tm = TieredCacheManager(hot=hot, cold=cold)

        ok = tm.store(b"\x04" * 32, ["data"])
        assert ok is True
        cold.save_block.assert_called_once()

    def test_store_no_cold(self):
        hot = self._make_hot()
        tm = TieredCacheManager(hot=hot, cold=None)
        ok = tm.store(b"\x04" * 32, ["data"])
        assert ok is False

    def test_evict_from_cold(self):
        hot = self._make_hot()
        cold = self._make_cold()
        cold.evict = MagicMock(return_value=True)
        tm = TieredCacheManager(hot=hot, cold=cold)

        ok = tm.evict(b"\x05" * 32)
        assert ok is True

    def test_clear(self):
        hot = self._make_hot()
        cold = self._make_cold()
        cold.clear = MagicMock(return_value=5)
        tm = TieredCacheManager(hot=hot, cold=cold)

        count = tm.clear()
        assert count >= 5
        cold.clear.assert_called_once()

    def test_size_property(self):
        hot = self._make_hot()
        tm = TieredCacheManager(hot=hot, cold=None)
        assert tm.size >= 0

    def test_max_size_property(self):
        hot = self._make_hot()
        tm = TieredCacheManager(hot=hot, cold=None)
        assert tm.max_size == hot.max_blocks * hot.block_size

    def test_maybe_demote_no_cold(self):
        hot = self._make_hot()
        tm = TieredCacheManager(hot=hot, cold=None)
        assert tm.maybe_demote() == 0

    def test_maybe_demote_below_threshold(self):
        hot = self._make_hot()
        cold = self._make_cold()
        tm = TieredCacheManager(hot=hot, cold=cold, demotion_threshold=0.99)
        assert tm.maybe_demote() == 0

    def test_get_tier_stats(self):
        hot = self._make_hot()
        cold = self._make_cold()
        tm = TieredCacheManager(hot=hot, cold=cold)

        stats = tm.get_tier_stats()
        assert "tiered" in stats
        assert "hot" in stats
        assert "cold" in stats

    def test_implements_cache_manager(self):
        hot = self._make_hot()
        tm = TieredCacheManager(hot=hot, cold=None)
        assert hasattr(tm, "fetch")
        assert hasattr(tm, "store")
        assert hasattr(tm, "evict")
        assert hasattr(tm, "clear")
        assert hasattr(tm, "get_stats")
        assert hasattr(tm, "size")
        assert hasattr(tm, "max_size")
