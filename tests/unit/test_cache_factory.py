from unittest.mock import MagicMock, patch

from fusion_mlx.cache.factory import CacheFactory


class TestCacheFactory:
    def test_create_paged_ssd_cache_disabled(self):
        config = MagicMock()
        config.paged_ssd_cache_dir = None
        result = CacheFactory.create_paged_ssd_cache(config)
        assert result is None

    def test_create_paged_ssd_cache_enabled(self):
        config = MagicMock()
        config.paged_ssd_cache_dir = "/tmp/test_cache"
        config.max_paged_ssd_cache_size = 1024

        # Patch at source since it's a local import inside the method
        with patch(
            "fusion_mlx.cache.paged_ssd_cache.PagedSSDCacheManager"
        ) as MockCache:
            result = CacheFactory.create_paged_ssd_cache(config)
            MockCache.assert_called_once()
            call_kwargs = MockCache.call_args
            assert call_kwargs.kwargs["max_cache_size"] == 1024
