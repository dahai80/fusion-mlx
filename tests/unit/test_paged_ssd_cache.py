import tempfile
from pathlib import Path

from fusion_mlx.cache.paged_ssd_cache import PagedSSDCacheManager


class TestPagedSSDCacheManager:
    def test_init_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "sub" / "cache"
            mgr = PagedSSDCacheManager(
                cache_dir=cache_dir,
                max_size_bytes=10 * 1024 * 1024,
            )
            assert cache_dir.exists()
            mgr.close()

    def test_has_block_returns_false_for_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = PagedSSDCacheManager(
                cache_dir=Path(tmp),
                max_size_bytes=10 * 1024 * 1024,
            )
            assert mgr.has_block(b"\x00" * 16) is False
            mgr.close()

    def test_max_size_bytes_param(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = PagedSSDCacheManager(
                cache_dir=Path(tmp),
                max_size_bytes=2048,
            )
            assert mgr.configured_max_size == 2048
            mgr.close()
