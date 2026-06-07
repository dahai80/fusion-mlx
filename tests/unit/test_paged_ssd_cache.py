import tempfile
from pathlib import Path
from fusion_mlx.cache.paged_ssd_cache import PagedSSDCacheManager


class TestPagedSSDCacheManager:
     def test_init_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "sub" / "cache"
            mgr = PagedSSDCacheManager(cache_dir=str(cache_dir))
            assert cache_dir.exists()

     def test_store_block_returns_false_for_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = PagedSSDCacheManager(cache_dir=tmp)
            assert mgr.store_block(0, []) is False

     def test_store_block_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = PagedSSDCacheManager(cache_dir=tmp)
             # Use a simple list instead of numpy to avoid MLX dependency
            layers = [[1.0, 2.0, 3.0]]
            result = mgr.store_block(1, layers)
            assert result is True
            assert (Path(tmp) / "block_1.safetensors").exists()

     def test_max_cache_size_param(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = PagedSSDCacheManager(cache_dir=tmp, max_cache_size=2048)
            assert mgr.max_cache_size == 2048
