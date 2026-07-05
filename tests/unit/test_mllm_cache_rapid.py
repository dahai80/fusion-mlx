# SPDX-License-Identifier: Apache-2.0
"""Tests for MLLM (Multimodal Language Model) KV cache functionality
(migrated from Rapid-MLX)."""

import os
import tempfile

import pytest

from fusion_mlx.cache.mllm_cache import (
    MLLMCacheStats,
    MLLMPrefixCacheEntry,
    MLLMPrefixCacheManager,
    compute_image_hash,
    compute_images_hash,
)

MLLMCacheEntry = MLLMPrefixCacheEntry
MLLMCacheManager = MLLMPrefixCacheManager


class TestMLLMCacheStats:

    def test_initial_stats(self):
        stats = MLLMCacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.tokens_saved == 0
        assert stats.image_cache_hits == 0
        assert stats.total_queries == 0
        assert stats.evictions == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        stats = MLLMCacheStats(hits=3, misses=7, total_queries=10)
        assert stats.hit_rate == 0.3

    def test_hit_rate_zero_queries(self):
        stats = MLLMCacheStats()
        assert stats.hit_rate == 0.0

    def test_to_dict(self):
        stats = MLLMCacheStats(
            hits=5,
            misses=5,
            tokens_saved=100,
            image_cache_hits=3,
            total_queries=10,
            evictions=2,
        )
        d = stats.to_dict()
        assert d["hits"] == 5
        assert d["misses"] == 5
        assert d["hit_rate"] == 0.5
        assert d["tokens_saved"] == 100
        assert d["image_cache_hits"] == 3
        assert d["total_queries"] == 10
        assert d["evictions"] == 2


class TestImageHashing:

    def test_compute_image_hash_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake image content for testing")
            temp_path = f.name

        try:
            hash1 = compute_image_hash(temp_path)
            hash2 = compute_image_hash(temp_path)

            assert hash1 == hash2
            assert len(hash1) == 16
        finally:
            os.unlink(temp_path)

    def test_compute_image_hash_different_content(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1:
            f1.write(b"content 1")
            path1 = f1.name

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
            f2.write(b"content 2")
            path2 = f2.name

        try:
            hash1 = compute_image_hash(path1)
            hash2 = compute_image_hash(path2)
            assert hash1 != hash2
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_compute_image_hash_url(self):
        url = "https://example.com/image.jpg"
        hash1 = compute_image_hash(url)
        hash2 = compute_image_hash(url)

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_compute_images_hash_empty(self):
        result = compute_images_hash([])
        assert result == "no_images"

    def test_compute_images_hash_single(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"test content")
            path = f.name

        try:
            hash_single = compute_images_hash([path])
            assert len(hash_single) == 16
        finally:
            os.unlink(path)

    def test_compute_images_hash_multiple(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1:
            f1.write(b"image 1")
            path1 = f1.name

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
            f2.write(b"image 2")
            path2 = f2.name

        try:
            hash_a = compute_images_hash([path1, path2])
            hash_b = compute_images_hash([path2, path1])
            assert (
                hash_a != hash_b
            ), "Different image orders must produce different hashes"
            hash_c = compute_images_hash([path1, path2])
            assert hash_a == hash_c
        finally:
            os.unlink(path1)
            os.unlink(path2)


class TestMLLMCacheEntry:

    def test_cache_entry_creation(self):
        entry = MLLMPrefixCacheEntry(
            image_hash="abc123",
            prompt_hash="def456",
            kv_cache=["mock_kv_cache"],
            prompt_tokens=50,
        )
        assert entry.kv_cache == ["mock_kv_cache"]
        assert entry.image_hash == "abc123"
        assert entry.prompt_hash == "def456"
        assert entry.prompt_tokens == 50
        assert entry.hit_count == 0

    def test_cache_entry_hit_count_increment(self):
        entry = MLLMPrefixCacheEntry(
            image_hash="xyz",
            prompt_hash="abc",
            kv_cache=["cache"],
            prompt_tokens=10,
        )
        entry.hit_count += 1
        assert entry.hit_count == 1


class TestMLLMCacheManager:

    @pytest.fixture
    def cache_manager(self):
        return MLLMCacheManager(max_entries=10)

    @pytest.fixture
    def temp_image(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"test image content")
            path = f.name
        yield path
        os.unlink(path)

    def test_initialization(self):
        manager = MLLMCacheManager(max_entries=50)
        assert manager.max_size == 50
        assert len(manager) == 0

    def test_fetch_empty_cache(self, cache_manager):
        cache, hit = cache_manager.fetch_cache(["image.jpg"], "Describe this")

        assert cache is None
        assert hit is False
        assert cache_manager.stats.misses == 1
        assert cache_manager.stats.hits == 0

    def test_store_and_fetch_exact_match(self, cache_manager, temp_image):
        images = [temp_image]
        prompt = "Describe this image"
        mock_cache = ["kv_layer_1", "kv_layer_2"]

        cache_manager.store_cache(images, prompt, mock_cache, num_tokens=100)
        assert len(cache_manager) == 1

        cache, hit = cache_manager.fetch_cache(images, prompt)

        assert cache is not None
        assert hit is True
        assert cache_manager.stats.hits == 1
        assert cache_manager.stats.tokens_saved == 100

    def test_different_prompt_different_cache(self, cache_manager, temp_image):
        images = [temp_image]

        cache_manager.store_cache(images, "Describe this", ["cache1"], num_tokens=50)

        cache, hit = cache_manager.fetch_cache(images, "What is in this image?")

        assert cache is None
        assert hit is False
        assert cache_manager.stats.misses == 1

    def test_different_image_different_cache(self, cache_manager):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1:
            f1.write(b"image 1 content")
            img1 = f1.name

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
            f2.write(b"image 2 content")
            img2 = f2.name

        try:
            prompt = "Describe this"

            cache_manager.store_cache([img1], prompt, ["cache1"], num_tokens=50)

            cache, hit = cache_manager.fetch_cache([img2], prompt)
            assert cache is None
            assert hit is False
        finally:
            os.unlink(img1)
            os.unlink(img2)

    def test_video_cache_key(self, cache_manager):
        video_source_1 = "video:test.mp4:fps2.0:max32"
        video_source_2 = "video:test.mp4:fps1.0:max64"
        prompt = "Describe this video"

        cache_manager.store_cache([video_source_1], prompt, ["cache1"], num_tokens=100)

        cache, hit = cache_manager.fetch_cache([video_source_1], prompt)
        assert hit is True

        cache, hit = cache_manager.fetch_cache([video_source_2], prompt)
        assert hit is False

    def test_lru_eviction(self):
        manager = MLLMCacheManager(max_entries=3)

        manager.store_cache(["img1.jpg"], "prompt1", ["cache1"])
        manager.store_cache(["img2.jpg"], "prompt2", ["cache2"])
        manager.store_cache(["img3.jpg"], "prompt3", ["cache3"])
        assert len(manager) == 3

        manager.store_cache(["img4.jpg"], "prompt4", ["cache4"])
        assert len(manager) == 3
        assert manager.stats.evictions == 1

        cache, hit = manager.fetch_cache(["img1.jpg"], "prompt1")
        assert cache is None
        assert hit is False

    def test_lru_touch_on_access(self):
        manager = MLLMCacheManager(max_entries=3)

        manager.store_cache(["img1.jpg"], "p1", ["cache1"])
        manager.store_cache(["img2.jpg"], "p2", ["cache2"])
        manager.store_cache(["img3.jpg"], "p3", ["cache3"])

        manager.fetch_cache(["img1.jpg"], "p1")

        manager.store_cache(["img4.jpg"], "p4", ["cache4"])

        cache, hit = manager.fetch_cache(["img1.jpg"], "p1")
        assert hit is True

        cache, hit = manager.fetch_cache(["img2.jpg"], "p2")
        assert hit is False

    def test_store_empty_cache(self, cache_manager):
        cache_manager.store_cache(["img.jpg"], "prompt", [])
        assert len(cache_manager) == 0

    def test_store_none_cache(self, cache_manager):
        cache_manager.store_cache(["img.jpg"], "prompt", None)
        assert len(cache_manager) == 0

    def test_get_stats(self, cache_manager, temp_image):
        cache_manager.store_cache([temp_image], "Describe", ["cache1"], num_tokens=50)
        cache_manager.fetch_cache([temp_image], "Describe")
        cache_manager.fetch_cache(["other.jpg"], "Describe")

        stats = cache_manager.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["total_queries"] == 2
        assert stats["image_cache_hits"] == 1

    def test_reset_stats(self, cache_manager):
        cache_manager.stats.hits = 10
        cache_manager.stats.misses = 5
        cache_manager.reset_stats()

        assert cache_manager.stats.hits == 0
        assert cache_manager.stats.misses == 0

    def test_clear(self, cache_manager, temp_image):
        cache_manager.store_cache([temp_image], "p1", ["cache1"])
        cache_manager.store_cache(["img2.jpg"], "p2", ["cache2"])
        assert len(cache_manager) == 2

        cache_manager.clear()
        assert len(cache_manager) == 0

        assert cache_manager.stats.hits == 0

    def test_cache_returns_deep_copy(self, cache_manager, temp_image):
        original = [[1, 2, 3]]
        cache_manager.store_cache([temp_image], "prompt", original)

        cache, hit = cache_manager.fetch_cache([temp_image], "prompt")
        assert hit is True
        assert cache is not None
        assert cache[0] == [1, 2, 3]

        cache2, _ = cache_manager.fetch_cache([temp_image], "prompt")
        assert cache2 is not cache

    def test_repr(self, cache_manager):
        repr_str = repr(cache_manager)
        assert "MLLMPrefixCacheManager" in repr_str
        assert "entries=0" in repr_str
        assert "memory=" in repr_str

    def test_multi_image_cache(self, cache_manager):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1:
            f1.write(b"img1")
            img1 = f1.name

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
            f2.write(b"img2")
            img2 = f2.name

        try:
            prompt = "Compare these images"
            images = [img1, img2]

            cache_manager.store_cache(images, prompt, ["multi_cache"], num_tokens=200)

            cache, hit = cache_manager.fetch_cache(images, prompt)
            assert hit is True

            cache, hit = cache_manager.fetch_cache([img2, img1], prompt)
            assert hit is False, "Different image order should produce cache miss"
        finally:
            os.unlink(img1)
            os.unlink(img2)


class TestMLXMultimodalLMCache:
    """Tests for cache integration with MLXMultimodalLM — skipped
    (vllm_mlx.models.mllm not available in fusion-mlx)."""

    @pytest.mark.skip(reason="vllm_mlx.models.mllm not available in fusion-mlx")
    def test_mllm_cache_enabled_by_default(self):
        pass

    @pytest.mark.skip(reason="vllm_mlx.models.mllm not available in fusion-mlx")
    def test_mllm_cache_disabled(self):
        pass

    @pytest.mark.skip(reason="vllm_mlx.models.mllm not available in fusion-mlx")
    def test_mllm_cache_custom_size(self):
        pass

    @pytest.mark.skip(reason="vllm_mlx.models.mllm not available in fusion-mlx")
    def test_mllm_get_cache_stats_disabled(self):
        pass

    @pytest.mark.skip(reason="vllm_mlx.models.mllm not available in fusion-mlx")
    def test_mllm_get_cache_stats_enabled(self):
        pass

    @pytest.mark.skip(reason="vllm_mlx.models.mllm not available in fusion-mlx")
    def test_mllm_clear_cache(self):
        pass
