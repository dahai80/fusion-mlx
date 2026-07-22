# SPDX-License-Identifier: Apache-2.0

import mlx.core as mx

from fusion_mlx.cache.latent_cache import (
    get_image_latent_cache,
    image_latent_key,
    latent_cache_enabled,
)
from fusion_mlx.cache.radix_diffusion_cache import all_cache_stats


class TestImageLatentKey:
    def test_deterministic_same_args(self):
        k1 = image_latent_key("ltx2", "/img/a.png", 512, 512, mx.bfloat16)
        k2 = image_latent_key("ltx2", "/img/a.png", 512, 512, mx.bfloat16)
        assert k1 == k2

    def test_prefix_is_latent(self):
        k = image_latent_key("ltx2", "/img/a.png", 512, 512, mx.bfloat16)
        assert k.startswith("latent:ltx2:")

    def test_different_resolution_distinct(self):
        k1 = image_latent_key("m", "a", 256, 256, mx.bfloat16)
        k2 = image_latent_key("m", "a", 512, 512, mx.bfloat16)
        assert k1 != k2

    def test_different_dtype_distinct(self):
        k1 = image_latent_key("m", "a", 512, 512, mx.bfloat16)
        k2 = image_latent_key("m", "a", 512, 512, mx.float16)
        assert k1 != k2

    def test_different_model_distinct(self):
        k1 = image_latent_key("ltx2", "a", 512, 512, mx.bfloat16)
        k2 = image_latent_key("wan2", "a", 512, 512, mx.bfloat16)
        assert k1 != k2

    def test_different_source_distinct(self):
        k1 = image_latent_key("m", "/img/a.png", 512, 512, mx.bfloat16)
        k2 = image_latent_key("m", "/img/b.png", 512, 512, mx.bfloat16)
        assert k1 != k2


class TestImageLatentCache:
    def test_get_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.setenv("FUSION_LATENT_CACHE", "0")
        assert get_image_latent_cache("disabled-uniq-1") is None

    def test_enabled_by_default(self):
        assert latent_cache_enabled() is True

    def test_memoized_same_instance(self):
        c1 = get_image_latent_cache("memo-uniq-1")
        c2 = get_image_latent_cache("memo-uniq-1")
        assert c1 is c2

    def test_hit_returns_same_array(self):
        c = get_image_latent_cache("hit-uniq-1", max_mb=1)
        arr = mx.ones((4, 4), dtype=mx.bfloat16)
        key = image_latent_key("hit-uniq-1", "/img/x.png", 512, 512, mx.bfloat16)
        c.put(key, arr)
        got = c.get(key)
        assert got is not None
        assert mx.array_equal(got, arr)
        assert c.stats()["hits"] == 1

    def test_miss_returns_none(self):
        c = get_image_latent_cache("miss-uniq-1", max_mb=1)
        key = image_latent_key("miss-uniq-1", "/img/y.png", 512, 512, mx.bfloat16)
        assert c.get(key) is None
        assert c.stats()["misses"] == 1

    def test_eviction_under_max_mb(self):
        c = get_image_latent_cache("evict-uniq-1", max_mb=1)
        big = 600 * 1024
        k1 = image_latent_key("evict-uniq-1", "a", 512, 512, mx.bfloat16)
        k2 = image_latent_key("evict-uniq-1", "b", 512, 512, mx.bfloat16)
        c.put(k1, mx.zeros((1, big // 4)), size_bytes=big)
        c.put(k2, mx.zeros((1, big // 4)), size_bytes=big)
        assert c.get(k1) is None
        assert c.get(k2) is not None
        assert c.stats()["evictions"] >= 1

    def test_pin_prevents_eviction(self):
        c = get_image_latent_cache("pin-uniq-1", max_mb=1)
        big = 600 * 1024
        k1 = image_latent_key("pin-uniq-1", "a", 512, 512, mx.bfloat16)
        c.put(k1, mx.zeros((1, big // 4)), size_bytes=big)
        assert c.pin(k1) is True
        c.put(
            image_latent_key("pin-uniq-1", "b", 512, 512, mx.bfloat16),
            mx.zeros((1, big // 4)),
            size_bytes=big,
        )
        assert c.get(k1) is not None
        assert c.unpin(k1) is True


class TestLatentCacheRegistry:
    def test_appears_in_all_cache_stats(self):
        c = get_image_latent_cache("registry-uniq-1", max_mb=1)
        c.put(
            image_latent_key("registry-uniq-1", "z", 64, 64, mx.bfloat16),
            mx.ones((2, 2)),
        )
        stats = all_cache_stats()
        names = {s["name"] for s in stats}
        assert "latent:registry-uniq-1" in names

    def test_name_labels_model_id(self):
        c = get_image_latent_cache("registry-uniq-2", max_mb=1)
        assert c.name == "latent:registry-uniq-2"
