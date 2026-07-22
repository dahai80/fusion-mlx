# SPDX-License-Identifier: Apache-2.0

import mlx.core as mx

from fusion_mlx.cache.radix_diffusion_cache import (
    DiffusionRadixCache,
    RadixCacheStats,
)
from fusion_mlx.video.skyreels_v3.text_encoder import UMT5Encoder


class TestDiffusionRadixCache:
    def test_put_get_hit(self):
        c = DiffusionRadixCache(max_mb=1)
        arr = mx.ones((4, 4))
        c.put("a", arr)
        got = c.get("a")
        assert got is not None
        assert mx.array_equal(got, arr)
        s = c.stats()
        assert s["hits"] == 1
        assert s["insertions"] == 1
        assert s["leaf_count"] == 1

    def test_get_miss(self):
        c = DiffusionRadixCache(max_mb=1)
        assert c.get("missing") is None
        assert c.stats()["misses"] == 1

    def test_prefix_keys_no_collision(self):
        c = DiffusionRadixCache(max_mb=1)
        v1 = mx.ones((2, 2))
        v2 = mx.ones((2, 2)) * 2
        c.put("prompt:hello world", v1)
        c.put("prompt:hello there", v2)
        assert mx.array_equal(c.get("prompt:hello world"), v1)
        assert mx.array_equal(c.get("prompt:hello there"), v2)
        assert c.stats()["leaf_count"] == 2

    def test_update_existing_key(self):
        c = DiffusionRadixCache(max_mb=1)
        v1 = mx.ones((4, 4))
        v2 = mx.ones((4, 4)) * 2
        c.put("k", v1, size_bytes=64)
        c.put("k", v2, size_bytes=64)
        assert mx.array_equal(c.get("k"), v2)
        s = c.stats()
        assert s["leaf_count"] == 1
        assert s["total_bytes"] == 64
        assert s["insertions"] == 1

    def test_lru_eviction_by_bytes(self):
        c = DiffusionRadixCache(max_mb=1)
        big = 600 * 1024
        c.put("a", mx.zeros((1, big // 4)), size_bytes=big)
        c.put("b", mx.zeros((1, big // 4)), size_bytes=big)
        assert c.get("a") is None
        assert c.get("b") is not None
        s = c.stats()
        assert s["evictions"] >= 1
        assert s["leaf_count"] == 1

    def test_pin_prevents_eviction(self):
        c = DiffusionRadixCache(max_mb=1)
        big = 600 * 1024
        c.put("a", mx.zeros((1, big // 4)), size_bytes=big)
        assert c.pin("a") is True
        c.put("b", mx.zeros((1, big // 4)), size_bytes=big)
        c.put("c", mx.zeros((1, big // 4)), size_bytes=big)
        assert c.get("a") is not None
        assert c.unpin("a") is True

    def test_pin_missing_key(self):
        c = DiffusionRadixCache(max_mb=1)
        assert c.pin("nope") is False
        assert c.unpin("nope") is False

    def test_infer_size_mx_array(self):
        c = DiffusionRadixCache(max_mb=1)
        arr = mx.zeros((10, 10))
        c.put("a", arr)
        assert c.stats()["total_bytes"] == arr.nbytes

    def test_clear(self):
        c = DiffusionRadixCache(max_mb=1)
        c.put("a", mx.ones((2, 2)))
        c.clear()
        assert c.get("a") is None
        s = c.stats()
        assert s["leaf_count"] == 0
        assert s["total_bytes"] == 0

    def test_hit_rate(self):
        c = DiffusionRadixCache(max_mb=1)
        c.put("a", mx.ones((2, 2)))
        c.get("a")
        c.get("miss")
        s = c.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 0.5

    def test_stats_dataclass_defaults(self):
        s = RadixCacheStats()
        assert s.hits == 0
        assert s.misses == 0
        assert s.evictions == 0
        assert s.leaf_count == 0
        assert s.total_bytes == 0


class _MockEncoder:
    def __init__(self, d_model):
        self.d_model = d_model
        self.call_count = 0

    def __call__(self, ids, attention_mask=None):
        self.call_count += 1
        length = ids.shape[1]
        return mx.ones((1, length, self.d_model)) * self.call_count


def _wire_mock(enc, valid_tokens=8, max_length=16):
    enc._tokenizer = "fake"
    enc.encoder = _MockEncoder(enc.config.d_model)

    def fake_tokenize(prompt, *, max_length=512):
        ids = mx.zeros((1, max_length), dtype=mx.int32)
        mask = mx.zeros((1, max_length))
        mask[:, :valid_tokens] = 1.0
        return ids, mask

    enc.tokenize = fake_tokenize
    return enc.encoder


class TestUMT5EncoderCache:
    def test_second_call_hits_cache(self):
        enc = UMT5Encoder()
        mock = _wire_mock(enc)
        r1 = enc.encode_text("hello world", max_length=16)
        r2 = enc.encode_text("hello world", max_length=16)
        assert mock.call_count == 1
        assert mx.array_equal(r1, r2)
        s = enc._text_cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1

    def test_zero_copy_same_ref(self):
        enc = UMT5Encoder()
        _wire_mock(enc)
        r1 = enc.encode_text("same prompt", max_length=16)
        r2 = enc.encode_text("same prompt", max_length=16)
        assert r1 is r2

    def test_different_prompts_no_hit(self):
        enc = UMT5Encoder()
        mock = _wire_mock(enc)
        enc.encode_text("prompt one", max_length=16)
        enc.encode_text("prompt two", max_length=16)
        assert mock.call_count == 2
        assert enc._text_cache.stats()["hits"] == 0

    def test_different_max_length_no_hit(self):
        enc = UMT5Encoder()
        mock = _wire_mock(enc)
        enc.encode_text("same text", max_length=16)
        enc.encode_text("same text", max_length=32)
        assert mock.call_count == 2

    def test_env_disabled_no_cache(self, monkeypatch):
        monkeypatch.setenv("FUSION_DIFFUSION_TEXT_CACHE", "0")
        enc = UMT5Encoder()
        assert enc._text_cache is None
        mock = _wire_mock(enc)
        enc.encode_text("hello", max_length=16)
        enc.encode_text("hello", max_length=16)
        assert mock.call_count == 2

    def test_stub_mode_does_not_pollute_cache(self):
        enc = UMT5Encoder()
        enc._tokenizer = "stub"
        enc.encoder = None
        r1 = enc.encode_text("stubbed", max_length=16)
        r2 = enc.encode_text("stubbed", max_length=16)
        assert mx.array_equal(r1, r2)
        assert enc._text_cache.stats()["insertions"] == 0


# #178 Phase-2: weakref registry + all_cache_stats() aggregator
class TestCacheRegistry:
    def test_all_cache_stats_registry(self):
        from fusion_mlx.cache.radix_diffusion_cache import all_cache_stats

        c1 = DiffusionRadixCache(max_mb=1, name="umt5")
        c2 = DiffusionRadixCache(max_mb=1, name="clip")
        c1.put("a", mx.ones((2, 2)))
        c2.get("miss")
        stats = all_cache_stats()
        names = {s["name"] for s in stats}
        assert "umt5" in names
        assert "clip" in names
        umt5 = next(s for s in stats if s["name"] == "umt5")
        clip = next(s for s in stats if s["name"] == "clip")
        assert umt5["insertions"] == 1
        assert clip["misses"] == 1

    def test_registry_weakref_gc(self):
        import gc

        from fusion_mlx.cache.radix_diffusion_cache import all_cache_stats

        c = DiffusionRadixCache(max_mb=1, name="gc_target")
        assert any(s["name"] == "gc_target" for s in all_cache_stats())
        del c
        gc.collect()
        assert not any(s["name"] == "gc_target" for s in all_cache_stats())

    def test_unnamed_cache_has_none_name(self):
        from fusion_mlx.cache.radix_diffusion_cache import all_cache_stats

        c = DiffusionRadixCache(max_mb=1)  # default name=None
        stats = all_cache_stats()
        assert any(s.get("name") is None for s in stats)


# #178 Phase-2: CLIP text-encoding cache wiring
class TestCLIPTextEncoderCache:
    def test_clip_encoder_has_named_cache(self):
        from fusion_mlx.video.skyreels_v3.text_encoder import CLIPTextEncoder

        enc = CLIPTextEncoder()
        assert enc._text_cache is not None
        assert enc._text_cache.name == "clip"

    def test_clip_cache_hit_skips_load(self):
        from fusion_mlx.video.skyreels_v3.text_encoder import (
            CLIPTextEncoder,
            _prompt_hash,
        )

        enc = CLIPTextEncoder()
        # pre-populate cache with the exact key encode_text computes
        key = f"clip:77:{_prompt_hash('a cat')}"
        sentinel = mx.ones((1, enc.EMBED_DIM)) * 7.0
        enc._text_cache.put(key, sentinel)
        # hit path must NOT call _ensure_loaded -> backend/model stay unloaded
        out = enc.encode_text("a cat")
        assert enc._backend is None
        assert enc._clip_model is None
        assert mx.array_equal(out, sentinel)
        assert enc._text_cache.stats()["hits"] == 1

    def test_clip_stub_mode_not_cached(self):
        from fusion_mlx.video.skyreels_v3.text_encoder import CLIPTextEncoder

        enc = CLIPTextEncoder()
        # force stub without real load: _ensure_loaded early-returns when
        # _clip_model is already set, leaving _backend="stub"
        enc._clip_model = "stub"
        enc._backend = "stub"
        r1 = enc.encode_text("stubbed clip")
        r2 = enc.encode_text("stubbed clip")
        assert mx.array_equal(r1, r2)
        assert enc._text_cache.stats()["insertions"] == 0

    def test_clip_different_prompts_no_hit(self):
        from fusion_mlx.video.skyreels_v3.text_encoder import CLIPTextEncoder

        enc = CLIPTextEncoder()
        enc._clip_model = "stub"
        enc._backend = "stub"
        enc.encode_text("prompt one")
        enc.encode_text("prompt two")
        assert enc._text_cache.stats()["hits"] == 0


# #178 Phase-2: /v1/cache/stats endpoint aggregation
class TestCacheStatsEndpoint:
    def test_stats_endpoint_aggregates_registry(self):
        import asyncio

        from fusion_mlx.routes_internal.cache import cache_stats

        c = DiffusionRadixCache(max_mb=1, name="endpoint_test")
        c.put("k", mx.ones((2, 2)))
        result = asyncio.run(cache_stats(is_admin=True))
        assert result["cache_type"] == "diffusion_text_encoding"
        names = {s["name"] for s in result["caches"]}
        assert "endpoint_test" in names
        totals = result["totals"]
        assert totals["cache_count"] >= 1
        assert "hit_rate" in totals
        assert totals["insertions"] >= 1
