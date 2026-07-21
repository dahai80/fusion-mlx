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
