# SPDX-License-Identifier: Apache-2.0
"""Tests for fusion_mlx.cache.response_cache."""

import threading
import time

import pytest

from fusion_mlx.cache.response_cache import (
    CachePolicy,
    ResponseCache,
    ResponseCacheStats,
    get_response_cache,
    reset_response_cache,
)


@pytest.fixture(autouse=True)
def _clean_global():
    reset_response_cache()
    yield
    reset_response_cache()


class TestFingerprint:
    def test_same_inputs_same_key(self):
        c = ResponseCache()
        k1 = c.fingerprint("m1", [{"role": "user", "content": "hi"}], temperature=0.0)
        k2 = c.fingerprint("m1", [{"role": "user", "content": "hi"}], temperature=0.0)
        assert k1 == k2

    def test_different_model_different_key(self):
        c = ResponseCache()
        k1 = c.fingerprint("m1", [{"role": "user", "content": "hi"}])
        k2 = c.fingerprint("m2", [{"role": "user", "content": "hi"}])
        assert k1 != k2

    def test_different_messages_different_key(self):
        c = ResponseCache()
        k1 = c.fingerprint("m1", [{"role": "user", "content": "hi"}])
        k2 = c.fingerprint("m1", [{"role": "user", "content": "bye"}])
        assert k1 != k2

    def test_different_temp_different_key(self):
        c = ResponseCache()
        k1 = c.fingerprint("m1", [{"role": "user", "content": "hi"}], temperature=0.0)
        k2 = c.fingerprint("m1", [{"role": "user", "content": "hi"}], temperature=0.7)
        assert k1 != k2

    def test_tools_included(self):
        c = ResponseCache()
        tools = [{"type": "function", "function": {"name": "f"}}]
        k1 = c.fingerprint("m1", [{"role": "user", "content": "hi"}], tools=None)
        k2 = c.fingerprint("m1", [{"role": "user", "content": "hi"}], tools=tools)
        assert k1 != k2


class TestResolvePolicy:
    def test_temp_zero_is_force(self):
        c = ResponseCache()
        assert c.resolve_policy(0.0) == CachePolicy.FORCE

    def test_temp_nonzero_is_bypass(self):
        c = ResponseCache()
        assert c.resolve_policy(0.7) == CachePolicy.BYPASS

    def test_header_bypass(self):
        c = ResponseCache()
        assert c.resolve_policy(0.0, {"X-Cache": "bypass"}) == CachePolicy.BYPASS

    def test_header_force(self):
        c = ResponseCache()
        assert c.resolve_policy(0.7, {"X-Cache": "force"}) == CachePolicy.FORCE

    def test_header_only_if_cached(self):
        c = ResponseCache()
        assert (
            c.resolve_policy(0.7, {"X-Cache": "only-if-cached"})
            == CachePolicy.ONLY_IF_CACHED
        )

    def test_header_no_store(self):
        c = ResponseCache()
        assert c.resolve_policy(0.0, {"X-Cache": "no-store"}) == CachePolicy.NO_STORE

    def test_none_temp_defaults_bypass(self):
        c = ResponseCache()
        assert c.resolve_policy(None) == CachePolicy.BYPASS


class TestGetPut:
    def test_put_and_get(self):
        c = ResponseCache()
        key = c.fingerprint("m1", [{"role": "user", "content": "hi"}], temperature=0.0)
        resp = {"id": "chatcmpl-abc", "choices": [{"text": "hello"}]}
        assert c.put(key, resp) is True
        result = c.get(key)
        assert result == resp

    def test_miss_returns_none(self):
        c = ResponseCache()
        assert c.get("nonexistent") is None
        assert c.stats.misses == 1

    def test_ttl_expiry(self):
        c = ResponseCache(default_ttl=0.05)
        key = "test-key"
        c.put(key, {"data": 1})
        time.sleep(0.1)
        assert c.get(key) is None
        assert c.stats.misses == 1

    def test_hit_increments_stats(self):
        c = ResponseCache()
        key = "test-key"
        c.put(key, {"data": 1})
        c.get(key)
        assert c.stats.hits == 1
        assert c.stats.misses == 0

    def test_oversized_entry_rejected(self):
        c = ResponseCache(max_entry_bytes=10)
        key = "test-key"
        big_resp = {"data": "x" * 1000}
        assert c.put(key, big_resp) is False

    def test_non_serializable_rejected(self):
        c = ResponseCache()
        key = "test-key"
        assert c.put(key, object()) is False

    def test_lru_eviction(self):
        c = ResponseCache(max_entries=3, max_total_bytes=1024 * 1024)
        for i in range(5):
            c.put(f"key-{i}", {"i": i})
        assert c.stats.entry_count <= 3
        assert c.stats.evictions >= 2


class TestInvalidate:
    def test_invalidate_existing(self):
        c = ResponseCache()
        c.put("k", {"v": 1})
        assert c.invalidate("k") is True
        assert c.get("k") is None

    def test_invalidate_nonexistent(self):
        c = ResponseCache()
        assert c.invalidate("nope") is False


class TestClear:
    def test_clear_empties_cache(self):
        c = ResponseCache()
        c.put("k1", {"v": 1})
        c.put("k2", {"v": 2})
        c.clear()
        assert c.stats.entry_count == 0
        assert c.stats.size_bytes == 0


class TestGetByResponseId:
    def test_found(self):
        c = ResponseCache()
        c.put("k1", {"id": "resp-abc123", "content": "hello"})
        result = c.get_by_response_id("resp-abc123")
        assert result is not None
        assert result["content"] == "hello"

    def test_not_found(self):
        c = ResponseCache()
        assert c.get_by_response_id("nope") is None


class TestThreadSafety:
    def test_concurrent_access(self):
        c = ResponseCache()
        errors = []

        def writer(n):
            try:
                for i in range(50):
                    c.put(f"key-{n}-{i}", {"val": i})
            except Exception as e:
                errors.append(e)

        def reader(n):
            try:
                for i in range(50):
                    c.get(f"key-{n}-{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for n in range(4):
            threads.append(threading.Thread(target=writer, args=(n,)))
            threads.append(threading.Thread(target=reader, args=(n,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


class TestGlobalCache:
    def test_singleton(self):
        c1 = get_response_cache()
        c2 = get_response_cache()
        assert c1 is c2

    def test_reset_clears(self):
        c = get_response_cache()
        c.put("k", {"v": 1})
        reset_response_cache()
        c2 = get_response_cache()
        assert c2.stats.entry_count == 0


class TestStats:
    def test_to_dict(self):
        s = ResponseCacheStats(hits=10, misses=5, size_bytes=1024, entry_count=3)
        d = s.to_dict()
        assert d["hits"] == 10
        assert d["misses"] == 5
        assert abs(d["hit_rate"] - 10 / 15) < 0.001
        assert d["size_bytes"] == 1024
        assert d["entry_count"] == 3

    def test_hit_rate_zero_queries(self):
        s = ResponseCacheStats()
        assert s.hit_rate == 0.0
