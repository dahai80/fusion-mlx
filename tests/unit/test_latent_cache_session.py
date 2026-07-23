# SPDX-License-Identifier: Apache-2.0
# Unit tests for Phase-2 session tail cache (latent_cache.py extensions).

import os
import pytest

from fusion_mlx.cache.latent_cache import (
    session_tail_key,
    session_tail_cache_enabled,
    put_session_tail,
    get_session_tail,
)


class TestSessionTailKey:
    def test_format(self):
        key = session_tail_key("sess-123", "ltx-2-dev")
        assert key == "session_tail:ltx-2-dev:sess-123"

    def test_different_sessions_isolated(self):
        k1 = session_tail_key("sess-a", "model-x")
        k2 = session_tail_key("sess-b", "model-x")
        assert k1 != k2

    def test_different_models_isolated(self):
        k1 = session_tail_key("sess-a", "model-x")
        k2 = session_tail_key("sess-a", "model-y")
        assert k1 != k2


class TestSessionTailCacheEnabled:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("FUSION_SESSION_TAIL_CACHE", raising=False)
        monkeypatch.setenv("FUSION_LATENT_CACHE", "1")
        assert not session_tail_cache_enabled()

    def test_enabled_when_both_on(self, monkeypatch):
        monkeypatch.setenv("FUSION_LATENT_CACHE", "1")
        monkeypatch.setenv("FUSION_SESSION_TAIL_CACHE", "1")
        assert session_tail_cache_enabled()

    def test_disabled_when_master_off(self, monkeypatch):
        monkeypatch.setenv("FUSION_LATENT_CACHE", "0")
        monkeypatch.setenv("FUSION_SESSION_TAIL_CACHE", "1")
        assert not session_tail_cache_enabled()


class TestSessionTailPutGet:
    def test_round_trip(self, monkeypatch):
        monkeypatch.setenv("FUSION_LATENT_CACHE", "1")
        monkeypatch.setenv("FUSION_SESSION_TAIL_CACHE", "1")
        import mlx.core as mx

        tail = mx.ones((1, 128, 1, 16, 32))
        ok = put_session_tail("sess-1", "model-a", tail)
        assert ok
        result = get_session_tail("sess-1", "model-a")
        assert result is not None
        assert result.shape == tail.shape

    def test_miss_returns_none(self, monkeypatch):
        monkeypatch.setenv("FUSION_LATENT_CACHE", "1")
        monkeypatch.setenv("FUSION_SESSION_TAIL_CACHE", "1")
        result = get_session_tail("nonexistent", "model-a")
        assert result is None

    def test_disabled_returns_false_put(self, monkeypatch):
        monkeypatch.setenv("FUSION_LATENT_CACHE", "1")
        monkeypatch.delenv("FUSION_SESSION_TAIL_CACHE", raising=False)
        import mlx.core as mx

        tail = mx.ones((1, 128, 1, 16, 32))
        ok = put_session_tail("sess-1", "model-a", tail)
        assert not ok

    def test_disabled_returns_none_get(self, monkeypatch):
        monkeypatch.setenv("FUSION_LATENT_CACHE", "1")
        monkeypatch.delenv("FUSION_SESSION_TAIL_CACHE", raising=False)
        result = get_session_tail("sess-1", "model-a")
        assert result is None

    def test_sessions_isolated(self, monkeypatch):
        monkeypatch.setenv("FUSION_LATENT_CACHE", "1")
        monkeypatch.setenv("FUSION_SESSION_TAIL_CACHE", "1")
        import mlx.core as mx

        tail_a = mx.ones((1, 128, 1, 16, 32)) * 1.0
        tail_b = mx.ones((1, 128, 1, 16, 32)) * 2.0
        put_session_tail("sess-a", "model-x", tail_a)
        put_session_tail("sess-b", "model-x", tail_b)
        ra = get_session_tail("sess-a", "model-x")
        rb = get_session_tail("sess-b", "model-x")
        assert ra is not None
        assert rb is not None
