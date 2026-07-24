# SPDX-License-Identifier: Apache-2.0
# Unit tests for Phase-2 session tail cache (latent_cache.py extensions).


from fusion_mlx.cache.latent_cache import (
    get_session_tail,
    put_session_tail,
    session_tail_cache_enabled,
    session_tail_key,
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
    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("FUSION_SESSION_TAIL_CACHE", raising=False)
        monkeypatch.setenv("FUSION_LATENT_CACHE", "1")
        assert session_tail_cache_enabled()

    def test_explicit_disable(self, monkeypatch):
        monkeypatch.setenv("FUSION_LATENT_CACHE", "1")
        monkeypatch.setenv("FUSION_SESSION_TAIL_CACHE", "0")
        assert not session_tail_cache_enabled()

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
        monkeypatch.setenv("FUSION_SESSION_TAIL_CACHE", "0")
        import mlx.core as mx

        tail = mx.ones((1, 128, 1, 16, 32))
        ok = put_session_tail("sess-1", "model-a", tail)
        assert not ok

    def test_disabled_returns_none_get(self, monkeypatch):
        monkeypatch.setenv("FUSION_LATENT_CACHE", "1")
        monkeypatch.setenv("FUSION_SESSION_TAIL_CACHE", "0")
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


class TestVideoGenEngineSessionIdFlow:
    def test_session_id_flows_to_params(self):
        from fusion_mlx.engines.video_backends.base import VideoGenParams

        params = VideoGenParams(
            prompt="test",
            n=1,
            num_frames=25,
            width=512,
            height=512,
            fps=24,
            seed=42,
            session_id="sess-integration",
        )
        assert params.session_id == "sess-integration"

    def test_session_id_none_by_default(self):
        from fusion_mlx.engines.video_backends.base import VideoGenParams

        params = VideoGenParams(
            prompt="test",
            n=1,
            num_frames=25,
            width=512,
            height=512,
            fps=24,
            seed=42,
        )
        assert params.session_id is None

    def test_session_id_from_kwargs(self):
        from fusion_mlx.engines.video_backends.base import VideoGenParams

        kwargs = {"session_id": "sess-from-api"}
        params = VideoGenParams(
            prompt="test",
            n=1,
            num_frames=25,
            width=512,
            height=512,
            fps=24,
            seed=42,
            session_id=kwargs.get("session_id"),
        )
        assert params.session_id == "sess-from-api"

    def test_missing_session_id_yields_none(self):
        from fusion_mlx.engines.video_backends.base import VideoGenParams

        kwargs = {}
        params = VideoGenParams(
            prompt="test",
            n=1,
            num_frames=25,
            width=512,
            height=512,
            fps=24,
            seed=42,
            session_id=kwargs.get("session_id"),
        )
        assert params.session_id is None
