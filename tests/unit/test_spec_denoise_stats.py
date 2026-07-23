# SPDX-License-Identifier: Apache-2.0
# Tests for issue #177 Phase-3: speculative-denoise stats stage surface.
# Covers SpecStats.to_dict, VideoBackend base default, SkyReelsBackend
# override (pipeline None / with run), VideoGenEngine delegation, and the
# GET /v1/videos/denoise-stats route. Hard-imports mlx via speculative_denoise
# + videos_routes -> macOS-only (conftest _OPT_DEP_SUITES "mlx" skip-list).

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api.videos_routes import router as videos_router
from fusion_mlx.api.videos_routes import set_videos_context
from fusion_mlx.engines.video import VideoGenEngine
from fusion_mlx.engines.video_backends.base import VideoBackend, VideoConstraints
from fusion_mlx.engines.video_backends.skyreels import SkyReelsBackend
from fusion_mlx.exceptions import ModelNotFoundError
from fusion_mlx.video.skyreels_v3.speculative_denoise import SpecStats

logger = logging.getLogger(__name__)

_SPEC_ENV_VARS = (
    "FUSION_SPECULATIVE_DENOISE",
    "FUSION_SPEC_K",
    "FUSION_SPEC_EPSILON",
    "FUSION_SPEC_DRAFT_BLOCKS",
    "FUSION_SPEC_EVAL_STEPS",
    "FUSION_ASYNC_DENOISE",
)


@pytest.fixture(autouse=True)
def _clean_spec_env(monkeypatch):
    for var in _SPEC_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class _MinimalBackend(VideoBackend):
    name = "minimal"

    @classmethod
    def detect(cls, model_path):
        return False

    async def start(self, model_path, **kwargs):
        return None

    async def stop(self):
        return None

    async def generate(self, params):
        return []

    def constraints(self):
        return VideoConstraints()


class _FakePipeline:
    def __init__(self, stats):
        self._last_spec_stats = stats


def _make_app(pool) -> TestClient:
    app = FastAPI()
    app.include_router(videos_router)
    set_videos_context(pool)
    return TestClient(app, raise_server_exceptions=False)


def _stats_engine(stats_dict):
    engine = MagicMock(spec=VideoGenEngine)
    engine.last_denoise_stats = MagicMock(return_value=stats_dict)
    return engine


class TestSpecStatsToDict:
    def test_roundtrip_with_values(self):
        stats = SpecStats(
            macro_steps=3,
            accepted=[2, 0, 1],
            full_forwards=2,
            draft_forwards=6,
            baseline_steps=5,
        )
        data = stats.to_dict()
        assert data["macro_steps"] == 3
        assert data["accepted"] == [2, 0, 1]
        assert data["avg_accept"] == 1.0
        assert data["full_forwards"] == 2
        assert data["draft_forwards"] == 6
        assert data["baseline_steps"] == 5
        assert data["speedup"] == 2.5

    def test_defaults_empty(self):
        data = SpecStats().to_dict()
        assert data["macro_steps"] == 0
        assert data["accepted"] == []
        assert data["avg_accept"] == 0.0
        assert data["full_forwards"] == 0
        assert data["draft_forwards"] == 0
        assert data["baseline_steps"] == 0
        assert data["speedup"] == 0.0

    def test_to_dict_does_not_alias_accepted_list(self):
        stats = SpecStats(accepted=[1, 2])
        data = stats.to_dict()
        data["accepted"].append(99)
        assert stats.accepted == [1, 2]


class TestVideoBackendBaseDefault:
    def test_base_default_returns_empty_dict(self):
        backend = _MinimalBackend()
        assert backend.last_denoise_stats() == {}


class TestSkyReelsBackendStats:
    def test_no_pipeline_available_false(self):
        backend = SkyReelsBackend("r2v_14b")
        assert backend._pipeline is None
        data = backend.last_denoise_stats()
        assert data["available"] is False
        assert data["enabled"] is False
        assert data["macro_steps"] == 0
        assert data["accepted"] == []
        assert data["avg_accept"] == 0.0
        assert data["speedup"] == 0.0
        assert data["config"] == {
            "K": 4,
            "epsilon": 0.1,
            "relative": True,
            "eval_steps": True,
            "draft_strategy": "extrapolation",
        }

    def test_with_run_available_true(self):
        backend = SkyReelsBackend("r2v_14b")
        backend._pipeline = _FakePipeline(
            SpecStats(
                macro_steps=3,
                accepted=[2, 0, 1],
                full_forwards=2,
                draft_forwards=6,
                baseline_steps=5,
            )
        )
        data = backend.last_denoise_stats()
        assert data["available"] is True
        assert data["enabled"] is False
        assert data["macro_steps"] == 3
        assert data["accepted"] == [2, 0, 1]
        assert data["avg_accept"] == 1.0
        assert data["full_forwards"] == 2
        assert data["draft_forwards"] == 6
        assert data["baseline_steps"] == 5
        assert data["speedup"] == 2.5
        assert data["config"]["K"] == 4

    def test_enabled_reflects_env(self, monkeypatch):
        monkeypatch.setenv("FUSION_SPECULATIVE_DENOISE", "1")
        backend = SkyReelsBackend("r2v_14b")
        data = backend.last_denoise_stats()
        assert data["enabled"] is True
        assert data["available"] is False

    def test_config_reflects_env(self, monkeypatch):
        monkeypatch.setenv("FUSION_SPEC_K", "8")
        monkeypatch.setenv("FUSION_SPEC_EPSILON", "0.25")
        monkeypatch.setenv("FUSION_SPEC_EVAL_STEPS", "0")
        backend = SkyReelsBackend("r2v_14b")
        data = backend.last_denoise_stats()
        assert data["config"]["K"] == 8
        assert data["config"]["epsilon"] == 0.25
        assert data["config"]["eval_steps"] is False

    def test_pipeline_without_stats_attr_available_false(self):
        backend = SkyReelsBackend("r2v_14b")

        class _NoAttrPipeline:
            pass

        backend._pipeline = _NoAttrPipeline()
        data = backend.last_denoise_stats()
        assert data["available"] is False
        assert data["macro_steps"] == 0


class TestVideoGenEngineDelegation:
    def test_delegates_to_backend(self):
        engine = VideoGenEngine.__new__(VideoGenEngine)
        engine._backend = MagicMock()
        engine._backend.last_denoise_stats.return_value = {
            "enabled": False,
            "available": False,
        }
        assert engine.last_denoise_stats() == {"enabled": False, "available": False}
        engine._backend.last_denoise_stats.assert_called_once()

    def test_delegates_backend_default_empty(self):
        engine = VideoGenEngine.__new__(VideoGenEngine)
        engine._backend = _MinimalBackend()
        assert engine.last_denoise_stats() == {}


class TestDenoiseStatsRoute:
    def test_route_ok_with_stats(self):
        stats = {"enabled": False, "available": True, "avg_accept": 1.0}
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=_stats_engine(stats))
        client = _make_app(pool)
        resp = client.get("/v1/videos/denoise-stats", params={"model": "r2v_14b"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "r2v_14b"
        assert body["stats"] == stats
        pool.get_engine.assert_awaited_once_with("r2v_14b")

    def test_route_default_model_ltx2(self):
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=_stats_engine({}))
        client = _make_app(pool)
        resp = client.get("/v1/videos/denoise-stats")
        assert resp.status_code == 200
        pool.get_engine.assert_awaited_once_with("ltx-2")

    def test_route_404_engine_none(self):
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=None)
        client = _make_app(pool)
        resp = client.get("/v1/videos/denoise-stats")
        assert resp.status_code == 404
        assert "not loaded" in resp.json()["detail"].lower()

    def test_route_404_not_video_gen(self):
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=MagicMock())
        client = _make_app(pool)
        resp = client.get("/v1/videos/denoise-stats")
        assert resp.status_code == 404

    def test_route_404_model_not_found(self):
        pool = MagicMock()
        pool.get_engine = AsyncMock(side_effect=ModelNotFoundError("nope"))
        client = _make_app(pool)
        resp = client.get("/v1/videos/denoise-stats", params={"model": "ghost"})
        assert resp.status_code == 404

    def test_route_503_pool_not_initialized(self, monkeypatch):
        from fusion_mlx.api import videos_routes

        monkeypatch.setattr(videos_routes, "_pool", None)
        app = FastAPI()
        app.include_router(videos_router)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/videos/denoise-stats")
        assert resp.status_code == 503

    def test_route_500_when_stats_raises(self):
        engine = MagicMock(spec=VideoGenEngine)
        engine.last_denoise_stats = MagicMock(side_effect=RuntimeError("boom"))
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=engine)
        client = _make_app(pool)
        resp = client.get("/v1/videos/denoise-stats")
        assert resp.status_code == 500
        assert "boom" in resp.json()["detail"]
