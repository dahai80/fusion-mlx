# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1044.

#1044: skyreels_v3 speculative_denoise was falsified (0% acceptance, 0.2-0.4x
slower than baseline). The module, all call sites, the /denoise-stats API
surface, and the last_denoise_stats accessor chain were removed. The
SPECULATIVE_DENOISE.md conclusion doc is kept. The separate async double-buffer
feature (#180, FUSION_ASYNC_DENOISE) is preserved — its helper moved into
pipelines/__init__.py.
"""

from __future__ import annotations

import importlib
import inspect

import pytest


class TestSpeculativeDenoiseRemoved:
    """The falsified speculative_denoise module and surface are gone."""

    def test_module_deleted(self):
        try:
            importlib.import_module("fusion_mlx.video.skyreels_v3.speculative_denoise")
        except ModuleNotFoundError:
            return
        pytest.fail("speculative_denoise module should have been removed")

    def test_no_denoise_stats_route(self):
        from fusion_mlx.api import videos_routes

        source = inspect.getsource(videos_routes)
        assert "/denoise-stats" not in source
        assert "video_denoise_stats" not in source
        assert "last_denoise_stats" not in source

    def test_engine_no_last_denoise_stats(self):
        from fusion_mlx.engines.video import VideoGenEngine

        assert not hasattr(VideoGenEngine, "last_denoise_stats")

    def test_backend_base_no_last_denoise_stats(self):
        from fusion_mlx.engines.video_backends.base import VideoBackend

        assert not hasattr(VideoBackend, "last_denoise_stats")

    def test_skyreels_backend_no_last_denoise_stats(self):
        from fusion_mlx.engines.video_backends.skyreels import SkyReelsBackend

        assert not hasattr(SkyReelsBackend, "last_denoise_stats")

    def test_pipelines_no_spec_references(self):
        from fusion_mlx.video.skyreels_v3 import pipelines

        source = inspect.getsource(pipelines)
        assert "speculative_enabled" not in source
        assert "SpeculativeConfig" not in source
        assert "speculative_denoise" not in source
        assert "_denoise_sample_speculative" not in source
        assert "_last_spec_stats" not in source

    def test_doc_conclusion_kept(self):
        from pathlib import Path

        doc = (
            Path(__file__).resolve().parents[2]
            / "fusion_mlx"
            / "video"
            / "skyreels_v3"
            / "SPECULATIVE_DENOISE.md"
        )
        assert doc.exists(), "SPECULATIVE_DENOISE.md conclusion doc must be kept"


class TestAsyncDenoisePreserved:
    """The separate #180 async double-buffer feature is preserved."""

    def test_async_helper_moved_to_pipelines(self):
        from fusion_mlx.video.skyreels_v3.pipelines import async_denoise_enabled

        assert callable(async_denoise_enabled)
        assert async_denoise_enabled() is False

    def test_async_flag_env_on(self, monkeypatch):
        from fusion_mlx.video.skyreels_v3.pipelines import async_denoise_enabled

        monkeypatch.setenv("FUSION_ASYNC_DENOISE", "1")
        assert async_denoise_enabled() is True

    def test_async_flag_env_off(self, monkeypatch):
        from fusion_mlx.video.skyreels_v3.pipelines import async_denoise_enabled

        monkeypatch.delenv("FUSION_ASYNC_DENOISE", raising=False)
        assert async_denoise_enabled() is False


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
