# SPDX-License-Identifier: Apache-2.0
"""Timeout-gap tests for issue #1191.

Covers the three timeout gaps closed by this PR:
  - music generate: bounded by get_music_gen_timeout() / FUSION_MUSIC_TIMEOUT
  - laya /v1/laya/decide router dispatch: bounded by _laya_dispatch_timeout()
    / FUSION_LAYA_DISPATCH_TIMEOUT, TimeoutError -> 503
  - uniworld backend: uses the shared get_video_gen_timeout() knob instead of
    a hardcoded 600s

Also pins the ServerConfig.default_timeout deprecation status.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api.decisions_routes import (
    _laya_dispatch_timeout,
)
from fusion_mlx.api.decisions_routes import (
    router as decisions_router,
)
from fusion_mlx.engines.music import MusicGenEngine, get_music_gen_timeout
from fusion_mlx.middleware.auth import verify_api_key


def _client():
    app = FastAPI()
    app.include_router(decisions_router)
    app.dependency_overrides[verify_api_key] = lambda: True
    return TestClient(app)


class _SleepyOrchestrator:
    def __init__(self, delay_s):
        self._delay_s = delay_s

    def generate_music(self, **kwargs):
        time.sleep(self._delay_s)
        wav = np.zeros((2, 4), dtype=np.float32)
        return (wav, 48000)


class _SleepyRouter:
    def predict(self, prompt, questions, lang=None):
        time.sleep(1.0)
        return {"answers": {}, "routing": {"model": "test-router"}}


def test_music_timeout_env_default_and_override(monkeypatch):
    monkeypatch.delenv("FUSION_MUSIC_TIMEOUT", raising=False)
    assert get_music_gen_timeout() == 600.0
    monkeypatch.setenv("FUSION_MUSIC_TIMEOUT", "5")
    assert get_music_gen_timeout() == 5.0
    monkeypatch.setenv("FUSION_MUSIC_TIMEOUT", "junk")
    assert get_music_gen_timeout() == 600.0


def test_music_generate_bounded(monkeypatch):
    monkeypatch.setenv("FUSION_MUSIC_TIMEOUT", "0.2")
    eng = MusicGenEngine.__new__(MusicGenEngine)
    eng._orchestrator = _SleepyOrchestrator(delay_s=1.0)
    with pytest.raises(RuntimeError, match="exceeded"):
        asyncio.run(eng.generate_music("test caption"))


def test_music_generate_success_path(monkeypatch):
    monkeypatch.setenv("FUSION_MUSIC_TIMEOUT", "5")
    eng = MusicGenEngine.__new__(MusicGenEngine)
    eng._orchestrator = _SleepyOrchestrator(delay_s=0.0)
    wav, sr = asyncio.run(eng.generate_music("test caption"))
    assert sr == 48000
    assert wav is not None


def test_laya_dispatch_timeout_helper(monkeypatch):
    monkeypatch.delenv("FUSION_LAYA_DISPATCH_TIMEOUT", raising=False)
    assert _laya_dispatch_timeout() == 120.0
    monkeypatch.setenv("FUSION_LAYA_DISPATCH_TIMEOUT", "5")
    assert _laya_dispatch_timeout() == 5.0
    monkeypatch.setenv("FUSION_LAYA_DISPATCH_TIMEOUT", "junk")
    assert _laya_dispatch_timeout() == 120.0


def test_laya_decide_timeout_503(monkeypatch):
    monkeypatch.setenv("FUSION_LAYA_DISPATCH_TIMEOUT", "0.1")
    fake = _SleepyRouter()
    with patch("fusion_mlx.api.decisions_routes._router_instance", fake):
        client = _client()
        resp = client.post(
            "/v1/laya/decide",
            json={"prompt": "hi", "preset": "router", "use_router": True},
        )
    assert resp.status_code == 503
    assert "exceeded" in resp.json()["detail"]


def test_uniworld_uses_shared_knob():
    import fusion_mlx.video.uniworld.backend as backend_mod

    src = Path(backend_mod.__file__).read_text()
    assert "get_video_gen_timeout" in src
    assert "timeout=600.0" not in src


def test_video_gen_timeout_env_governor(monkeypatch):
    from fusion_mlx.engine_core import get_video_gen_timeout

    monkeypatch.delenv("FUSION_VIDEO_GEN_TIMEOUT", raising=False)
    assert get_video_gen_timeout() == 7200.0
    monkeypatch.setenv("FUSION_VIDEO_GEN_TIMEOUT", "0.2")
    assert get_video_gen_timeout() == 0.2


def test_default_timeout_deprecated_not_removed():
    from fusion_mlx import config as config_mod
    from fusion_mlx.cli_serve import (
        audio_mode,  # noqa: F401
        config_resolve,  # noqa: F401
    )
    from fusion_mlx.config import ServerConfig

    cfg = ServerConfig()
    assert hasattr(cfg, "default_timeout")
    src = Path(config_mod.__file__).read_text()
    assert "deprecated: never read" in src
