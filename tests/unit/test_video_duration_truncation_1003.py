# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1003.

#1003: ``POST /v1/videos/generate`` silently truncated requested duration
(15s request -> 2.33s output, 2s -> 0.69s) with no warning or 422. Root
cause: ``duration`` and ``resolution`` were not declared fields on
``VideoGenerateRequest``, so pydantic silently dropped them and the model
fell back to its native dims + default num_frames. Worse, the response
body carried no output metadata, so a caller had no way to detect that
the generated mp4 was shorter than planned.

Fix (#1003): ``duration`` and ``resolution`` are now first-class request
params. ``duration`` derives ``num_frames`` (round(duration*fps/8)*8+1,
honoring the LTX 1+8k constraint); ``resolution`` derives width/height
(rounded to the backend's dim_divisibility). The ACTUAL output duration
is probed from the generated mp4's moov/mvhd box and echoed back as
``VideoOutput.duration_seconds``; a >15% mismatch is logged loudly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api.videos_routes import (
    _duration_to_num_frames,
    _probe_mp4_duration_seconds,
    _resolution_to_dims,
    set_videos_context,
)
from fusion_mlx.api.videos_routes import (
    router as videos_router,
)
from fusion_mlx.engines.video import VideoGenEngine
from fusion_mlx.engines.video_backends import VideoConstraints


def _synth_mp4(duration_s: float, timescale: int = 24) -> bytes:
    # Build a minimal ISOBMFF mp4 whose moov/mvhd declares the given
    # duration. Used as a fake engine.generate() return value so the route
    # can probe the real container duration without ffmpeg or model loading.
    duration_units = int(round(duration_s * timescale))
    # mvhd version 0: version(1)+flags(3)+creation(4)+mod(4)+timescale(4)+duration(4)
    mvhd_body = (
        bytes([0, 0, 0, 0])
        + b"\x00\x00\x00\x00"
        + b"\x00\x00\x00\x00"
        + timescale.to_bytes(4, "big")
        + duration_units.to_bytes(4, "big")
    )
    mvhd = (8 + len(mvhd_body)).to_bytes(4, "big") + b"mvhd" + mvhd_body
    moov = (8 + len(mvhd)).to_bytes(4, "big") + b"moov" + mvhd
    ftyp_body = b"isom" + b"\x00\x00\x02\x00"
    ftyp = (8 + len(ftyp_body)).to_bytes(4, "big") + b"ftyp" + ftyp_body
    return ftyp + moov


def _make_video_engine(byte_sequences):
    engine = MagicMock(spec=VideoGenEngine)
    payload = list(byte_sequences)

    async def _generate(**kwargs):
        return list(payload)

    engine.generate = AsyncMock(side_effect=_generate)
    return engine


def _make_app(pool) -> TestClient:
    app = FastAPI()
    app.include_router(videos_router)
    set_videos_context(pool)
    return TestClient(app, raise_server_exceptions=False)


class TestDurationToNumFrames:
    def test_5s_at_24fps(self):
        # 5*24/8 = 15 -> 15*8+1 = 121
        assert _duration_to_num_frames(5.0, 24) == 121

    def test_2s_at_24fps(self):
        # 2*24/8 = 6 -> 6*8+1 = 49
        assert _duration_to_num_frames(2.0, 24) == 49

    def test_15s_at_24fps(self):
        # 15*24/8 = 45 -> 45*8+1 = 361
        assert _duration_to_num_frames(15.0, 24) == 361

    def test_result_satisfies_ltx_constraint(self):
        # num_frames must be 1 + 8*k for all durations.
        for d in (0.5, 1.0, 3.7, 10.0, 29.9):
            nf = _duration_to_num_frames(d, 24)
            assert nf % 8 == 1, (d, nf)

    def test_clamped_to_minimum_one(self):
        assert _duration_to_num_frames(0.1, 24) >= 1


class TestResolutionToDims:
    def test_1080p_div32_matches_model_native(self):
        # The issue reported 1920x1088 — round(1080/32)*32 = 1088.
        w, h = _resolution_to_dims("1080p", 32)
        assert w == 1920
        assert h == 1088

    def test_720p_div32(self):
        w, h = _resolution_to_dims("720p", 32)
        assert w % 32 == 0 and h % 32 == 0
        assert w == 1280

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown resolution"):
            _resolution_to_dims("4k", 32)

    def test_case_insensitive(self):
        w1, h1 = _resolution_to_dims("1080p", 32)
        w2, h2 = _resolution_to_dims("1080P", 32)
        assert (w1, h1) == (w2, h2)

    def test_div1_no_rounding(self):
        w, h = _resolution_to_dims("720p", 1)
        assert (w, h) == (1280, 720)


class TestProbeMp4Duration:
    def test_reads_declared_duration(self):
        mp4 = _synth_mp4(5.0, timescale=24)
        assert _probe_mp4_duration_seconds(mp4) == 5.0

    def test_fractional_duration(self):
        mp4 = _synth_mp4(2.33, timescale=1000)
        assert abs(_probe_mp4_duration_seconds(mp4) - 2.33) < 1e-6

    def test_v1_header(self):
        # mvhd version 1: 8-byte creation/mod + 4 timescale + 8 duration.
        timescale = 24000
        duration_units = timescale * 3  # 3.0s
        mvhd_body = (
            bytes([1, 0, 0, 0])
            + b"\x00" * 16
            + timescale.to_bytes(4, "big")
            + duration_units.to_bytes(8, "big")
        )
        mvhd = (8 + len(mvhd_body)).to_bytes(4, "big") + b"mvhd" + mvhd_body
        moov = (8 + len(mvhd)).to_bytes(4, "big") + b"moov" + mvhd
        ftyp = (16).to_bytes(4, "big") + b"ftyp" + b"isom" + b"\x00\x00\x02\x00"
        mp4 = ftyp + moov
        assert _probe_mp4_duration_seconds(mp4) == 3.0

    def test_no_moov_returns_none(self):
        ftyp = (16).to_bytes(4, "big") + b"ftyp" + b"isom" + b"\x00\x00\x02\x00"
        assert _probe_mp4_duration_seconds(ftyp) is None

    def test_empty_bytes_returns_none(self):
        assert _probe_mp4_duration_seconds(b"") is None

    def test_truncated_box_returns_none(self):
        mp4 = _synth_mp4(5.0)
        assert _probe_mp4_duration_seconds(mp4[:12]) is None


class TestRouteDurationResolution:
    # End-to-end route behavior: duration/resolution are honored, actual
    # duration is echoed, mismatch is logged.

    def _setup(self, monkeypatch, mp4_bytes):
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=_make_video_engine([mp4_bytes]))
        # Avoid coupling to resolve_backend/model loading: return a generic
        # constraint with the LTX-2.5 dim_divisibility (32).
        monkeypatch.setattr(
            "fusion_mlx.api.videos_routes.constraints_for",
            lambda *a, **k: VideoConstraints(
                supports_i2v=True, max_n=4, dim_divisibility=32
            ),
        )
        return _make_app(pool)

    def test_duration_derives_num_frames(self, monkeypatch):
        mp4 = _synth_mp4(5.0)
        client = self._setup(monkeypatch, mp4)
        resp = client.post(
            "/v1/videos/generate",
            json={"prompt": "p", "model": "ltx-2", "duration": 5.0},
        )
        assert resp.status_code == 200, resp.text
        out = resp.json()["data"][0]
        assert out["num_frames"] == 121
        assert out["duration_seconds"] == 5.0

    def test_resolution_derives_dims(self, monkeypatch):
        mp4 = _synth_mp4(5.0)
        client = self._setup(monkeypatch, mp4)
        resp = client.post(
            "/v1/videos/generate",
            json={"prompt": "p", "model": "ltx-2", "resolution": "1080p"},
        )
        assert resp.status_code == 200, resp.text
        out = resp.json()["data"][0]
        assert out["width"] == 1920
        assert out["height"] == 1088

    def test_actual_duration_echoed_on_truncation(self, monkeypatch, caplog):
        # Request 15s but the (fake) mp4 only contains 2.33s — the route
        # must echo the ACTUAL 2.33s, not the requested 15s, and log a
        # warning (#1003 core complaint: silent truncation).
        mp4 = _synth_mp4(2.33, timescale=1000)
        client = self._setup(monkeypatch, mp4)
        with caplog.at_level("WARNING", logger="fusion_mlx.api.videos_routes"):
            resp = client.post(
                "/v1/videos/generate",
                json={"prompt": "p", "model": "ltx-2", "duration": 15.0},
            )
        assert resp.status_code == 200, resp.text
        out = resp.json()["data"][0]
        assert abs(out["duration_seconds"] - 2.33) < 1e-3
        assert any("truncated" in r.getMessage() for r in caplog.records)

    def test_unknown_resolution_422(self, monkeypatch):
        mp4 = _synth_mp4(5.0)
        client = self._setup(monkeypatch, mp4)
        resp = client.post(
            "/v1/videos/generate",
            json={"prompt": "p", "model": "ltx-2", "resolution": "8k"},
        )
        assert resp.status_code == 422
        assert "unknown resolution" in resp.text

    def test_duration_overrides_default_num_frames(self, monkeypatch):
        # Without duration, default num_frames=97 is used.
        mp4 = _synth_mp4(4.0)
        client = self._setup(monkeypatch, mp4)
        resp = client.post(
            "/v1/videos/generate",
            json={"prompt": "p", "model": "ltx-2"},
        )
        assert resp.json()["data"][0]["num_frames"] == 97
        # With duration=2, num_frames becomes 49 (overrides the 97 default).
        client2 = self._setup(monkeypatch, mp4)
        resp2 = client2.post(
            "/v1/videos/generate",
            json={"prompt": "p", "model": "ltx-2", "duration": 2.0},
        )
        assert resp2.json()["data"][0]["num_frames"] == 49

    def test_response_has_metadata_fields(self, monkeypatch):
        mp4 = _synth_mp4(5.0)
        client = self._setup(monkeypatch, mp4)
        resp = client.post(
            "/v1/videos/generate",
            json={"prompt": "p", "model": "ltx-2", "duration": 5.0, "fps": 30},
        )
        out = resp.json()["data"][0]
        for field in ("num_frames", "fps", "width", "height", "duration_seconds"):
            assert field in out, field
        assert out["fps"] == 30


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
