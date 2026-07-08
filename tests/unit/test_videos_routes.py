# SPDX-License-Identifier: Apache-2.0
# Tests for POST /v1/videos/generate. Uses a minimal FastAPI app with the
# videos router and a mocked EnginePool - no mlx-video or model loading.

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api.videos_routes import (
    router as videos_router,
)
from fusion_mlx.api.videos_routes import (
    set_videos_context,
)
from fusion_mlx.engines.video import VideoGenEngine


def _make_video_engine(byte_sequences):
    engine = MagicMock(spec=VideoGenEngine)
    engine.generate = AsyncMock(return_value=list(byte_sequences))
    return engine


def _make_app(pool) -> TestClient:
    app = FastAPI()
    app.include_router(videos_router)
    set_videos_context(pool)
    return TestClient(app, raise_server_exceptions=False)


class TestVideoGenerateB64:
    def test_b64_json_response(self):
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=_make_video_engine([b"MP4_X"]))
        client = _make_app(pool)
        resp = client.post(
            "/v1/videos/generate",
            json={"prompt": "a cat", "response_format": "b64_json"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "created" in body
        assert len(body["data"]) == 1
        import base64

        assert base64.b64decode(body["data"][0]["b64_json"]) == b"MP4_X"
        assert body["data"][0]["url"] is None

    def test_url_response_is_data_uri(self):
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=_make_video_engine([b"MP4_Y"]))
        client = _make_app(pool)
        resp = client.post(
            "/v1/videos/generate",
            json={"prompt": "a dog"},
        )
        assert resp.status_code == 200
        url = resp.json()["data"][0]["url"]
        assert url.startswith("data:video/mp4;base64,")
        assert resp.json()["data"][0]["b64_json"] is None

    def test_multiple_videos_n(self):
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=_make_video_engine([b"A", b"B", b"C"]))
        client = _make_app(pool)
        resp = client.post(
            "/v1/videos/generate",
            json={"prompt": "p", "n": 3, "response_format": "b64_json"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) == 3


class TestVideoGenerateDefaults:
    def test_default_model_is_ltx2(self):
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=_make_video_engine([b"M"]))
        client = _make_app(pool)
        client.post("/v1/videos/generate", json={"prompt": "p"})
        pool.get_engine.assert_awaited_once_with("ltx-2")

    def test_explicit_model_passed_through(self):
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=_make_video_engine([b"M"]))
        client = _make_app(pool)
        client.post(
            "/v1/videos/generate",
            json={"prompt": "p", "model": "wan2.1"},
        )
        pool.get_engine.assert_awaited_once_with("wan2.1")

    def test_generate_receives_request_params(self):
        engine = _make_video_engine([b"M"])
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=engine)
        client = _make_app(pool)
        client.post(
            "/v1/videos/generate",
            json={
                "prompt": "hello",
                "num_frames": 16,
                "width": 512,
                "height": 512,
                "fps": 12,
                "seed": 99,
                "n": 1,
            },
        )
        engine.generate.assert_awaited_once()
        _, kwargs = engine.generate.call_args
        assert kwargs["prompt"] == "hello"
        assert kwargs["num_frames"] == 16
        assert kwargs["width"] == 512
        assert kwargs["height"] == 512
        assert kwargs["fps"] == 12
        assert kwargs["seed"] == 99
        assert kwargs["n"] == 1


class TestVideoGenerateErrors:
    def test_404_when_engine_not_video_gen(self):
        pool = MagicMock()
        not_video = MagicMock()  # not a VideoGenEngine instance
        pool.get_engine = AsyncMock(return_value=not_video)
        client = _make_app(pool)
        resp = client.post("/v1/videos/generate", json={"prompt": "p"})
        assert resp.status_code == 404
        assert "not loaded" in resp.json()["detail"].lower()

    def test_404_when_engine_none(self):
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=None)
        client = _make_app(pool)
        resp = client.post("/v1/videos/generate", json={"prompt": "p"})
        assert resp.status_code == 404

    def test_500_when_engine_generate_raises(self):
        engine = MagicMock(spec=VideoGenEngine)
        engine.generate = AsyncMock(side_effect=RuntimeError("boom"))
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=engine)
        client = _make_app(pool)
        resp = client.post("/v1/videos/generate", json={"prompt": "p"})
        assert resp.status_code == 500
        assert "boom" in resp.json()["detail"]


class TestVideoGenerateValidation:
    @pytest.mark.parametrize(
        "payload",
        [
            {"prompt": "p", "n": 0},
            {"prompt": "p", "n": 5},
            {"prompt": "p", "num_frames": 0},
            {"prompt": "p", "width": 100},
            {"prompt": "p", "height": 5000},
            {"prompt": "p", "fps": 0},
            {"prompt": "p", "response_format": "weird"},
        ],
    )
    def test_invalid_payload_returns_422(self, payload):
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=_make_video_engine([b"M"]))
        client = _make_app(pool)
        resp = client.post("/v1/videos/generate", json=payload)
        assert resp.status_code == 422
