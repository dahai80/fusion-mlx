# SPDX-License-Identifier: Apache-2.0
# Tests for POST /v1/videos/generate. Uses a minimal FastAPI app with the
# videos router and a mocked EnginePool - no mlx-video or model loading.

import base64
import os
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
from fusion_mlx.exceptions import ModelNotFoundError

# Exact kwargs the route is allowed to forward to engine.generate. A mock that
# accepts anything silently masks route/engine signature drift, so the strict
# side_effect below raises on any unexpected kwarg (#12). Optional knobs are
# forwarded only when set, so requests that omit them still produce the core 7.
_VALID_GENERATE_KWARGS = {
    "prompt",
    "num_frames",
    "width",
    "height",
    "fps",
    "seed",
    "n",
    "image",
    "negative_prompt",
    "num_inference_steps",
    "scheduler",
    "cfg_scale",
    "guide_scale",
    "shift",
}


def _make_video_engine(byte_sequences):
    engine = MagicMock(spec=VideoGenEngine)
    payload = list(byte_sequences)

    async def _generate(**kwargs):
        unexpected = set(kwargs) - _VALID_GENERATE_KWARGS
        if unexpected:
            raise AssertionError(f"unexpected generate kwargs: {sorted(unexpected)}")
        prompt = kwargs.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise AssertionError(f"generate called with bad prompt: {prompt!r}")
        return list(payload)

    engine.generate = AsyncMock(side_effect=_generate)
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
                "num_frames": 17,
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
        assert kwargs["num_frames"] == 17
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

    def test_404_when_model_not_found(self):
        pool = MagicMock()
        pool.get_engine = AsyncMock(side_effect=ModelNotFoundError("nope"))
        client = _make_app(pool)
        resp = client.post("/v1/videos/generate", json={"prompt": "p"})
        assert resp.status_code == 404

    def test_503_when_pool_not_initialized(self, monkeypatch):
        from fusion_mlx.api import videos_routes

        monkeypatch.setattr(videos_routes, "_pool", None)
        app = FastAPI()
        app.include_router(videos_router)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/videos/generate", json={"prompt": "p"})
        assert resp.status_code == 503

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
            {"prompt": "p", "num_frames": 16},  # 16 % 8 != 1
            {"prompt": "p", "width": 100},
            {"prompt": "p", "width": 300},  # 300 % 64 != 0
            {"prompt": "p", "height": 5000},
            {"prompt": "p", "height": 300},  # 300 % 64 != 0
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


class TestVideoGenerateBackendAware:
    def test_wan_valid_payload_passes(self):
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=_make_video_engine([b"M"]))
        client = _make_app(pool)
        # 41 = 4k+1, 768/512 % 16 == 0 -> Wan2 constraints satisfied
        resp = client.post(
            "/v1/videos/generate",
            json={"prompt": "p", "model": "wan2.1", "num_frames": 41},
        )
        assert resp.status_code == 200
        pool.get_engine.assert_awaited_once_with("wan2.1")

    def test_wan_invalid_num_frames_422(self):
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=_make_video_engine([b"M"]))
        client = _make_app(pool)
        # 40 -> (40-1)%4 != 0 -> Wan2 rejects
        resp = client.post(
            "/v1/videos/generate",
            json={"prompt": "p", "model": "wan2.1", "num_frames": 40},
        )
        assert resp.status_code == 422
        pool.get_engine.assert_not_awaited()

    def test_wan_invalid_dim_422(self):
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=_make_video_engine([b"M"]))
        client = _make_app(pool)
        # 260 >= 256 (field ok) but 260 % 16 != 0 -> Wan2 rejects
        resp = client.post(
            "/v1/videos/generate",
            json={"prompt": "p", "model": "wan2.1", "width": 260},
        )
        assert resp.status_code == 422

    def test_ltx_default_constraints_still_enforced(self):
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=_make_video_engine([b"M"]))
        client = _make_app(pool)
        # 16 % 8 != 1 -> LTX-2 (default model) rejects via backend validation
        resp = client.post(
            "/v1/videos/generate", json={"prompt": "p", "num_frames": 16}
        )
        assert resp.status_code == 422


class TestVideoGenerateI2V:
    def test_ltx_rejects_image_422(self):
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=_make_video_engine([b"M"]))
        client = _make_app(pool)
        # LTX-2 backend does not support I2V (Phase 1) -> 422
        resp = client.post(
            "/v1/videos/generate",
            json={"prompt": "p", "model": "ltx-2", "image": "/tmp/x.png"},
        )
        assert resp.status_code == 422
        pool.get_engine.assert_not_awaited()

    def test_wan_image_path_forwarded(self):
        engine = _make_video_engine([b"M"])
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=engine)
        client = _make_app(pool)
        resp = client.post(
            "/v1/videos/generate",
            json={"prompt": "p", "model": "wan2.1", "image": "/tmp/clip.png"},
        )
        assert resp.status_code == 200
        _, kwargs = engine.generate.call_args
        assert kwargs["image"] == "/tmp/clip.png"

    def test_wan_image_data_uri_resolved_to_temp(self):
        engine = _make_video_engine([b"M"])
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=engine)
        client = _make_app(pool)
        b64 = base64.b64encode(b"\x89PNG fake").decode()
        data_uri = f"data:image/png;base64,{b64}"
        resp = client.post(
            "/v1/videos/generate",
            json={"prompt": "p", "model": "wan2.1", "image": data_uri},
        )
        assert resp.status_code == 200
        _, kwargs = engine.generate.call_args
        # data URI must be decoded to a temp file path, not passed through raw
        assert "fusion_i2v_" in kwargs["image"]
        assert not kwargs["image"].startswith("data:")
        # temp file must be cleaned up after generation
        assert not os.path.exists(kwargs["image"])

    def test_optional_knobs_forwarded_when_set(self):
        engine = _make_video_engine([b"M"])
        pool = MagicMock()
        pool.get_engine = AsyncMock(return_value=engine)
        client = _make_app(pool)
        resp = client.post(
            "/v1/videos/generate",
            json={
                "prompt": "p",
                "model": "wan2.1",
                "negative_prompt": "blurry",
                "num_inference_steps": 15,
                "scheduler": "unipc",
                "guide_scale": 3.5,
                "shift": 1.0,
            },
        )
        assert resp.status_code == 200
        _, kwargs = engine.generate.call_args
        assert kwargs["negative_prompt"] == "blurry"
        assert kwargs["num_inference_steps"] == 15
        assert kwargs["scheduler"] == "unipc"
        assert kwargs["guide_scale"] == 3.5
        assert kwargs["shift"] == 1.0
        # unset optional knobs must not be forwarded
        assert "image" not in kwargs
        assert "cfg_scale" not in kwargs
