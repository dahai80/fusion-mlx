# SPDX-License-Identifier: Apache-2.0
# #1486 item 2: POST /v1/videos/generations alias. Downstream clients
# (ai-ads-videobuilder) send {model, prompt, image_b64?, duration} and read
# the OpenAI-style envelope back. Verifies the mapping onto /generate:
# duration -> num_frames at 24 fps, raw b64 -> png data URI, response shape.

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


@pytest.fixture(autouse=True)
def _reset_videos_pool(monkeypatch):
    # Routes verify Bearer auth; in a bare TestClient there is no configured
    # key and loopback anonymous access is fail-closed (#350) — the dev/test
    # override is the supported bypass.
    monkeypatch.setenv("FUSION_ALLOW_ANONYMOUS", "1")
    # The heavy-admission gate is host-memory dependent — a 1000+ frame
    # projection fails on a 7GB CI runner but not locally. This is a wiring
    # test, not an admission test: stub the gate as always-admitting.
    from types import SimpleNamespace

    import fusion_mlx.admission as admission_mod

    monkeypatch.setattr(
        admission_mod,
        "get_heavy_admission_gate",
        lambda: SimpleNamespace(
            admit_media=lambda *a, **k: SimpleNamespace(admitted=True, reason="")
        ),
    )
    yield
    import fusion_mlx.api.videos_routes as vr

    vr._pool = None


def _make_app(pool) -> TestClient:
    app = FastAPI()
    app.include_router(videos_router)
    set_videos_context(pool)
    return TestClient(app, raise_server_exceptions=False)


def _pool_capturing():
    # Route isinstance-checks the engine, so spec against VideoGenEngine.
    # The pipeline resolves image inputs to temp files before the engine
    # sees them and unlinks them after — snapshot the bytes inside generate.
    captured = {}
    image_bytes = {"data": None}

    async def _generate(**kwargs):
        captured.update(kwargs)
        if captured.get("image") and os.path.isfile(captured["image"]):
            with open(captured["image"], "rb") as f:
                image_bytes["data"] = f.read()
        return [b"MP4DATA"]

    eng = MagicMock(spec=VideoGenEngine)
    eng.generate = AsyncMock(side_effect=_generate)
    pool = MagicMock()
    pool.get_engine = AsyncMock(return_value=eng)
    return pool, captured, image_bytes


class TestGenerationsAlias:
    def test_minimal_request_maps_prompt(self):
        # /generate forces b64_json internally (the #1254 job flow writes
        # outputs to disk first) — bytes come back in data[0].b64_json.
        pool, captured, _ = _pool_capturing()
        client = _make_app(pool)
        resp = client.post(
            "/v1/videos/generations",
            json={"model": "wan2", "prompt": "a cat walking"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]) == 1
        assert base64.b64decode(body["data"][0]["b64_json"]) == b"MP4DATA"
        assert captured["prompt"] == "a cat walking"

    def test_duration_maps_to_num_frames(self):
        pool, captured, _ = _pool_capturing()
        client = _make_app(pool)
        resp = client.post(
            "/v1/videos/generations",
            json={"prompt": "a cat", "duration": 2.0},
        )
        assert resp.status_code == 200
        # 2s * 24fps = 48 → snapped to the 1+8k frame constraint.
        assert captured["num_frames"] == 49

    def test_duration_clamped_to_schema_ceiling(self):
        pool, captured, _ = _pool_capturing()
        client = _make_app(pool)
        resp = client.post(
            "/v1/videos/generations",
            json={"prompt": "a cat", "duration": 45.0},
        )
        assert resp.status_code == 200
        # 45s*24 = 1080 → clamped to the largest 1+8k frame count ≤1024.
        assert captured["num_frames"] == 1017

    def test_image_b64_wraps_png_data_uri(self):
        pool, captured, image_bytes = _pool_capturing()
        client = _make_app(pool)
        raw = base64.b64encode(b"FAKEPNG").decode()
        resp = client.post(
            "/v1/videos/generations",
            json={"prompt": "make it move", "image_b64": raw},
        )
        assert resp.status_code == 200
        assert captured["image"].endswith(".png")
        assert image_bytes["data"] == b"FAKEPNG"

    def test_image_data_uri_passthrough(self):
        pool, captured, image_bytes = _pool_capturing()
        client = _make_app(pool)
        uri = "data:image/jpeg;base64,QUJD"
        resp = client.post(
            "/v1/videos/generations",
            json={"prompt": "make it move", "image_b64": uri},
        )
        assert resp.status_code == 200
        assert captured["image"].endswith(".jpg")
        assert image_bytes["data"] == b"ABC"

    def test_b64_json_response_format(self):
        pool, _, _ = _pool_capturing()
        client = _make_app(pool)
        resp = client.post(
            "/v1/videos/generations",
            json={"prompt": "a cat", "response_format": "b64_json"},
        )
        assert resp.status_code == 200
        assert (
            resp.json()["data"][0]["b64_json"] == base64.b64encode(b"MP4DATA").decode()
        )

    def test_invalid_duration_rejected(self):
        pool, _, _ = _pool_capturing()
        client = _make_app(pool)
        resp = client.post(
            "/v1/videos/generations",
            json={"prompt": "a cat", "duration": 0},
        )
        assert resp.status_code == 422

    def test_missing_prompt_rejected(self):
        pool, _, _ = _pool_capturing()
        client = _make_app(pool)
        resp = client.post(
            "/v1/videos/generations",
            json={"model": "wan2"},
        )
        assert resp.status_code == 422
