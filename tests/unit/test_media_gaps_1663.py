# SPDX-License-Identifier: Apache-2.0
"""#1663: media engine gaps — image matting route/engine, /v1/audio/voices
flat catalog, and model-config.json ``models.<name>.type`` overrides.

Contract:
- ``get_model_type_overrides()`` returns only valid engine-kind strings from
  model-config.json; invalid values are logged and ignored (never crash).
- ``detect_model_type`` honors the override first, then detects BiRefNet
  matting checkpoints (config.json ``model_type: birefnet``) as
  ``image_matting``.
- ``/v1/images/matting`` composites an RGBA cutout (or a mask-only PNG)
  through a MattingEngine and rejects non-matting engines with 400.
- ``/v1/audio/voices`` exposes a flat ``catalog`` of builtin + cloned
  voices with {id, name, lang, kind, preview_url}.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image


class _FakeMattingEngine:
    engine_type = "image_matting"
    loaded = True

    async def matte(self, rgb):
        h, w, _ = rgb.shape
        alpha = np.full((h, w), 200, dtype=np.uint8)
        alpha[:, : w // 2] = 10
        return alpha


class _FakeImageEngine:
    engine_type = "image_gen"


class _FakePool:
    def __init__(self, engine):
        self._engine = engine

    async def get_engine(self, model_name):
        return self._engine


@pytest.fixture
def matting_client(monkeypatch):
    import fusion_mlx.api.images as images_mod
    import fusion_mlx.server as server_mod
    from fusion_mlx.api.images import router as images_router
    from fusion_mlx.middleware.auth import check_rate_limit, verify_api_key

    monkeypatch.setattr(server_mod, "resolve_model_id", lambda model_id: model_id)
    monkeypatch.setattr(images_mod, "_pool", _FakePool(_FakeMattingEngine()))

    app = FastAPI()
    app.include_router(images_router)
    app.dependency_overrides[verify_api_key] = lambda: True
    app.dependency_overrides[check_rate_limit] = lambda: True
    return TestClient(app)


def _png_bytes(size=(8, 6), color=(255, 0, 0)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestMattingRoute:
    def test_cutout_rgba(self, matting_client):
        resp = matting_client.post(
            "/v1/images/matting",
            files={"image": ("in.png", _png_bytes(), "image/png")},
            data={"output": "cutout"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model"] == "birefnet-lite"
        assert body["requested_model"] == "birefnet-lite"
        raw = body["data"][0]["b64_json"]
        import base64

        out = Image.open(io.BytesIO(base64.b64decode(raw)))
        assert out.mode == "RGBA"
        arr = np.array(out)
        # right half alpha=200, left half alpha=10
        assert arr[0, -1, 3] == 200
        assert arr[0, 0, 3] == 10

    def test_mask_output_l_mode(self, matting_client):
        resp = matting_client.post(
            "/v1/images/matting",
            files={"image": ("in.png", _png_bytes(), "image/png")},
            data={"output": "mask"},
        )
        assert resp.status_code == 200, resp.text
        import base64

        out = Image.open(
            io.BytesIO(base64.b64decode(resp.json()["data"][0]["b64_json"]))
        )
        assert out.mode == "L"
        assert np.array(out)[0, -1] == 200

    def test_invalid_output_rejected(self, matting_client):
        resp = matting_client.post(
            "/v1/images/matting",
            files={"image": ("in.png", _png_bytes(), "image/png")},
            data={"output": "bogus"},
        )
        assert resp.status_code == 422

    def test_invalid_response_format_rejected(self, matting_client):
        resp = matting_client.post(
            "/v1/images/matting",
            files={"image": ("in.png", _png_bytes(), "image/png")},
            data={"response_format": "bogus"},
        )
        assert resp.status_code == 422

    def test_non_matting_engine_400(self, monkeypatch):
        import fusion_mlx.api.images as images_mod
        import fusion_mlx.server as server_mod
        from fusion_mlx.api.images import router as images_router
        from fusion_mlx.middleware.auth import check_rate_limit, verify_api_key

        monkeypatch.setattr(server_mod, "resolve_model_id", lambda m: m)
        monkeypatch.setattr(images_mod, "_pool", _FakePool(_FakeImageEngine()))
        app = FastAPI()
        app.include_router(images_router)
        app.dependency_overrides[verify_api_key] = lambda: True
        app.dependency_overrides[check_rate_limit] = lambda: True
        client = TestClient(app)
        resp = client.post(
            "/v1/images/matting",
            files={"image": ("in.png", _png_bytes(), "image/png")},
        )
        assert resp.status_code == 400
        assert "not a matting engine" in resp.json()["detail"]


class TestModelTypeOverrides:
    def test_valid_override_returned(self, monkeypatch):
        import fusion_mlx.config as config_mod

        monkeypatch.setattr(
            config_mod,
            "_model_config",
            {"models": {"flux2-klein-9b-4bit": {"type": "image"}}},
        )
        assert config_mod.get_model_type_overrides() == {"flux2-klein-9b-4bit": "image"}

    def test_invalid_type_ignored(self, monkeypatch):
        import fusion_mlx.config as config_mod

        monkeypatch.setattr(
            config_mod,
            "_model_config",
            {"models": {"m1": {"type": "not-a-kind"}, "m2": {"type": "video"}}},
        )
        assert config_mod.get_model_type_overrides() == {"m2": "video"}

    def test_non_dict_spec_skipped(self, monkeypatch):
        import fusion_mlx.config as config_mod

        monkeypatch.setattr(
            config_mod, "_model_config", {"models": {"m1": "image", "m2": 3}}
        )
        assert config_mod.get_model_type_overrides() == {}

    def test_detect_model_type_override_wins(self, monkeypatch, tmp_path):
        from fusion_mlx.pool.model_discovery import detect_model_type

        monkeypatch.setattr(
            "fusion_mlx.config.get_model_type_overrides",
            lambda: {tmp_path.name: "video"},
        )
        assert detect_model_type(tmp_path) == "video"

    def test_detect_model_type_birefnet(self, monkeypatch, tmp_path):
        from fusion_mlx.pool.model_discovery import detect_model_type

        monkeypatch.setattr("fusion_mlx.config.get_model_type_overrides", lambda: {})
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "birefnet", "backbone": "swin_v1_t"})
        )
        assert detect_model_type(tmp_path) == "image_matting"


class TestVoicesCatalog:
    async def test_catalog_shape(self, monkeypatch):
        import fusion_mlx.audio.registry as registry_mod
        import fusion_mlx.audio.tts as tts_mod
        import fusion_mlx.audio.voice_bundles as bundles_mod
        import fusion_mlx.training.voice_service as voice_service_mod
        from fusion_mlx.api.audio_routes import list_voices

        monkeypatch.setattr(registry_mod, "list_audio_aliases", lambda: [])
        monkeypatch.setattr(tts_mod, "KOKORO_VOICES", ["af_heart", "zf_xiaobei"])
        monkeypatch.setattr(
            voice_service_mod,
            "list_trained_voices",
            lambda: [
                {"voice_id": "tenant_a_pack", "ready": True},
                {"voice_id": "draft_pack", "ready": False},
            ],
        )
        monkeypatch.setattr(
            bundles_mod,
            "list_voice_bundles",
            lambda: [{"name": "narrator-bundle"}],
        )

        result = await list_voices()
        catalog = result["catalog"]
        by_id = {v["id"]: v for v in catalog}

        assert by_id["af_heart"]["kind"] == "builtin"
        assert by_id["af_heart"]["lang"] == "en-US"
        assert by_id["zf_xiaobei"]["lang"] == "zh-CN"
        assert by_id["tenant_a_pack"]["kind"] == "cloned"
        assert "draft_pack" not in by_id
        assert by_id["narrator-bundle"]["kind"] == "cloned"
        for v in catalog:
            assert set(v) == {"id", "name", "lang", "kind", "preview_url"}


class TestMattingEngine:
    async def test_engine_type_and_lifecycle(self, monkeypatch):
        import fusion_mlx.image.matting.vendor_birefnet_lite as vendor
        from fusion_mlx.engines.matting import MattingEngine

        loaded_from = []

        def fake_load(folder):
            loaded_from.append(folder)
            return object()

        def fake_matte(model, rgb):
            return np.full(rgb.shape[:2], 255, dtype=np.uint8)

        monkeypatch.setattr(vendor, "load", fake_load)
        monkeypatch.setattr(vendor, "matte", fake_matte)
        monkeypatch.setattr("fusion_mlx.engine_core.get_executor", lambda name: None)

        engine = MattingEngine(model_name="/models/birefnet-lite")
        assert engine.engine_type == "image_matting"
        assert engine.tokenizer is None
        assert not engine.loaded

        alpha = await engine.matte(np.zeros((4, 4, 3), dtype=np.uint8))
        assert isinstance(alpha, np.ndarray)

    async def test_engine_starts_and_mattes(self, monkeypatch):
        import fusion_mlx.image.matting.vendor_birefnet_lite as vendor
        from fusion_mlx.engines.matting import MattingEngine

        def fake_load(folder):
            return {"loaded": True}

        def fake_matte(model, rgb):
            return np.full(rgb.shape[:2], 128, dtype=np.uint8)

        monkeypatch.setattr(vendor, "load", fake_load)
        monkeypatch.setattr(vendor, "matte", fake_matte)
        monkeypatch.setattr("fusion_mlx.engine_core.get_executor", lambda name: None)

        engine = MattingEngine(model_name="/models/birefnet-lite")
        await engine.start()
        assert engine.loaded
        alpha = await engine.matte(np.zeros((4, 4, 3), dtype=np.uint8))
        assert alpha[0, 0] == 128
        await engine.stop()
        assert not engine.loaded
