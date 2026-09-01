import hashlib
import json
import os
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fusion_mlx.api.watermark_models import (
    WatermarkEmbedRequest,
    WatermarkVerifyRequest,
)
from fusion_mlx.api.watermark_routes import router
from fusion_mlx.watermark.lsb import compute_signature


def test_embed_request_minimal():
    req = WatermarkEmbedRequest(model="org/repo", payload={"a": 1}, secret="nondefault")
    assert req.bits_per_weight == 1
    assert req.in_place is False
    assert req.layers is None


def test_embed_request_bits_per_weight_range():
    with pytest.raises(ValidationError):
        WatermarkEmbedRequest(model="m", payload={}, secret="s", bits_per_weight=4)
    with pytest.raises(ValidationError):
        WatermarkEmbedRequest(model="m", payload={}, secret="s", bits_per_weight=0)


def test_embed_request_output_path_required_when_not_in_place():
    # in_place=False (default) without output_path is allowed at model level;
    # the route enforces output_path presence. Here we just validate the
    # path-prefix constraint when a path IS given.
    home = os.path.expanduser("~/.fusion-mlx/models")
    req = WatermarkEmbedRequest(
        model="m", payload={}, secret="s", output_path=home + "/wm-out"
    )
    assert req.output_path.startswith(home)


def test_verify_request_minimal():
    req = WatermarkVerifyRequest(model="m", secret="s")
    assert req.bits_per_weight == 1
    assert req.layers is None


def _app():
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client():
    app = _app()
    from fusion_mlx.admin.auth import require_admin
    from fusion_mlx.middleware.auth import require_model_hub_source

    app.dependency_overrides[require_admin] = lambda: True
    app.dependency_overrides[require_model_hub_source] = lambda: True
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_embed_rejects_default_secret(client):
    with patch.dict(
        "os.environ", {"FMH_WATERMARK_SECRET": "fusion-model-hub-default-secret"}
    ):
        r = client.post(
            "/v1/watermark/embed",
            json={
                "model": "m",
                "payload": {"a": 1},
                "secret": "fusion-model-hub-default-secret",
            },
        )
        assert r.status_code == 503


def test_embed_rejects_empty_secret(client):
    r = client.post(
        "/v1/watermark/embed",
        json={"model": "m", "payload": {"a": 1}, "secret": ""},
    )
    assert r.status_code == 503


def test_verify_rejects_default_secret(client):
    r = client.post(
        "/v1/watermark/verify",
        json={"model": "m", "secret": "fusion-model-hub-default-secret"},
    )
    assert r.status_code == 503


def test_signature_format_route_aligned():
    sig = compute_signature("nondefault", "org/repo", {"owner": "x"})

    expected = hashlib.sha256(
        f"nondefault:org/repo::{json.dumps({'owner': 'x'}, sort_keys=True)}".encode()
    ).hexdigest()[:32]
    assert sig == expected


def _no_deps_app():
    app = FastAPI()
    app.include_router(router)
    return app


def test_embed_rejects_without_admin():
    app = _no_deps_app()
    client = TestClient(app)
    with patch.dict("os.environ", {"FMH_WATERMARK_SECRET": "nondefault"}):
        r = client.post(
            "/v1/watermark/embed",
            json={"model": "m", "payload": {"a": 1}, "secret": "nondefault"},
        )
    assert r.status_code == 401


def test_embed_rejects_without_hub_source():
    app = _no_deps_app()
    from fusion_mlx.admin.auth import require_admin

    app.dependency_overrides[require_admin] = lambda: True
    client = TestClient(app)
    with patch.dict("os.environ", {"FMH_WATERMARK_SECRET": "nondefault"}):
        r = client.post(
            "/v1/watermark/embed",
            json={"model": "m", "payload": {"a": 1}, "secret": "nondefault"},
        )
    assert r.status_code == 403


class _StubModel:
    def __init__(self, weights):
        self._weights = weights

    def parameters(self):
        return self._weights

    def update_weights(self, new_weights):
        self._weights = new_weights


def _stub_weights():
    import mlx.core as mx

    row = [float(i + 1) for i in range(64)]
    return {
        "layers.0.w": mx.array([row] * 64, dtype=mx.float32),
        "layers.1.w": mx.array([row[:32]] * 32, dtype=mx.float32),
    }


def _wire_admin(app):
    from fusion_mlx.admin.auth import require_admin
    from fusion_mlx.middleware.auth import require_model_hub_source

    app.dependency_overrides[require_admin] = lambda: True
    app.dependency_overrides[require_model_hub_source] = lambda: True


def _shared_model_fakes(resolve_path: str):
    import mlx_lm.utils as mlx_utils

    shared = _StubModel(_stub_weights())

    def fake_load(path, return_config=False):
        if return_config:
            return shared, None, {}
        return shared, None

    def fake_save(save_path, base_model, model, tokenizer, config, donate_model=False):
        return None

    def fake_resolve(name):
        return resolve_path

    return shared, fake_load, fake_save, fake_resolve, mlx_utils


def test_embed_then_verify_roundtrip(tmp_path):
    shared, fake_load, fake_save, fake_resolve, mlx_utils = _shared_model_fakes(
        "stub/path/model"
    )
    with (
        patch.object(mlx_utils, "load", fake_load),
        patch.object(mlx_utils, "save", fake_save),
        patch("fusion_mlx.model_aliases.resolve_model", fake_resolve),
    ):
        app = _app()
        _wire_admin(app)
        client = TestClient(app)
        payload = {"owner": "dahai80", "purpose": "provenance"}
        r1 = client.post(
            "/v1/watermark/embed",
            json={
                "model": "org/provenance-model",
                "payload": payload,
                "secret": "nondefault-secret",
                "in_place": True,
            },
        )
        assert r1.status_code == 200, r1.text
        embed = r1.json()
        assert embed["carrier_count"] > 0
        assert embed["signature"]

        r2 = client.post(
            "/v1/watermark/verify",
            json={"model": "org/provenance-model", "secret": "nondefault-secret"},
        )
        assert r2.status_code == 200, r2.text
        verify = r2.json()
        assert verify["verified"] is True, verify
        assert verify["payload"] == payload
        assert verify["signature"] == embed["signature"]

    expected = hashlib.sha256(
        f"nondefault-secret:org/provenance-model::{json.dumps(payload, sort_keys=True)}".encode()
    ).hexdigest()[:32]
    assert embed["signature"] == expected


def test_embed_verify_signature_stable_across_relocation(tmp_path):
    # F1 regression guard: signature must depend on request.model (logical id),
    # NOT the resolved filesystem path. Embed and verify resolve to DIFFERENT
    # paths but use the SAME logical id -> signatures must match.
    shared, fake_load, fake_save, fake_resolve_embed, mlx_utils = _shared_model_fakes(
        "source/path/model"
    )
    with (
        patch.object(mlx_utils, "load", fake_load),
        patch.object(mlx_utils, "save", fake_save),
        patch("fusion_mlx.model_aliases.resolve_model", fake_resolve_embed),
    ):
        app = _app()
        _wire_admin(app)
        client = TestClient(app)
        payload = {"owner": "x"}
        r1 = client.post(
            "/v1/watermark/embed",
            json={
                "model": "stable-id",
                "payload": payload,
                "secret": "s",
                "in_place": True,
            },
        )
        assert r1.status_code == 200, r1.text
        embed_sig = r1.json()["signature"]

    def fake_resolve_verify(name):
        return "relocated/elsewhere/model"

    with (
        patch.object(mlx_utils, "load", fake_load),
        patch.object(mlx_utils, "save", fake_save),
        patch("fusion_mlx.model_aliases.resolve_model", fake_resolve_verify),
    ):
        app = _app()
        _wire_admin(app)
        client = TestClient(app)
        r2 = client.post(
            "/v1/watermark/verify",
            json={"model": "stable-id", "secret": "s"},
        )
        assert r2.status_code == 200, r2.text
        verify_sig = r2.json()["signature"]

    assert embed_sig == verify_sig


def test_verify_corrupted_returns_false(tmp_path):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    shared, fake_load, fake_save, fake_resolve, mlx_utils = _shared_model_fakes(
        "stub/path/model"
    )
    with (
        patch.object(mlx_utils, "load", fake_load),
        patch.object(mlx_utils, "save", fake_save),
        patch("fusion_mlx.model_aliases.resolve_model", fake_resolve),
    ):
        app = _app()
        _wire_admin(app)
        client = TestClient(app)
        payload = {"owner": "dahai80"}
        r1 = client.post(
            "/v1/watermark/embed",
            json={
                "model": "org/model",
                "payload": payload,
                "secret": "s",
                "in_place": True,
            },
        )
        assert r1.status_code == 200, r1.text

        flat = tree_flatten(shared.parameters())
        name, w = flat[0]
        w_np = np.array(mx.array(w))
        w_np = np.zeros_like(w_np)
        shared._weights = {name: mx.array(w_np)}

        r2 = client.post(
            "/v1/watermark/verify",
            json={"model": "org/model", "secret": "s"},
        )
        assert r2.status_code == 200, r2.text
        verify = r2.json()
        assert verify["verified"] is False
