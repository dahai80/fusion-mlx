import pytest
from pydantic import ValidationError

from fusion_mlx.api.watermark_models import (
    WatermarkEmbedRequest,
    WatermarkVerifyRequest,
)


def test_embed_request_minimal():
    req = WatermarkEmbedRequest(
        model="org/repo", payload={"a": 1}, secret="nondefault"
    )
    assert req.bits_per_weight == 1
    assert req.in_place is False
    assert req.layers is None


def test_embed_request_bits_per_weight_range():
    with pytest.raises(ValidationError):
        WatermarkEmbedRequest(
            model="m", payload={}, secret="s", bits_per_weight=4
        )
    with pytest.raises(ValidationError):
        WatermarkEmbedRequest(
            model="m", payload={}, secret="s", bits_per_weight=0
        )


def test_embed_request_output_path_required_when_not_in_place():
    # in_place=False (default) without output_path is allowed at model level;
    # the route enforces output_path presence. Here we just validate the
    # path-prefix constraint when a path IS given.
    import os

    home = os.path.expanduser("~/.fusion-mlx/models")
    req = WatermarkEmbedRequest(
        model="m", payload={}, secret="s", output_path=home + "/wm-out"
    )
    assert req.output_path.startswith(home)


def test_verify_request_minimal():
    req = WatermarkVerifyRequest(model="m", secret="s")
    assert req.bits_per_weight == 1
    assert req.layers is None


import hashlib
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api.watermark_routes import router


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
    from fusion_mlx.watermark.lsb import compute_signature

    sig = compute_signature("nondefault", "org/repo", {"owner": "x"})
    import json

    expected = hashlib.sha256(
        f"nondefault:org/repo::{json.dumps({'owner': 'x'}, sort_keys=True)}".encode()
    ).hexdigest()[:32]
    assert sig == expected
