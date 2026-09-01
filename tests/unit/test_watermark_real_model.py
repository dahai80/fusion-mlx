import os
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("FUSION_MLX_REAL_MODEL_TESTS"),
    reason="set FUSION_MLX_REAL_MODEL_TESTS to run real-model watermark round-trip",
)

_SECRET = "real-model-test-secret-nondefault"


def _find_small_model() -> str | None:
    base = Path.home() / ".fusion-mlx" / "models"
    if not base.exists():
        return None
    candidates = []
    for p in base.iterdir():
        if (p / "config.json").exists() and (p / "model.safetensors").exists():
            candidates.append(p)
    candidates.sort(key=lambda p: (p / "model.safetensors").stat().st_size)
    return str(candidates[0]) if candidates else None


def test_embed_verify_real_model_roundtrip(tmp_path):
    model_path = _find_small_model()
    if model_path is None:
        pytest.skip("no small safetensors model in ~/.fusion-mlx/models")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from fusion_mlx.admin.auth import require_admin
    from fusion_mlx.api.watermark_routes import router
    from fusion_mlx.middleware.auth import require_model_hub_source

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_admin] = lambda: True
    app.dependency_overrides[require_model_hub_source] = lambda: True
    client = TestClient(app)

    out = tmp_path / "wm-out"
    out.mkdir()
    payload = {"owner": "dahai80", "purpose": "provenance-test"}

    r = client.post(
        "/v1/watermark/embed",
        json={
            "model": model_path,
            "payload": payload,
            "secret": _SECRET,
            "output_path": str(out),
        },
    )
    assert r.status_code == 200, r.text
    embed = r.json()
    assert embed["signature"]
    assert embed["carrier_count"] > 0

    r2 = client.post(
        "/v1/watermark/verify",
        json={"model": str(out), "secret": _SECRET},
    )
    assert r2.status_code == 200, r2.text
    verify = r2.json()
    assert verify["verified"] is True, verify
    assert verify["payload"] == payload
    assert verify["signature"] == embed["signature"]
    shutil.rmtree(out, ignore_errors=True)
