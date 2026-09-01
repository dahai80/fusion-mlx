import base64
import io

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api import images_sr


def _stub_resolve(images, model_path=None, scale=4, tile_size=512,
                  tile_overlap=64, config=None):
    n, h, w, c = images.shape
    rep = scale
    out = np.repeat(np.repeat(images, rep, axis=1), rep, axis=2)
    return out.astype(np.float32)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(images_sr, "super_resolve", _stub_resolve)
    monkeypatch.setattr(images_sr.os.path, "exists", lambda p: True)
    app = FastAPI()
    app.include_router(images_sr.router)
    return TestClient(app)


def _png_bytes(arr_hwc):
    from PIL import Image
    pil = Image.fromarray((np.clip(arr_hwc, 0, 1) * 255).astype(np.uint8)[:, :, :3])
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def test_sr_endpoint_roundtrip(client):
    img = _png_bytes(np.random.rand(32, 40, 3).astype(np.float32))
    resp = client.post(
        "/v1/images/super-resolution",
        files={"image": ("frame.png", img, "image/png")},
        data={"scale": "4", "tile_size": "512"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["in_width"] == 40 and body["in_height"] == 32
    assert body["width"] == 160 and body["height"] == 128
    dec = base64.b64decode(body["image_b64"])
    from PIL import Image
    out = np.array(Image.open(io.BytesIO(dec)))
    assert out.shape[0] == 128 and out.shape[1] == 160
