# SPDX-License-Identifier: Apache-2.0
# Tests for POST /v1/merge-adapter (#584). _run_merge_sync is mocked so no real
# mlx_lm load/fuse/save happens; we assert routing, the sync-await contract
# (output_path returned in body), model alias resolution, output_path
# validation, the slash-in-model body form, and the model-hub-source guard.

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from fusion_mlx.admin.auth import require_admin
from fusion_mlx.api import convert_routes
from fusion_mlx.middleware import require_model_hub_source


@pytest.fixture(scope="session")
def client():
    app = FastAPI()

    async def _ok():
        return True

    app.dependency_overrides[require_admin] = _ok
    app.dependency_overrides[require_model_hub_source] = _ok
    app.include_router(convert_routes.router)
    with TestClient(app) as c:
        yield c


def test_merge_adapter_returns_output_path(client, monkeypatch):
    captured: dict = {}

    def _fake_merge(base_model, adapter_path, output_path, dequantize, upload_repo):
        captured.update(
            base_model=base_model,
            adapter_path=adapter_path,
            output_path=output_path,
            dequantize=dequantize,
            upload_repo=upload_repo,
        )
        return str(output_path)

    monkeypatch.setattr(convert_routes, "_run_merge_sync", _fake_merge)
    r = client.post(
        "/v1/merge-adapter",
        json={"model": "test/repo", "adapter_path": "/data/adapters"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "test/repo"
    assert body["output_path"]
    assert captured["base_model"] == "test/repo"
    assert captured["adapter_path"] == "/data/adapters"
    assert captured["dequantize"] is False
    assert captured["upload_repo"] is None


def test_merge_adapter_accepts_slash_in_model(client, monkeypatch):
    # HF repos are org/name with a slash; the body form avoids URL encoding.
    captured: dict = {}
    monkeypatch.setattr(
        convert_routes,
        "_run_merge_sync",
        lambda *a: captured.update(base_model=a[0]) or str(a[2]),
    )
    r = client.post(
        "/v1/merge-adapter",
        json={"model": "mlx-community/Qwen2.5-0.5B-4bit", "adapter_path": "/adapters"},
    )
    assert r.status_code == 200, r.text
    assert captured["base_model"] == "mlx-community/Qwen2.5-0.5B-4bit"


def test_merge_adapter_resolves_alias(client, monkeypatch):
    monkeypatch.setattr(
        "fusion_mlx.model_aliases.resolve_model",
        lambda name: "mlx-community/real-model",
    )
    monkeypatch.setattr(convert_routes, "_run_merge_sync", lambda *a: str(a[2]))
    r = client.post(
        "/v1/merge-adapter",
        json={"model": "my-alias", "adapter_path": "/adapters"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "my-alias"


def test_merge_adapter_dequantize_flag(client, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        convert_routes,
        "_run_merge_sync",
        lambda *a: captured.update(dequantize=a[3]) or str(a[2]),
    )
    r = client.post(
        "/v1/merge-adapter",
        json={"model": "test/repo", "adapter_path": "/adapters", "dequantize": True},
    )
    assert r.status_code == 200, r.text
    assert captured["dequantize"] is True


def test_merge_adapter_default_output_path(client, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        convert_routes,
        "_run_merge_sync",
        lambda *a: captured.update(output_path=a[2]) or str(a[2]),
    )
    r = client.post(
        "/v1/merge-adapter",
        json={"model": "test/repo", "adapter_path": "/adapters"},
    )
    assert r.status_code == 200, r.text
    # default save path derives from model basename
    assert captured["output_path"] == "repo-fused"


def test_merge_adapter_rejects_output_path_outside_allowed_dirs(client):
    # /etc is not an allowed output prefix -> validator rejects (422)
    r = client.post(
        "/v1/merge-adapter",
        json={
            "model": "test/repo",
            "adapter_path": "/adapters",
            "output_path": "/etc/evil",
        },
    )
    assert r.status_code == 422


def test_merge_adapter_missing_model_rejected(client):
    r = client.post("/v1/merge-adapter", json={"adapter_path": "/adapters"})
    assert r.status_code == 422


def test_merge_adapter_failure_returns_500(client, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("adapter missing")

    monkeypatch.setattr(convert_routes, "_run_merge_sync", _boom)
    r = client.post(
        "/v1/merge-adapter",
        json={"model": "test/repo", "adapter_path": "/adapters"},
    )
    assert r.status_code == 500
    assert "adapter missing" in r.json()["detail"]


def test_merge_adapter_requires_model_hub_source(monkeypatch):
    # TestClient uses loopback (127.0.0.1) which require_model_hub_source
    # treats as dev-mode-allowed. To assert the real guard, swap in a strict
    # dependency that requires the X-Fusion-Source: model-hub header.
    async def _strict_source(request: Request):
        if request.headers.get("x-fusion-source", "").lower() != "model-hub":
            raise HTTPException(
                403, "Model management requires X-Fusion-Source: model-hub"
            )
        return True

    monkeypatch.setattr(convert_routes, "_run_merge_sync", lambda *a: str(a[2]))
    app = FastAPI()

    async def _ok():
        return True

    app.dependency_overrides[require_admin] = _ok
    app.dependency_overrides[require_model_hub_source] = _strict_source
    app.include_router(convert_routes.router)
    with TestClient(app) as c:
        r = c.post(
            "/v1/merge-adapter",
            json={"model": "test/repo", "adapter_path": "/adapters"},
        )
        assert r.status_code == 403
        assert "X-Fusion-Source" in r.json()["detail"]
        # WITH the header it passes the source guard and returns output_path
        r2 = c.post(
            "/v1/merge-adapter",
            json={"model": "test/repo", "adapter_path": "/adapters"},
            headers={"X-Fusion-Source": "model-hub"},
        )
        assert r2.status_code == 200, r2.text
        assert r2.json()["output_path"]
