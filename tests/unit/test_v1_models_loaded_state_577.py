# SPDX-License-Identifier: Apache-2.0
"""#577 — /v1/models must distinguish ``loaded`` (resident in memory) from
``registered`` (on-disk, discovered, not yet loaded).

Pre-fix the route only listed loaded models; a non-empty list made gateway /
studio / design checkers assume every id was immediately servable — a
502-on-generate "fake green". The fix adds a ``loaded`` bool + ``state`` field
to every entry and appends pool-discovered-but-not-loaded models as
``state="registered"``.

This file pins:
  * served (single-model) entries carry ``loaded=True`` / ``state="loaded"``.
  * pool-discovered models with a live engine surface as ``loaded=True``.
  * pool-discovered models with ``engine=None`` surface as ``loaded=False`` /
    ``state="registered"`` AND are listed (not invisible).
  * no pool wired -> only the served model appears (single-route mounts
    unchanged).
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@contextmanager
def _mounted(*, model_name=None, model_registry=None, pool=None):
    from fusion_mlx.config import get_config
    from fusion_mlx.routes_internal import models as models_route

    app = FastAPI()
    app.include_router(models_route.router)

    cfg = get_config()
    saved = {
        k: getattr(cfg, k, None)
        for k in (
            "model_name",
            "model_alias",
            "model_registry",
            "embedding_model_locked",
            "api_key",
        )
    }
    saved_pool = models_route._pool
    cfg.model_name = model_name
    cfg.model_alias = None
    cfg.model_registry = model_registry
    cfg.embedding_model_locked = None
    cfg.api_key = None
    models_route.set_models_context(pool)
    try:
        yield TestClient(app)
    finally:
        for k, v in saved.items():
            setattr(cfg, k, v)
        models_route.set_models_context(saved_pool)


def _make_pool_entry(model_id, *, loaded):
    entry = MagicMock()
    entry.model_id = model_id
    entry.engine = MagicMock() if loaded else None
    entry.model_type = "llm"
    return entry


def _make_pool(entries):
    pool = MagicMock()
    pool.list_models.return_value = [e.model_id for e in entries]
    pool.get_entry.side_effect = lambda mid: next(
        (e for e in entries if e.model_id == mid), None
    )
    return pool


def _by_id(body, model_id):
    for entry in body["data"]:
        if entry["id"] == model_id:
            return entry
    raise AssertionError(f"id {model_id!r} not in /v1/models: {body}")


def test_served_model_is_loaded():
    raw_id = "mlx-community/Qwen3-0.6B-bf16"
    with _mounted(model_name=raw_id) as client:
        body = client.get("/v1/models").json()
    entry = _by_id(body, raw_id)
    assert entry["loaded"] is True
    assert entry["state"] == "loaded"


def test_pool_loaded_model_surfaces_loaded():
    loaded = _make_pool_entry("mlx-community/Loaded-1B-4bit", loaded=True)
    registered = _make_pool_entry("mlx-community/Registered-1B-4bit", loaded=False)
    pool = _make_pool([loaded, registered])
    with _mounted(model_name="mlx-community/Qwen3-0.6B-bf16", pool=pool) as client:
        body = client.get("/v1/models").json()
    loaded_entry = _by_id(body, "mlx-community/Loaded-1B-4bit")
    assert loaded_entry["loaded"] is True
    assert loaded_entry["state"] == "loaded"


def test_pool_registered_model_surfaces_registered_and_listed():
    # #577 core: an on-disk-registered model (engine=None) must NOT be
    # invisible — it appears with loaded=False / state="registered".
    registered = _make_pool_entry("mlx-community/Registered-1B-4bit", loaded=False)
    pool = _make_pool([registered])
    with _mounted(model_name="mlx-community/Qwen3-0.6B-bf16", pool=pool) as client:
        body = client.get("/v1/models").json()
    ids = [e["id"] for e in body["data"]]
    assert (
        "mlx-community/Registered-1B-4bit" in ids
    ), "registered-but-not-loaded model must appear in /v1/models (#577 fake-green)"
    entry = _by_id(body, "mlx-community/Registered-1B-4bit")
    assert entry["loaded"] is False
    assert entry["state"] == "registered"


def test_no_pool_single_model_unchanged():
    # Single-route mounts (no pool wired) keep the pre-fix shape: only the
    # served model appears, no registered-only entries leak in.
    raw_id = "mlx-community/Qwen3-0.6B-bf16"
    with _mounted(model_name=raw_id, pool=None) as client:
        body = client.get("/v1/models").json()
    ids = [e["id"] for e in body["data"]]
    assert ids == [raw_id]
    entry = _by_id(body, raw_id)
    assert entry["loaded"] is True
    assert entry["state"] == "loaded"


def test_no_duplicate_when_pool_also_discovers_served():
    # The served model is both in the single-model branch AND in the pool;
    # the pool branch must skip ids already listed (no duplicate card).
    served = _make_pool_entry("mlx-community/Qwen3-0.6B-bf16", loaded=True)
    pool = _make_pool([served])
    with _mounted(model_name="mlx-community/Qwen3-0.6B-bf16", pool=pool) as client:
        body = client.get("/v1/models").json()
    served_cards = [
        e for e in body["data"] if e["id"] == "mlx-community/Qwen3-0.6B-bf16"
    ]
    assert len(served_cards) == 1, f"served model duplicated: {served_cards}"


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
