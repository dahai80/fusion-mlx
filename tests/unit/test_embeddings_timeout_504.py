# SPDX-License-Identifier: Apache-2.0
"""Embedding per-batch timeout must surface as a retryable 504, not a 500.

Reproducer (prod 2026-10-31 22:21-22:30): a heavy in-process DPO job on a
27B model saturated the GPU; every POST /v1/embeddings exceeded the 30s
per-batch cap in EmbeddingEngine.embed, the raw asyncio.TimeoutError fell
through to the unhandled-exception handler, and the client got a 500 plus
a full traceback in the server log — with no log line naming the batch,
the model, or the timeout. Callers retrying (semantic-cache embedFn) had
no way to tell "retry later" from "your request is malformed".
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestEmbeddingRouteTimeout504:
    def test_route_maps_timeout_to_504(self, monkeypatch):
        from fusion_mlx.api import embeddings_routes
        from fusion_mlx.config import get_config

        app = FastAPI()
        app.include_router(embeddings_routes.router)

        cfg = get_config()
        saved = {
            "api_key": cfg.api_key,
            "embedding_model_locked": getattr(cfg, "embedding_model_locked", None),
        }
        cfg.api_key = None

        async def _engine(_model_id):
            engine = MagicMock()

            async def _embed(*_a, **_kw):
                raise TimeoutError()

            engine.embed = _embed
            return engine

        monkeypatch.setattr(embeddings_routes, "get_embedding_engine", _engine)
        try:
            client = TestClient(app, raise_server_exceptions=False)
            r = client.post(
                "/v1/embeddings", json={"model": "bge-m3", "input": ["a", "b"]}
            )
        finally:
            for k, v in saved.items():
                setattr(cfg, k, v)

        assert r.status_code == 504, r.text
        detail = r.json()["detail"].lower()
        assert "timed out" in detail
        assert "retry" in detail

    def test_route_validation_error_still_400(self, monkeypatch):
        from fusion_mlx.api import embeddings_routes
        from fusion_mlx.config import get_config

        app = FastAPI()
        app.include_router(embeddings_routes.router)

        cfg = get_config()
        saved = {
            "api_key": cfg.api_key,
            "embedding_model_locked": getattr(cfg, "embedding_model_locked", None),
        }
        cfg.api_key = None

        async def _engine(_model_id):
            engine = MagicMock()

            async def _embed(*_a, **_kw):
                raise ValueError("bad input")

            engine.embed = _embed
            return engine

        monkeypatch.setattr(embeddings_routes, "get_embedding_engine", _engine)
        try:
            client = TestClient(app, raise_server_exceptions=False)
            r = client.post("/v1/embeddings", json={"model": "bge-m3", "input": ["a"]})
        finally:
            for k, v in saved.items():
                setattr(cfg, k, v)

        assert r.status_code == 400, r.text


class TestEmbeddingEngineTimeoutLog:
    def test_engine_logs_batch_and_reraises(self, caplog):
        from fusion_mlx.engines.embedding import EmbeddingEngine

        eng = EmbeddingEngine.__new__(EmbeddingEngine)
        eng._model_name = "stub-embed"
        eng._batch_size = 2

        class _StubModel:
            def embed(self, inputs, max_length=None, padding=True, truncation=True):
                raise TimeoutError()

        eng._model = _StubModel()
        eng._begin_activity = lambda *a, **k: None
        eng._end_activity = lambda *a: None

        with caplog.at_level(logging.ERROR, logger="fusion_mlx.engines.embedding"):
            with pytest.raises(TimeoutError):
                asyncio.run(eng.embed(["a", "b"]))

        assert "timed out after 30s" in caplog.text
        assert "stub-embed" in caplog.text
        assert "batch=2" in caplog.text
