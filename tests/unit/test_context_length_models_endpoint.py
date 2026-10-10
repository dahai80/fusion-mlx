# SPDX-License-Identifier: Apache-2.0
"""Regression tests for context_length in /v1/models, /api/tags, /api/show.

Claude Code reads ``context_length`` from the OpenAI-compatible
``/v1/models`` response to compute precompact thresholds. When the field
was absent, Claude Code rejected prompts during precompact with
"prompt is too long". Same gap existed in Ollama ``/api/tags`` and
``/api/show``.

Fix: ``ModelInfo`` gains ``context_length: int | None``; the list
handler resolves it from ``entry.model_context_length`` (discovery) or
``get_model_max_context(engine)`` (loaded engine fallback). Ollama
``/api/tags`` + ``/api/show`` gain the same engine fallback via
``_entry_context_length``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api.openai import router as openai_router
from fusion_mlx.api.openai._common import set_openai_context


@pytest.fixture(autouse=True)
def _reset_openai_pool():
    yield
    set_openai_context(None, None)


class _Entry:
    def __init__(
        self,
        mid: str,
        mt: str = "llm",
        ctx: int | None = None,
        loaded: bool = False,
    ):
        self.model_type = mt
        self.config_model_type = mt
        self.model_context_length = ctx
        self.engine = SimpleNamespace() if loaded else None
        self.estimated_size = 1000
        self.actual_size = None
        self.last_observed_size = 1200


class _FakePool:
    def __init__(self, entries: dict[str, _Entry]):
        self._entries = entries

    def list_models(self):
        return list(self._entries.keys())

    def get_entry(self, mid):
        return self._entries.get(mid)


def _openai_client(pool) -> TestClient:
    app = FastAPI()
    app.include_router(openai_router)
    from fusion_mlx.middleware.auth import verify_api_key

    app.dependency_overrides[verify_api_key] = lambda: True
    from fusion_mlx.api._concurrency import init_request_semaphore

    init_request_semaphore(8)
    set_openai_context(pool, SimpleNamespace())
    return TestClient(app)


class TestV1ModelsContextLength:
    """/v1/models includes context_length resolved from pool entry."""

    def test_from_discovery(self):
        pool = _FakePool(
            {"Qwen3-8B-4bit": _Entry("Qwen3-8B-4bit", ctx=40960, loaded=True)}
        )
        with patch(
            "fusion_mlx._cli_base.resolve_default_chat_model", return_value=None
        ):
            client = _openai_client(pool)
            resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) == 1
        assert data[0]["context_length"] == 40960

    def test_from_engine_fallback(self):
        pool = _FakePool(
            {"Qwen3-8B-4bit": _Entry("Qwen3-8B-4bit", ctx=None, loaded=True)}
        )
        with (
            patch("fusion_mlx._cli_base.resolve_default_chat_model", return_value=None),
            patch(
                "fusion_mlx.service.helpers.get_model_max_context", return_value=131072
            ),
        ):
            client = _openai_client(pool)
            resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data[0]["context_length"] == 131072

    def test_none_when_not_loaded(self):
        pool = _FakePool(
            {"Qwen3-8B-4bit": _Entry("Qwen3-8B-4bit", ctx=None, loaded=False)}
        )
        with patch(
            "fusion_mlx._cli_base.resolve_default_chat_model", return_value=None
        ):
            client = _openai_client(pool)
            resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data[0].get("context_length") is None

    def test_multiple_models(self):
        pool = _FakePool(
            {
                "Qwen3-8B-4bit": _Entry("Qwen3-8B-4bit", ctx=40960, loaded=True),
                "Llama-3.2-1B": _Entry("Llama-3.2-1B", ctx=131072, loaded=True),
            }
        )
        with patch(
            "fusion_mlx._cli_base.resolve_default_chat_model", return_value=None
        ):
            client = _openai_client(pool)
            resp = client.get("/v1/models")
        assert resp.status_code == 200
        by_id = {m["id"]: m for m in resp.json()["data"]}
        assert by_id["Qwen3-8B-4bit"]["context_length"] == 40960
        assert by_id["Llama-3.2-1B"]["context_length"] == 131072


class TestOllamaContextLength:
    """/api/tags and /api/show include context_length."""

    def test_tags_context_length(self):
        from fusion_mlx.api import ollama_routes

        pool = _FakePool({"test-model": _Entry("test-model", ctx=262144, loaded=True)})
        app = FastAPI()
        from fusion_mlx.middleware.auth import verify_api_key

        app.dependency_overrides[verify_api_key] = lambda: True
        app.include_router(ollama_routes.router)
        ollama_routes._pool = pool
        try:
            client = TestClient(app)
            resp = client.get("/api/tags")
        finally:
            ollama_routes._pool = None
        assert resp.status_code == 200
        models = resp.json()["models"]
        assert len(models) == 1
        assert models[0]["details"]["context_length"] == 262144

    def test_tags_no_context_length_when_none(self):
        from fusion_mlx.api import ollama_routes

        pool = _FakePool({"test-model": _Entry("test-model", ctx=None, loaded=False)})
        app = FastAPI()
        from fusion_mlx.middleware.auth import verify_api_key

        app.dependency_overrides[verify_api_key] = lambda: True
        app.include_router(ollama_routes.router)
        ollama_routes._pool = pool
        try:
            client = TestClient(app)
            resp = client.get("/api/tags")
        finally:
            ollama_routes._pool = None
        assert resp.status_code == 200
        models = resp.json()["models"]
        assert "context_length" not in models[0]["details"]

    def test_show_context_length(self):
        from fusion_mlx.api import ollama_routes

        pool = _FakePool({"test-model": _Entry("test-model", ctx=262144, loaded=True)})
        app = FastAPI()
        from fusion_mlx.middleware.auth import verify_api_key

        app.dependency_overrides[verify_api_key] = lambda: True
        app.include_router(ollama_routes.router)
        ollama_routes._pool = pool
        with patch(
            "fusion_mlx.server.resolve_model_with_profile",
            return_value=("test-model", {}),
        ):
            try:
                client = TestClient(app)
                resp = client.post("/api/show", json={"name": "test-model"})
            finally:
                ollama_routes._pool = None
        assert resp.status_code == 200
        body = resp.json()
        assert body["details"]["context_length"] == 262144
        assert body["model_info"]["llm.context_length"] == 262144

    def test_show_engine_fallback(self):
        from fusion_mlx.api import ollama_routes

        pool = _FakePool({"test-model": _Entry("test-model", ctx=None, loaded=True)})
        app = FastAPI()
        from fusion_mlx.middleware.auth import verify_api_key

        app.dependency_overrides[verify_api_key] = lambda: True
        app.include_router(ollama_routes.router)
        ollama_routes._pool = pool
        with (
            patch(
                "fusion_mlx.server.resolve_model_with_profile",
                return_value=("test-model", {}),
            ),
            patch(
                "fusion_mlx.service.helpers.get_model_max_context", return_value=131072
            ),
        ):
            try:
                client = TestClient(app)
                resp = client.post("/api/show", json={"name": "test-model"})
            finally:
                ollama_routes._pool = None
        assert resp.status_code == 200
        body = resp.json()
        assert body["details"]["context_length"] == 131072
        assert body["model_info"]["llm.context_length"] == 131072
