# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1087.

#1087: /v1/models did not mark which model is the default for chat, and
/v1/chat/completions error responses did not echo the resolved model id.
A downstream client (fusion-k12-teacher) couldn't tell its "auto" traffic
was landing on a non-chat model (#1086) until it 400'd — neither the
models list nor the error body hinted at the resolved model.

Fix:
  1. /v1/models marks the resolved default chat model with
     ``default_for_chat: true`` (None/absent otherwise).
  2. /v1/chat/completions 404 (model not available) + 400 (capability
     mismatch) error bodies include ``resolved_model`` +
     ``requested_model`` so a client sees exactly which model was chosen.
  3. ``resolve_default_chat_model()`` returns the resolved default id or
     None (never raises) so the listing stays 200.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx._cli_base import resolve_default_chat_model
from fusion_mlx.api.openai import router as openai_router
from fusion_mlx.api.openai._common import set_openai_context


def _make_pool(model_ids: list[str], model_types: dict[str, str] | None = None):
    model_types = model_types or {}

    class _Entry:
        def __init__(self, mid, mt):
            self.model_type = mt
            self.engine = SimpleNamespace()  # loaded

    class _FakePool:
        def list_models(self):
            return list(model_ids)

        def get_entry(self, mid):
            return _Entry(mid, model_types.get(mid, "llm"))

    return _FakePool()


def _client(pool) -> TestClient:
    app = FastAPI()
    app.include_router(openai_router)
    from fusion_mlx.middleware.auth import verify_api_key

    app.dependency_overrides[verify_api_key] = lambda: True
    from fusion_mlx.api._concurrency import init_request_semaphore

    init_request_semaphore(8)
    set_openai_context(pool, SimpleNamespace())
    return TestClient(app)


class TestResolveDefaultChatModel:
    """resolve_default_chat_model returns the resolved id or None (never raises)."""

    def test_returns_none_when_ambiguous(self):
        # resolve_model("default") raises ValueError when ambiguous; the
        # helper must swallow it and return None.
        with patch(
            "fusion_mlx.model_aliases.resolve_model",
            side_effect=ValueError("ambiguous"),
        ):
            assert resolve_default_chat_model() is None

    def test_returns_resolved_id(self):
        with patch(
            "fusion_mlx.model_aliases.resolve_model", return_value="Qwen3-8B-4bit"
        ):
            assert resolve_default_chat_model() == "Qwen3-8B-4bit"

    def test_returns_auto_detected(self):
        with patch(
            "fusion_mlx.model_aliases.resolve_model", return_value="Llama-3.2-1B"
        ):
            assert resolve_default_chat_model() == "Llama-3.2-1B"

    def test_never_raises_on_resolve_failure(self):
        with patch(
            "fusion_mlx.model_aliases.resolve_model",
            side_effect=ValueError("ambiguous"),
        ):
            assert resolve_default_chat_model() is None


class TestModelsListMarksDefault:
    """/v1/models marks default_for_chat:true on the resolved default entry."""

    def test_default_marked_true(self):
        pool = _make_pool(["Qwen3-8B-4bit", "Llama-3.2-1B"])
        with patch(
            "fusion_mlx._cli_base.resolve_default_chat_model",
            return_value="Qwen3-8B-4bit",
        ):
            client = _client(pool)
            resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()["data"]
        by_id = {m["id"]: m for m in data}
        assert by_id["Qwen3-8B-4bit"].get("default_for_chat") is True
        assert by_id["Llama-3.2-1B"].get("default_for_chat") is None

    def test_no_default_all_none(self):
        pool = _make_pool(["Qwen3-8B-4bit", "Llama-3.2-1B"])
        with patch(
            "fusion_mlx._cli_base.resolve_default_chat_model", return_value=None
        ):
            client = _client(pool)
            resp = client.get("/v1/models")
        assert resp.status_code == 200
        for m in resp.json()["data"]:
            assert m.get("default_for_chat") is None

    def test_default_not_in_list_all_none(self):
        # default resolves to a model NOT in the pool → no entry marked
        pool = _make_pool(["Qwen3-8B-4bit"])
        with patch(
            "fusion_mlx._cli_base.resolve_default_chat_model",
            return_value="Some-Other-Model",
        ):
            client = _client(pool)
            resp = client.get("/v1/models")
        assert resp.status_code == 200
        for m in resp.json()["data"]:
            assert m.get("default_for_chat") is None


class TestChatErrorEchoesResolvedModel:
    """/v1/chat/completions 404/400 include resolved_model + requested_model."""

    def test_404_includes_resolved_model(self):
        # pool has no engine for the requested model → get_engine returns None → 404
        pool = _make_pool([])

        class _NoEnginePool:
            def list_models(self):
                return []

            def get_entry(self, mid):
                return None

            async def get_engine(self, *a, **k):
                return None

            async def release_engine(self, *a, **k):
                pass

        from fusion_mlx.api.openai import _common

        client_app = FastAPI()
        client_app.include_router(openai_router)
        from fusion_mlx.api._concurrency import init_request_semaphore
        from fusion_mlx.middleware.auth import verify_api_key

        init_request_semaphore(8)
        client_app.dependency_overrides[verify_api_key] = lambda: True
        _common._pool = _NoEnginePool()
        _common._request_router = None
        client = TestClient(client_app)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "default", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 404
        body = resp.json()
        detail = body.get("detail", body)
        if isinstance(detail, dict) and "error" in detail:
            err = detail["error"]
        else:
            err = detail
        assert err.get("resolved_model") is not None
        assert err.get("requested_model") == "default"

    def test_source_has_resolved_model_in_error_paths(self):
        from fusion_mlx.api.openai import chat as chat_mod

        src = inspect.getsource(chat_mod._run_chat)
        assert "resolved_model" in src
        assert "requested_model" in src
        # streaming path too
        from fusion_mlx.api.openai import streaming as stream_mod

        ssrc = inspect.getsource(stream_mod._stream_chat)
        assert "resolved_model" in ssrc


class TestRoutesInternalMarksDefault:
    """routes_internal/models.py _entry_payload supports default_for_chat."""

    def test_entry_payload_includes_default_when_true(self):
        from fusion_mlx.routes_internal.models import _entry_payload

        payload = _entry_payload("qwen", None, None, default_for_chat=True)
        assert payload.get("default_for_chat") is True

    def test_entry_payload_omits_default_when_false(self):
        from fusion_mlx.routes_internal.models import _entry_payload

        payload = _entry_payload("qwen", None, None, default_for_chat=False)
        assert "default_for_chat" not in payload

    def test_entry_payload_omits_default_when_none(self):
        from fusion_mlx.routes_internal.models import _entry_payload

        payload = _entry_payload("qwen", None, None)
        assert "default_for_chat" not in payload


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
