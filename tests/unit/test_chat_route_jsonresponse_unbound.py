# SPDX-License-Identifier: Apache-2.0
"""Regression: non-stream /v1/chat/completions with unrecognized params must
not 500 with ``UnboundLocalError: cannot access local variable 'JSONResponse'``.

Pre-fix (main): ``chat_completions`` imported ``JSONResponse`` only inside
three conditional blocks (cache HIT, ONLY_IF_CACHED, cache-store). Python
treats a name assigned anywhere in a function as local for the whole scope,
so the post-dispatch line ``isinstance(result, JSONResponse)`` (which attaches
``X-Fusion-Ignored-Params``) referenced an unbound local whenever the cache
check took the MISS path AND the request carried extra params
(``__pydantic_extra__`` non-empty → ``_ignored_header`` truthy). That path is
the common one for Claude Code / Ollama clients that send ``enable_thinking``
or other non-standard top-level keys, so a plain non-stream chat call 500'd.

Fix: one unconditional ``from starlette.responses import JSONResponse`` at the
top of ``chat_completions``; the three conditional imports removed.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api.openai import router as openai_router
from fusion_mlx.api.openai import set_openai_context
from fusion_mlx.engine.base import GenerationOutput


class _PlainChatEngine:
    """Mock non-streaming chat engine returning plain content (no reasoning)."""

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    async def chat(self, messages, **kwargs):
        return GenerationOutput(
            text="hello",
            prompt_tokens=2,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
        )

    async def abort_request(self, request_id):
        return None


class _MockPool:
    def __init__(self, engine):
        self._engine = engine

    async def get_engine(self, model_name, _lease=False, adapter_path=None):
        return self._engine

    async def release_engine(self, model_name, adapter_path=None):
        return None


def _make_client(engine) -> TestClient:
    set_openai_context(_MockPool(engine), None)
    app = FastAPI()
    app.include_router(openai_router)
    return TestClient(app)


def test_non_stream_chat_with_extra_params_does_not_500():
    """Non-stream chat + unrecognized top-level param (enable_thinking) must
    return 200, not 500 UnboundLocalError on the JSONResponse reference."""
    client = _make_client(_PlainChatEngine())
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
            "enable_thinking": True,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "hello"
    # X-Fusion-Ignored-Params must be attached (proves the unbound line ran).
    assert "X-Fusion-Ignored-Params" in resp.headers
    assert "enable_thinking" in resp.headers["X-Fusion-Ignored-Params"]


def test_non_stream_chat_plain_request_still_200():
    """Sanity: no extra params → no ignored header, still 200 (the
    unconditional import must not break the plain path)."""
    client = _make_client(_PlainChatEngine())
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    assert "X-Fusion-Ignored-Params" not in resp.headers
