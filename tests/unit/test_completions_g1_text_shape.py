# SPDX-License-Identifier: Apache-2.0
"""Regression for G-1 (#0912 audit): ``/v1/completions`` must return the
text-completion shape (``choices[0].text``, ``object: text_completion``),
not the chat-completion shape (``choices[0].message.content``).

Pre-fix the route wrapped the prompt into a chat message, ran it through
``_run_chat``/``_stream_chat``, and returned the ``ChatCompletionResponse``
directly. OpenAI SDK clients reading ``choices[0].text`` got ``None``.

Fix: single message→text mapping in ``api/openai/completions.py`` —
non-stream remaps ``ChatCompletionResponse`` (and the context-budget
``JSONResponse`` wrapper) to ``CompletionResponse``; stream wraps the
chat ``StreamingResponse`` body to rewrite ``choices[0].delta.content``
SSE chunks as ``choices[0].text`` with ``object: text_completion``.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api.openai import router as openai_router
from fusion_mlx.api.openai import set_openai_context
from fusion_mlx.engine.base import GenerationOutput


class _PlainChatEngine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    async def chat(self, messages, **kwargs):
        return GenerationOutput(
            text="2 + 2 equals 4.",
            prompt_tokens=3,
            completion_tokens=5,
            finished=True,
            finish_reason="stop",
        )

    async def abort_request(self, request_id):
        return None


class _StreamChatEngine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self):
        self._tokens = ["Hello", " world", "!"]

    async def stream_chat(self, messages, **kwargs):
        for i, tok in enumerate(self._tokens):
            yield GenerationOutput(
                text="",
                new_text=tok,
                finished=False,
                finish_reason=None,
                prompt_tokens=2,
                completion_tokens=i + 1,
            )
        yield GenerationOutput(
            text="",
            new_text="",
            finished=True,
            finish_reason="stop",
            prompt_tokens=2,
            completion_tokens=3,
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


def test_completions_returns_text_completion_shape():
    client = _make_client(_PlainChatEngine())
    resp = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "prompt": "What is 2+2?",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["object"] == "text_completion", body
    choices = body["choices"]
    assert len(choices) == 1
    # choices[0].text must carry the content (not choices[0].message.content).
    assert "message" not in choices[0]
    assert choices[0]["text"] == "2 + 2 equals 4."
    assert choices[0]["finish_reason"] == "stop"
    assert body["model"] == "test-model"
    assert body["usage"]["prompt_tokens"] == 3
    assert body["usage"]["completion_tokens"] == 5


def test_completions_no_message_key_in_choice():
    """The legacy envelope has no ``message`` key — only ``text``."""
    client = _make_client(_PlainChatEngine())
    resp = client.post(
        "/v1/completions",
        json={"model": "test-model", "max_tokens": 8, "prompt": "hi"},
    )
    assert resp.status_code == 200, resp.text
    choice = resp.json()["choices"][0]
    assert "message" not in choice
    assert "text" in choice
    assert "finish_reason" in choice
    assert "index" in choice


def test_completions_stream_emits_text_completion_chunks():
    client = _make_client(_StreamChatEngine())
    with client.stream(
        "POST",
        "/v1/completions",
        json={"model": "test-model", "max_tokens": 16, "prompt": "hi", "stream": True},
    ) as resp:
        assert resp.status_code == 200, resp.text
        chunks = []
        for line in resp.iter_lines():
            if line.startswith("data: "):
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                import json

                chunks.append(json.loads(payload))
    assert chunks, "no SSE chunks received"
    texts = []
    for ch in chunks:
        assert ch["object"] == "text_completion", ch
        assert "text" in ch["choices"][0]
        assert "message" not in ch["choices"][0]
        texts.append(ch["choices"][0]["text"])
    joined = "".join(texts)
    assert "Hello world" in joined, joined
    # Final chunk must carry finish_reason + usage.
    last = chunks[-1]
    assert last["choices"][0]["finish_reason"] == "stop", last
    assert last.get("usage") is not None
    assert last["usage"]["completion_tokens"] == 3


def teardown_module():
    # The module-level ``set_openai_context(_MockPool(engine), None)`` wires
    # a mock engine pool into the shared OpenAI route context. Without this
    # teardown the mock leaks into every later test module in the session —
    # their ``_resolve_engine`` finds our mock (engine_type=unknown) instead
    # of their own ``cfg.engine`` stub (test_diffusion_engine et al.).
    from fusion_mlx.api.openai import set_openai_context

    set_openai_context(None, None)
