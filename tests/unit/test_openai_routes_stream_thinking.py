# SPDX-License-Identifier: Apache-2.0
"""Issue #21: streaming chat must split <think...</think > blocks into
``reasoning_content`` deltas, not dump thinking text into ``content``.

Qwen3.5 chat template injects an open ``<think`` tag so the model
generates ``<think ... </think > real answer``. Before the fix the
streaming fast path emitted every token as ``content`` (the engine
never set ``reasoning_content``), so OpenAI-compatible clients saw
thinking mixed into ``content`` and misjudged terminal turns.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api.openai_routes import router as openai_router
from fusion_mlx.api.openai_routes import set_openai_context
from fusion_mlx.api.thinking import _CLOSE_TAG, _OPEN_TAG
from fusion_mlx.engine.base import GenerationOutput


class _ThinkEngine:
    """Mock engine emitting a <think block then the real answer."""

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self) -> None:
        self.stream_calls: list[dict[str, Any]] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        # Tags deliberately split across chunks to exercise the parser's
        # partial-tag buffering.
        yield GenerationOutput(
            text=_OPEN_TAG, new_text=_OPEN_TAG, completion_tokens=1, finished=False
        )
        yield GenerationOutput(
            text=_OPEN_TAG + "Let me think.",
            new_text="Let me think.",
            completion_tokens=2,
            finished=False,
        )
        yield GenerationOutput(
            text=_OPEN_TAG + "Let me think." + _CLOSE_TAG,
            new_text=_CLOSE_TAG,
            completion_tokens=3,
            finished=False,
        )
        yield GenerationOutput(
            text=_OPEN_TAG + "Let me think." + _CLOSE_TAG + "Hi!",
            new_text="Hi!",
            completion_tokens=4,
            finished=True,
            finish_reason="stop",
        )

    async def abort_request(self, request_id):
        return None


class _NoThinkEngine:
    """Mock engine emitting plain content with no thinking tags."""

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    async def stream_chat(self, messages, **kwargs):
        yield GenerationOutput(
            text="Hello", new_text="Hello", completion_tokens=1, finished=False
        )
        yield GenerationOutput(
            text="Hello world",
            new_text=" world",
            completion_tokens=2,
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


@pytest.fixture(autouse=True)
def _reset_pool():
    yield
    set_openai_context(None, None)


def _parse_sse_deltas(body: str) -> list[dict]:
    deltas: list[dict] = []
    for raw_event in body.split("\n\n"):
        for line in raw_event.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line.removeprefix("data: ")
            if payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices", []) or []:
                if "delta" in choice:
                    deltas.append(choice["delta"])
    return deltas


def test_streaming_splits_thinking_into_reasoning_content():
    """Issue #21: <think block text must surface as reasoning_content,
    not content."""
    engine = _ThinkEngine()
    client = _make_client(engine)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text

    deltas = _parse_sse_deltas(resp.text)
    assert deltas, "expected streaming deltas"

    reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
    content = "".join(d.get("content", "") or "" for d in deltas)

    assert (
        "Let me think." in reasoning
    ), f"thinking text leaked out of reasoning_content; reasoning={reasoning!r}"
    assert "Hi!" in content, f"real answer missing from content; content={content!r}"
    # Thinking text must NOT appear in content.
    assert (
        "Let me think." not in content
    ), f"#21 regression: thinking text in content; content={content!r}"
    # Raw think tags must not leak into either channel.
    assert (
        _OPEN_TAG not in content
    ), f"open tag leaked into content; content={content!r}"
    assert (
        _CLOSE_TAG not in content
    ), f"close tag leaked into content; content={content!r}"
    assert (
        _OPEN_TAG not in reasoning
    ), f"open tag leaked into reasoning; reasoning={reasoning!r}"


def test_streaming_plain_content_no_reasoning():
    """Tag-free output passes through as content only; no spurious
    reasoning_content deltas."""
    engine = _NoThinkEngine()
    client = _make_client(engine)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text

    deltas = _parse_sse_deltas(resp.text)
    content = "".join(d.get("content", "") or "" for d in deltas)
    reasoning = "".join(d.get("reasoning_content", "") for d in deltas)

    assert content == "Hello world", f"plain content corrupted; got {content!r}"
    assert (
        reasoning == ""
    ), f"no-think output must not emit reasoning_content; got {reasoning!r}"
