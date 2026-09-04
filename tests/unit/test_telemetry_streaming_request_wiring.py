# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

logger = logging.getLogger(__name__)


class _FakeStreamGen:
    def __init__(
        self,
        new_text="hello",
        completion_tokens=5,
        prompt_tokens=10,
        cached_tokens=0,
        finish_reason="stop",
    ):
        self.new_text = new_text
        self.completion_tokens = completion_tokens
        self.prompt_tokens = prompt_tokens
        self.cached_tokens = cached_tokens
        self.finish_reason = finish_reason
        self.finished = True
        self.tool_calls = None
        self.logprobs = None


class _FakeEngine:
    is_mllm = False
    supports_guided_generation = False

    def __init__(self, chunks):
        self._chunks = chunks

    async def stream_chat(self, messages, **kwargs):
        for c in self._chunks:
            yield c


def _stub_server_module(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("fusion_mlx.server")
    fake.resolve_model_with_profile = lambda model_id: (model_id, {})  # type: ignore[attr-defined]
    fake.get_max_context_window = lambda model_id: 0  # type: ignore[attr-defined]
    fake.get_settings = MagicMock(return_value=MagicMock(sse_keepalive_seconds=0))  # type: ignore[attr-defined]
    fake._server_state = {}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fusion_mlx.server", fake)


def _make_request():
    from fusion_mlx.api.openai_models import ChatCompletionRequest

    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )


@pytest.fixture
def telemetry_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("FUSION_MLX_TELEMETRY", raising=False)
    monkeypatch.setenv("FUSION_MLX_TELEMETRY_REQUEST_SAMPLE", "1.0")
    import importlib

    import fusion_mlx.telemetry.state as state

    importlib.reload(state)
    import fusion_mlx.telemetry.emit as emit

    importlib.reload(emit)
    emit._reset_for_tests()
    yield


def _patch_routes(monkeypatch, chunks):
    import fusion_mlx.api.openai_routes as routes

    _stub_server_module(monkeypatch)
    monkeypatch.setattr(routes, "_inject_web_search", lambda req: _async_none())
    import fusion_mlx.tool_parsers.ui_tars_tool_parser as _uitp

    monkeypatch.setattr(
        _uitp, "inject_ui_tars_sysprompt_for_lane", lambda msgs, **kw: msgs
    )
    monkeypatch.setattr(
        routes,
        "_messages_for_engine",
        lambda req_msgs, is_mllm: list(req_msgs),
    )
    monkeypatch.setattr(
        routes,
        "_build_sampling_params",
        lambda req, **kw: MagicMock(
            max_tokens=16,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
            min_p=0.0,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            stop=None,
        ),
    )
    monkeypatch.setattr(routes, "_resolve_streaming_reasoning_parser", lambda m: None)
    monkeypatch.setattr(routes, "_resolve_streaming_tool_parser", lambda e, m: None)
    monkeypatch.setattr(routes, "_compile_grammar_for_request", lambda e, r: None)
    monkeypatch.setattr(routes, "record_chat_session", lambda *a, **kw: None)
    monkeypatch.setattr(routes, "record_llm_metrics", lambda *a, **kw: None)
    monkeypatch.setattr(routes, "record_llm_disconnect_cancel", lambda *a, **kw: None)
    monkeypatch.setattr(routes, "is_claude_code_request", lambda h: False)
    monkeypatch.setattr(
        routes,
        "_adapter",
        MagicMock(
            format_stream_chunk=lambda *a, **kw: "data: {}\n\n",
            format_stream_end=lambda *a, **kw: "data: [DONE]\n\n",
        ),
    )
    import fusion_mlx.service.helpers as helpers

    monkeypatch.setattr(
        helpers, "compute_prompt_tokens_for_messages", lambda e, m, **kw: 5
    )
    engine = _FakeEngine(chunks)
    return routes, engine


async def _async_none():
    return None


async def _drive(routes, engine, req, headers):
    chunks_out: list[str] = []
    async for piece in routes._stream_chat_generator(
        req,
        engine,
        "test-model",
        None,
        principal=None,
        headers=headers,
    ):
        chunks_out.append(piece)
    return chunks_out


def test_streaming_emit_request_and_activation_when_consent_on(
    telemetry_env, monkeypatch
):
    from fusion_mlx.telemetry import emit
    from fusion_mlx.telemetry.state import record_consent

    record_consent(True, fusion_mlx_version="0.0.0+test")

    captured: list[dict] = []

    class _StubQueue:
        def enqueue(self, payload):
            captured.append(payload)

    monkeypatch.setattr(emit, "get_queue", lambda: _StubQueue())
    monkeypatch.setattr(emit, "claim_activation_marker", lambda kind: True)

    chunks = [
        _FakeStreamGen(new_text="hello ", completion_tokens=3, prompt_tokens=10),
        _FakeStreamGen(new_text="world", completion_tokens=5, prompt_tokens=10),
    ]
    routes, engine = _patch_routes(monkeypatch, chunks)
    req = _make_request()

    import asyncio

    out = asyncio.new_event_loop().run_until_complete(
        _drive(routes, engine, req, {"user-agent": "test-agent/1.0"})
    )
    asyncio.set_event_loop(asyncio.new_event_loop())

    assert out, "stream should yield chunks"
    logger.info("captured payloads: %d", len(captured))
    request_payloads = [p for p in captured if p.get("event") == "request"]
    activation_payloads = [p for p in captured if p.get("event") == "activation"]
    assert (
        len(request_payloads) == 1
    ), f"expected 1 request, got {len(request_payloads)}"
    r = request_payloads[0]["request"]
    assert r["endpoint"] == "/v1/chat/completions"
    assert r["stream"] is True
    assert r["status"] == 200
    assert r["completion_empty"] is False
    assert r["output_degenerate"] is False
    assert r["completion_abnormally_short"] is False
    assert r["tool_call_used"] is False
    assert isinstance(r["caller_agent"], str)
    assert (
        len(activation_payloads) == 1
    ), f"expected 1 activation, got {len(activation_payloads)}"
    assert activation_payloads[0]["activation"]["activation_kind"] == "first_inference"


def test_streaming_no_emit_when_consent_off(telemetry_env, monkeypatch):
    from fusion_mlx.telemetry import emit

    captured: list[dict] = []

    class _StubQueue:
        def enqueue(self, payload):
            captured.append(payload)

    monkeypatch.setattr(emit, "get_queue", lambda: _StubQueue())

    chunks = [
        _FakeStreamGen(new_text="hello", completion_tokens=5, prompt_tokens=10),
    ]
    routes, engine = _patch_routes(monkeypatch, chunks)
    req = _make_request()

    import asyncio

    out = asyncio.new_event_loop().run_until_complete(
        _drive(routes, engine, req, {"user-agent": "test-agent/1.0"})
    )
    asyncio.set_event_loop(asyncio.new_event_loop())

    assert out, "stream should yield chunks even without telemetry"
    assert captured == [], f"expected zero emits when consent off, got {captured}"


def test_streaming_telemetry_failure_does_not_break_stream(telemetry_env, monkeypatch):
    from fusion_mlx.telemetry import emit
    from fusion_mlx.telemetry.state import record_consent

    record_consent(True, fusion_mlx_version="0.0.0+test")

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic telemetry failure")

    monkeypatch.setattr(emit, "request", boom)

    chunks = [
        _FakeStreamGen(new_text="hello ", completion_tokens=3, prompt_tokens=10),
        _FakeStreamGen(new_text="world", completion_tokens=5, prompt_tokens=10),
    ]
    routes, engine = _patch_routes(monkeypatch, chunks)
    req = _make_request()

    import asyncio

    out = asyncio.new_event_loop().run_until_complete(
        _drive(routes, engine, req, {"user-agent": "test-agent/1.0"})
    )
    asyncio.set_event_loop(asyncio.new_event_loop())

    assert out, "stream must complete even if telemetry raises"
    assert len(out) >= 2, f"expected multiple chunks, got {len(out)}"
