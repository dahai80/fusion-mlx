# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

logger = logging.getLogger(__name__)


class _FakeEngine:
    is_mllm = False
    supports_guided_generation = False

    def __init__(self, gen):
        self._gen = gen

    async def chat(self, messages, **kwargs):
        return self._gen


def _stub_server_module(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("fusion_mlx.server")
    fake.resolve_model_with_profile = lambda model_id: (model_id, {})  # type: ignore[attr-defined]
    fake.get_max_context_window = lambda model_id: 0  # type: ignore[attr-defined]
    fake.get_settings = MagicMock(return_value=MagicMock(sse_keepalive_seconds=0))  # type: ignore[attr-defined]
    fake._server_state = {}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fusion_mlx.server", fake)


def _make_gen(text="hello world", completion_tokens=5, prompt_tokens=10):
    from fusion_mlx.engines.base import GenerationOutput

    return GenerationOutput(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason="stop",
        tool_calls=None,
        cached_tokens=0,
    )


def _make_request():
    from fusion_mlx.api.openai_models import ChatCompletionRequest

    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
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


def _patch_routes(monkeypatch, gen):
    import fusion_mlx.api.openai_routes as routes

    _stub_server_module(monkeypatch)
    engine = _FakeEngine(gen)
    monkeypatch.setattr(
        routes, "_resolve_engine", lambda *a, **kw: _async_return(engine)
    )
    monkeypatch.setattr(routes, "_release_engine", lambda *a, **kw: _async_return(None))
    monkeypatch.setattr(routes, "record_chat_session", lambda *a, **kw: None)
    monkeypatch.setattr(routes, "record_llm_metrics", lambda *a, **kw: None)
    monkeypatch.setattr(routes, "_pool", None)
    monkeypatch.setattr(routes, "_request_router", object())
    return routes


async def _async_return(value):
    return value


def test_emit_request_and_activation_when_consent_on(telemetry_env, monkeypatch):
    from fusion_mlx.telemetry import emit
    from fusion_mlx.telemetry.state import record_consent

    record_consent(True, fusion_mlx_version="0.0.0+test")

    captured: list[dict] = []

    class _StubQueue:
        def enqueue(self, payload):
            captured.append(payload)

    monkeypatch.setattr(emit, "get_queue", lambda: _StubQueue())
    monkeypatch.setattr(emit, "claim_activation_marker", lambda kind: True)

    gen = _make_gen()
    routes = _patch_routes(monkeypatch, gen)
    req = _make_request()

    import asyncio

    resp = asyncio.new_event_loop().run_until_complete(
        routes._run_chat(
            req, _skip_cap_check=True, headers={"user-agent": "test-agent/1.0"}
        )
    )
    asyncio.set_event_loop(asyncio.new_event_loop())

    assert resp is not None
    logger.info("captured payloads: %d", len(captured))
    request_payloads = [p for p in captured if p.get("event") == "request"]
    activation_payloads = [p for p in captured if p.get("event") == "activation"]
    assert (
        len(request_payloads) == 1
    ), f"expected 1 request event, got {len(request_payloads)}"
    r = request_payloads[0]["request"]
    assert r["endpoint"] == "/v1/chat/completions"
    assert r["stream"] is False
    assert r["status"] == 200
    assert "completion_tokens_bucket" in r
    assert isinstance(r["caller_agent"], str)
    assert (
        len(activation_payloads) == 1
    ), f"expected 1 activation event, got {len(activation_payloads)}"
    assert activation_payloads[0]["activation"]["activation_kind"] == "first_inference"


def test_no_emit_when_consent_off(telemetry_env, monkeypatch):
    from fusion_mlx.telemetry import emit

    captured: list[dict] = []

    class _StubQueue:
        def enqueue(self, payload):
            captured.append(payload)

    monkeypatch.setattr(emit, "get_queue", lambda: _StubQueue())

    gen = _make_gen()
    routes = _patch_routes(monkeypatch, gen)
    req = _make_request()

    import asyncio

    resp = asyncio.new_event_loop().run_until_complete(
        routes._run_chat(
            req, _skip_cap_check=True, headers={"user-agent": "test-agent/1.0"}
        )
    )
    asyncio.set_event_loop(asyncio.new_event_loop())

    assert resp is not None
    assert captured == [], f"expected zero emits when consent off, got {captured}"


def test_telemetry_failure_does_not_break_request(telemetry_env, monkeypatch):
    from fusion_mlx.telemetry import emit
    from fusion_mlx.telemetry.state import record_consent

    record_consent(True, fusion_mlx_version="0.0.0+test")

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic telemetry failure")

    monkeypatch.setattr(emit, "request", boom)

    gen = _make_gen()
    routes = _patch_routes(monkeypatch, gen)
    req = _make_request()

    import asyncio

    resp = asyncio.new_event_loop().run_until_complete(
        routes._run_chat(
            req, _skip_cap_check=True, headers={"user-agent": "test-agent/1.0"}
        )
    )
    asyncio.set_event_loop(asyncio.new_event_loop())

    assert resp is not None
    if hasattr(resp, "choices"):
        assert resp.choices is not None
    else:
        assert hasattr(resp, "body"), f"unexpected response type: {type(resp)}"
