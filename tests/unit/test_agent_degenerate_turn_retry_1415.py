# SPDX-License-Identifier: Apache-2.0
# Tests for #1415: degenerate agent turn detection + single retry on the
# non-stream chat route. Long-context thinking models intermittently emit
# intent prose then EOS with no tool_calls — the agent client stalls. See
# fusion_mlx/api/openai/chat.py _is_degenerate_agent_turn.
from __future__ import annotations

import asyncio
import logging
import sys
import types
from unittest.mock import MagicMock

import pytest


def _make_gen(text="hello", completion_tokens=5, prompt_tokens=10, **kw):
    from fusion_mlx.engines.base import GenerationOutput

    fields = dict(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason="stop",
        tool_calls=None,
        cached_tokens=0,
    )
    fields.update(kw)
    return GenerationOutput(**fields)


def _make_request(tools=None, **kw):
    from fusion_mlx.api.openai_models import ChatCompletionRequest

    base = dict(
        model="test-model",
        messages=[{"role": "user", "content": "run the probe"}],
        stream=False,
    )
    base.update(kw)
    if tools is not None:
        base["tools"] = tools
    return ChatCompletionRequest(**base)


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


def _healthy_tool_call_gen():
    return _make_gen(
        text="",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "Bash",
                    "arguments": '{"command": "echo hi"}',
                },
            }
        ],
    )


@pytest.fixture(autouse=True)
def _retry_env_on(monkeypatch):
    monkeypatch.delenv("FUSION_MLX_AGENT_DEGENERATE_RETRY", raising=False)


# ---------------------------------------------------------------------------
# Decision helpers
# ---------------------------------------------------------------------------


def test_degenerate_detected_tools_stop_short_text():
    from fusion_mlx.api.openai.chat import _is_degenerate_agent_turn

    req = _make_request(tools=_TOOLS)
    gen = _make_gen(text="Batch probe: CI log + branch state + issue #31")
    assert _is_degenerate_agent_turn(req, gen) is True


def test_not_degenerate_when_tool_calls_present():
    from fusion_mlx.api.openai.chat import _is_degenerate_agent_turn

    req = _make_request(tools=_TOOLS)
    assert _is_degenerate_agent_turn(req, _healthy_tool_call_gen()) is False


def test_not_degenerate_when_finish_length():
    from fusion_mlx.api.openai.chat import _is_degenerate_agent_turn

    req = _make_request(tools=_TOOLS)
    gen = _make_gen(text="Con", finish_reason="length")
    assert _is_degenerate_agent_turn(req, gen) is False


def test_not_degenerate_when_no_tools():
    from fusion_mlx.api.openai.chat import _is_degenerate_agent_turn

    req = _make_request()
    gen = _make_gen(text="Con")
    assert _is_degenerate_agent_turn(req, gen) is False


def test_not_degenerate_when_long_genuine_answer():
    # The 932-token status report stall shape: long content must NOT be
    # retried — only the announcement-without-call shape is.
    from fusion_mlx.api.openai.chat import _is_degenerate_agent_turn

    req = _make_request(tools=_TOOLS)
    gen = _make_gen(text="Status: " + "all systems verified. " * 40)
    assert len(gen.text) > 256
    assert _is_degenerate_agent_turn(req, gen) is False


def test_degenerate_with_think_block_in_text():
    # Engine did not split reasoning_content; raw text carries the think
    # block — detection must look at the post-thinking content only.
    from fusion_mlx.api.openai.chat import _is_degenerate_agent_turn

    req = _make_request(tools=_TOOLS)
    gen = _make_gen(
        text=" <think>\nI should probe the CI log now.\n</think>\n\nBatch probe: CI log."
    )
    assert _is_degenerate_agent_turn(req, gen) is True


def test_post_think_content_prefers_reasoning_content_attr():
    from fusion_mlx.api.openai.chat import _post_think_content

    gen = _make_gen(
        text="Short final answer.",
        reasoning_content="long reasoning " * 10,
    )
    assert _post_think_content(gen) == "Short final answer."


def test_retry_env_kill_switch(monkeypatch):
    from fusion_mlx.api.openai.chat import _agent_degenerate_retry_enabled

    assert _agent_degenerate_retry_enabled() is True
    monkeypatch.setenv("FUSION_MLX_AGENT_DEGENERATE_RETRY", "0")
    assert _agent_degenerate_retry_enabled() is False


# ---------------------------------------------------------------------------
# _run_chat integration: retry once, log evidence
# ---------------------------------------------------------------------------


class _SequenceEngine:
    is_mllm = False
    supports_guided_generation = False

    def __init__(self, gens):
        self._gens = list(gens)
        self.calls = 0

    async def chat(self, messages, **kwargs):
        self.calls += 1
        return self._gens[min(self.calls - 1, len(self._gens) - 1)]


def _patch_routes(monkeypatch, engine):
    import fusion_mlx.api.openai._common as _common_mod
    import fusion_mlx.api.openai.chat as routes
    import fusion_mlx.service.helpers as helpers

    fake = types.ModuleType("fusion_mlx.server")
    fake.resolve_model_with_profile = lambda model_id: (model_id, {})  # type: ignore[attr-defined]
    fake.get_max_context_window = lambda model_id: 0  # type: ignore[attr-defined]
    fake.get_settings = MagicMock(  # type: ignore[attr-defined]
        return_value=MagicMock(sse_keepalive_seconds=0)
    )
    fake._server_state = {}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fusion_mlx.server", fake)

    async def _ret(value):
        return value

    monkeypatch.setattr(routes, "_resolve_engine", lambda *a, **kw: _ret(engine))
    monkeypatch.setattr(routes, "_release_engine", lambda *a, **kw: _ret(None))
    monkeypatch.setattr(routes, "record_chat_session", lambda *a, **kw: None)
    monkeypatch.setattr(routes, "record_llm_metrics", lambda *a, **kw: None)
    monkeypatch.setattr(_common_mod, "_pool", None)
    monkeypatch.setattr(_common_mod, "_request_router", object())
    # Fake engine has no tokenizer/context metadata — skip token pre-checks.
    monkeypatch.setattr(
        helpers, "compute_prompt_tokens_for_messages", lambda *a, **kw: 0
    )
    monkeypatch.setattr(helpers, "get_model_max_context", lambda *a, **kw: 0)
    return routes


def _run(routes, req):
    return asyncio.new_event_loop().run_until_complete(
        routes._run_chat(req, _skip_cap_check=True, headers=None)
    )


def test_retry_recovers_tool_call(monkeypatch, caplog):
    engine = _SequenceEngine(
        [
            _make_gen(text="Batch probe: CI log + branch state"),
            _healthy_tool_call_gen(),
        ]
    )
    routes = _patch_routes(monkeypatch, engine)
    req = _make_request(tools=_TOOLS)

    with caplog.at_level(logging.WARNING, logger="fusion_mlx.api.openai._common"):
        resp = _run(routes, req)

    assert engine.calls == 2, "degenerate turn must be retried exactly once"
    assert resp.choices[0].message.tool_calls, "retry result must carry the tool call"
    warns = [r for r in caplog.records if "degenerate agent turn" in r.getMessage()]
    assert warns, "degenerate turn must be logged loudly"
    assert "Batch probe" in warns[0].getMessage()


def test_long_answer_not_retried(monkeypatch, caplog):
    long_gen = _make_gen(text="Status report: " + "all verified. " * 60)
    engine = _SequenceEngine([long_gen])
    routes = _patch_routes(monkeypatch, engine)
    req = _make_request(tools=_TOOLS)

    with caplog.at_level(logging.WARNING, logger="fusion_mlx.api.openai._common"):
        resp = _run(routes, req)

    assert engine.calls == 1, "genuine long answer must not be retried"
    assert not any("degenerate agent turn" in r.getMessage() for r in caplog.records)
    assert resp.choices[0].message.content.startswith("Status report:")


def test_retry_env_off_disables_retry(monkeypatch):
    monkeypatch.setenv("FUSION_MLX_AGENT_DEGENERATE_RETRY", "0")
    engine = _SequenceEngine(
        [
            _make_gen(text="Con"),
            _healthy_tool_call_gen(),
        ]
    )
    routes = _patch_routes(monkeypatch, engine)
    req = _make_request(tools=_TOOLS)

    resp = _run(routes, req)

    assert engine.calls == 1, "kill switch must disable the retry"
    assert resp.choices[0].message.content == "Con"


def test_persistent_degenerate_returned_as_is(monkeypatch, caplog):
    engine = _SequenceEngine([_make_gen(text="Con"), _make_gen(text="Con")])
    routes = _patch_routes(monkeypatch, engine)
    req = _make_request(tools=_TOOLS)

    with caplog.at_level(logging.WARNING, logger="fusion_mlx.api.openai._common"):
        resp = _run(routes, req)

    assert engine.calls == 2, "exactly one retry, never more"
    assert resp.choices[0].message.content == "Con"
    persisted = [r for r in caplog.records if "persisted after retry" in r.getMessage()]
    assert persisted, "persistent degeneration must be logged"


def test_no_tools_no_retry(monkeypatch):
    engine = _SequenceEngine([_make_gen(text="Con")])
    routes = _patch_routes(monkeypatch, engine)
    req = _make_request()

    resp = _run(routes, req)

    assert engine.calls == 1, "non-agent requests must never be retried"
    assert resp.choices[0].message.content == "Con"
