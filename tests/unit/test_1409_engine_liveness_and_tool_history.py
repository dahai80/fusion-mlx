# SPDX-License-Identifier: Apache-2.0
# #1409: two defects behind the Claude Code "death loop".
#   1. EnginePool's get_engine fast path only consulted is_dead, so an engine
#      whose teardown was cancelled after stop() stayed attached and was
#      handed out forever (every request 500'd "engine is not running").
#   2. _messages_for_engine dropped assistant tool_calls, so the model saw a
#      transcript with tool results but no record of its own actions and
#      re-issued the same tool call indefinitely.

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from fusion_mlx.api.openai._common import (
    _messages_for_engine,
    _tool_calls_for_engine,
)
from fusion_mlx.engines.vlm import VLMBatchedEngine
from fusion_mlx.pool.engine_pool import EngineEntry, EnginePool


class _FakeEngine:
    def __init__(self, running: bool = True):
        self.running = running
        self.stop_calls = 0

    def is_running(self) -> bool:
        return self.running

    def is_dead(self) -> bool:
        return False

    async def stop(self) -> None:
        self.stop_calls += 1
        self.running = False


class _RaisingEngine:
    def is_running(self) -> bool:
        raise RuntimeError("probe exploded")


def _make_pool() -> EnginePool:
    pool = EnginePool()
    pool._get_final_ceiling = lambda: 0
    return pool


def _register(pool: EnginePool, model_id: str, engine) -> EngineEntry:
    entry = EngineEntry(
        model_id=model_id,
        model_path=f"/nonexistent/{model_id}",
        engine_type="batched",
        model_type="llm",
        estimated_size=1024,
    )
    entry.engine = engine
    pool._entries[model_id] = entry
    return entry


def test_engine_is_running_true_without_probe():
    assert EnginePool._engine_is_running(object()) is True


def test_engine_is_running_false_when_probe_says_stopped():
    assert EnginePool._engine_is_running(_FakeEngine(running=False)) is False


def test_engine_is_running_true_when_probe_raises():
    assert EnginePool._engine_is_running(_RaisingEngine()) is True


@pytest.mark.asyncio
async def test_get_engine_serves_running_engine_without_reload():
    pool = _make_pool()
    engine = _FakeEngine(running=True)
    _register(pool, "live-model", engine)
    reload_calls = []
    pool._load_engine = AsyncMock(side_effect=lambda *a, **k: reload_calls.append(a))

    got = await pool.get_engine("live-model")

    assert got is engine
    assert reload_calls == []
    assert engine.stop_calls == 0


@pytest.mark.asyncio
async def test_get_engine_reloads_engine_that_is_attached_but_stopped():
    pool = _make_pool()
    stale = _FakeEngine(running=False)
    entry = _register(pool, "stopped-model", stale)
    fresh = _FakeEngine(running=True)

    async def _fake_load(entry_key, force_lm=False, runtime_settings=None):
        pool._entries[entry_key].engine = fresh

    pool._load_engine = AsyncMock(side_effect=_fake_load)
    pool._settle_unloaded_engine = AsyncMock(return_value=None)

    got = await pool.get_engine("stopped-model")

    assert got is fresh
    assert pool._load_engine.await_count == 1
    assert entry.engine is fresh


@pytest.mark.asyncio
async def test_get_engine_still_serves_engine_without_liveness_surface():
    pool = _make_pool()
    engine = MagicMock(spec=["stop", "safe_evict", "has_active_requests"])
    engine.has_active_requests = MagicMock(return_value=False)
    _register(pool, "media-model", engine)
    pool._load_engine = AsyncMock()

    got = await pool.get_engine("media-model")

    assert got is engine
    pool._load_engine.assert_not_awaited()


def _vlm_with_inner(inner):
    engine = object.__new__(VLMBatchedEngine)
    engine._engine = inner
    return engine


def test_vlm_engine_reports_not_running_when_hollow():
    assert _vlm_with_inner(None).is_running() is False


def test_vlm_engine_reports_inner_liveness():
    inner = MagicMock()
    inner.engine = MagicMock()
    inner.engine.is_running = MagicMock(return_value=False)
    assert _vlm_with_inner(inner).is_running() is False
    inner.engine.is_running = MagicMock(return_value=True)
    assert _vlm_with_inner(inner).is_running() is True


def test_vlm_engine_is_dead_mirrors_inner_core():
    inner = MagicMock()
    inner.engine = MagicMock()
    inner.engine.is_dead = MagicMock(return_value=True)
    inner.engine.dead_reason = "circuit breaker"
    engine = _vlm_with_inner(inner)
    assert engine.is_dead() is True
    assert engine.dead_reason == "circuit breaker"


def _msg(**kwargs):
    msg = MagicMock()
    for key, value in kwargs.items():
        setattr(msg, key, value)
    return msg


def test_tool_calls_arguments_are_parsed_into_a_mapping():
    msg = _msg(role="assistant", content="", tool_calls=None)
    msg.tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "Bash", "arguments": '{"command": "git status"}'},
        }
    ]
    calls = _tool_calls_for_engine(msg)
    assert calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "Bash", "arguments": {"command": "git status"}},
        }
    ]


def test_tool_calls_unparseable_arguments_stay_a_string():
    msg = _msg()
    msg.tool_calls = [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "Read", "arguments": "not json"},
        }
    ]
    calls = _tool_calls_for_engine(msg)
    assert calls[0]["function"]["arguments"] == "not json"


def test_messages_for_engine_keeps_tool_history():
    assistant = _msg(role="assistant", content="", tool_call_id=None, name=None)
    assistant.tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "Bash", "arguments": json.dumps({"command": "ls"})},
        }
    ]
    tool_result = _msg(
        role="tool",
        content="a.py\nb.py",
        tool_call_id="call_1",
        name="Bash",
    )
    tool_result.tool_calls = None
    user = _msg(role="user", content="next step", tool_call_id=None, name=None)
    user.tool_calls = None

    out = _messages_for_engine([assistant, tool_result, user], is_mllm=False)

    assert out[0]["role"] == "assistant"
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
    assert out[1]["tool_call_id"] == "call_1"
    assert out[1]["name"] == "Bash"
    assert "tool_calls" not in out[1]
    assert out[2] == {"role": "user", "content": "next step"}


def test_messages_for_engine_keeps_tool_history_for_multimodal():
    assistant = _msg(role="assistant", content=None, tool_call_id=None, name=None)
    assistant.tool_calls = [
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "Read", "arguments": {"path": "a.py"}},
        }
    ]
    media = _msg(role="user", content=[{"type": "text", "text": "see"}])
    media.tool_calls = None
    media.tool_call_id = None
    media.name = None

    out = _messages_for_engine([assistant, media], is_mllm=True)

    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"path": "a.py"}
    assert out[1]["content"] == [{"type": "text", "text": "see"}]
