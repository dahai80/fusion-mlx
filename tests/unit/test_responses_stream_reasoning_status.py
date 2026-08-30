# SPDX-License-Identifier: Apache-2.0
"""Regression guard for streaming ``/v1/responses`` reasoning-item status
under budget exhaustion (#707).

Pre-fix, ``_stream_responses`` hardcoded the terminal reasoning-item
``status`` to ``"completed"`` regardless of ``finish_reason``. The
non-stream adapter (``build_responses_response``) flips reasoning-item
status to ``"incomplete"`` when ``finish_reason == "length"`` AND no
downstream output (message body or tool call) shipped — i.e. a
reasoning-only cutoff. The stream path silently diverged from that
contract (git blame: never shipped, c98892e5). This file pins the
cross-path parity restored by #707.

Also pins ``usage.output_tokens_details.reasoning_tokens`` emission on
the stream path: pre-fix the stream never reported reasoning token
breakdown even when the engine supplied ``completion_tokens_details``.
"""

import json
import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class _Tokenizer:
    chat_template = ""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


class _BaseEngine:
    pass


@dataclass
class _CompletionTokensDetails:
    reasoning_tokens: int = 0


@dataclass
class _GenerationOutput:
    text: str
    raw_text: str = ""
    tokens: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = "stop"
    new_text: str = ""
    finished: bool = True
    logprobs: Any = None
    channel: str | None = None
    tool_calls: list | None = None
    reasoning_text: str = ""
    completion_tokens_details: Any = None


class _Engine:
    preserve_native_tool_format = False

    def __init__(self, scenario: str = "reasoning_only_length"):
        self.tokenizer = _Tokenizer()
        self.scenario = scenario

    async def chat(self, messages, **kwargs):
        return _GenerationOutput(
            text="hello",
            prompt_tokens=3,
            completion_tokens=1,
            finish_reason="stop",
        )

    async def stream_chat(self, messages, **kwargs):
        if self.scenario == "reasoning_only_length":
            # Reasoning bytes, then cut by max_output_tokens BEFORE any
            # message body ships -> reasoning-only cutoff.
            yield _GenerationOutput(
                text="",
                new_text="Thinking hard about this",
                channel="reasoning",
                prompt_tokens=4,
                completion_tokens=1,
                finish_reason=None,
                completion_tokens_details=_CompletionTokensDetails(reasoning_tokens=7),
            )
            yield _GenerationOutput(
                text="",
                new_text=" but ran out",
                channel="reasoning",
                completion_tokens=2,
                finish_reason="length",
                completion_tokens_details=_CompletionTokensDetails(reasoning_tokens=7),
            )
        elif self.scenario == "message_then_length":
            # Reasoning THEN a message body ships, then cut by length.
            # Downstream output WAS seen -> reasoning stays completed.
            yield _GenerationOutput(
                text="",
                new_text="Reasoning here",
                channel="reasoning",
                prompt_tokens=4,
                completion_tokens=1,
                finish_reason=None,
                completion_tokens_details=_CompletionTokensDetails(reasoning_tokens=3),
            )
            yield _GenerationOutput(
                text="Answer",
                new_text="Answer",
                channel=None,
                completion_tokens=2,
                finish_reason="length",
                completion_tokens_details=_CompletionTokensDetails(reasoning_tokens=3),
            )
        elif self.scenario == "reasoning_only_stop":
            # Reasoning then stop normally -> completed, no cutoff.
            yield _GenerationOutput(
                text="",
                new_text="Quick thought",
                channel="reasoning",
                prompt_tokens=4,
                completion_tokens=1,
                finish_reason=None,
                completion_tokens_details=_CompletionTokensDetails(reasoning_tokens=2),
            )
            yield _GenerationOutput(
                text="",
                new_text="",
                channel=None,
                completion_tokens=1,
                finish_reason="stop",
                completion_tokens_details=_CompletionTokensDetails(reasoning_tokens=2),
            )


_IMPORTED = (
    "vllm_mlx.config",
    "vllm_mlx.config.server_config",
    "vllm_mlx.engine",
    "vllm_mlx.engine.base",
    "vllm_mlx.middleware.auth",
    "vllm_mlx.service.helpers",
    "vllm_mlx.routes.responses",
)
_PARENT_ATTRS = (
    ("vllm_mlx", "config"),
    ("vllm_mlx", "engine"),
    ("vllm_mlx.config", "server_config"),
    ("vllm_mlx.engine", "base"),
    ("vllm_mlx.middleware", "auth"),
    ("vllm_mlx.service", "helpers"),
    ("vllm_mlx.routes", "responses"),
)
_MISSING = object()


def _install_lightweight_engine_modules(monkeypatch):
    engine_pkg = types.ModuleType("vllm_mlx.engine")
    engine_pkg.BaseEngine = _BaseEngine
    engine_pkg.GenerationOutput = _GenerationOutput

    base_mod = types.ModuleType("vllm_mlx.engine.base")
    base_mod.BaseEngine = _BaseEngine
    base_mod.GenerationOutput = _GenerationOutput

    monkeypatch.setitem(sys.modules, "vllm_mlx.engine", engine_pkg)
    monkeypatch.setitem(sys.modules, "vllm_mlx.engine.base", base_mod)


@pytest.fixture
def responses_client(monkeypatch):
    previous_modules = {n: sys.modules.get(n, _MISSING) for n in _IMPORTED}
    previous_attrs = {}
    for module_name, attr in _PARENT_ATTRS:
        module = sys.modules.get(module_name)
        previous_attrs[(module_name, attr)] = (
            getattr(module, attr, _MISSING) if module is not None else _MISSING
        )

    _install_lightweight_engine_modules(monkeypatch)

    from fusion_mlx.config import reset_config
    from fusion_mlx.middleware.auth import rate_limiter
    from fusion_mlx.middleware.exception_handlers import install_exception_handlers
    from fusion_mlx.routes_internal.responses import router

    cfg = reset_config()
    cfg.api_key = "test-secret"
    cfg.model_name = "test-model"
    cfg.model_registry = None

    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    def _make(scenario: str):
        cfg.engine = _Engine(scenario=scenario)
        app = FastAPI()
        install_exception_handlers(app)
        app.include_router(router)
        return SimpleNamespace(client=TestClient(app), engine=cfg.engine, cfg=cfg)

    yield _make

    reset_config()
    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    for name, previous in previous_modules.items():
        if previous is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous

    for (module_name, attr), previous in previous_attrs.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if previous is _MISSING:
            if hasattr(module, attr):
                delattr(module, attr)
        else:
            setattr(module, attr, previous)


def _parse_sse(body_text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_name = None
        data_text = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_text = line[len("data:") :].strip()
        if event_name and data_text is not None:
            events.append((event_name, json.loads(data_text)))
    return events


HEADERS = {"Authorization": "Bearer test-secret"}


def _stream_payload(**overrides):
    base = {"model": "test-model", "input": "Hello, world", "stream": True}
    base.update(overrides)
    return base


def _collect_events(client, **payload_overrides):
    with client.stream(
        "POST",
        "/v1/responses",
        json=_stream_payload(**payload_overrides),
        headers=HEADERS,
    ) as resp:
        assert resp.status_code == 200
        body = b"".join(resp.iter_bytes()).decode()
    return _parse_sse(body)


def _terminal_response(events: list[tuple[str, dict]]) -> dict:
    for name, data in events:
        if name == "response.completed":
            return data["response"]
    raise AssertionError("no response.completed event emitted")


def _reasoning_done_item(events: list[tuple[str, dict]]) -> dict:
    for name, data in events:
        if name == "response.output_item.done":
            item = data.get("item", {})
            if item.get("type") == "reasoning":
                return item
    raise AssertionError("no reasoning output_item.done emitted")


# =============================================================================
# #707 — reasoning-item status parity between stream and non-stream paths
# =============================================================================


class TestResponsesStreamReasoningStatus:
    def test_reasoning_only_length_flip_to_incomplete(self, responses_client):
        """Reasoning-only cutoff: no message body, finish_reason=length.

        Mirrors the non-stream adapter contract
        (build_responses_response lines 321-328): when finish_reason is
        "length" AND no downstream output shipped, the reasoning item
        MUST flip to "incomplete". Pre-fix the stream hardcoded
        "completed" here.
        """
        ns = responses_client("reasoning_only_length")
        events = _collect_events(ns.client, max_output_tokens=2)

        reasoning_item = _reasoning_done_item(events)
        assert reasoning_item["status"] == "incomplete", (
            "stream reasoning item must flip to 'incomplete' on a "
            "reasoning-only budget exhaust (finish_reason=length, no "
            "message/tool shipped) — matching the non-stream adapter. "
            f"Got status={reasoning_item['status']!r}."
        )

        response = _terminal_response(events)
        assert response["status"] == "incomplete"
        assert response["incomplete_details"] == {"reason": "max_output_tokens"}

    def test_message_then_length_keeps_reasoning_completed(self, responses_client):
        """Message body DID ship under length: reasoning stays completed.

        downstream_output_seen == True once message content or tool calls
        ship, so the reasoning item is NOT the incomplete deliverable —
        the message is. Pre-fix and post-fix agree here; this pins that
        the fix did not over-flip.
        """
        ns = responses_client("message_then_length")
        events = _collect_events(ns.client, max_output_tokens=4)

        reasoning_item = _reasoning_done_item(events)
        assert reasoning_item["status"] == "completed", (
            "reasoning item must stay 'completed' when a message body "
            "shipped even under finish_reason=length (downstream output "
            "seen). Got status=" + repr(reasoning_item["status"]) + "."
        )

        response = _terminal_response(events)
        assert response["status"] == "incomplete"
        assert response["incomplete_details"] == {"reason": "max_output_tokens"}

    def test_reasoning_only_stop_stays_completed(self, responses_client):
        """Normal stop (no cutoff): reasoning is completed.

        Guards against the fix accidentally flipping on finish_reason
        other than "length".
        """
        ns = responses_client("reasoning_only_stop")
        events = _collect_events(ns.client)

        reasoning_item = _reasoning_done_item(events)
        assert reasoning_item["status"] == "completed"

        response = _terminal_response(events)
        assert response["status"] == "completed"
        assert response.get("incomplete_details") is None


# =============================================================================
# #707 — usage.output_tokens_details.reasoning_tokens emission on stream
# =============================================================================


class TestResponsesStreamReasoningTokens:
    def test_usage_reports_reasoning_tokens(self, responses_client):
        """Stream usage MUST echo engine-reported reasoning_tokens.

        Pre-fix the stream path never populated
        ``output_tokens_details`` on the terminal ResponsesUsage, so
        reasoning-token breakdown was silently dropped even when the
        engine supplied completion_tokens_details.reasoning_tokens.
        """
        ns = responses_client("reasoning_only_stop")
        events = _collect_events(ns.client)

        response = _terminal_response(events)
        usage = response["usage"]
        assert "output_tokens_details" in usage, (
            "stream usage must include output_tokens_details (#707). " f"usage={usage}"
        )
        assert usage["output_tokens_details"]["reasoning_tokens"] == 2, (
            "reasoning_tokens must be threaded from engine "
            "completion_tokens_details. "
            f"output_tokens_details={usage['output_tokens_details']}"
        )


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
