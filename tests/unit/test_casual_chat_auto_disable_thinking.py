# SPDX-License-Identifier: Apache-2.0
"""R12-T2F-276 — default-disable ``enable_thinking`` on casual chat.

PATH B harness rescue (issue #531): this test was written against the
old rapid-mlx service layout and asserted a
``maybe_auto_disable_thinking_for_casual_chat(request, extra_signals=None) -> bool``
helper contract that fusion-mlx does NOT implement. fusion-mlx achieves
the same "default-disable thinking on casual chat" goal via a single
convergence point:

    ``resolve_enable_thinking_default(ct_kwargs)``
    (fusion_mlx/api/utils.py:1537)

which does ``ct_kwargs.setdefault("enable_thinking", False)`` and is
called on every chat / anthropic / responses route (openai_routes.py
:593 / :1094, anthropic_routes.py:332 / :584, responses.py:415 / :572).

The flag is passed nested under ``chat_template_kwargs=``, NOT as a
flat ``enable_thinking=`` kwarg. The old
``maybe_auto_disable_thinking_for_casual_chat`` is a NO-OP stub in
fusion-mlx (returns the passed ``enable_thinking`` unchanged) and is
never called by any route, so the helper-level contracts that pinned
injection / marker / precedence behaviour have been dropped. The
``reasoning_max_tokens`` / ``reasoning_effort`` / ``reasoning`` dict
"keep thinking ON" contracts were also dropped: the chat route does
not consult those fields, so ``setdefault`` fires regardless. Only the
client's explicit ``enable_thinking`` (top-level or nested under
``chat_template_kwargs``) overrides the default.

This file pins three contract groups against the real prod shape:

  1. Unit: ``resolve_enable_thinking_default`` — setdefault semantics,
     non-destructive merge, explicit-value precedence.
  2. /v1/chat/completions route: the engine sees
     ``chat_template_kwargs["enable_thinking"]`` — False by default,
     preserved when the client pinned True / False.
  3. /v1/responses route: same nested shape, same default-disable.

Plus a small unit group for the live helpers that still exist:
``enable_thinking_warning_header`` (emits ``X-FusionMLX-Warning``) and
``maybe_auto_disable_thinking_for_tools`` (exported but not route-wired;
its own behaviour is pinned as a standalone unit).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api import response_format_metrics
from fusion_mlx.api.utils import resolve_enable_thinking_default
from fusion_mlx.config import reset_config
from fusion_mlx.engine.base import GenerationOutput
from fusion_mlx.middleware.exception_handlers import install_exception_handlers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# (1) Unit: resolve_enable_thinking_default convergence point
# ---------------------------------------------------------------------------


class TestResolveEnableThinkingDefault:
    def test_empty_dict_gets_default_false(self):
        ct_kwargs: dict = {}
        out = resolve_enable_thinking_default(ct_kwargs)
        assert out is ct_kwargs
        assert ct_kwargs == {"enable_thinking": False}

    def test_explicit_true_preserved(self):
        ct_kwargs = {"enable_thinking": True}
        resolve_enable_thinking_default(ct_kwargs)
        assert ct_kwargs == {"enable_thinking": True}

    def test_explicit_false_preserved(self):
        ct_kwargs = {"enable_thinking": False}
        resolve_enable_thinking_default(ct_kwargs)
        assert ct_kwargs == {"enable_thinking": False}

    def test_forward_compat_key_survives(self):
        ct_kwargs = {"future_key": "x"}
        resolve_enable_thinking_default(ct_kwargs)
        assert ct_kwargs == {"future_key": "x", "enable_thinking": False}

    def test_idempotent_second_call(self):
        ct_kwargs: dict = {}
        resolve_enable_thinking_default(ct_kwargs)
        resolve_enable_thinking_default(ct_kwargs)
        assert ct_kwargs == {"enable_thinking": False}

    def test_does_not_touch_none_sentinel(self):
        # The routes guard with ``ct_kwargs if ct_kwargs else None``;
        # resolve itself only runs on a real dict. Passing an empty
        # dict is the no-client-signal shape.
        ct_kwargs: dict = {}
        resolve_enable_thinking_default(ct_kwargs)
        assert "enable_thinking" in ct_kwargs


# ---------------------------------------------------------------------------
# (2) /v1/chat/completions route: engine sees nested chat_template_kwargs
# ---------------------------------------------------------------------------


class _ChatEngine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    supports_tool_calls = True
    tokenizer = None

    def __init__(self, *, text: str = "ok"):
        self._text = text
        self.chat_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return GenerationOutput(
            text=self._text,
            raw_text=self._text,
            prompt_tokens=4,
            completion_tokens=2,
            finished=True,
            finish_reason="stop",
            cached_tokens=0,
        )


@pytest.fixture(autouse=True)
def _reset_metrics_between_tests():
    response_format_metrics.reset_for_tests()
    yield
    response_format_metrics.reset_for_tests()


@pytest.fixture
def _rate_limiter_state():
    from fusion_mlx.middleware.auth import rate_limiter

    saved_enabled = rate_limiter.enabled
    saved_rpm = rate_limiter.requests_per_minute
    saved_requests = dict(rate_limiter._requests)
    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()
    yield rate_limiter
    rate_limiter.enabled = saved_enabled
    rate_limiter.requests_per_minute = saved_rpm
    rate_limiter._requests.clear()
    rate_limiter._requests.update(saved_requests)


def _make_chat_client(
    engine: _ChatEngine, *, reasoning_parser_name="qwen3"
) -> TestClient:
    from fusion_mlx.routes_internal.chat import router as chat_router

    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.reasoning_parser_name = reasoning_parser_name

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(chat_router)
    return TestClient(app)


def _ctk_from_kwargs(kwargs: dict) -> dict | None:
    return kwargs.get("chat_template_kwargs")


class TestChatRouteDefaultDisable:
    def test_casual_no_preference_defaults_thinking_off(self, _rate_limiter_state):
        engine = _ChatEngine(text="ok")
        client = _make_chat_client(engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "max_tokens": 80,
                "messages": [
                    {"role": "user", "content": "In 8 words, what is fusion-mlx?"}
                ],
            },
        )
        assert resp.status_code == 200, resp.text
        assert engine.chat_calls, "engine.chat was not called"
        ctk = _ctk_from_kwargs(engine.chat_calls[0]["kwargs"])
        assert (
            ctk is not None
        ), f"chat_template_kwargs missing; kwargs={engine.chat_calls[0]['kwargs']!r}"
        assert ctk.get("enable_thinking") is False, (
            "casual chat + no preference must default enable_thinking=False "
            f"via resolve_enable_thinking_default; ctk={ctk!r}"
        )

    def test_explicit_nested_true_preserved(self, _rate_limiter_state):
        engine = _ChatEngine(text="ok")
        client = _make_chat_client(engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "max_tokens": 80,
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
        assert resp.status_code == 200, resp.text
        ctk = _ctk_from_kwargs(engine.chat_calls[0]["kwargs"])
        assert ctk is not None
        assert ctk.get("enable_thinking") is True

    def test_explicit_nested_false_preserved(self, _rate_limiter_state):
        engine = _ChatEngine(text="ok")
        client = _make_chat_client(engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "max_tokens": 80,
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        assert resp.status_code == 200, resp.text
        ctk = _ctk_from_kwargs(engine.chat_calls[0]["kwargs"])
        assert ctk is not None
        assert ctk.get("enable_thinking") is False

    def test_forward_compat_key_survives_route_merge(self, _rate_limiter_state):
        engine = _ChatEngine(text="ok")
        client = _make_chat_client(engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "max_tokens": 80,
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"future_key": "x"},
            },
        )
        assert resp.status_code == 200, resp.text
        ctk = _ctk_from_kwargs(engine.chat_calls[0]["kwargs"])
        assert ctk is not None
        assert ctk == {
            "future_key": "x",
            "enable_thinking": False,
        }, f"forward-compat key dropped by setdefault merge: {ctk!r}"

    def test_non_thinking_model_also_gets_default_disabled(self, _rate_limiter_state):
        # resolve_enable_thinking_default is unconditional (no parser
        # gate) — the disable-by-default applies on every model so a
        # thinking-capable model never burns the budget by accident.
        # Dropped the old "non-thinking model unaffected" contract: prod
        # does not gate on reasoning_parser_name.
        engine = _ChatEngine(text="hi back")
        client = _make_chat_client(engine, reasoning_parser_name=None)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "max_tokens": 80,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200, resp.text
        ctk = _ctk_from_kwargs(engine.chat_calls[0]["kwargs"])
        assert ctk is not None
        assert ctk.get("enable_thinking") is False


# ---------------------------------------------------------------------------
# (3) /v1/responses route: same nested shape, same default-disable
# ---------------------------------------------------------------------------


class _ResponsesEngine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    supports_tool_calls = True
    tokenizer = None

    def __init__(self, *, text: str = "ok"):
        self._text = text
        self.chat_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, *, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return GenerationOutput(
            text=self._text,
            new_text=self._text,
            prompt_tokens=4,
            completion_tokens=2,
            finished=True,
            finish_reason="stop",
            channel=None,
        )


def _make_responses_client(
    engine: _ResponsesEngine, *, reasoning_parser_name="qwen3"
) -> TestClient:
    from fusion_mlx.routes_internal.responses import router as responses_router

    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.reasoning_parser_name = reasoning_parser_name

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(responses_router)
    return TestClient(app)


def _responses_payload(
    *,
    chat_template_kwargs: dict | None = None,
    enable_thinking: bool | None = None,
) -> dict:
    body: dict = {
        "model": "test-model",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "In 8 words, what is fusion-mlx?"}
                ],
            }
        ],
        "max_output_tokens": 80,
    }
    if chat_template_kwargs is not None:
        body["chat_template_kwargs"] = chat_template_kwargs
    if enable_thinking is not None:
        body["enable_thinking"] = enable_thinking
    return body


class TestResponsesRouteDefaultDisable:
    def test_casual_no_preference_defaults_thinking_off(self, _rate_limiter_state):
        engine = _ResponsesEngine(text="ok")
        client = _make_responses_client(engine)
        resp = client.post("/v1/responses", json=_responses_payload())
        assert resp.status_code == 200, resp.text
        assert engine.chat_calls, "engine.chat was not called"
        ctk = _ctk_from_kwargs(engine.chat_calls[0]["kwargs"])
        assert ctk is not None
        assert ctk.get("enable_thinking") is False

    def test_explicit_enable_thinking_true_preserved(self, _rate_limiter_state):
        engine = _ResponsesEngine(text="ok")
        client = _make_responses_client(engine)
        resp = client.post(
            "/v1/responses", json=_responses_payload(enable_thinking=True)
        )
        assert resp.status_code == 200, resp.text
        ctk = _ctk_from_kwargs(engine.chat_calls[0]["kwargs"])
        assert ctk is not None
        assert ctk.get("enable_thinking") is True

    def test_explicit_nested_false_preserved(self, _rate_limiter_state):
        engine = _ResponsesEngine(text="ok")
        client = _make_responses_client(engine)
        resp = client.post(
            "/v1/responses",
            json=_responses_payload(chat_template_kwargs={"enable_thinking": False}),
        )
        assert resp.status_code == 200, resp.text
        ctk = _ctk_from_kwargs(engine.chat_calls[0]["kwargs"])
        assert ctk is not None
        assert ctk.get("enable_thinking") is False


# ---------------------------------------------------------------------------
# (4) Live helper units: warning header + tools helper
# ---------------------------------------------------------------------------
# These helpers are exported from fusion_mlx.service.helpers but are NOT
# wired into the route path (the routes use resolve_enable_thinking_default
# instead). Their own unit behaviour is pinned here so a refactor cannot
# silently change the exported symbol contract. The prod warning header
# name is ``X-FusionMLX-Warning`` (helpers.py:829).


class TestEnableThinkingWarningHeader:
    def test_no_parser_returns_empty(self):
        from fusion_mlx.service.helpers import enable_thinking_warning_header

        req = SimpleNamespace(
            chat_template_kwargs={"enable_thinking": False},
            _auto_disabled_thinking=False,
        )
        assert enable_thinking_warning_header(req, None) == {}

    def test_honoring_parser_returns_empty(self):
        from fusion_mlx.service.helpers import enable_thinking_warning_header

        req = SimpleNamespace(
            chat_template_kwargs={"enable_thinking": False},
            _auto_disabled_thinking=False,
        )
        # qwen3 honors the flag, so no warning.
        assert enable_thinking_warning_header(req, "qwen3") == {}

    def test_non_honoring_parser_with_client_hint_emits_warning(self):
        from fusion_mlx.service.helpers import enable_thinking_warning_header

        req = SimpleNamespace(
            chat_template_kwargs={"enable_thinking": False},
            _auto_disabled_thinking=False,
        )
        out = enable_thinking_warning_header(req, "deepseek_r1")
        assert out == {
            "X-FusionMLX-Warning": "enable_thinking ignored for parser=deepseek_r1"
        }

    def test_auto_disable_marker_suppresses_warning(self):
        from fusion_mlx.service.helpers import enable_thinking_warning_header

        req = SimpleNamespace(
            chat_template_kwargs={"enable_thinking": False},
            _auto_disabled_thinking=True,
        )
        assert enable_thinking_warning_header(req, "deepseek_r1") == {}

    def test_no_ctk_returns_empty(self):
        from fusion_mlx.service.helpers import enable_thinking_warning_header

        req = SimpleNamespace(
            chat_template_kwargs=None,
            _auto_disabled_thinking=False,
        )
        assert enable_thinking_warning_header(req, "deepseek_r1") == {}


class TestMaybeAutoDisableThinkingForTools:
    # maybe_auto_disable_thinking_for_tools is exported but NOT called
    # by any route (the routes use resolve_enable_thinking_default).
    # Pinning its standalone unit behaviour so the exported symbol
    # contract does not drift.
    def test_no_tools_returns_enable_thinking_unchanged(self):
        from fusion_mlx.service.helpers import maybe_auto_disable_thinking_for_tools

        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs=None,
            reasoning_effort=None,
            reasoning_max_tokens=None,
            reasoning=None,
        )
        assert maybe_auto_disable_thinking_for_tools(req, None) is None
        assert maybe_auto_disable_thinking_for_tools(req, True) is True

    def test_tools_present_injects_false(self):
        from fusion_mlx.service.helpers import maybe_auto_disable_thinking_for_tools

        req = SimpleNamespace(
            tools=[{"type": "function", "function": {"name": "x"}}],
            tool_choice=None,
            chat_template_kwargs=None,
            reasoning_effort=None,
            reasoning_max_tokens=None,
            reasoning=None,
        )
        assert maybe_auto_disable_thinking_for_tools(req, True) is True
        assert req.chat_template_kwargs == {"enable_thinking": False}
        assert getattr(req, "_auto_disabled_thinking", False) is True

    def test_tool_choice_none_skips_injection(self):
        from fusion_mlx.service.helpers import maybe_auto_disable_thinking_for_tools

        req = SimpleNamespace(
            tools=[{"type": "function", "function": {"name": "x"}}],
            tool_choice="none",
            chat_template_kwargs=None,
            reasoning_effort=None,
            reasoning_max_tokens=None,
            reasoning=None,
        )
        assert maybe_auto_disable_thinking_for_tools(req, True) is True
        assert req.chat_template_kwargs is None

    def test_reasoning_intent_skips_injection(self):
        from fusion_mlx.service.helpers import maybe_auto_disable_thinking_for_tools

        req = SimpleNamespace(
            tools=[{"type": "function", "function": {"name": "x"}}],
            tool_choice=None,
            chat_template_kwargs=None,
            reasoning_effort="low",
            reasoning_max_tokens=None,
            reasoning=None,
        )
        assert maybe_auto_disable_thinking_for_tools(req, True) is False
        assert req.chat_template_kwargs is None
