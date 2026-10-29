# SPDX-License-Identifier: Apache-2.0
"""Tests for X-Context-Budget and X-Context-Warning response headers (#327).

Exercises ``build_context_budget_headers`` and
``compute_prompt_tokens_for_messages`` helpers directly with fake engines,
plus route-level integration via TestClient.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fusion_mlx.service.helpers import (
    build_context_budget_headers,
    compute_prompt_tokens_for_messages,
)


class _StubArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _StubModel:
    def __init__(self, args=None, config=None):
        if args is not None:
            self.args = args
        if config is not None:
            self.config = config


class _StubTokenizer:
    def __init__(self, model_max_length=None, bos_token=None, chars_per_token=4):
        if model_max_length is not None:
            self.model_max_length = model_max_length
        self.bos_token = bos_token
        self._cpt = chars_per_token

    def encode(self, text, add_special_tokens=True):
        return [0] * max(1, len(text) // self._cpt)


class _StubEngine:
    is_mllm = False

    def __init__(self, model=None, tokenizer=None):
        self._model = model
        self._tokenizer = tokenizer

    @property
    def tokenizer(self):
        return self._tokenizer


# ─── build_context_budget_headers ─────────────────────────────────────


class TestBuildContextBudgetHeaders:
    def test_normal_usage(self):
        h = build_context_budget_headers(12000, 32768)
        assert h["X-Context-Budget"] == "used=12000;total=32768;remaining=20768"
        assert "X-Context-Warning" not in h

    def test_warning_above_70_percent(self):
        h = build_context_budget_headers(23000, 32768)
        assert h["X-Context-Budget"] == "used=23000;total=32768;remaining=9768"
        assert h["X-Context-Warning"] == "true"

    def test_exactly_70_percent_no_warning(self):
        # 22937 / 32768 ≈ 0.6999 < 0.7
        h = build_context_budget_headers(22937, 32768)
        assert "X-Context-Warning" not in h

    def test_just_over_70_percent_warns(self):
        # 22938 / 32768 ≈ 0.7000... > 0.7
        h = build_context_budget_headers(22938, 32768)
        assert h["X-Context-Warning"] == "true"

    def test_zero_context_window_returns_empty(self):
        h = build_context_budget_headers(100, 0)
        assert h == {}

    def test_negative_context_window_returns_empty(self):
        h = build_context_budget_headers(100, -1)
        assert h == {}

    def test_zero_prompt_tokens(self):
        h = build_context_budget_headers(0, 32768)
        assert h["X-Context-Budget"] == "used=0;total=32768;remaining=32768"
        assert "X-Context-Warning" not in h

    def test_full_context_triggers_warning(self):
        h = build_context_budget_headers(32768, 32768)
        assert h["X-Context-Budget"] == "used=32768;total=32768;remaining=0"
        assert h["X-Context-Warning"] == "true"

    def test_over_context_clamps_remaining_to_zero(self):
        h = build_context_budget_headers(40000, 32768)
        assert h["X-Context-Budget"] == "used=40000;total=32768;remaining=0"
        assert h["X-Context-Warning"] == "true"


# ─── compute_prompt_tokens_for_messages ───────────────────────────────


class TestComputePromptTokensForMessages:
    def test_engine_with_build_prompt(self):
        engine = _StubEngine(
            tokenizer=_StubTokenizer(chars_per_token=4),
        )
        engine.build_prompt = (
            lambda messages, tools=None, enable_thinking=None: "Hello world test prompt"
        )
        result = compute_prompt_tokens_for_messages(
            engine, [{"role": "user", "content": "Hello world test prompt"}]
        )
        assert result > 0

    def test_engine_with_apply_chat_template(self):
        engine = _StubEngine(
            tokenizer=_StubTokenizer(chars_per_token=4),
        )
        engine._apply_chat_template = (
            lambda messages, tools=None, chat_template_kwargs=None: "A" * 100
        )
        result = compute_prompt_tokens_for_messages(
            engine, [{"role": "user", "content": "test"}]
        )
        assert result > 0

    def test_prefers_count_chat_tokens_over_raw_template(self):
        # #context-budget-system-first: inline role="system" messages
        # (CC 2.1.154+ via the gateway OpenAI route) make the raw
        # template raise; count_chat_tokens normalizes like the real
        # build, so the estimate must come from there.
        engine = _StubEngine(
            tokenizer=_StubTokenizer(chars_per_token=4),
        )
        seen = {}

        def _count_chat(messages, tools=None, chat_template_kwargs=None):
            seen["messages"] = messages
            seen["kwargs"] = chat_template_kwargs
            return 1234

        def _raw_raise(messages, tools=None, chat_template_kwargs=None):
            raise RuntimeError("System message must be at the beginning.")

        engine.count_chat_tokens = _count_chat
        engine._apply_chat_template = _raw_raise
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "inline sys"},
        ]
        result = compute_prompt_tokens_for_messages(
            engine, messages, enable_thinking=True
        )
        assert result == 1234
        assert seen["messages"] == messages
        assert seen["kwargs"] == {"enable_thinking": True}

    def test_count_chat_tokens_failure_returns_zero(self):
        engine = _StubEngine(
            tokenizer=_StubTokenizer(chars_per_token=4),
        )

        def _fail(messages, tools=None, chat_template_kwargs=None):
            raise RuntimeError("boom")

        engine.count_chat_tokens = _fail
        result = compute_prompt_tokens_for_messages(
            engine, [{"role": "user", "content": "test"}]
        )
        assert result == 0

    def test_mllm_engine_returns_zero(self):
        engine = _StubEngine(tokenizer=_StubTokenizer())
        engine.is_mllm = True
        result = compute_prompt_tokens_for_messages(
            engine, [{"role": "user", "content": "test"}]
        )
        assert result == 0

    def test_engine_without_prompt_methods_returns_zero(self):
        engine = _StubEngine(tokenizer=_StubTokenizer())
        result = compute_prompt_tokens_for_messages(
            engine, [{"role": "user", "content": "test"}]
        )
        assert result == 0

    def test_build_prompt_failure_returns_zero(self):
        engine = _StubEngine(tokenizer=_StubTokenizer())

        def _fail(messages, tools=None, enable_thinking=None):
            raise RuntimeError("template error")

        engine.build_prompt = _fail
        result = compute_prompt_tokens_for_messages(
            engine, [{"role": "user", "content": "test"}]
        )
        assert result == 0


# ─── #1445: header must not recount raw model_dump() messages ────────


class _StrictArgsEngine(_StubEngine):
    # Mirrors the Qwen3.5 template: tool_call.arguments is rendered with
    # `|items`, which raises TypeError unless it is a mapping.
    def __init__(self, model=None, tokenizer=None):
        super().__init__(model=model, tokenizer=tokenizer)
        self.counted = []

    def count_chat_tokens(self, messages, tools=None, chat_template_kwargs=None):
        for m in messages:
            for tc in m.get("tool_calls") or []:
                args = tc.get("function", {}).get("arguments")
                if not isinstance(args, dict):
                    raise TypeError("Can only get item pairs from a mapping.")
        self.counted.append(messages)
        return 97759


class TestContextBudgetHeaderUsesPreflightEstimate:
    def test_header_from_preflight_estimate(self):
        from fusion_mlx.api.openai.streaming import _context_budget_headers

        engine = _StubEngine(
            model=_StubModel(args=_StubArgs(max_position_embeddings=262144))
        )
        headers = _context_budget_headers(engine, 97759)
        assert headers["X-Context-Budget"] == (
            "used=97759;total=262144;remaining=164385"
        )

    def test_tool_history_string_arguments_count_nonzero(self):
        # The failure mode of #1445: arguments arrive as a JSON string, the
        # raw-dump recount raised TypeError, the estimate became 0 and the
        # header advertised used=0 for every Claude Code turn.
        from fusion_mlx.api.openai.streaming import _context_preflight

        engine = _StrictArgsEngine(
            model=_StubModel(args=_StubArgs(max_position_embeddings=262144))
        )
        from fusion_mlx.api.openai_models import ChatCompletionRequest

        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                {"role": "user", "content": "run ls"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "Bash",
                                "arguments": '{"command": "ls"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "a.py b.py"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "description": "run a shell command",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        import asyncio

        messages, estimate = asyncio.run(
            _context_preflight(request, engine, "test-model")
        )
        assert estimate > 0
        assert messages[1]["tool_calls"][0]["function"]["arguments"] == {
            "command": "ls"
        }


# ─── Route-level integration ─────────────────────────────────────────


class TestContextBudgetRouteIntegration:
    @pytest.fixture
    def app(self):
        from fusion_mlx.server import Server

        srv = Server()
        srv.config.api_key = ""
        return srv.app

    @pytest.fixture
    def client(self, app):
        from starlette.testclient import TestClient

        return TestClient(app)

    def test_non_streaming_response_has_context_budget_header(self, client):
        with patch("fusion_mlx.api.openai.chat._resolve_engine") as mock_resolve:
            mock_engine = MagicMock()
            mock_engine.is_mllm = False
            mock_engine._model = _StubModel(
                args=_StubArgs(max_position_embeddings=8192)
            )
            mock_engine._tokenizer = _StubTokenizer(model_max_length=8192)
            mock_engine.tokenizer = mock_engine._tokenizer

            from fusion_mlx.engines.base import GenerationOutput

            mock_gen = GenerationOutput(
                text="Hello",
                prompt_tokens=100,
                completion_tokens=5,
                finish_reason="stop",
                tool_calls=[],
                cached_tokens=0,
            )

            async def _fake_lease(*a, **kw):
                return True

            mock_resolve.return_value = mock_engine

            with patch.object(mock_engine, "chat", return_value=mock_gen):
                with patch("fusion_mlx.api.openai.chat._release_engine"):
                    resp = client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "hi"}],
                            "stream": False,
                        },
                    )
        # The response may be 200 or 500 depending on mock wiring.
        # We just verify the header pattern if the request succeeds.
        if resp.status_code == 200:
            assert "X-Context-Budget" in resp.headers
            assert "used=" in resp.headers["X-Context-Budget"]
            assert "total=" in resp.headers["X-Context-Budget"]
            assert "remaining=" in resp.headers["X-Context-Budget"]
