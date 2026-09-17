# SPDX-License-Identifier: Apache-2.0
"""Tests for #0916: Anthropic-style thinking dict → enable_thinking mapping
on the OpenAI /v1/chat/completions route.

Regression: the OpenAI chat route called resolve_enable_thinking_default(ct_kwargs)
without client_thinking, hitting the legacy force-False path and silently
overriding an explicit {"thinking":{"type":"enabled"}} request. The thinking
dict was captured by extra="allow" but never mapped to enable_thinking.
"""

from __future__ import annotations

from types import SimpleNamespace

from fusion_mlx.api.openai_models import ChatCompletionRequest
from fusion_mlx.api.utils import (
    client_thinking_from_request,
    resolve_enable_thinking_default,
)


class TestThinkingDictMapping:
    def test_enabled_sets_enable_thinking_true(self):
        r = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "enabled", "budget_tokens": 256},
        )
        assert r.enable_thinking is True
        assert r.reasoning_max_tokens == 256

    def test_disabled_sets_enable_thinking_false(self):
        r = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "disabled"},
        )
        assert r.enable_thinking is False

    def test_no_thinking_leaves_enable_thinking_none(self):
        r = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert r.enable_thinking is None
        assert r.thinking is None

    def test_explicit_enable_thinking_not_overridden_by_thinking_dict(self):
        r = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            enable_thinking=False,
            thinking={"type": "enabled", "budget_tokens": 100},
        )
        assert r.enable_thinking is False

    def test_budget_only_set_when_reasoning_max_tokens_none(self):
        r = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "enabled", "budget_tokens": 512},
            reasoning_max_tokens=128,
        )
        assert r.reasoning_max_tokens == 128


class TestClientThinkingFromRequest:
    def test_true_returns_enabled(self):
        req = SimpleNamespace(enable_thinking=True)
        assert client_thinking_from_request(req) == "enabled"

    def test_false_returns_disabled(self):
        req = SimpleNamespace(enable_thinking=False)
        assert client_thinking_from_request(req) == "disabled"

    def test_none_returns_none(self):
        req = SimpleNamespace(enable_thinking=None)
        assert client_thinking_from_request(req) is None

    def test_missing_attr_returns_none(self):
        req = SimpleNamespace()
        assert client_thinking_from_request(req) is None


class TestResolverHonorsClientThinking:
    def test_enabled_overrides_legacy_force_false(self):
        ct_kwargs: dict = {}
        resolve_enable_thinking_default(ct_kwargs, client_thinking="enabled")
        assert ct_kwargs.get("enable_thinking") is True

    def test_disabled_overrides_legacy_force_false(self):
        ct_kwargs: dict = {}
        resolve_enable_thinking_default(ct_kwargs, client_thinking="disabled")
        assert ct_kwargs.get("enable_thinking") is False

    def test_none_falls_back_to_legacy_force_false(self):
        ct_kwargs: dict = {}
        resolve_enable_thinking_default(ct_kwargs, client_thinking=None)
        assert ct_kwargs.get("enable_thinking") is False

    def test_explicit_ct_kwargs_not_overridden(self):
        ct_kwargs = {"enable_thinking": True}
        resolve_enable_thinking_default(ct_kwargs, client_thinking="disabled")
        assert ct_kwargs.get("enable_thinking") is True
