# SPDX-License-Identifier: Apache-2.0
"""Tests for #241: prompt caching support on /v1/chat/completions."""

import pytest
from unittest.mock import MagicMock

from fusion_mlx.api.openai_routes import _detect_prefix_cache_boundary
from fusion_mlx.api.adapters.openai import OpenAIAdapter
from fusion_mlx.api.adapters.base import InternalResponse
from fusion_mlx.api.openai_models import ChatCompletionRequest, Message


class TestDetectPrefixCacheBoundary:
    def test_no_cache_control_returns_none(self):
        msgs = [
            MagicMock(role="system", content="You are helpful.", cache_control=None),
            MagicMock(role="user", content="Hi"),
        ]
        assert _detect_prefix_cache_boundary(msgs) is None

    def test_system_string_with_cache_control(self):
        msgs = [
            MagicMock(
                role="system",
                content="A" * 40,
                cache_control={"type": "ephemeral"},
            ),
            MagicMock(role="user", content="Hi", cache_control=None),
        ]
        result = _detect_prefix_cache_boundary(msgs)
        assert result == 10  # 40 chars / 4

    def test_system_list_with_cache_control_on_part(self):
        part_no_cc = MagicMock(type="text", text="Hello ", cache_control=None)
        part_cc = MagicMock(type="text", text="world!", cache_control={"type": "ephemeral"})
        msgs = [
            MagicMock(role="system", content=[part_no_cc, part_cc], cache_control=None),
            MagicMock(role="user", content="Hi", cache_control=None),
        ]
        result = _detect_prefix_cache_boundary(msgs)
        assert result == 3  # 12 chars / 4

    def test_dict_content_with_cache_control(self):
        msgs = [
            MagicMock(
                role="system",
                content=[
                    {"type": "text", "text": "System prompt", "cache_control": {"type": "ephemeral"}},
                ],
                cache_control=None,
            ),
            MagicMock(role="user", content="Hi", cache_control=None),
        ]
        result = _detect_prefix_cache_boundary(msgs)
        assert result == 3  # 13 chars / 4

    def test_non_system_message_stops_scan(self):
        msgs = [
            MagicMock(role="user", content="Hi", cache_control={"type": "ephemeral"}),
        ]
        assert _detect_prefix_cache_boundary(msgs) is None

    def test_empty_messages_returns_none(self):
        assert _detect_prefix_cache_boundary([]) is None

    def test_multiple_system_second_has_cc(self):
        msgs = [
            MagicMock(role="system", content="First", cache_control=None),
            MagicMock(
                role="system",
                content="Second",
                cache_control={"type": "ephemeral"},
            ),
            MagicMock(role="user", content="Hi", cache_control=None),
        ]
        result = _detect_prefix_cache_boundary(msgs)
        # Only "Second" (6) counted since first had no cc
        assert result == 1

    def test_min_boundary_is_1(self):
        msgs = [
            MagicMock(role="system", content="A", cache_control={"type": "ephemeral"}),
            MagicMock(role="user", content="Hi", cache_control=None),
        ]
        result = _detect_prefix_cache_boundary(msgs)
        assert result == 1


class TestAdapterCachedTokens:
    """Verify PromptTokensDetails.cached_tokens appears in adapter response."""

    def test_format_response_includes_cached_tokens(self):
        adapter = OpenAIAdapter()
        internal = InternalResponse(
            text="Hello",
            finish_reason="stop",
            prompt_tokens=100,
            completion_tokens=5,
            cached_tokens=80,
            request_id="chatcmpl-test",
            model="test-model",
        )
        req = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
        )
        resp = adapter.format_response(internal, req)
        assert resp.usage.prompt_tokens_details is not None
        assert resp.usage.prompt_tokens_details.cached_tokens == 80

    def test_format_response_zero_cached_tokens(self):
        adapter = OpenAIAdapter()
        internal = InternalResponse(
            text="Hello",
            finish_reason="stop",
            prompt_tokens=100,
            completion_tokens=5,
            cached_tokens=0,
            request_id="chatcmpl-test",
            model="test-model",
        )
        req = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
        )
        resp = adapter.format_response(internal, req)
        assert resp.usage.prompt_tokens_details is not None
        assert resp.usage.prompt_tokens_details.cached_tokens == 0


class TestCacheControlField:
    """Verify cache_control field is accepted on ContentPart and Message."""

    def test_content_part_accepts_cache_control(self):
        from fusion_mlx.api.openai_models import ContentPart

        part = ContentPart(type="text", text="system prompt", cache_control={"type": "ephemeral"})
        assert part.cache_control == {"type": "ephemeral"}

    def test_message_accepts_cache_control(self):
        msg = Message(role="system", content="prompt", cache_control={"type": "ephemeral"})
        assert msg.cache_control == {"type": "ephemeral"}

    def test_content_part_default_none(self):
        from fusion_mlx.api.openai_models import ContentPart

        part = ContentPart(type="text", text="hello")
        assert part.cache_control is None

    def test_message_default_none(self):
        msg = Message(role="user", content="hi")
        assert msg.cache_control is None
