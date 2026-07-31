# SPDX-License-Identifier: Apache-2.0
"""Tests for reasoning_routes — /v1/reasoning endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fusion_mlx.api.reasoning_routes import (
    EFFORT_TOKEN_MAP,
    ReasoningRequest,
    ReasoningResponse,
    ReasoningUsage,
    _resolve_engine,
    set_reasoning_context,
)


class TestEffortTokenMap:
    def test_low(self):
        assert EFFORT_TOKEN_MAP["low"] == 512

    def test_medium(self):
        assert EFFORT_TOKEN_MAP["medium"] == 2048

    def test_high(self):
        assert EFFORT_TOKEN_MAP["high"] == 8192


class TestReasoningRequest:
    def test_defaults(self):
        r = ReasoningRequest(model="test", prompt="hello")
        assert r.reasoning_effort == "medium"
        assert r.max_reasoning_tokens is None
        assert r.temperature == 0.6
        assert r.max_tokens == 4096
        assert r.stream is False

    def test_custom_effort(self):
        r = ReasoningRequest(model="test", prompt="hello", reasoning_effort="high")
        assert r.reasoning_effort == "high"

    def test_invalid_effort(self):
        with pytest.raises(Exception):
            ReasoningRequest(model="test", prompt="hello", reasoning_effort="extreme")


class TestResolveEngine:
    def test_no_pool_raises_503(self):
        set_reasoning_context(None)
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _resolve_engine("any")
        assert exc_info.value.status_code == 503

    def test_missing_model_raises_404(self):
        pool = MagicMock()
        pool.get.return_value = None
        set_reasoning_context(pool)
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _resolve_engine("missing")
        assert exc_info.value.status_code == 404

    def test_found_engine(self):
        engine = MagicMock()
        pool = MagicMock()
        pool.get.return_value = engine
        set_reasoning_context(pool)
        result = _resolve_engine("test-model")
        assert result is engine


class TestReasoningResponse:
    def test_structure(self):
        resp = ReasoningResponse(
            id="reason-abc",
            model="test",
            reasoning_content="thinking...",
            content="answer",
            usage=ReasoningUsage(
                prompt_tokens=10,
                completion_tokens=20,
                reasoning_tokens=5,
                total_tokens=30,
            ),
        )
        assert resp.object == "reasoning_result"
        assert resp.reasoning_content == "thinking..."
        assert resp.content == "answer"
        assert resp.usage.reasoning_tokens == 5
