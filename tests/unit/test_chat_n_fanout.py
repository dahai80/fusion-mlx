# SPDX-License-Identifier: Apache-2.0
"""n>1 chat fan-out: merge n independent samplings into one response."""

import asyncio
import json

import pytest
from starlette.responses import JSONResponse

from fusion_mlx.api.openai.chat import _fan_out_chat_choices
from fusion_mlx.api.openai_models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
)


def _resp(token_count: int, content: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model="m",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=token_count,
            total_tokens=10 + token_count,
        ),
    )


def _body(out) -> dict:
    assert isinstance(out, JSONResponse)
    return json.loads(out.body)


class TestFanOut:
    def test_merges_choices_with_reindexed_index(self, monkeypatch):
        async def fake_run(request, **kwargs):
            return _resp(20, "sample")

        monkeypatch.setattr("fusion_mlx.api.openai.chat._run_chat", fake_run)
        req = ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}], n=4
        )
        out = asyncio.run(_fan_out_chat_choices(req, principal=None, headers={}, n=4))
        body = _body(out)
        assert len(body["choices"]) == 4
        assert [c["index"] for c in body["choices"]] == [0, 1, 2, 3]

    def test_usage_sums_and_timing_cleared(self, monkeypatch):
        async def fake_run(request, **kwargs):
            r = _resp(30, "x")
            r.usage.time_to_first_token = 0.5
            r.usage.generation_tokens_per_second = 99.0
            return r

        monkeypatch.setattr("fusion_mlx.api.openai.chat._run_chat", fake_run)
        req = ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}], n=3
        )
        out = asyncio.run(_fan_out_chat_choices(req, principal=None, headers={}, n=3))
        u = _body(out)["usage"]
        assert u["prompt_tokens"] == 30
        assert u["completion_tokens"] == 90
        assert u["total_tokens"] == 120
        assert u["time_to_first_token"] is None
        assert u["generation_tokens_per_second"] is None

    def test_children_get_n1(self, monkeypatch):
        seen = []

        async def fake_run(request, **kwargs):
            seen.append(request.n)
            return _resp(5, "x")

        monkeypatch.setattr("fusion_mlx.api.openai.chat._run_chat", fake_run)
        req = ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}], n=2
        )
        asyncio.run(_fan_out_chat_choices(req, principal=None, headers={}, n=2))
        assert seen == [1, 1]

    def test_failed_child_propagates_error(self, monkeypatch):
        async def fake_run(request, **kwargs):
            if seen_run[0]:
                return JSONResponse(
                    content={"error": {"message": "boom"}}, status_code=500
                )
            seen_run[0] = True
            return _resp(5, "x")

        seen_run = [False]
        monkeypatch.setattr("fusion_mlx.api.openai.chat._run_chat", fake_run)
        req = ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}], n=2
        )
        with pytest.raises(Exception):
            asyncio.run(_fan_out_chat_choices(req, principal=None, headers={}, n=2))

    def test_n_field_validated(self):
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="m", messages=[{"role": "user", "content": "hi"}], n=0
            )
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="m", messages=[{"role": "user", "content": "hi"}], n=33
            )
