# SPDX-License-Identifier: Apache-2.0
# #1763: /v1/completions forwards logprobs/top_logprobs to the chat lane
# (follow-up to #1756/#1759, which only fixed /v1/chat/completions) and
# converts the chat ChoiceLogProbs shape into the legacy completions
# 4-array shape (tokens/token_logprobs/top_logprobs/text_offset). echo is
# rejected visibly (400) because prompt-token scoring is not implemented
# engine-side; silently ignoring it would mis-score echo clients.

import pytest
from pydantic import ValidationError

from fusion_mlx.api.openai.completions import (
    _chat_dict_to_completion_dict,
    _legacy_logprobs_from_chat,
)
from fusion_mlx.api.openai_models import CompletionRequest


class TestCompletionRequestLogprobs:
    def test_logprobs_int_accepted(self):
        req = CompletionRequest(model="m", prompt="hi", logprobs=2)
        assert req.logprobs == 2

    def test_logprobs_bool_rejected(self):
        with pytest.raises(ValidationError) as ctx:
            CompletionRequest(model="m", prompt="hi", logprobs=True)
        assert "must be an integer" in str(ctx.value)

    def test_logprobs_range(self):
        with pytest.raises(ValidationError):
            CompletionRequest(model="m", prompt="hi", logprobs=6)

    def test_top_logprobs_range(self):
        with pytest.raises(ValidationError):
            CompletionRequest(model="m", prompt="hi", top_logprobs=21)
        assert (
            CompletionRequest(model="m", prompt="hi", top_logprobs=20).top_logprobs
            == 20
        )

    def test_echo_declared(self):
        assert CompletionRequest(model="m", prompt="hi", echo=True).echo is True


class TestLegacyLogprobsConversion:
    def test_chat_shape_to_four_array(self):
        chat_lp = {
            "content": [
                {
                    "token": "The",
                    "logprob": -1.0,
                    "top_logprobs": [
                        {"token": "The", "logprob": -1.0},
                        {"token": "A", "logprob": -2.0},
                    ],
                },
                {"token": " cap", "logprob": -0.5, "top_logprobs": []},
            ]
        }
        legacy, offset = _legacy_logprobs_from_chat(chat_lp, 0)
        assert legacy["tokens"] == ["The", " cap"]
        assert legacy["token_logprobs"] == [-1.0, -0.5]
        assert legacy["top_logprobs"] == [{"The": -1.0, "A": -2.0}, {}]
        assert legacy["text_offset"] == [0, 3]
        assert offset == 7

    def test_running_offset(self):
        chat_lp = {"content": [{"token": "x", "logprob": -0.1, "top_logprobs": []}]}
        legacy, offset = _legacy_logprobs_from_chat(chat_lp, 12)
        assert legacy["text_offset"] == [12]
        assert offset == 13

    def test_none_content(self):
        legacy, offset = _legacy_logprobs_from_chat(None, 0)
        assert legacy["tokens"] == []
        assert offset == 0


class TestChatDictRemap:
    def test_logprobs_converted_in_remap(self):
        chat = {
            "id": "c1",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "The cap"},
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {"token": "The", "logprob": -1.0, "top_logprobs": []},
                            {"token": " cap", "logprob": -0.5, "top_logprobs": []},
                        ]
                    },
                }
            ],
        }
        out = _chat_dict_to_completion_dict(chat, "m")
        lp = out["choices"][0]["logprobs"]
        assert lp["tokens"] == ["The", " cap"]
        assert lp["token_logprobs"] == [-1.0, -0.5]
        assert lp["text_offset"] == [0, 3]

    def test_null_logprobs_stays_null(self):
        chat = {
            "id": "c1",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "x"},
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
        }
        out = _chat_dict_to_completion_dict(chat, "m")
        assert out["choices"][0]["logprobs"] is None


class TestEchoRejection:
    @pytest.mark.asyncio
    async def test_echo_400(self):
        from fastapi import HTTPException

        from fusion_mlx.api.openai.completions import completions

        req = CompletionRequest(model="m", prompt="hi", echo=True)
        http_req = type("R", (), {})()
        with pytest.raises(HTTPException) as ctx:
            await completions(req, http_req)
        assert ctx.value.status_code == 400
        assert "echo is not supported" in str(ctx.value.detail)
