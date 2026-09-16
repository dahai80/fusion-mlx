# SPDX-License-Identifier: Apache-2.0
"""Tests for #0916: /v1/completions content-length fix.

Regression: _to_completion_result copied stale content-length from the
chat-shaped JSONResponse when remapping to the completion shape. The
remapped body is a different length, so the stale content-length caused
"Too little data for declared Content-Length" (peer closed connection).
"""

from __future__ import annotations

import json

from starlette.responses import JSONResponse

from fusion_mlx.api.openai.completions import _to_completion_result


def _make_chat_jsonresponse() -> JSONResponse:
    chat_body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello world"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    return JSONResponse(
        content=chat_body,
        headers={
            "content-length": str(len(json.dumps(chat_body))),
            "x-context-budget": "12345",
        },
    )


class TestToCompletionResultContentLength:
    def test_strips_content_length_from_jsonresponse(self):
        src = _make_chat_jsonresponse()
        stale_cl = src.headers.get("content-length") or src.headers.get(
            "Content-Length"
        )
        result = _to_completion_result(src, "test-model")
        assert isinstance(result, JSONResponse)
        # Starlette recomputes content-length for the new body. Verify the
        # result's content-length matches the NEW body, not the stale original.
        new_cl = result.headers.get("content-length") or result.headers.get(
            "Content-Length"
        )
        assert new_cl is not None
        assert new_cl == str(len(result.body))
        assert new_cl != stale_cl

    def test_preserves_other_headers(self):
        result = _to_completion_result(_make_chat_jsonresponse(), "test-model")
        assert any(k.lower() == "x-context-budget" for k in result.headers)

    def test_body_is_completion_shape(self):
        result = _to_completion_result(_make_chat_jsonresponse(), "test-model")
        body = json.loads(result.body)
        assert body["object"] == "text_completion"
        assert body["choices"][0]["text"] == "hello world"
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_plain_model_dump_path_unaffected(self):
        from fusion_mlx.api.models import ChatCompletionResponse

        chat = ChatCompletionResponse(
            id="chatcmpl-1",
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
        result = _to_completion_result(chat, "test-model")
        assert not isinstance(result, JSONResponse)
        assert result.choices[0].text == "hi"
