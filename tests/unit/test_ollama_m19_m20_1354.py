# SPDX-License-Identifier: Apache-2.0
"""#1354 audit M19/M20: Ollama option mapping + stream fidelity.

M19: options.num_ctx is the context window, not an output cap; it must not
overwrite max_tokens (which num_predict owns).
M20: streaming /api/chat + /api/generate must forward upstream error frames,
reasoning and tool_calls instead of reporting a clean done_reason "stop".
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api import ollama_routes
from fusion_mlx.api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    FunctionCall,
    ToolCall,
    Usage,
)
from fusion_mlx.api.ollama_routes import (
    _ollama_error_frame,
    _options_to_params,
    _tool_calls_to_ollama,
)
from fusion_mlx.middleware.auth import verify_api_key


@pytest.fixture
def client():
    app = FastAPI()

    async def _fake_auth():
        return True

    app.dependency_overrides[verify_api_key] = _fake_auth
    app.include_router(ollama_routes.router)
    return TestClient(app)


def _sse(*payloads):
    return [f"data: {json.dumps(p)}\n\n" for p in payloads]


def _install_fake_stream(monkeypatch, chunks):
    async def _gen():
        for chunk in chunks:
            yield chunk

    async def _fake_call(model, messages, stream, params):
        return _gen(), "req-1354", object()

    monkeypatch.setattr(ollama_routes, "_call_openai_chat", _fake_call)


def _ndjson(resp):
    lines = [ln for ln in resp.iter_lines() if ln and ln.startswith("{")]
    return [json.loads(ln) for ln in lines]


class TestM19NumCtx:
    def test_num_ctx_not_mapped_to_max_tokens(self):
        assert _options_to_params({"num_ctx": 8192}) == {}

    def test_num_ctx_does_not_clobber_num_predict(self):
        assert _options_to_params({"num_ctx": 8192, "num_predict": 64}) == {
            "max_tokens": 64
        }


class TestErrorFrameHelper:
    def test_dict_error_extracts_message(self):
        frame = json.loads(_ollama_error_frame({"message": "boom", "status": 500}))
        assert frame == {"error": "boom"}

    def test_string_error_passthrough(self):
        assert json.loads(_ollama_error_frame("kaput")) == {"error": "kaput"}


class TestToolCallConversion:
    def test_arguments_string_parsed_to_object(self):
        calls = _tool_calls_to_ollama(
            [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"P"}'},
                }
            ]
        )
        assert calls == [
            {"function": {"name": "get_weather", "arguments": {"city": "P"}}}
        ]

    def test_invalid_arguments_fall_back_to_empty_dict(self):
        calls = _tool_calls_to_ollama(
            [{"function": {"name": "f", "arguments": "not-json"}}]
        )
        assert calls == [{"function": {"name": "f", "arguments": {}}}]

    def test_missing_name_skipped(self):
        assert _tool_calls_to_ollama([{"function": {"arguments": "{}"}}]) == []


class TestGenerateStream:
    def test_error_frame_forwarded_without_done(self, client, monkeypatch):
        _install_fake_stream(
            monkeypatch,
            _sse({"error": {"message": "upstream exploded", "status": 500}}),
        )
        with client.stream(
            "POST", "/api/generate", json={"model": "m", "prompt": "hi"}
        ) as resp:
            frames = _ndjson(resp)
        assert frames == [{"error": "upstream exploded"}]

    def test_length_finish_reason_maps_to_done_reason(self, client, monkeypatch):
        _install_fake_stream(
            monkeypatch,
            _sse(
                {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "length"}]},
            ),
        )
        with client.stream(
            "POST", "/api/generate", json={"model": "m", "prompt": "hi"}
        ) as resp:
            frames = _ndjson(resp)
        assert frames[0]["response"] == "hi"
        assert frames[0]["done"] is False
        assert frames[-1]["done"] is True
        assert frames[-1]["done_reason"] == "length"


class TestChatStream:
    def test_error_frame_forwarded_without_done(self, client, monkeypatch):
        _install_fake_stream(
            monkeypatch,
            _sse({"error": {"message": "upstream exploded", "status": 500}}),
        )
        with client.stream(
            "POST",
            "/api/chat",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        ) as resp:
            frames = _ndjson(resp)
        assert frames == [{"error": "upstream exploded"}]

    def test_thinking_delta_forwarded(self, client, monkeypatch):
        _install_fake_stream(
            monkeypatch,
            _sse(
                {
                    "choices": [
                        {"delta": {"reasoning_content": "hmm"}, "finish_reason": None}
                    ]
                },
                {"choices": [{"delta": {"content": "answer"}, "finish_reason": None}]},
            ),
        )
        with client.stream(
            "POST",
            "/api/chat",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        ) as resp:
            frames = _ndjson(resp)
        assert frames[0]["message"]["thinking"] == "hmm"
        assert frames[0]["message"]["content"] == ""
        assert frames[1]["message"]["content"] == "answer"
        assert "thinking" not in frames[1]["message"]

    def test_tool_call_fragments_accumulated_into_one_message(
        self, client, monkeypatch
    ):
        _install_fake_stream(
            monkeypatch,
            _sse(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "c1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city":',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": '"Paris"}'},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ),
        )
        with client.stream(
            "POST",
            "/api/chat",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        ) as resp:
            frames = _ndjson(resp)
        tool_frames = [f for f in frames if "tool_calls" in f.get("message", {})]
        assert len(tool_frames) == 1
        assert tool_frames[0]["message"]["tool_calls"] == [
            {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
        ]
        assert frames[-1]["done"] is True
        assert frames[-1]["done_reason"] == "stop"

    def test_length_finish_reason_maps_to_done_reason(self, client, monkeypatch):
        _install_fake_stream(
            monkeypatch,
            _sse({"choices": [{"delta": {}, "finish_reason": "length"}]}),
        )
        with client.stream(
            "POST",
            "/api/chat",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        ) as resp:
            frames = _ndjson(resp)
        assert frames[-1]["done"] is True
        assert frames[-1]["done_reason"] == "length"


class TestChatNonStreaming:
    def test_tool_calls_and_thinking_in_response(self, client, monkeypatch):
        result = ChatCompletionResponse(
            id="chatcmpl-x",
            created=0,
            model="m",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(
                        content="",
                        reasoning_content="plan",
                        tool_calls=[
                            ToolCall(
                                id="c1",
                                type="function",
                                function=FunctionCall(
                                    name="get_weather",
                                    arguments='{"city":"P"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=Usage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
        )

        async def _fake_call(model, messages, stream, params):
            return result

        monkeypatch.setattr(ollama_routes, "_call_openai_chat", _fake_call)
        r = client.post(
            "/api/chat",
            json={
                "model": "m",
                "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 200
        message = r.json()["message"]
        assert message["thinking"] == "plan"
        assert message["tool_calls"] == [
            {"function": {"name": "get_weather", "arguments": {"city": "P"}}}
        ]
