# SPDX-License-Identifier: Apache-2.0
"""#385 增量 tool_call_delta 流式: 校验流式路径在生成期间增量发射
tool_call_delta SSE (而非全部等到 gen.finished)。

验收:
- tool_use 调用期间有流式输出 (mid-stream tool_call_delta)
- SSE 事件格式向后兼容 (无 tools 时路径不变)
- 不传 stream=true 时行为不变 (此处聚焦 stream=true 增量)
"""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api.openai_routes import router as chat_router
from fusion_mlx.config import reset_config
from fusion_mlx.engines.base import GenerationOutput

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


class _ToolStreamEngine:
    # 模拟流式引擎: 逐 delta 输出含工具调用标记的文本。
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, deltas: list[str]):
        self._deltas = deltas
        self.stream_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        accumulated = ""
        for i, delta in enumerate(self._deltas):
            accumulated += delta
            is_last = i == len(self._deltas) - 1
            yield GenerationOutput(
                text=accumulated,
                new_text=delta,
                prompt_tokens=4,
                completion_tokens=i + 1,
                finished=is_last,
                finish_reason="stop" if is_last else None,
                channel=None,
            )


def _make_client(engine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


def _parse_sse_events(text: str) -> tuple[list[dict], bool]:
    events: list[dict] = []
    saw_done = False
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return events, saw_done


def _tool_call_chunks(events: list[dict]) -> list[dict]:
    out = []
    for e in events:
        choices = e.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if delta.get("tool_calls"):
            out.append(e)
    return out


def _content_text(events: list[dict]) -> str:
    parts = []
    for e in events:
        choices = e.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        c = delta.get("content")
        if c:
            parts.append(c)
    return "".join(parts)


def test_incremental_tool_call_delta_emitted_mid_stream():
    """#385: 工具调用标记在生成期间到达时, tool_call_delta 应在 finished
    之前增量发射, 且工具标记文本不应作为 content 泄漏。"""
    # Qwen/Hermes XML 风格: 工具调用跨多个 delta 到达, 收尾 marker 后还有正文。
    # auto 解析器 (无 tool_call_parser 配置时的回退) 原生识别 <tool_call>{...}</tool_call> wire format。
    deltas = [
        "Let me check. ",
        '<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>',
        " Done.",
    ]
    engine = _ToolStreamEngine(deltas)
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": _TOOLS,
        },
    )
    assert resp.status_code == 200, resp.text
    events, saw_done = _parse_sse_events(resp.text)
    assert saw_done

    tc_chunks = _tool_call_chunks(events)
    assert tc_chunks, "expected at least one incremental tool_call_delta chunk"
    # 第一个 tool_call_delta 必须出现在 finished chunk 之前 (增量, 非全量收尾)
    finish_idx = next(
        (
            i
            for i, e in enumerate(events)
            if (e.get("choices") or [{}])[0].get("finish_reason")
        ),
        len(events),
    )
    tc_idx = events.index(tc_chunks[0])
    assert tc_idx < finish_idx, (
        f"tool_call_delta must arrive before finish: tc_idx={tc_idx} "
        f"finish_idx={finish_idx}"
    )
    # 校验 tool_call_delta 内容
    first_tc = (tc_chunks[0]["choices"][0]["delta"]["tool_calls"])[0]
    assert first_tc["function"]["name"] == "get_weather"
    # 工具标记文本不应泄漏到 content
    content = _content_text(events)
    assert "<tool_call>" not in content, f"tool marker leaked into content: {content!r}"
    assert "get_weather" not in content, f"tool name leaked into content: {content!r}"
    # finish_reason 应为 tool_calls
    finish_reasons = [
        (e.get("choices") or [{}])[0].get("finish_reason")
        for e in events
        if (e.get("choices") or [{}])[0].get("finish_reason")
    ]
    assert "tool_calls" in finish_reasons, finish_reasons


def test_no_tools_path_unchanged():
    """#385 向后兼容: 无 tools 时流式路径与改造前一致, 无 tool_call_delta。"""
    engine = _ToolStreamEngine(["Hello", " world."])
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "say hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    events, saw_done = _parse_sse_events(resp.text)
    assert saw_done
    assert _tool_call_chunks(events) == [], "no tool_call_delta without tools"
    assert _content_text(events) == "Hello world."


def test_multi_tool_incremental_interleave():
    """#385 多工具: 两个工具调用应各自增量发射, index 单调。"""
    deltas = [
        '<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>',
        '<tool_call>{"name":"get_time","arguments":{"zone":"PST"}}</tool_call>',
    ]
    engine = _ToolStreamEngine(deltas)
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "weather and time"}],
            "tools": _TOOLS,
        },
    )
    assert resp.status_code == 200, resp.text
    events, _ = _parse_sse_events(resp.text)
    tc_chunks = _tool_call_chunks(events)
    names = []
    for e in tc_chunks:
        for tc in e["choices"][0]["delta"]["tool_calls"]:
            names.append(tc["function"]["name"])
    assert names == ["get_weather", "get_time"], names
