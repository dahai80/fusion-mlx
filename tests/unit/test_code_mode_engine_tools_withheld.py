# SPDX-License-Identifier: Apache-2.0
"""M10 code-mode: engine must not receive native tools when code_mode is on.

With tools passed to the engine the chat template advertises the native
tool-call channel and the model answers with tool_calls (content=None)
instead of the code block the caller's harness expects. Regression for
the production smoke finding on 2026-11-17.
"""

import asyncio
import logging
import sys
import types
from unittest.mock import MagicMock

logger = logging.getLogger(__name__)

_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        "x-return-shape": {"temp_c": "float", "cond": "str"},
    },
}


class _RecordingEngine:
    is_mllm = False
    supports_guided_generation = False

    def __init__(self):
        self.chat_kwargs = None

    async def chat(self, messages, **kwargs):
        self.chat_kwargs = kwargs

        class _Gen:
            text = "```python\nresult = get_weather('Oslo')\n```"
            reasoning_content = None
            tool_calls = None
            finish_reason = "stop"
            prompt_tokens = 0
            cached_tokens = 0
            completion_tokens = 10
            generation_tokens = 10
            generation_tps = 0.0
            token_ids = []

        return _Gen()


def _patch_routes(monkeypatch, engine):
    import fusion_mlx.api.openai._common as _common_mod
    import fusion_mlx.api.openai.chat as routes
    import fusion_mlx.service.helpers as helpers

    fake = types.ModuleType("fusion_mlx.server")
    fake.resolve_model_with_profile = lambda model_id: (model_id, {})
    fake.get_max_context_window = lambda model_id: 0
    fake.get_settings = MagicMock(return_value=MagicMock(sse_keepalive_seconds=0))
    fake._server_state = {}
    monkeypatch.setitem(sys.modules, "fusion_mlx.server", fake)

    async def _ret(value):
        return value

    monkeypatch.setattr(routes, "_resolve_engine", lambda *a, **kw: _ret(engine))
    monkeypatch.setattr(routes, "_release_engine", lambda *a, **kw: _ret(None))
    monkeypatch.setattr(routes, "record_chat_session", lambda *a, **kw: None)
    monkeypatch.setattr(routes, "record_llm_metrics", lambda *a, **kw: None)
    monkeypatch.setattr(_common_mod, "_pool", None)
    monkeypatch.setattr(_common_mod, "_request_router", object())
    monkeypatch.setattr(
        helpers, "compute_prompt_tokens_for_messages", lambda *a, **kw: 0
    )
    monkeypatch.setattr(helpers, "get_model_max_context", lambda *a, **kw: 0)
    return routes


def _request(code_mode, tools):
    from fusion_mlx.api.openai.chat import ChatCompletionRequest

    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Weather in Oslo?"}],
        tools=tools,
        code_mode=code_mode,
        max_tokens=64,
    )


def _run(routes, req):
    return asyncio.new_event_loop().run_until_complete(
        routes._run_chat(req, _skip_cap_check=True, headers=None)
    )


def test_code_mode_withholds_tools_from_engine(monkeypatch):
    engine = _RecordingEngine()
    routes = _patch_routes(monkeypatch, engine)
    _run(routes, _request(code_mode=True, tools=[_TOOL]))
    assert engine.chat_kwargs is not None
    assert engine.chat_kwargs.get("tools") is None


def test_no_code_mode_passes_tools_through(monkeypatch):
    engine = _RecordingEngine()
    routes = _patch_routes(monkeypatch, engine)
    _run(routes, _request(code_mode=False, tools=[_TOOL]))
    assert engine.chat_kwargs is not None
    passed = engine.chat_kwargs.get("tools")
    assert passed is not None and len(passed) == 1
    assert passed[0].function["name"] == "get_weather"
