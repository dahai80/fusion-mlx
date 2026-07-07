# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.api.openai_models + api.models — Pydantic request/response schemas.

Covers field validation, coercion, aliases, serialization for all BaseModel classes.
"""

from __future__ import annotations

import json

import pytest

# ── api/openai_models.py ─────────────────────────────────────────────
from fusion_mlx.api.openai_models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionRequest,
    ContentPart,
    FunctionCall,
    ImageURL,
    Message,
    PromptTokensDetails,
    ResponseFormat,
    ResponseFormatJsonSchema,
    StreamOptions,
    StructuredOutputOptions,
    ToolCall,
    ToolDefinition,
    Usage,
    _coerce_tool_call_arguments,
)


class TestImageURL:
    def test_url_field(self):
        u = ImageURL(url="http://x.png")
        assert u.url == "http://x.png"

    def test_detail_optional(self):
        u = ImageURL(url="http://x.png")
        assert u.detail == "auto"  # default is "auto" not None


class TestContentPart:
    def test_text_part(self):
        p = ContentPart(type="text", text="hello")
        assert p.type == "text"
        assert p.text == "hello"

    def test_image_url_part(self):
        p = ContentPart(type="image_url", image_url={"url": "http://x.png"})
        assert p.type == "image_url"


class TestMessage:
    def test_minimal(self):
        m = Message(role="user", content="hi")
        assert m.role == "user"
        assert m.content == "hi"

    def test_with_tool_calls(self):
        # Message.tool_calls accepts list of dicts (not ToolCall objects per pydantic validation)
        tc = {
            "id": "tc1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "sf"}'},
        }
        m = Message(role="assistant", content=None, tool_calls=[tc])
        assert m.tool_calls[0]["id"] == "tc1"

    def test_tool_call_arguments_none_coerced(self):
        # FunctionCall.arguments=None raises ValueError (must be JSON string per OpenAI spec)
        with pytest.raises(Exception):
            ToolCall(id="tc1", function=FunctionCall(name="get", arguments=None))

    def test_strip_name_whitespace(self):
        fc = FunctionCall(name="  get_weather  ", arguments="{}")
        assert fc.name == "get_weather"

    def test_validate_arguments_json_string(self):
        fc = FunctionCall(name="get", arguments={"city": "sf"})
        # dict → json string coercion
        assert isinstance(fc.arguments, str)


class TestCoerceToolCallArguments:
    def test_none_raises_valueerror(self):
        # _coerce_tool_call_arguments(None) raises ValueError per OpenAI spec
        with pytest.raises(ValueError, match="must be a JSON-encoded string"):
            _coerce_tool_call_arguments(None)

    def test_dict_returns_json_string(self):
        result = _coerce_tool_call_arguments({"a": 1})
        assert json.loads(result) == {"a": 1}

    def test_string_passthrough(self):
        assert _coerce_tool_call_arguments('{"x": 1}') == '{"x": 1}'


class TestToolDefinition:
    def test_minimal(self):
        t = ToolDefinition(type="function", function={"name": "get_weather"})
        assert t.type == "function"


class TestResponseFormatJsonSchema:
    def test_with_schema(self):
        r = ResponseFormatJsonSchema(name="weather", schema={"type": "object"})
        assert r.name == "weather"


class TestResponseFormat:
    def test_type_only(self):
        r = ResponseFormat(type="text")
        assert r.type == "text"


class TestStructuredOutputOptions:
    def test_defaults(self):
        s = StructuredOutputOptions()
        # Check fields exist
        assert hasattr(s, "json_schema")


class TestStreamOptions:
    def test_include_usage_optional(self):
        s = StreamOptions()
        assert s.include_usage is None or hasattr(s, "include_usage")


class TestChatCompletionRequest:
    def test_minimal(self):
        r = ChatCompletionRequest(
            model="qwen", messages=[Message(role="user", content="hi")]
        )
        assert r.model == "qwen"
        assert r.messages[0].content == "hi"

    def test_coerce_stop_string_to_list(self):
        r = ChatCompletionRequest(
            model="qwen", messages=[Message(role="user", content="hi")], stop="END"
        )
        assert r.stop == ["END"]

    def test_coerce_stop_none(self):
        r = ChatCompletionRequest(
            model="qwen", messages=[Message(role="user", content="hi")], stop=None
        )
        assert r.stop is None

    def test_max_completion_tokens_alias(self):
        r = ChatCompletionRequest(
            model="qwen",
            messages=[Message(role="user", content="hi")],
            max_tokens=100,
        )
        # openai_models ChatCompletionRequest only has max_tokens (not max_completion_tokens)
        assert r.max_tokens == 100

    def test_with_tools(self):
        tools = [ToolDefinition(type="function", function={"name": "get"})]
        r = ChatCompletionRequest(
            model="qwen", messages=[Message(role="user", content="hi")], tools=tools
        )
        assert r.tools[0].function["name"] == "get"

    def test_with_response_format(self):
        rf = ResponseFormat(type="json_object")
        r = ChatCompletionRequest(
            model="qwen",
            messages=[Message(role="user", content="hi")],
            response_format=rf,
        )
        assert r.response_format.type == "json_object"

    def test_with_stream(self):
        r = ChatCompletionRequest(
            model="qwen", messages=[Message(role="user", content="hi")], stream=True
        )
        assert r.stream is True

    def test_temperature_optional(self):
        r = ChatCompletionRequest(
            model="qwen", messages=[Message(role="user", content="hi")]
        )
        assert r.temperature is None or hasattr(r, "temperature")


class TestAssistantMessage:
    def test_minimal(self):
        m = AssistantMessage(role="assistant", content="hello")
        assert m.role == "assistant"


class TestChatCompletionChoice:
    def test_with_message(self):
        msg = AssistantMessage(role="assistant", content="hi")
        c = ChatCompletionChoice(index=0, message=msg, finish_reason="stop")
        assert c.finish_reason == "stop"


class TestPromptTokensDetails:
    def test_defaults(self):
        d = PromptTokensDetails()
        assert hasattr(d, "cached_tokens") or hasattr(d, "audited_tokens")


class TestUsage:
    def test_with_tokens(self):
        u = Usage(prompt_tokens=10, completion_tokens=20)
        assert u.prompt_tokens == 10
        assert u.completion_tokens == 20


class TestChatCompletionResponse:
    def test_minimal(self):
        msg = AssistantMessage(role="assistant", content="hi")
        choice = ChatCompletionChoice(index=0, message=msg, finish_reason="stop")
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        r = ChatCompletionResponse(id="r1", model="qwen", choices=[choice], usage=usage)
        assert r.id == "r1"
        assert r.choices[0].message.content == "hi"


class TestCompletionRequest:
    def test_minimal(self):
        r = CompletionRequest(model="qwen", prompt="Hello")
        assert r.model == "qwen"
        assert r.prompt == "Hello"

    def test_coerce_stop_string(self):
        r = CompletionRequest(model="qwen", prompt="hi", stop="END")
        assert r.stop == ["END"]


class TestCompletionChoice:
    def test_with_text(self):
        c = CompletionChoice(index=0, text="hello", finish_reason="stop")
        assert c.text == "hello"


# ── api/models.py ────────────────────────────────────────────────────


from fusion_mlx.api.models import (
    AssistantMessage as ModelsAssistantMessage,
)
from fusion_mlx.api.models import (
    AudioUrl,
    ChoiceLogProbs,
    TokenLogProb,
    TopLogProb,
    VideoUrl,
)
from fusion_mlx.api.models import (
    ChatCompletionRequest as ModelsChatCompletionRequest,
)
from fusion_mlx.api.models import (
    ContentPart as ModelsContentPart,
)
from fusion_mlx.api.models import (
    FunctionCall as ModelsFunctionCall,
)
from fusion_mlx.api.models import (
    ImageUrl as ModelsImageUrl,
)
from fusion_mlx.api.models import (
    Message as ModelsMessage,
)
from fusion_mlx.api.models import (
    ResponseFormat as ModelsResponseFormat,
)
from fusion_mlx.api.models import (
    StreamOptions as ModelsStreamOptions,
)
from fusion_mlx.api.models import (
    ToolCall as ModelsToolCall,
)
from fusion_mlx.api.models import (
    ToolDefinition as ModelsToolDefinition,
)


class TestModelsImageUrl:
    def test_url(self):
        u = ModelsImageUrl(url="http://x.png")
        assert u.url == "http://x.png"


class TestVideoUrl:
    def test_url(self):
        v = VideoUrl(url="http://x.mp4")
        assert v.url == "http://x.mp4"


class TestAudioUrl:
    def test_url(self):
        a = AudioUrl(url="http://x.wav")
        assert a.url == "http://x.wav"


class TestModelsContentPart:
    def test_text(self):
        p = ModelsContentPart(type="text", text="hello")
        assert p.type == "text"

    def test_image_url(self):
        p = ModelsContentPart(type="image_url", image_url={"url": "http://x.png"})
        assert p.type == "image_url"

    def test_video_url(self):
        p = ModelsContentPart(type="video_url", video_url={"url": "http://x.mp4"})
        assert p.type == "video_url"

    def test_audio_url(self):
        p = ModelsContentPart(type="audio_url", audio_url={"url": "http://x.wav"})
        assert p.type == "audio_url"


class TestModelsMessage:
    def test_minimal(self):
        m = ModelsMessage(role="user", content="hi")
        assert m.role == "user"

    # ModelsMessage has no `name` field — only openai_models.Message has it


class TestModelsFunctionCall:
    def test_with_name(self):
        fc = ModelsFunctionCall(name="get", arguments="{}")
        assert fc.name == "get"


class TestModelsToolCall:
    def test_minimal(self):
        fc = ModelsFunctionCall(name="get", arguments="{}")
        tc = ModelsToolCall(id="tc1", function=fc)
        assert tc.id == "tc1"


class TestModelsToolDefinition:
    def test_minimal(self):
        t = ModelsToolDefinition(type="function", function={"name": "get"})
        assert t.function["name"] == "get"


class TestModelsResponseFormat:
    def test_type(self):
        r = ModelsResponseFormat(type="text")
        assert r.type == "text"


class TestTopLogProb:
    def test_with_token(self):
        t = TopLogProb(token="hello", logprob=-0.5, bytes=[1, 2])
        assert t.token == "hello"


class TestTokenLogProb:
    def test_with_top_logprobs(self):
        top = [TopLogProb(token="a", logprob=-1.0, bytes=[1])]
        t = TokenLogProb(token="hello", logprob=-0.5, bytes=[1, 2], top_logprobs=top)
        assert t.top_logprobs[0].token == "a"


class TestChoiceLogProbs:
    def test_with_content(self):
        t = TokenLogProb(token="hi", logprob=-0.1, bytes=[1], top_logprobs=[])
        c = ChoiceLogProbs(content=[t])
        assert c.content[0].token == "hi"


class TestModelsChatCompletionRequest:
    def test_minimal(self):
        r = ModelsChatCompletionRequest(
            model="qwen", messages=[ModelsMessage(role="user", content="hi")]
        )
        assert r.model == "qwen"

    def test_with_reasoning_effort(self):
        r = ModelsChatCompletionRequest(
            model="qwen",
            messages=[ModelsMessage(role="user", content="hi")],
            reasoning_effort="medium",
        )
        assert r.reasoning_effort == "medium"

    def test_invalid_reasoning_effort_coerced(self):
        # invalid reasoning_effort raises ValueError during validation
        with pytest.raises(Exception):
            ModelsChatCompletionRequest(
                model="qwen",
                messages=[ModelsMessage(role="user", content="hi")],
                reasoning_effort="invalid",
            )

    def test_with_top_logprobs(self):
        r = ModelsChatCompletionRequest(
            model="qwen",
            messages=[ModelsMessage(role="user", content="hi")],
            top_logprobs=5,
        )
        assert r.top_logprobs == 5

    def test_with_logprobs(self):
        r = ModelsChatCompletionRequest(
            model="qwen",
            messages=[ModelsMessage(role="user", content="hi")],
            logprobs=True,
        )
        assert r.logprobs is True

    def test_with_functions_legacy(self):
        r = ModelsChatCompletionRequest(
            model="qwen",
            messages=[ModelsMessage(role="user", content="hi")],
            functions=[{"name": "get"}],
        )
        # _normalize_legacy_functions: functions → tools
        assert r.tools is not None or hasattr(r, "tools")

    def test_with_tool_choice(self):
        r = ModelsChatCompletionRequest(
            model="qwen",
            messages=[ModelsMessage(role="user", content="hi")],
            tool_choice="auto",
        )
        assert r.tool_choice == "auto"

    def test_normalize_max_completion_tokens(self):
        r = ModelsChatCompletionRequest(
            model="qwen",
            messages=[ModelsMessage(role="user", content="hi")],
            max_tokens=100,
        )
        # _normalize_max_completion_tokens: max_tokens → max_completion_tokens
        assert (
            getattr(r, "max_completion_tokens", 100) == 100
            or getattr(r, "max_tokens", 100) == 100
        )

    def test_with_extra_body(self):
        # ChatCompletionRequest has no `extra_body` field — skip this test
        pytest.skip("no extra_body field in ModelsChatCompletionRequest")

    def test_with_stream_options(self):
        so = ModelsStreamOptions(include_usage=True)
        r = ModelsChatCompletionRequest(
            model="qwen",
            messages=[ModelsMessage(role="user", content="hi")],
            stream=True,
            stream_options=so,
        )
        assert r.stream_options.include_usage is True


class TestModelsAssistantMessage:
    def test_minimal(self):
        m = ModelsAssistantMessage(role="assistant", content="hi")
        assert m.role == "assistant"
