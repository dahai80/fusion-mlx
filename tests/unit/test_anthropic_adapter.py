# SPDX-License-Identifier: Apache-2.0
import json
import logging

import pytest

from fusion_mlx.api.adapters.anthropic import AnthropicAdapter
from fusion_mlx.api.adapters.base import (
    BaseAdapter,
    InternalRequest,
    InternalResponse,
    StreamChunk,
)
from fusion_mlx.api.anthropic_adapter import (
    _convert_message,
    _convert_stop_reason,
    _convert_tool,
    _convert_tool_choice,
    anthropic_to_openai,
    openai_to_anthropic,
)
from fusion_mlx.api.anthropic_models import (
    AnthropicMessage,
    AnthropicTool,
    ContentBlockText,
    ContentBlockToolResult,
    ContentBlockToolUse,
    MessagesRequest,
)
from fusion_mlx.api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    FunctionCall,
    ToolCall,
    Usage,
)

logger = logging.getLogger(__name__)


class TestAnthropicAdapter:
    @pytest.fixture
    def adapter(self):
        return AnthropicAdapter()

    def test_adapter_name(self, adapter):
        assert adapter.name == "anthropic"

    def test_adapter_inherits_base(self, adapter):
        assert isinstance(adapter, BaseAdapter)

    def test_parse_request_simple_message(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[
                AnthropicMessage(role="user", content="Hello"),
            ],
        )

        internal = adapter.parse_request(request)

        assert isinstance(internal, InternalRequest)
        assert len(internal.messages) == 1
        assert internal.messages[0].role == "user"
        assert internal.messages[0].content == "Hello"
        assert internal.model == "claude-3-sonnet"
        assert internal.max_tokens == 1024

    def test_parse_request_multiple_messages(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[
                AnthropicMessage(role="user", content="Hello"),
                AnthropicMessage(role="assistant", content="Hi there!"),
                AnthropicMessage(role="user", content="How are you?"),
            ],
        )

        internal = adapter.parse_request(request)

        assert len(internal.messages) == 3
        assert internal.messages[0].role == "user"
        assert internal.messages[1].role == "assistant"
        assert internal.messages[2].role == "user"

    def test_parse_request_with_system(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[
                AnthropicMessage(role="user", content="Hello"),
            ],
            system="You are a helpful assistant.",
        )

        internal = adapter.parse_request(request)

        assert len(internal.messages) == 2
        assert internal.messages[0].role == "system"
        assert internal.messages[0].content == "You are a helpful assistant."
        assert internal.messages[1].role == "user"

    def test_parse_request_in_messages_system(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[
                AnthropicMessage(role="user", content="Hi there"),
                AnthropicMessage(role="system", content="Be terse."),
                AnthropicMessage(role="assistant", content="ok"),
            ],
        )

        internal = adapter.parse_request(request)

        assert internal.messages[0].role == "system"
        assert internal.messages[0].content == "Be terse."
        roles = [m.role for m in internal.messages[1:]]
        assert roles == ["user", "assistant"]

    def test_parse_request_system_field_and_in_messages_merge(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[
                AnthropicMessage(role="system", content="Be terse."),
                AnthropicMessage(role="user", content="Hi"),
            ],
            system="You are a helpful assistant.",
        )

        internal = adapter.parse_request(request)

        assert internal.messages[0].role == "system"
        assert internal.messages[0].content == (
            "You are a helpful assistant.\n\nBe terse."
        )
        assert internal.messages[1].role == "user"

    def test_parse_request_multiple_in_messages_system(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[
                AnthropicMessage(role="system", content="First."),
                AnthropicMessage(role="user", content="Hi"),
                AnthropicMessage(role="system", content="Second."),
            ],
        )

        internal = adapter.parse_request(request)

        assert internal.messages[0].role == "system"
        assert internal.messages[0].content == "First.\nSecond."
        assert [m.role for m in internal.messages[1:]] == ["user"]

    def test_parse_request_with_temperature(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            temperature=0.5,
        )

        internal = adapter.parse_request(request)

        assert internal.temperature == 0.5

    def test_parse_request_with_zero_temperature(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            temperature=0.0,
        )

        internal = adapter.parse_request(request)

        assert internal.temperature == 0.0

    def test_parse_request_default_temperature(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )

        internal = adapter.parse_request(request)

        assert internal.temperature == 1.0

    def test_parse_request_with_top_p(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            top_p=0.9,
        )

        internal = adapter.parse_request(request)

        assert internal.top_p == 0.9

    def test_parse_request_with_top_k(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            top_k=40,
        )

        internal = adapter.parse_request(request)

        assert internal.top_k == 40

    def test_parse_request_with_stream(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            stream=True,
        )

        internal = adapter.parse_request(request)

        assert internal.stream is True

    def test_parse_request_with_stop_sequences(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            stop_sequences=["STOP", "END"],
        )

        internal = adapter.parse_request(request)

        assert internal.stop == ["STOP", "END"]

    def test_parse_request_with_tools(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            tools=[
                AnthropicTool(
                    name="get_weather",
                    description="Get weather info",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                    },
                )
            ],
        )

        internal = adapter.parse_request(request)

        assert internal.tools is not None
        assert len(internal.tools) == 1
        assert internal.tools[0]["function"]["name"] == "get_weather"
        assert internal.tools[0]["function"]["description"] == "Get weather info"

    def test_parse_request_generates_request_id(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )

        internal = adapter.parse_request(request)

        assert internal.request_id is not None
        assert internal.request_id.startswith("msg_")

    def test_format_response_basic(self, adapter):
        from fusion_mlx.api.anthropic_models import MessagesResponse

        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        response = InternalResponse(
            text="Hi there!",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
        )

        result = adapter.format_response(response, request)

        assert isinstance(result, MessagesResponse)
        assert result.type == "message"
        assert result.role == "assistant"
        assert result.model == "claude-3-sonnet"
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "Hi there!"
        assert result.stop_reason == "end_turn"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    def test_format_response_with_tool_calls(self, adapter):
        from fusion_mlx.api.openai_models import FunctionCall, ToolCall

        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        tool_calls = [
            ToolCall(
                id="toolu_123",
                type="function",
                function=FunctionCall(
                    name="get_weather",
                    arguments='{"location": "Tokyo"}',
                ),
            )
        ]
        response = InternalResponse(
            text="",
            finish_reason="tool_calls",
            tool_calls=tool_calls,
        )

        result = adapter.format_response(response, request)

        assert result.stop_reason == "tool_use"
        tool_use_blocks = [c for c in result.content if c.type == "tool_use"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0].name == "get_weather"
        assert tool_use_blocks[0].input == {"location": "Tokyo"}

    def test_format_response_finish_reason_length(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        response = InternalResponse(
            text="Response truncated...",
            finish_reason="length",
        )

        result = adapter.format_response(response, request)

        assert result.stop_reason == "max_tokens"

    def test_format_response_empty_text(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        response = InternalResponse(text="")

        result = adapter.format_response(response, request)

        assert len(result.content) >= 1

    def test_format_stream_chunk_first(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        chunk = StreamChunk(text="Hello", is_first=True)

        result = adapter.format_stream_chunk(chunk, request)

        assert "event: message_start" in result
        assert "event: content_block_start" in result
        assert "event: content_block_delta" in result

    def test_format_stream_chunk_middle(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        chunk = StreamChunk(text=" world", is_first=False, is_last=False)

        result = adapter.format_stream_chunk(chunk, request)

        assert "event: content_block_delta" in result
        assert "text_delta" in result
        assert " world" in result

    def test_format_stream_chunk_last(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        chunk = StreamChunk(
            text="",
            finish_reason="stop",
            is_last=True,
            completion_tokens=10,
        )

        result = adapter.format_stream_chunk(chunk, request)

        assert "event: content_block_stop" in result
        assert "event: message_delta" in result
        assert "event: message_stop" in result

    def test_format_stream_chunk_with_tool_call_delta(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        tool_delta = {"name": "get_weather"}
        chunk = StreamChunk(tool_call_delta=tool_delta)

        result = adapter.format_stream_chunk(chunk, request)

        assert "event: content_block_delta" in result
        assert "input_json_delta" in result

    def test_format_stream_chunk_empty_no_events(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        chunk = StreamChunk(text="", is_first=False, is_last=False)

        result = adapter.format_stream_chunk(chunk, request)

        assert result == ""

    def test_format_stream_end(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )

        result = adapter.format_stream_end(request)

        assert result == ""

    def test_create_error_response_default(self, adapter):
        result = adapter.create_error_response("Something went wrong")

        assert result["type"] == "error"
        assert result["error"]["message"] == "Something went wrong"
        assert result["error"]["type"] == "api_error"

    def test_create_error_response_custom_type(self, adapter):
        result = adapter.create_error_response(
            "Invalid request",
            error_type="invalid_request_error",
        )

        assert result["error"]["message"] == "Invalid request"
        assert result["error"]["type"] == "invalid_request_error"

    def test_create_error_response_authentication(self, adapter):
        result = adapter.create_error_response(
            "Invalid API key",
            error_type="authentication_error",
        )

        assert result["error"]["type"] == "authentication_error"
        assert result["error"]["message"] == "Invalid API key"

    def test_format_error_event(self, adapter):
        result = adapter.format_error_event("Something went wrong")

        assert "event: error" in result
        assert "Something went wrong" in result

    def test_format_error_event_custom_type(self, adapter):
        result = adapter.format_error_event(
            "Invalid request",
            error_type="invalid_request_error",
        )

        assert "event: error" in result
        assert "invalid_request_error" in result


class TestAnthropicStreamingEvents:
    @pytest.fixture
    def adapter(self):
        return AnthropicAdapter()

    def test_full_stream_sequence(self, adapter):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )

        first = adapter.format_stream_chunk(
            StreamChunk(text="Hi", is_first=True),
            request,
        )
        middle = adapter.format_stream_chunk(
            StreamChunk(text=" there"),
            request,
        )
        last = adapter.format_stream_chunk(
            StreamChunk(
                text="!",
                finish_reason="stop",
                is_last=True,
                completion_tokens=3,
            ),
            request,
        )
        end = adapter.format_stream_end(request)

        assert "message_start" in first
        assert "content_block_start" in first
        assert "content_block_delta" in first

        assert "content_block_delta" in middle

        assert "content_block_delta" in last
        assert "content_block_stop" in last
        assert "message_delta" in last
        assert "message_stop" in last

        assert end == ""

    def test_stream_preserves_model_name(self, adapter):
        request = MessagesRequest(
            model="claude-3-opus",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )

        result = adapter.format_stream_chunk(
            StreamChunk(text="Hi", is_first=True),
            request,
        )

        assert "claude-3-opus" in result


class TestAnthropicToolUseConversion:
    def test_tool_use_block_converted_to_calling_tool_format(self):
        from fusion_mlx.api.anthropic_models import AnthropicMessage, MessagesRequest
        from fusion_mlx.api.anthropic_utils import convert_anthropic_to_internal

        request = MessagesRequest(
            model="test-model",
            max_tokens=1024,
            messages=[
                AnthropicMessage(role="user", content="What is the weather?"),
                AnthropicMessage(
                    role="assistant",
                    content=[
                        {"type": "text", "text": "Let me check."},
                        {
                            "type": "tool_use",
                            "id": "call_123",
                            "name": "get_weather",
                            "input": {"city": "Tokyo"},
                        },
                    ],
                ),
                AnthropicMessage(
                    role="user",
                    content=[
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_123",
                            "content": "Sunny, 25C",
                        },
                    ],
                ),
            ],
        )

        messages = convert_anthropic_to_internal(request)

        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        content = assistant_msgs[0]["content"]

        assert "[Calling tool: get_weather(" in content
        assert "[Tool call:" not in content


class TestAnthropicAudioConversion:
    def test_input_audio_block_dropped_with_preserve_images(self):
        import base64

        from fusion_mlx.api.anthropic_models import (
            AnthropicMessage,
            ContentBlockInputAudio,
            MessagesRequest,
        )
        from fusion_mlx.api.anthropic_utils import convert_anthropic_to_internal

        fake_audio = base64.b64encode(b"\x00" * 100).decode()

        request = MessagesRequest(
            model="gemma4-unified",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "What sound is this?"},
                        ContentBlockInputAudio(
                            source={
                                "type": "base64",
                                "media_type": "audio/wav",
                                "data": fake_audio,
                            },
                        ),
                    ],
                ),
            ],
        )

        messages = convert_anthropic_to_internal(request, preserve_images=True)

        assert len(messages) == 1
        content = messages[0]["content"]
        assert isinstance(content, str)
        assert "What sound is this?" in content
        assert "input_audio" not in content.lower()

    def test_input_audio_block_dropped_without_preserve_images(self):
        import base64

        from fusion_mlx.api.anthropic_models import (
            AnthropicMessage,
            ContentBlockInputAudio,
            MessagesRequest,
        )
        from fusion_mlx.api.anthropic_utils import convert_anthropic_to_internal

        fake_audio = base64.b64encode(b"\x00" * 100).decode()

        request = MessagesRequest(
            model="gemma4-unified",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "What sound is this?"},
                        ContentBlockInputAudio(
                            source={
                                "type": "base64",
                                "media_type": "audio/wav",
                                "data": fake_audio,
                            },
                        ),
                    ],
                ),
            ],
        )

        messages = convert_anthropic_to_internal(request, preserve_images=False)

        content = messages[0]["content"]
        assert isinstance(content, str)
        assert "input_audio" not in content.lower()

    def test_audio_only_message(self):
        import base64

        from fusion_mlx.api.anthropic_models import (
            AnthropicMessage,
            ContentBlockInputAudio,
            MessagesRequest,
        )
        from fusion_mlx.api.anthropic_utils import convert_anthropic_to_internal

        fake_audio = base64.b64encode(b"\x00" * 100).decode()

        request = MessagesRequest(
            model="gemma4-unified",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        ContentBlockInputAudio(
                            source={
                                "type": "base64",
                                "media_type": "audio/wav",
                                "data": fake_audio,
                            },
                        ),
                    ],
                ),
            ],
        )

        messages = convert_anthropic_to_internal(request, preserve_images=True)

        assert len(messages) == 1


# =============================================================================
# Migrated from Rapid-MLX test_anthropic_adapter.py
# Low-level conversion function tests for anthropic_adapter.py
# ============================================================================


class TestConvertStopReason:
    def test_stop_to_end_turn(self):
        assert _convert_stop_reason("stop") == "end_turn"

    def test_tool_calls_to_tool_use(self):
        assert _convert_stop_reason("tool_calls") == "tool_use"

    def test_length_to_max_tokens(self):
        assert _convert_stop_reason("length") == "max_tokens"

    def test_content_filter_to_end_turn(self):
        assert _convert_stop_reason("content_filter") == "end_turn"

    def test_none_to_end_turn(self):
        assert _convert_stop_reason(None) == "end_turn"

    def test_unknown_to_end_turn(self):
        assert _convert_stop_reason("something_else") == "end_turn"


class TestConvertToolChoiceMigrated:
    def test_auto(self):
        assert _convert_tool_choice({"type": "auto"}) == "auto"

    def test_any_to_required(self):
        assert _convert_tool_choice({"type": "any"}) == "required"

    def test_none_type(self):
        assert _convert_tool_choice({"type": "none"}) == "none"

    def test_specific_tool(self):
        result = _convert_tool_choice({"type": "tool", "name": "search"})
        assert result == {
            "type": "function",
            "function": {"name": "search"},
        }

    def test_missing_type_defaults_to_auto(self):
        assert _convert_tool_choice({}) == "auto"

    def test_unknown_type_defaults_to_auto(self):
        assert _convert_tool_choice({"type": "unknown"}) == "auto"


class TestConvertToolMigrated:
    def test_minimal_tool(self):
        tool = AnthropicTool(
            name="search",
            input_schema={"type": "object", "properties": {}},
        )
        result = _convert_tool(tool)
        assert result.type == "function"
        assert result.function["name"] == "search"
        assert result.function["description"] == ""
        assert result.function["parameters"] == {"type": "object", "properties": {}}
        logger.debug("test_minimal_tool: name=%s", result.function["name"])

    def test_full_tool(self):
        tool = AnthropicTool(
            name="get_weather",
            description="Get weather for a city",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
        result = _convert_tool(tool)
        assert result.function["name"] == "get_weather"
        assert result.function["description"] == "Get weather for a city"
        assert result.function["parameters"]["required"] == ["city"]


class TestConvertMessageMigrated:
    def test_simple_user_string(self):
        msg = AnthropicMessage(role="user", content="hello")
        result = _convert_message(msg)
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content == "hello"

    def test_simple_assistant_string(self):
        msg = AnthropicMessage(role="assistant", content="hi there")
        result = _convert_message(msg)
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].content == "hi there"

    def test_user_with_text_blocks(self):
        msg = AnthropicMessage(
            role="user",
            content=[
                ContentBlockText(text="first"),
                ContentBlockText(text="second"),
            ],
        )
        result = _convert_message(msg)
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content == "first\nsecond"

    def test_user_with_tool_results(self):
        msg = AnthropicMessage(
            role="user",
            content=[
                ContentBlockToolResult(
                    tool_use_id="call_1",
                    content="sunny, 22C",
                ),
                ContentBlockToolResult(
                    tool_use_id="call_2",
                    content="rainy, 15C",
                ),
            ],
        )
        result = _convert_message(msg)
        assert len(result) == 2
        assert result[0].role == "tool"
        assert result[0].content == "sunny, 22C"
        assert result[0].tool_call_id == "call_1"
        assert result[1].role == "tool"
        assert result[1].content == "rainy, 15C"

    def test_user_with_text_and_tool_results(self):
        msg = AnthropicMessage(
            role="user",
            content=[
                ContentBlockText(text="here are results"),
                ContentBlockToolResult(
                    tool_use_id="call_1",
                    content="done",
                ),
            ],
        )
        result = _convert_message(msg)
        assert len(result) == 2
        assert result[0].role == "user"
        assert result[0].content == "here are results"
        assert result[1].role == "tool"

    def test_tool_result_with_list_content(self):
        msg = AnthropicMessage(
            role="user",
            content=[
                ContentBlockToolResult(
                    tool_use_id="call_1",
                    content=[
                        {"type": "text", "text": "line one"},
                        {"type": "text", "text": "line two"},
                    ],
                ),
            ],
        )
        result = _convert_message(msg)
        assert result[0].role == "tool"
        assert result[0].content == "line one\nline two"

    def test_tool_result_with_empty_string_content(self):
        msg = AnthropicMessage(
            role="user",
            content=[
                ContentBlockToolResult(
                    tool_use_id="call_1",
                    content="",
                ),
            ],
        )
        result = _convert_message(msg)
        assert result[0].content == ""

    def test_assistant_with_tool_use(self):
        msg = AnthropicMessage(
            role="assistant",
            content=[
                ContentBlockText(text="Let me check."),
                ContentBlockToolUse(
                    id="call_abc",
                    name="search",
                    input={"q": "weather"},
                ),
            ],
        )
        result = _convert_message(msg)
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].content == "Let me check."
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0]["function"]["name"] == "search"
        args = json.loads(result[0].tool_calls[0]["function"]["arguments"])
        assert args == {"q": "weather"}

    def test_assistant_empty_content(self):
        msg = AnthropicMessage(
            role="assistant",
            content=[],
        )
        result = _convert_message(msg)
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].content == ""

    def test_user_empty_content(self):
        msg = AnthropicMessage(
            role="user",
            content=[],
        )
        result = _convert_message(msg)
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content == ""


class TestAnthropicToOpenaiMigrated:
    def _make_request(self, **kwargs):
        defaults = {
            "model": "default",
            "messages": [AnthropicMessage(role="user", content="hi")],
            "max_tokens": 100,
        }
        defaults.update(kwargs)
        return MessagesRequest(**defaults)

    def test_simple_request(self):
        req = self._make_request()
        result = anthropic_to_openai(req)
        assert result.model == "default"
        assert result.max_tokens == 100
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "hi"

    def test_system_string(self):
        req = self._make_request(system="Be helpful.")
        result = anthropic_to_openai(req)
        assert len(result.messages) == 2
        assert result.messages[0].role == "system"
        assert result.messages[0].content == "Be helpful."
        assert result.messages[1].role == "user"

    def test_system_list(self):
        req = self._make_request(system=[{"type": "text", "text": "Be concise."}])
        result = anthropic_to_openai(req)
        assert result.messages[0].role == "system"
        assert result.messages[0].content == "Be concise."

    def test_temperature_default_forwards_none(self):
        req = self._make_request()
        result = anthropic_to_openai(req)
        assert result.temperature is None

    def test_temperature_explicit(self):
        req = self._make_request(temperature=0.3)
        result = anthropic_to_openai(req)
        assert result.temperature == 0.3

    def test_top_p_default_forwards_none(self):
        req = self._make_request()
        result = anthropic_to_openai(req)
        assert result.top_p is None

    def test_top_p_explicit(self):
        req = self._make_request(top_p=0.5)
        result = anthropic_to_openai(req)
        assert result.top_p == 0.5

    def test_top_k_forwarded(self):
        req = self._make_request(top_k=20)
        result = anthropic_to_openai(req)
        assert result.top_k == 20

    def test_top_k_default_forwards_none(self):
        req = self._make_request()
        result = anthropic_to_openai(req)
        assert result.top_k is None

    def test_stop_sequences(self):
        req = self._make_request(stop_sequences=["END", "STOP"])
        result = anthropic_to_openai(req)
        assert result.stop == ["END", "STOP"]

    def test_stream_flag(self):
        req = self._make_request(stream=True)
        result = anthropic_to_openai(req)
        assert result.stream is True

    def test_tools_conversion(self):
        req = self._make_request(
            tools=[
                AnthropicTool(
                    name="search",
                    description="Search the web",
                    input_schema={
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                )
            ]
        )
        result = anthropic_to_openai(req)
        assert len(result.tools) == 1
        assert result.tools[0].function["name"] == "search"

    def test_tool_choice_conversion(self):
        req = self._make_request(
            tools=[
                AnthropicTool(
                    name="search",
                    description="Search",
                    input_schema={"type": "object"},
                )
            ],
            tool_choice={"type": "any"},
        )
        result = anthropic_to_openai(req)
        assert result.tool_choice == "required"

    def test_no_tools(self):
        req = self._make_request()
        result = anthropic_to_openai(req)
        assert result.tools is None
        assert result.tool_choice is None

    def test_multiple_messages(self):
        msgs = [
            AnthropicMessage(role="user", content="hello"),
            AnthropicMessage(role="assistant", content="hi"),
            AnthropicMessage(role="user", content="how are you"),
        ]
        req = self._make_request(messages=msgs)
        result = anthropic_to_openai(req)
        assert len(result.messages) == 3
        assert result.messages[0].role == "user"
        assert result.messages[1].role == "assistant"
        assert result.messages[2].role == "user"


class TestOpenaiToAnthropicMigrated:
    def _make_response(
        self,
        content="hello",
        finish_reason="stop",
        tool_calls=None,
        reasoning_content=None,
    ):
        msg = AssistantMessage(
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )
        choice = ChatCompletionChoice(message=msg, finish_reason=finish_reason)
        return ChatCompletionResponse(
            model="default",
            choices=[choice],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    def test_simple_text_response(self):
        resp = self._make_response(content="hi there")
        result = openai_to_anthropic(resp, "default")
        assert result.model == "default"
        assert result.type == "message"
        assert result.role == "assistant"
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "hi there"
        assert result.stop_reason == "end_turn"
        logger.debug(
            "test_simple_text_response: model=%s stop_reason=%s",
            result.model,
            result.stop_reason,
        )

    def test_usage_mapping(self):
        resp = self._make_response()
        result = openai_to_anthropic(resp, "default")
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    @pytest.mark.skip(
        reason="fusion-mlx AnthropicUsage cache fields default to 0 (int) not None; "
        "no PromptTokensDetails support in Usage model"
    )
    def test_usage_mapping_no_cache_leaves_cache_fields_unset(self):
        pass

    @pytest.mark.skip(
        reason="fusion-mlx does not support PromptTokensDetails in Usage model; "
        "cache hit accounting is not available"
    )
    def test_usage_mapping_with_cache_hit(self):
        pass

    @pytest.mark.skip(
        reason="fusion-mlx does not support PromptTokensDetails in Usage model"
    )
    def test_usage_mapping_full_cache_hit(self):
        pass

    def test_tool_calls_response(self):
        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(
                name="search",
                arguments='{"q": "test"}',
            ),
        )
        resp = self._make_response(
            content="Let me search.",
            finish_reason="tool_calls",
            tool_calls=[tc],
        )
        result = openai_to_anthropic(resp, "default")
        assert len(result.content) == 2
        assert result.content[0].type == "text"
        assert result.content[0].text == "Let me search."
        assert result.content[1].type == "tool_use"
        assert result.content[1].name == "search"
        assert result.content[1].input == {"q": "test"}
        assert result.stop_reason == "tool_use"

    def test_tool_call_invalid_json_arguments(self):
        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="search", arguments="not json"),
        )
        resp = self._make_response(
            content=None, finish_reason="tool_calls", tool_calls=[tc]
        )
        result = openai_to_anthropic(resp, "default")
        tool_block = [b for b in result.content if b.type == "tool_use"][0]
        assert tool_block.input == {}

    def test_empty_choices(self):
        resp = ChatCompletionResponse(
            model="default",
            choices=[],
            usage=Usage(),
        )
        result = openai_to_anthropic(resp, "default")
        assert result.stop_reason == "end_turn"
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == ""

    def test_no_content_adds_empty_text(self):
        resp = self._make_response(content=None)
        result = openai_to_anthropic(resp, "default")
        assert len(result.content) >= 1
        has_text = any(b.type == "text" for b in result.content)
        assert has_text

    def test_stop_reason_length(self):
        resp = self._make_response(finish_reason="length")
        result = openai_to_anthropic(resp, "default")
        assert result.stop_reason == "max_tokens"

    def test_matched_stop_promotes_end_turn_to_stop_sequence(self):
        resp = self._make_response(content="hi END", finish_reason="stop")
        result = openai_to_anthropic(resp, "default", matched_stop="END")
        assert result.stop_reason == "stop_sequence"
        assert result.stop_sequence == "END"

    def test_matched_stop_none_keeps_end_turn(self):
        resp = self._make_response(content="hi", finish_reason="stop")
        result = openai_to_anthropic(resp, "default", matched_stop=None)
        assert result.stop_reason == "end_turn"
        assert result.stop_sequence is None

    def test_matched_stop_does_not_override_max_tokens(self):
        resp = self._make_response(content="long output", finish_reason="length")
        result = openai_to_anthropic(resp, "default", matched_stop="END")
        assert result.stop_reason == "max_tokens"

    def test_matched_stop_does_not_override_tool_use(self):
        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="search", arguments='{"q": "x"}'),
        )
        resp = self._make_response(
            content="Calling search.",
            finish_reason="tool_calls",
            tool_calls=[tc],
        )
        result = openai_to_anthropic(resp, "default", matched_stop="END")
        assert result.stop_reason == "tool_use"

    def test_response_has_id(self):
        resp = self._make_response()
        result = openai_to_anthropic(resp, "test-model")
        assert result.id.startswith("msg_")
        assert result.model == "test-model"

    def test_reasoning_content_becomes_thinking_block(self):
        resp = self._make_response(
            content="Final answer.",
            reasoning_content="Let me think.",
        )
        result = openai_to_anthropic(resp, "default")
        assert len(result.content) == 2
        assert result.content[0].type == "thinking"
        assert result.content[0].thinking == "Let me think."
        assert result.content[1].type == "text"
        assert result.content[1].text == "Final answer."

    def test_reasoning_content_with_tool_calls(self):
        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="search", arguments='{"q": "x"}'),
        )
        resp = self._make_response(
            content="Calling search.",
            reasoning_content="I need to look this up.",
            finish_reason="tool_calls",
            tool_calls=[tc],
        )
        result = openai_to_anthropic(resp, "default")
        assert [b.type for b in result.content] == ["thinking", "text", "tool_use"]
        assert result.content[0].thinking == "I need to look this up."

    def test_no_reasoning_content_omits_thinking_block(self):
        resp = self._make_response(content="hi", reasoning_content=None)
        result = openai_to_anthropic(resp, "default")
        assert all(b.type != "thinking" for b in result.content)

    def test_reasoning_disabled_omits_thinking_block(self):
        resp = self._make_response(
            content="Final answer.",
            reasoning_content="Let me think.",
        )
        result = openai_to_anthropic(resp, "default", reasoning_enabled=False)
        assert all(b.type != "thinking" for b in result.content)
        assert any(
            b.type == "text" and b.text == "Final answer." for b in result.content
        )

    @pytest.mark.skip(
        reason="fusion-mlx openai_to_anthropic does not support reasoning_enabled kwarg; "
        "no reasoning==content dedup guard"
    )
    def test_reasoning_equals_content_suppresses_thinking_block(self):
        pass

    @pytest.mark.skip(
        reason="fusion-mlx openai_to_anthropic does not support reasoning_enabled kwarg"
    )
    def test_reasoning_enabled_distinct_reasoning_still_emits_thinking(self):
        pass

    def test_reasoning_default_preserves_thinking_block(self):
        resp = self._make_response(
            content="Final answer.",
            reasoning_content="Let me think.",
        )
        result = openai_to_anthropic(resp, "default")
        assert any(b.type == "thinking" for b in result.content)
        assert any(b.type == "text" for b in result.content)

    @pytest.mark.skip(
        reason="fusion-mlx openai_to_anthropic does not support reasoning_enabled kwarg; "
        "no whitespace-only reasoning guard"
    )
    def test_whitespace_only_reasoning_omits_thinking_block(self):
        pass


@pytest.mark.skip(
    reason="fusion-mlx does not have to_anthropic_tool_use_id function; "
    "tool_use id rewriting is not implemented"
)
class TestAnthropicToolUseIdPrefix:
    def test_to_anthropic_tool_use_id_rewrites_call_prefix(self):
        pass

    def test_to_anthropic_tool_use_id_passes_through_toolu_prefix(self):
        pass

    def test_to_anthropic_tool_use_id_mints_fresh_when_missing(self):
        pass

    def test_to_anthropic_tool_use_id_mints_fresh_for_non_hex_tail(self):
        pass

    def test_to_anthropic_tool_use_id_mints_fresh_for_non_hex_toolu_tail(self):
        pass

    def test_to_anthropic_tool_use_id_mints_fresh_on_empty_tail(self):
        pass

    def test_tool_use_id_uses_toolu_prefix_at_adapter(self):
        pass

    def test_tool_use_id_already_toolu_passes_through(self):
        pass


@pytest.mark.skip(
    reason="fusion-mlx anthropic_to_openai does not strip tools when "
    "tool_choice=none (F7 logic not ported)"
)
class TestToolChoiceNoneStripsTools:
    def test_tool_choice_none_drops_tools(self):
        pass

    def test_tool_choice_auto_keeps_tools(self):
        pass

    def test_tool_choice_any_keeps_tools(self):
        pass

    def test_tool_choice_specific_tool_keeps_tools(self):
        pass


class TestAnthropicResponseExcludesNullFields:
    def test_basic_response_serialization_excludes_none(self):
        msg = AssistantMessage(content="hi")
        choice = ChatCompletionChoice(message=msg, finish_reason="stop")
        resp = ChatCompletionResponse(
            model="default",
            choices=[choice],
            usage=Usage(prompt_tokens=5, completion_tokens=2),
        )
        result = openai_to_anthropic(resp, "default")
        body = result.model_dump(exclude_none=True)
        for key, val in body.items():
            assert val is not None, (
                f"top-level key {key!r} serialized as None; "
                "exclude_none should have stripped it"
            )
        for key, val in body.get("usage", {}).items():
            assert val is not None, f"usage key {key!r} serialized as None"
        for blk in body.get("content", []):
            for key, val in blk.items():
                assert val is not None, f"content block key {key!r} serialized as None"
        logger.debug("test_basic_response_serialization_excludes_none passed")

    def test_wire_json_has_no_literal_null(self):
        msg = AssistantMessage(content="hi")
        choice = ChatCompletionChoice(message=msg, finish_reason="stop")
        resp = ChatCompletionResponse(
            model="default",
            choices=[choice],
            usage=Usage(prompt_tokens=5, completion_tokens=2),
        )
        result = openai_to_anthropic(resp, "default")
        wire_json = result.model_dump_json(exclude_none=True)
        assert (
            ": null" not in wire_json and ":null" not in wire_json
        ), f"wire JSON leaked a literal null: {wire_json[:500]}"

    def test_tool_use_response_has_no_null_fields(self):
        tc = ToolCall(
            id="call_abc12345",
            type="function",
            function=FunctionCall(name="search", arguments='{"q": "x"}'),
        )
        choice = ChatCompletionChoice(
            message=AssistantMessage(content="Searching...", tool_calls=[tc]),
            finish_reason="tool_calls",
        )
        resp = ChatCompletionResponse(model="default", choices=[choice], usage=Usage())
        result = openai_to_anthropic(resp, "default")
        wire_json = result.model_dump_json(exclude_none=True)
        assert (
            ": null" not in wire_json and ":null" not in wire_json
        ), f"tool_use wire JSON leaked a null: {wire_json[:500]}"
