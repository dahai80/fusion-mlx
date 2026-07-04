# SPDX-License-Identifier: Apache-2.0
import logging

import pytest

from fusion_mlx.api.anthropic_models import (
    AnthropicMessage,
    AnthropicTool,
    ContentBlockDocument,
    ContentBlockText,
    ContentBlockThinking,
    ContentBlockToolResult,
    ContentBlockToolUse,
    MessagesRequest,
    SystemContent,
)
from fusion_mlx.api.anthropic_utils import (
    convert_anthropic_to_internal,
    convert_anthropic_tools_to_internal,
    convert_internal_to_anthropic_response,
    create_content_block_start_event,
    create_content_block_stop_event,
    create_error_event,
    create_input_json_delta_event,
    create_message_delta_event,
    create_message_start_event,
    create_message_stop_event,
    create_ping_event,
    create_text_delta_event,
    format_sse_event,
    map_finish_reason_to_stop_reason,
)
from fusion_mlx.api.openai_models import ContentPart, FunctionCall, Message, ToolCall
from fusion_mlx.api.utils import (
    SPECIAL_TOKENS_PATTERN,
    _chat_template_supports_tool_role,
    _consolidate_system_messages,
    _drop_void_assistant_messages,
    _extract_multimodal_content_list,
    _merge_consecutive_roles,
    clean_output_text,
    detect_and_strip_partial,
    extract_harmony_messages,
    extract_multimodal_content,
    extract_text_content,
)


class TestCleanOutputText:

    def test_clean_empty_text(self):
        result = clean_output_text("")
        assert result == ""

    def test_clean_none_text(self):
        result = clean_output_text(None)
        assert result is None

    def test_clean_text_no_special_tokens(self):
        result = clean_output_text("Hello, world!")
        assert result == "Hello, world!"

    def test_clean_im_end_token(self):
        result = clean_output_text("Hello<|im_end|>")
        assert result == "Hello"

    def test_clean_im_start_token(self):
        result = clean_output_text("<|im_start|>Hello")
        assert result == "Hello"

    def test_clean_endoftext_token(self):
        result = clean_output_text("Response</s>")
        assert result == "Response"

    def test_clean_eot_id_token(self):
        result = clean_output_text("Text<|eot_id|>")
        assert result == "Text"

    def test_clean_end_token(self):
        result = clean_output_text("Content<|end|>")
        assert result == "Content"

    def test_clean_header_tokens(self):
        result = clean_output_text("<|start_header_id|>assistant<|end_header_id|>Hello")
        assert result == "assistantHello"

    def test_clean_eos_bos_tokens(self):
        result = clean_output_text("<s>Hello</s>")
        assert result == "Hello"

    def test_clean_pad_token(self):
        result = clean_output_text("Hello<pad>World")
        assert result == "HelloWorld"

    def test_clean_bracket_tokens(self):
        result = clean_output_text("[CLS]Hello[SEP]World[PAD]")
        assert result == "HelloWorld"

    @pytest.mark.skip(
        reason="fusion-mlx SPECIAL_TOKENS_PATTERN lacks Gemma tokens "
        "(<eos>, <bos>, <end_of_turn>, <start_of_turn>)"
    )
    def test_clean_gemma_special_tokens(self):
        assert clean_output_text("answer<eos>") == "answer"
        assert clean_output_text("answer<end_of_turn>") == "answer"
        assert clean_output_text("<start_of_turn>hi") == "hi"
        assert clean_output_text("<bos>hello<eos>") == "hello"

    def test_clean_multiple_tokens(self):
        result = clean_output_text("<|im_start|>Hello<|im_end|>")
        assert result == "Hello"

    def test_removes_think_tags(self):
        result = clean_output_text("<think>reasoning</think>Answer")
        assert "<think>" not in result
        assert "</think>" not in result
        assert "reasoning" not in result
        assert result == "Answer"

    def test_removes_multiple_think_blocks(self):
        result = clean_output_text("<think>a</think><think>b</think>Text")
        assert "<think>" not in result
        assert result == "Text"

    def test_removes_partial_think_closing(self):
        result = clean_output_text("thinking content</think>Answer")
        assert "</think>" not in result
        assert result == "Answer"

    def test_removes_empty_think_blocks(self):
        result = clean_output_text("<think></think>Text")
        assert result == "Text"

    def test_preserves_text_without_think_tags(self):
        result = clean_output_text("Normal response text")
        assert result == "Normal response text"

    def test_removes_think_with_newlines(self):
        result = clean_output_text("<think>\nreasoning\nprocess\n</think>Answer")
        assert "<think>" not in result
        assert "reasoning" not in result
        assert result == "Answer"

    def test_clean_whitespace(self):
        result = clean_output_text("  Hello<|im_end|>  ")
        assert result == "Hello"


class TestSpecialTokensPattern:

    def test_pattern_matches_im_tokens(self):
        assert SPECIAL_TOKENS_PATTERN.search("<|im_end|>")
        assert SPECIAL_TOKENS_PATTERN.search("<|im_start|>")

    def test_pattern_matches_endoftext(self):
        assert SPECIAL_TOKENS_PATTERN.search("</s>")

    def test_pattern_matches_llama_tokens(self):
        assert SPECIAL_TOKENS_PATTERN.search("<|eot_id|>")
        assert SPECIAL_TOKENS_PATTERN.search("<|end|>")
        assert SPECIAL_TOKENS_PATTERN.search("<|start_header_id|>")
        assert SPECIAL_TOKENS_PATTERN.search("<|end_header_id|>")

    def test_pattern_matches_legacy_tokens(self):
        assert SPECIAL_TOKENS_PATTERN.search("</s>")
        assert SPECIAL_TOKENS_PATTERN.search("<s>")
        assert SPECIAL_TOKENS_PATTERN.search("<pad>")
        assert SPECIAL_TOKENS_PATTERN.search("[PAD]")
        assert SPECIAL_TOKENS_PATTERN.search("[SEP]")
        assert SPECIAL_TOKENS_PATTERN.search("[CLS]")


class TestExtractTextContent:

    def test_simple_text_message(self):
        messages = [Message(role="user", content="Hello")]
        result = extract_text_content(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_multiple_messages(self):
        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        result = extract_text_content(messages)
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"

    def test_content_array_message(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"},
                ],
            )
        ]
        result = extract_text_content(messages)
        assert len(result) == 1
        assert "Hello" in result[0]["content"]
        assert "World" in result[0]["content"]

    def test_content_array_with_pydantic(self):
        messages = [
            Message(
                role="user",
                content=[
                    ContentPart(type="text", text="Hello"),
                ],
            )
        ]
        result = extract_text_content(messages)
        assert "Hello" in result[0]["content"]
        assert isinstance(result[0]["content"], str)

    def test_none_content(self):
        messages = [Message(role="assistant", content=None)]
        result = extract_text_content(messages)
        assert len(result) == 0

    def test_none_content_non_assistant_preserved(self):
        messages = [Message(role="user", content=None)]
        result = extract_text_content(messages)
        assert len(result) == 1
        assert result[0]["content"] == ""

    def test_tool_response_message(self):
        messages = [
            Message(
                role="tool",
                content='{"result": "success"}',
                tool_call_id="call_123",
            )
        ]
        result = extract_text_content(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "call_123" in result[0]["content"]
        assert "success" in result[0]["content"]

    def test_tool_response_message_with_content_part_list(self):
        messages = [
            Message(
                role="tool",
                content=[ContentPart(type="text", text='{"result": "success"}')],
                tool_call_id="call_123",
            )
        ]
        result = extract_text_content(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "call_123" in result[0]["content"]
        assert "success" in result[0]["content"]
        assert isinstance(result[0]["content"], str)

    def test_tool_response_fallback_preserves_role_boundary(self):
        messages = [
            Message(role="user", content="Before"),
            Message(
                role="tool",
                content='{"result": "success"}',
                tool_call_id="call_123",
            ),
            Message(role="user", content="After"),
        ]
        result = extract_text_content(messages)
        assert len(result) == 3
        assert result[0]["content"] == "Before"
        assert "Tool Result" in result[1]["content"]
        assert result[2]["content"] == "After"

    def test_assistant_with_tool_calls(self):
        messages = [
            Message(
                role="assistant",
                content="Let me check.",
                tool_calls=[
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Tokyo"}',
                        }
                    }
                ],
            )
        ]
        result = extract_text_content(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "Let me check." in result[0]["content"]
        assert "get_weather" in result[0]["content"]

    def test_assistant_tool_call_fallback_preserves_role_boundary(self):
        messages = [
            Message(
                role="assistant",
                content="Let me check.",
                tool_calls=[
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Tokyo"}',
                        }
                    }
                ],
            ),
            Message(role="assistant", content="Done."),
        ]
        result = extract_text_content(messages)
        assert len(result) == 2
        assert "get_weather" in result[0]["content"]
        assert result[1]["content"] == "Done."

    def test_developer_role_normalized_to_system(self):
        messages = [
            Message(role="developer", content="You are a coding assistant."),
            Message(role="user", content="Hello"),
        ]
        result = extract_text_content(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a coding assistant."
        assert result[1]["role"] == "user"

    def test_assistant_tool_calls_with_content_array(self):
        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock(spec=[])
        mock_tokenizer.has_tool_calling = True

        messages = [
            Message(
                role="assistant",
                content=[
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "x", "name": "f", "input": {}},
                ],
                tool_calls=[{"function": {"name": "f", "arguments": "{}"}}],
            )
        ]
        result = extract_text_content(messages, tokenizer=mock_tokenizer)
        assert result[0]["content"] == "Let me check."
        assert "tool_use" not in str(result[0]["content"])

    def test_developer_role_in_harmony(self):
        messages = [
            Message(role="developer", content="You are a coding assistant."),
            Message(role="user", content="Hello"),
        ]
        result = extract_harmony_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a coding assistant."


class TestExtractTextContentReasoningReconstruction:

    def test_reasoning_and_content_merged_on_assistant(self):
        messages = [
            Message(role="assistant", reasoning_content="R", content="A"),
        ]
        result = extract_text_content(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "<think>\nR\n</think>\n\nA"

    def test_reasoning_with_none_content(self):
        messages = [
            Message(role="assistant", reasoning_content="R", content=None),
        ]
        result = extract_text_content(messages)
        assert len(result) == 1
        assert result[0]["content"] == "<think>\nR\n</think>\n\n"

    def test_reasoning_with_content_list(self):
        messages = [
            Message(
                role="assistant",
                reasoning_content="R",
                content=[{"type": "text", "text": "A"}],
            ),
        ]
        result = extract_text_content(messages)
        assert len(result) == 1
        assert result[0]["content"] == "<think>\nR\n</think>\n\nA"

    def test_reasoning_on_non_assistant_passthrough(self):
        messages = [
            Message(role="user", reasoning_content="R", content="A"),
        ]
        result = extract_text_content(messages)
        assert len(result) == 1
        assert result[0]["content"] == "A"

    def test_no_reasoning_content_passthrough(self):
        messages = [
            Message(role="assistant", content="A"),
        ]
        result = extract_text_content(messages)
        assert len(result) == 1
        assert result[0]["content"] == "A"


class TestExtractTextContentNativeReasoningContent:

    def test_native_mode_passes_reasoning_as_field(self):
        messages = [
            Message(role="assistant", reasoning_content="R", content="A"),
        ]
        result = extract_text_content(messages, native_reasoning_content=True)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "A"
        assert result[0]["reasoning_content"] == "R"
        assert "<think>" not in result[0]["content"]

    def test_native_mode_with_none_content(self):
        messages = [
            Message(role="assistant", reasoning_content="R", content=None),
        ]
        result = extract_text_content(messages, native_reasoning_content=True)
        assert len(result) == 1
        assert result[0]["reasoning_content"] == "R"

    def test_native_mode_with_list_content(self):
        messages = [
            Message(
                role="assistant",
                reasoning_content="R",
                content=[{"type": "text", "text": "A"}],
            ),
        ]
        result = extract_text_content(messages, native_reasoning_content=True)
        assert len(result) == 1
        assert result[0]["content"] == "A"
        assert result[0]["reasoning_content"] == "R"

    def test_native_mode_with_tool_calls(self):
        messages = [
            Message(
                role="assistant",
                reasoning_content="R",
                content="calling",
                tool_calls=[
                    {"id": "c1", "function": {"name": "fn", "arguments": "{}"}}
                ],
            ),
        ]

        class NativeToolTokenizer:
            has_tool_calling = True

        result = extract_text_content(
            messages,
            tokenizer=NativeToolTokenizer(),
            native_reasoning_content=True,
        )
        assert len(result) == 1
        assert result[0]["content"] == "calling"
        assert result[0]["reasoning_content"] == "R"
        assert result[0]["tool_calls"][0]["function"]["name"] == "fn"

    def test_native_mode_non_assistant_does_not_emit_field(self):
        messages = [
            Message(role="user", reasoning_content="R", content="A"),
        ]
        result = extract_text_content(messages, native_reasoning_content=True)
        assert len(result) == 1
        assert result[0]["content"] == "A"
        assert "reasoning_content" not in result[0]

    def test_native_mode_recovers_inline_thinking_from_history(self):
        messages = [
            Message(role="assistant", content="<think>\nR\n</think>\n\nA"),
        ]
        result = extract_text_content(messages, native_reasoning_content=True)
        assert len(result) == 1
        assert result[0]["content"] == "A"
        assert result[0]["reasoning_content"] == "R"
        assert "<think>" not in result[0]["content"]

    def test_native_mode_recovers_minimax_inline_thinking_from_history(self):
        messages = [
            Message(role="assistant", content="<mm:think>R</mm:think>A"),
        ]
        result = extract_text_content(messages, native_reasoning_content=True)
        assert len(result) == 1
        assert result[0]["content"] == "A"
        assert result[0]["reasoning_content"] == "R"


@pytest.mark.skip(reason="uses_native_reasoning_content not available in fusion-mlx")
class TestUsesNativeReasoningContent:
    pass


class TestConvertAnthropicToInternal:

    def test_simple_message(self):
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        result = convert_anthropic_to_internal(request)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_with_system_string(self):
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            system="Be helpful",
        )
        result = convert_anthropic_to_internal(request)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be helpful"
        assert result[1]["role"] == "user"

    def test_inline_system_position_can_be_deferred(self):
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(role="user", content="Hello"),
                AnthropicMessage(role="system", content="Cacheable tail note"),
            ],
        )
        result = convert_anthropic_to_internal(
            request,
            consolidate_system_messages=False,
        )
        assert [message["role"] for message in result] == ["user", "system"]
        assert result[1]["content"] == "Cacheable tail note"

    def test_content_blocks(self):
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        ContentBlockText(text="Hello"),
                        ContentBlockText(text="World"),
                    ],
                )
            ],
        )
        result = convert_anthropic_to_internal(request)
        assert "Hello" in result[0]["content"]
        assert "World" in result[0]["content"]

    def test_tool_use_block(self):
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        ContentBlockToolUse(
                            id="toolu_123",
                            name="get_weather",
                            input={"location": "Tokyo"},
                        )
                    ],
                )
            ],
        )
        result = convert_anthropic_to_internal(request)
        assert "get_weather" in result[0]["content"]

    def test_system_billing_header_filtered(self):
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            system=[
                SystemContent(
                    text="x-anthropic-billing-header: cc_version=2.1.37.3a3; cc_entrypoint=cli; cch=3217b;"
                ),
                SystemContent(
                    text="You are Claude Code.",
                    cache_control={"type": "ephemeral"},
                ),
                SystemContent(
                    text="Be helpful.",
                    cache_control={"type": "ephemeral"},
                ),
            ],
        )
        result = convert_anthropic_to_internal(request)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "x-anthropic-billing-header" not in result[0]["content"]
        assert "You are Claude Code." in result[0]["content"]
        assert "Be helpful." in result[0]["content"]

    def test_system_billing_header_only(self):
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            system=[
                SystemContent(
                    text="x-anthropic-billing-header: cc_version=2.1.37.3a3; cc_entrypoint=cli; cch=abc12;"
                ),
            ],
        )
        result = convert_anthropic_to_internal(request)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_tool_result_block(self):
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        ContentBlockToolResult(
                            tool_use_id="toolu_123",
                            content="The weather is sunny",
                        )
                    ],
                )
            ],
        )
        result = convert_anthropic_to_internal(request)
        assert "toolu_123" in result[0]["content"]
        assert "sunny" in result[0]["content"]

    def test_native_tool_calling_preserves_structured_tool_history(self):
        class NativeToolTokenizer:
            has_tool_calling = True

        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        ContentBlockText(text="Checking"),
                        ContentBlockToolUse(
                            id="toolu_123",
                            name="get_weather",
                            input={"location": "Tokyo"},
                        ),
                    ],
                ),
                AnthropicMessage(
                    role="user",
                    content=[
                        ContentBlockToolResult(
                            tool_use_id="toolu_123",
                            content="The weather is sunny",
                        )
                    ],
                ),
            ],
        )
        result = convert_anthropic_to_internal(
            request,
            tokenizer=NativeToolTokenizer(),
        )
        assert result[0]["role"] == "assistant"
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "toolu_123"
        assert result[1]["content"] == "The weather is sunny"

    def test_tool_result_with_image_preserve_images_nonnative(self):
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_img",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "iVBOR",
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": "screenshot.png",
                                },
                            ],
                        }
                    ],
                )
            ],
        )
        result = convert_anthropic_to_internal(request, preserve_images=True)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        image_parts = [p for p in content if p.get("type") == "image_url"]
        text_parts = [p for p in content if p.get("type") == "text"]
        assert len(image_parts) == 1
        assert "iVBOR" in image_parts[0]["image_url"]["url"]
        assert len(text_parts) == 1
        assert "toolu_img" in text_parts[0]["text"]

    def test_tool_result_with_image_no_preserve(self):
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_img",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "iVBOR",
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": "screenshot.png",
                                },
                            ],
                        }
                    ],
                )
            ],
        )
        result = convert_anthropic_to_internal(request, preserve_images=False)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, str)
        assert "screenshot.png" in content
        assert "iVBOR" not in content

    def test_tool_result_with_image_native_path(self):
        class NativeToolTokenizer:
            has_tool_calling = True

        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        ContentBlockToolUse(
                            id="toolu_img",
                            name="read_file",
                            input={"path": "/tmp/screenshot.png"},
                        ),
                    ],
                ),
                AnthropicMessage(
                    role="user",
                    content=[
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_img",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "iVBOR",
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": "screenshot.png",
                                },
                            ],
                        }
                    ],
                ),
            ],
        )
        result = convert_anthropic_to_internal(
            request,
            tokenizer=NativeToolTokenizer(),
            preserve_images=True,
        )
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert result[1]["content"] == "screenshot.png"
        assert result[2]["role"] == "user"
        content = result[2]["content"]
        assert isinstance(content, list)
        image_parts = [p for p in content if p.get("type") == "image_url"]
        assert len(image_parts) == 1
        assert "iVBOR" in image_parts[0]["image_url"]["url"]

    def test_document_block_text_plain(self):
        import base64

        text_data = base64.b64encode(b"Hello from document").decode()
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        ContentBlockDocument(
                            source={
                                "type": "base64",
                                "media_type": "text/plain",
                                "data": text_data,
                            },
                            title="notes.txt",
                        ),
                    ],
                ),
            ],
        )
        result = convert_anthropic_to_internal(request)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Hello from document" in result[0]["content"]
        assert "[Document: notes.txt]" in result[0]["content"]

    def test_document_block_pdf_placeholder(self):
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        ContentBlockDocument(
                            source={
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "JVBERi0xLjQ=",
                            },
                            title="manual.pdf",
                        ),
                    ],
                ),
            ],
        )
        result = convert_anthropic_to_internal(request)
        assert len(result) == 1
        content = result[0]["content"]
        assert "manual.pdf" in content
        assert "oMLX does not provide PDF parsing" in content

    def test_thinking_block_reconstructed_as_think_tag(self):
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        ContentBlockThinking(
                            type="thinking",
                            thinking="step by step",
                            signature="",
                        ),
                        ContentBlockText(text="Answer"),
                    ],
                ),
            ],
        )
        result = convert_anthropic_to_internal(request)
        assert len(result) == 1
        content = result[0]["content"]
        assert "<think>\nstep by step\n</think>" in content
        assert "Answer" in content
        assert content.index("<think>") < content.index("Answer")

    def test_multiple_thinking_blocks_preserve_source_order(self):
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        ContentBlockThinking(
                            type="thinking",
                            thinking="FIRST",
                            signature="",
                        ),
                        ContentBlockThinking(
                            type="thinking",
                            thinking="SECOND",
                            signature="",
                        ),
                        ContentBlockText(text="Answer"),
                    ],
                ),
            ],
        )
        result = convert_anthropic_to_internal(request)
        content = result[0]["content"]
        assert content.index("FIRST") < content.index("SECOND")
        assert content.index("SECOND") < content.index("Answer")

    def test_thinking_block_native_tool_calling_assistant(self):
        class NativeToolTokenizer:
            has_tool_calling = True

        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        ContentBlockThinking(
                            type="thinking",
                            thinking="deliberating",
                            signature="",
                        ),
                        ContentBlockText(text="Let me check."),
                        ContentBlockToolUse(
                            id="toolu_1",
                            name="get_weather",
                            input={"location": "Tokyo"},
                        ),
                    ],
                ),
            ],
        )
        result = convert_anthropic_to_internal(
            request,
            tokenizer=NativeToolTokenizer(),
        )
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        content = result[0]["content"]
        assert "<think>\ndeliberating\n</think>" in content
        assert "Let me check." in content

    def test_document_block_mixed_with_text(self):
        import base64

        text_data = base64.b64encode(b"Doc content here").decode()
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        ContentBlockText(text="Please read this:"),
                        ContentBlockDocument(
                            source={
                                "type": "base64",
                                "media_type": "text/plain",
                                "data": text_data,
                            },
                        ),
                    ],
                ),
            ],
        )
        result = convert_anthropic_to_internal(request)
        assert len(result) == 1
        content = result[0]["content"]
        assert "Please read this:" in content
        assert "Doc content here" in content


class TestConvertAnthropicToInternalNativeReasoning:

    def test_native_mode_thinking_becomes_reasoning_field(self):
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        ContentBlockThinking(
                            type="thinking",
                            thinking="step by step",
                            signature="",
                        ),
                        ContentBlockText(text="Answer"),
                    ],
                ),
            ],
        )
        result = convert_anthropic_to_internal(request, native_reasoning_content=True)
        assert len(result) == 1
        assert result[0]["content"] == "Answer"
        assert result[0]["reasoning_content"] == "step by step"
        assert "<think>" not in result[0]["content"]

    def test_native_mode_multiple_thinking_blocks_joined(self):
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        ContentBlockThinking(
                            type="thinking", thinking="FIRST", signature=""
                        ),
                        ContentBlockThinking(
                            type="thinking", thinking="SECOND", signature=""
                        ),
                        ContentBlockText(text="Answer"),
                    ],
                ),
            ],
        )
        result = convert_anthropic_to_internal(request, native_reasoning_content=True)
        assert result[0]["content"] == "Answer"
        assert result[0]["reasoning_content"] == "FIRST\nSECOND"

    def test_native_mode_tool_calling_assistant(self):
        class NativeToolTokenizer:
            has_tool_calling = True

        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        ContentBlockThinking(
                            type="thinking",
                            thinking="deliberating",
                            signature="",
                        ),
                        ContentBlockText(text="Let me check."),
                        ContentBlockToolUse(
                            id="toolu_1",
                            name="get_weather",
                            input={"location": "Tokyo"},
                        ),
                    ],
                ),
            ],
        )
        result = convert_anthropic_to_internal(
            request,
            tokenizer=NativeToolTokenizer(),
            native_reasoning_content=True,
        )
        assert len(result) == 1
        assert result[0]["content"] == "Let me check."
        assert result[0]["reasoning_content"] == "deliberating"
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert "<think>" not in result[0]["content"]

    def test_native_mode_no_thinking_no_field(self):
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[ContentBlockText(text="Just a reply")],
                ),
            ],
        )
        result = convert_anthropic_to_internal(request, native_reasoning_content=True)
        assert result[0]["content"] == "Just a reply"
        assert "reasoning_content" not in result[0]


class TestConvertAnthropicToolsToInternal:

    def test_none_tools(self):
        result = convert_anthropic_tools_to_internal(None)
        assert result is None

    def test_empty_tools(self):
        result = convert_anthropic_tools_to_internal([])
        assert result is None

    def test_single_tool(self):
        tools = [
            AnthropicTool(
                name="get_weather",
                description="Get weather info",
                input_schema={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            )
        ]
        result = convert_anthropic_tools_to_internal(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather info"
        assert "parameters" in result[0]["function"]

    def test_multiple_tools(self):
        tools = [
            AnthropicTool(name="tool1", input_schema={}),
            AnthropicTool(name="tool2", input_schema={}),
        ]
        result = convert_anthropic_tools_to_internal(tools)
        assert len(result) == 2

    def test_tool_as_dict(self):
        tools = [
            {
                "name": "search",
                "description": "Search for info",
                "input_schema": {"type": "object"},
            }
        ]
        result = convert_anthropic_tools_to_internal(tools)
        assert result[0]["function"]["name"] == "search"

    def test_drops_server_side_web_search(self):
        tools = [AnthropicTool(type="web_search_20250305", name="web_search")]
        result = convert_anthropic_tools_to_internal(tools)
        assert result is None

    def test_drops_server_side_code_execution(self):
        tools = [
            AnthropicTool(type="code_execution_20250825", name="code_execution"),
        ]
        result = convert_anthropic_tools_to_internal(tools)
        assert result is None

    @pytest.mark.parametrize(
        "tool_type,name",
        [
            ("bash_20250124", "bash"),
            ("text_editor_20250728", "str_replace_editor"),
            ("computer_20250124", "computer"),
        ],
    )
    def test_drops_bash_text_editor_computer(self, tool_type, name):
        tools = [AnthropicTool(type=tool_type, name=name)]
        result = convert_anthropic_tools_to_internal(tools)
        assert result is None

    def test_keeps_user_tools_drops_server_side(self):
        tools = [
            AnthropicTool(name="get_weather", input_schema={"type": "object"}),
            AnthropicTool(type="web_search_20250305", name="web_search"),
        ]
        result = convert_anthropic_tools_to_internal(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"

    def test_drop_logs_at_info(self, caplog):
        tools = [
            AnthropicTool(type="web_search_20250305", name="web_search"),
            AnthropicTool(type="code_execution_20250825", name="code_execution"),
        ]
        with caplog.at_level(logging.INFO, logger="fusion_mlx.api.anthropic_utils"):
            convert_anthropic_tools_to_internal(tools)
        joined = "\n".join(caplog.messages)
        assert "Dropped 2" in joined
        assert "web_search_20250305:web_search" in joined
        assert "code_execution_20250825:code_execution" in joined

    def test_unknown_type_prefix_is_treated_as_user_tool(self):
        tools = [
            AnthropicTool(
                name="custom",
                type="unknown_kind_v1",
                input_schema={"type": "object"},
            ),
        ]
        result = convert_anthropic_tools_to_internal(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "custom"
        assert result[0]["function"]["parameters"] == {"type": "object"}


class TestConvertInternalToAnthropicResponse:

    def test_basic_response(self):
        result = convert_internal_to_anthropic_response(
            text="Hello!",
            model="claude-3",
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop",
        )
        assert result.type == "message"
        assert result.role == "assistant"
        assert result.model == "claude-3"
        assert len(result.content) == 1
        assert result.content[0].text == "Hello!"
        assert result.stop_reason == "end_turn"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    def test_response_with_tool_calls(self):
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
        result = convert_internal_to_anthropic_response(
            text="",
            model="claude-3",
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="tool_calls",
            tool_calls=tool_calls,
        )
        assert result.stop_reason == "tool_use"
        tool_use_blocks = [c for c in result.content if c.type == "tool_use"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0].name == "get_weather"

    def test_response_empty_text(self):
        result = convert_internal_to_anthropic_response(
            text="",
            model="claude-3",
            prompt_tokens=0,
            completion_tokens=0,
            finish_reason="stop",
        )
        assert len(result.content) >= 1

    def test_no_cache_control_legacy_shape(self):
        result = convert_internal_to_anthropic_response(
            text="hi",
            model="claude-3",
            prompt_tokens=100,
            completion_tokens=5,
            finish_reason="stop",
            cached_tokens=40,
            prefix_cache_enabled=False,
        )
        assert result.usage.input_tokens == 100
        assert result.usage.cache_creation_input_tokens == 0
        assert result.usage.cache_read_input_tokens == 0

    def test_cache_control_cold_partitions_to_creation(self):
        result = convert_internal_to_anthropic_response(
            text="hi",
            model="claude-3",
            prompt_tokens=100,
            completion_tokens=5,
            finish_reason="stop",
            cached_tokens=0,
            prefix_cache_enabled=True,
        )
        assert result.usage.input_tokens == 0
        assert result.usage.cache_creation_input_tokens == 100
        assert result.usage.cache_read_input_tokens == 0

    def test_cache_control_warm_partitions_to_read(self):
        result = convert_internal_to_anthropic_response(
            text="hi",
            model="claude-3",
            prompt_tokens=100,
            completion_tokens=5,
            finish_reason="stop",
            cached_tokens=20,
            prefix_cache_enabled=True,
        )
        assert result.usage.input_tokens == 0
        assert result.usage.cache_creation_input_tokens == 80
        assert result.usage.cache_read_input_tokens == 20

    def test_usage_triple_is_disjoint_partition(self):
        for uses_cc in (True, False):
            for cached in (0, 25, 100, 200):
                result = convert_internal_to_anthropic_response(
                    text="hi",
                    model="claude-3",
                    prompt_tokens=100,
                    completion_tokens=5,
                    finish_reason="stop",
                    cached_tokens=cached,
                    prefix_cache_enabled=uses_cc,
                )
                u = result.usage
                assert (
                    u.input_tokens
                    + u.cache_creation_input_tokens
                    + u.cache_read_input_tokens
                    == 100
                ), (
                    f"partition broken at uses_cc={uses_cc}, cached={cached}: "
                    f"{u.input_tokens} + {u.cache_creation_input_tokens} + "
                    f"{u.cache_read_input_tokens} != 100"
                )
                assert u.cache_read_input_tokens <= 100


@pytest.mark.skip(reason="request_has_cache_control not available in fusion-mlx")
class TestRequestHasCacheControl:
    pass


class TestMapFinishReasonToStopReason:

    def test_stop_to_end_turn(self):
        result = map_finish_reason_to_stop_reason("stop", False)
        assert result == "end_turn"

    def test_length_to_max_tokens(self):
        result = map_finish_reason_to_stop_reason("length", False)
        assert result == "max_tokens"

    def test_tool_calls_to_tool_use(self):
        result = map_finish_reason_to_stop_reason("tool_calls", False)
        assert result == "tool_use"

    def test_has_tool_calls_overrides(self):
        result = map_finish_reason_to_stop_reason("stop", True)
        assert result == "tool_use"

    def test_none_reason(self):
        result = map_finish_reason_to_stop_reason(None, False)
        assert result is None

    def test_unknown_reason(self):
        result = map_finish_reason_to_stop_reason("unknown", False)
        assert result == "end_turn"


class TestSSEEventFormatters:

    def test_format_sse_event(self):
        result = format_sse_event("message_start", {"type": "message_start"})
        assert result.startswith("event: message_start\n")
        assert "data: " in result
        assert result.endswith("\n\n")

    def test_create_message_start_event(self):
        result = create_message_start_event("msg_123", "claude-3", input_tokens=10)
        assert "event: message_start" in result
        assert "msg_123" in result
        assert "claude-3" in result

    def test_create_content_block_start_event_text(self):
        result = create_content_block_start_event(0, "text")
        assert "event: content_block_start" in result
        assert '"index": 0' in result

    def test_create_content_block_start_event_tool_use(self):
        result = create_content_block_start_event(
            0, "tool_use", id="toolu_123", name="get_weather"
        )
        assert "event: content_block_start" in result
        assert "tool_use" in result

    def test_create_text_delta_event(self):
        result = create_text_delta_event(0, "Hello")
        assert "event: content_block_delta" in result
        assert "text_delta" in result
        assert "Hello" in result

    def test_create_input_json_delta_event(self):
        result = create_input_json_delta_event(0, '{"location":')
        assert "event: content_block_delta" in result
        assert "input_json_delta" in result

    def test_create_content_block_stop_event(self):
        result = create_content_block_stop_event(0)
        assert "event: content_block_stop" in result
        assert '"index": 0' in result

    def test_create_message_delta_event(self):
        result = create_message_delta_event("end_turn", 10)
        assert "event: message_delta" in result
        assert "end_turn" in result
        assert '"output_tokens": 10' in result

    def test_create_message_delta_event_with_input_tokens(self):
        result = create_message_delta_event("end_turn", 10, input_tokens=100)
        assert '"input_tokens": 100' in result

    def test_create_message_delta_event_cache_control_splits(self):
        result = create_message_delta_event(
            "end_turn",
            10,
            input_tokens=100,
            cached_tokens=30,
            prefix_cache_enabled=True,
        )
        assert '"input_tokens": 0' in result
        assert '"cache_creation_input_tokens": 70' in result
        assert '"cache_read_input_tokens": 30' in result

    def test_create_message_delta_event_no_cache_control_omits_cache_fields(self):
        result = create_message_delta_event(
            "end_turn",
            10,
            input_tokens=100,
            cached_tokens=30,
            prefix_cache_enabled=False,
        )
        assert '"input_tokens": 100' in result
        assert "cache_creation_input_tokens" not in result
        assert "cache_read_input_tokens" not in result

    def test_create_message_stop_event(self):
        result = create_message_stop_event()
        assert "event: message_stop" in result

    def test_create_ping_event(self):
        result = create_ping_event()
        assert "event: ping" in result

    def test_create_error_event(self):
        result = create_error_event("api_error", "Something went wrong")
        assert "event: error" in result
        assert "api_error" in result
        assert "Something went wrong" in result


class TestExtractHarmonyMessages:

    def test_simple_message(self):
        messages = [Message(role="user", content="Hello")]
        result = extract_harmony_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_tool_message_preserved(self):
        messages = [
            Message(
                role="tool",
                content='{"result": "success"}',
                tool_call_id="call_123",
            )
        ]
        result = extract_harmony_messages(messages)
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_123"

    def test_assistant_tool_calls_preserved(self):
        messages = [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Tokyo"}',
                        },
                    }
                ],
            )
        ]
        result = extract_harmony_messages(messages)
        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_tool_message_with_content_part_list(self):
        messages = [
            Message(
                role="tool",
                content=[ContentPart(type="text", text='{"result": "success"}')],
                tool_call_id="call_123",
            )
        ]
        result = extract_harmony_messages(messages)
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_123"
        assert not isinstance(result[0]["content"], list)

    def test_json_arguments_parsed(self):
        messages = [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "call_123",
                        "function": {
                            "name": "test",
                            "arguments": '{"key": "value"}',
                        },
                    }
                ],
            )
        ]
        result = extract_harmony_messages(messages)
        args = result[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, dict)
        assert args["key"] == "value"

    def test_tool_content_json_parsed(self):
        messages = [
            Message(
                role="tool",
                content='{"result": "success"}',
                tool_call_id="call_123",
            )
        ]
        result = extract_harmony_messages(messages)
        content = result[0]["content"]
        assert isinstance(content, dict)
        assert content["result"] == "success"

    def test_simple_dict_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = extract_harmony_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_tool_dict_message(self):
        messages = [
            {
                "role": "tool",
                "content": '{"result": "ok"}',
                "tool_call_id": "call_abc",
            }
        ]
        result = extract_harmony_messages(messages)
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_abc"
        assert isinstance(result[0]["content"], dict)

    def test_assistant_tool_calls_dict(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Tokyo"}',
                        },
                    }
                ],
            }
        ]
        result = extract_harmony_messages(messages)
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert isinstance(result[0]["tool_calls"][0]["function"]["arguments"], dict)

    def test_mixed_pydantic_and_dict_messages(self):
        messages = [
            Message(role="system", content="You are helpful."),
            {"role": "user", "content": "Hi"},
        ]
        result = extract_harmony_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"


class TestConsolidateSystemMessages:

    def test_no_system_messages(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = _consolidate_system_messages(msgs)
        assert result == msgs

    def test_system_already_first(self):
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = _consolidate_system_messages(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be helpful"

    def test_system_mid_conversation(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "How are you?"},
        ]
        result = _consolidate_system_messages(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be helpful"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Hello"

    def test_multiple_system_messages_merged(self):
        msgs = [
            {"role": "system", "content": "Instruction 1"},
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Instruction 2"},
        ]
        result = _consolidate_system_messages(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Instruction 1\n\nInstruction 2"
        assert result[1]["role"] == "user"

    def test_empty_system_content_skipped(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": ""},
            {"role": "system", "content": "Real instruction"},
        ]
        result = _consolidate_system_messages(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Real instruction"

    def test_all_empty_system_returns_original(self):
        msgs = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello"},
        ]
        result = _consolidate_system_messages(msgs)
        assert result == msgs

    def test_extract_text_content_developer_mid_conversation(self):
        messages = [
            Message(role="user", content="Hello"),
            Message(role="developer", content="New instructions"),
            Message(role="user", content="What now?"),
        ]
        result = extract_text_content(messages)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "New instructions"
        assert all(m["role"] != "system" for m in result[1:])

    def test_extract_text_content_preserves_tool_order(self):
        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Call tool"),
            Message(role="assistant", content="OK"),
            Message(role="system", content="Extra instruction"),
            Message(role="user", content="Continue"),
        ]
        result = extract_text_content(messages)
        assert result[0]["role"] == "system"
        assert "Be helpful" in result[0]["content"]
        assert "Extra instruction" in result[0]["content"]
        assert result[1]["role"] == "user"

    def test_system_message_with_list_content(self):
        msgs = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Be helpful"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            },
            {"role": "user", "content": "Hello"},
        ]
        result = _consolidate_system_messages(msgs)
        assert result[0]["role"] == "system"
        assert isinstance(result[0]["content"], str)
        assert "Be helpful" in result[0]["content"]


@pytest.mark.skip(
    reason="prepare_system_messages_for_template and chat_template_preserves_mid_system "
    "not available in fusion-mlx"
)
class TestPrepareSystemMessagesForTemplate:
    pass


class TestMergeConsecutiveRoles:

    def test_empty_list(self):
        assert _merge_consecutive_roles([]) == []

    def test_single_message(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    def test_consecutive_user_merged(self):
        msgs = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "First\n\nSecond"

    def test_preserve_role_boundary_skips_merge(self):
        msgs = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Tool", "_preserve_role_boundary": True},
            {"role": "user", "content": "Third"},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 3

    def test_three_consecutive_user_merged(self):
        msgs = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
            {"role": "user", "content": "Third"},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "First\n\nSecond\n\nThird"

    def test_consecutive_assistant_merged(self):
        msgs = [
            {"role": "assistant", "content": "Part 1"},
            {"role": "assistant", "content": "Part 2"},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "Part 1\n\nPart 2"

    def test_system_messages_not_merged(self):
        msgs = [
            {"role": "system", "content": "Instruction 1"},
            {"role": "system", "content": "Instruction 2"},
            {"role": "user", "content": "Hello"},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 3

    def test_tool_messages_not_merged(self):
        msgs = [
            {"role": "tool", "content": "Result 1", "tool_call_id": "a"},
            {"role": "tool", "content": "Result 2", "tool_call_id": "b"},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 2

    def test_alternating_roles_unchanged(self):
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 4

    def test_empty_content_merge(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": ""},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    def test_both_empty_content(self):
        msgs = [
            {"role": "user", "content": ""},
            {"role": "user", "content": ""},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        assert result[0]["content"] == ""

    def test_does_not_mutate_input(self):
        msgs = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
        ]
        original_first = msgs[0]["content"]
        _merge_consecutive_roles(msgs)
        assert msgs[0]["content"] == original_first
        assert len(msgs) == 2

    def test_merge_list_content_with_string(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            },
            {"role": "user", "content": "What do you think?"},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        types = [p["type"] for p in content]
        assert "image_url" in types
        assert "text" in types
        texts = [p["text"] for p in content if p["type"] == "text"]
        assert "Look at this" in texts
        assert "What do you think?" in texts

    def test_merge_string_with_list_content(self):
        msgs = [
            {"role": "user", "content": "Context text"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "See image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,def"},
                    },
                ],
            },
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 3

    def test_merge_two_list_contents(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,def"},
                    },
                ],
            },
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2

    def test_merge_empty_string_with_list_content(self):
        msgs = [
            {"role": "user", "content": ""},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            },
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)

    def test_extract_text_content_merges_consecutive_user(self):
        messages = [
            Message(role="user", content="Page content here"),
            Message(role="user", content="What is this about?"),
        ]
        result = extract_text_content(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Page content here" in result[0]["content"]
        assert "What is this about?" in result[0]["content"]

    def test_brave_leo_pattern(self):
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Here is some context: blah blah"),
            Message(role="user", content="What does this mean?"),
        ]
        result = extract_text_content(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert "blah blah" in result[1]["content"]
        assert "What does this mean?" in result[1]["content"]

    def test_extract_harmony_merges_consecutive_user(self):
        messages = [
            Message(role="user", content="First"),
            Message(role="user", content="Second"),
        ]
        result = extract_harmony_messages(messages)
        assert len(result) == 1
        assert result[0]["content"] == "First\n\nSecond"


class TestExtractMultimodalContent:

    def test_tool_message_with_content_part_list(self):
        messages = [
            Message(
                role="tool",
                content=[ContentPart(type="text", text='{"result": "success"}')],
                tool_call_id="call_123",
            )
        ]
        result = extract_multimodal_content(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "call_123" in result[0]["content"]
        assert "success" in result[0]["content"]
        assert isinstance(result[0]["content"], str)

    def test_converts_input_text_and_input_image(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "input_text", "text": "Describe this image"},
                    {"type": "input_image", "image_url": "/tmp/example.png"},
                ],
            )
        ]
        result = extract_multimodal_content(messages)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Describe this image"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "/tmp/example.png"

    def test_converts_input_image_dict_shape(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Analyze"},
                    {
                        "type": "input_image",
                        "image_url": {"url": "https://example.com/a.png"},
                    },
                ],
            )
        ]
        result = extract_multimodal_content(messages)
        content = result[0]["content"]
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "https://example.com/a.png"

    def test_normalizes_image_url_from_model_dump(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "text": None,
                        "image_url": {
                            "url": "data:image/png;base64,abc",
                            "detail": "auto",
                        },
                    },
                ],
            )
        ]
        result = extract_multimodal_content(messages)
        content = result[0]["content"]
        assert isinstance(content, list)
        img_part = content[1]
        assert img_part == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,abc"},
        }
        assert "text" not in img_part
        assert "detail" not in img_part.get("image_url", {})

    def test_normalizes_image_url_string_form(self):
        parts = _extract_multimodal_content_list(
            [
                {"type": "image_url", "image_url": "data:image/png;base64,abc"},
            ]
        )
        assert len(parts) == 1
        assert parts[0] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,abc"},
        }

    def test_image_url_missing_url_dropped(self):
        parts = _extract_multimodal_content_list(
            [
                {"type": "image_url", "image_url": None},
                {"type": "image_url"},
            ]
        )
        assert len(parts) == 0

    def test_input_audio_pass_through(self):
        parts = _extract_multimodal_content_list(
            [
                {
                    "type": "input_audio",
                    "input_audio": {"data": "abc", "format": "wav"},
                },
            ]
        )
        assert len(parts) == 1
        assert parts[0] == {
            "type": "input_audio",
            "input_audio": {"data": "abc", "format": "wav"},
        }

    def test_input_audio_non_dict_dropped(self):
        parts = _extract_multimodal_content_list(
            [
                {"type": "input_audio", "input_audio": None},
                {"type": "input_audio"},
            ]
        )
        assert len(parts) == 0

    def test_input_audio_preserved_with_image(self):
        parts = _extract_multimodal_content_list(
            [
                {"type": "text", "text": "Look and listen"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
                {
                    "type": "input_audio",
                    "input_audio": {"data": "xyz", "format": "mp3"},
                },
            ]
        )
        assert len(parts) == 3
        types = [p["type"] for p in parts]
        assert types == ["text", "image_url", "input_audio"]

    def test_input_audio_with_model_dump(self):
        from unittest.mock import MagicMock

        audio_part = MagicMock()
        audio_part.model_dump.return_value = {
            "type": "input_audio",
            "input_audio": {"data": "audio_data", "format": "wav"},
        }
        parts = _extract_multimodal_content_list([audio_part])
        assert len(parts) == 1
        assert parts[0]["type"] == "input_audio"
        assert parts[0]["input_audio"]["format"] == "wav"


class TestDetectAndStripPartial:

    def test_detects_partial_assistant(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "{", "partial": True},
        ]
        assert detect_and_strip_partial(messages) is True

    def test_ignores_partial_non_assistant(self):
        messages = [
            {"role": "user", "content": "Hello", "partial": True},
        ]
        assert detect_and_strip_partial(messages) is False

    def test_strips_partial_from_all_messages(self):
        messages = [
            {"role": "user", "content": "Hello", "partial": False},
            {"role": "assistant", "content": "{", "partial": True},
        ]
        detect_and_strip_partial(messages)
        for msg in messages:
            assert "partial" not in msg

    def test_empty_messages(self):
        assert detect_and_strip_partial([]) is False

    def test_no_partial_field(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        assert detect_and_strip_partial(messages) is False


class TestExtractTextContentPreservesNamePartial:

    def test_preserves_name_on_text_message(self):
        messages = [
            Message(role="assistant", content="Hello", name="Kimi"),
        ]
        result = extract_text_content(messages)
        assert result[0]["name"] == "Kimi"

    def test_preserves_partial_on_assistant(self):
        messages = [
            Message(role="assistant", content="{", partial=True),
        ]
        result = extract_text_content(messages)
        assert result[0].get("partial") is True

    def test_no_name_when_absent(self):
        messages = [
            Message(role="user", content="Hello"),
        ]
        result = extract_text_content(messages)
        assert "name" not in result[0]

    def test_no_partial_when_false(self):
        messages = [
            Message(role="user", content="Hello"),
        ]
        result = extract_text_content(messages)
        assert "partial" not in result[0]

    def test_preserves_name_on_tool_call_message(self):
        messages = [
            Message(
                role="assistant",
                content="Let me call a tool",
                name="Kimi",
                tool_calls=[
                    {"id": "1", "function": {"name": "search", "arguments": "{}"}}
                ],
            ),
        ]
        result = extract_text_content(messages)
        assert result[0].get("name") == "Kimi"

    def test_preserves_partial_on_tool_call_message(self):
        messages = [
            Message(
                role="assistant",
                content="Let me call a tool",
                partial=True,
                tool_calls=[
                    {"id": "1", "function": {"name": "search", "arguments": "{}"}}
                ],
            ),
        ]
        result = extract_text_content(messages)
        assert result[0].get("partial") is True

    def test_preserves_partial_on_tool_call_message_multimodal(self):
        messages = [
            Message(
                role="assistant",
                content="Let me call a tool",
                partial=True,
                tool_calls=[
                    {"id": "1", "function": {"name": "search", "arguments": "{}"}}
                ],
            ),
        ]
        result = extract_multimodal_content(messages)
        assert result[0].get("partial") is True

    def test_preserves_name_in_multimodal_extraction(self):
        messages = [
            Message(role="assistant", content="Hello", name="Kimi"),
        ]
        result = extract_multimodal_content(messages)
        assert result[0]["name"] == "Kimi"


class TestNameFieldSchemaAcceptance:

    def test_name_field_accepted_on_all_roles(self):
        msgs = [
            Message(
                role="system",
                content="This is a turn-based roleplaying session.",
            ),
            Message(
                role="user",
                content="*bangs on the door*",
                name="Arthur Dent",
            ),
            Message(
                role="assistant",
                content="*",
                name="Marvin the Paranoid Android",
                partial=True,
            ),
        ]
        assert msgs[1].name == "Arthur Dent"
        assert msgs[2].name == "Marvin the Paranoid Android"

    def test_name_field_survives_extraction_for_template(self):
        msgs = [
            Message(
                role="system",
                content="This is a turn-based roleplaying session.",
            ),
            Message(
                role="user",
                content="*bangs on the door*",
                name="Arthur Dent",
            ),
            Message(
                role="assistant",
                content="*",
                name="Marvin the Paranoid Android",
                partial=True,
            ),
        ]
        result = extract_text_content(msgs)
        assert result[1]["name"] == "Arthur Dent"
        assert result[2]["name"] == "Marvin the Paranoid Android"
        assert result[2]["partial"] is True

    def test_name_absent_when_not_provided(self):
        msgs = [Message(role="user", content="Hello")]
        result = extract_text_content(msgs)
        assert "name" not in result[0]


class TestDropVoidAssistantMessages:

    def test_drops_empty_content_no_tool_calls(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "Again"},
        ]
        result = _drop_void_assistant_messages(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "user"

    def test_drops_none_content_no_tool_calls(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "Again"},
        ]
        result = _drop_void_assistant_messages(msgs)
        assert len(result) == 2

    def test_keeps_assistant_with_content(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Thanks"},
        ]
        result = _drop_void_assistant_messages(msgs)
        assert len(result) == 3

    def test_keeps_assistant_with_tool_calls(self):
        msgs = [
            {"role": "user", "content": "List files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "function": {"name": "ls"}}],
            },
            {"role": "user", "content": "Thanks"},
        ]
        result = _drop_void_assistant_messages(msgs)
        assert len(result) == 3

    def test_preserves_other_roles(self):
        msgs = [
            {"role": "system", "content": ""},
            {"role": "user", "content": ""},
            {"role": "tool", "content": ""},
        ]
        result = _drop_void_assistant_messages(msgs)
        assert len(result) == 3

    def test_extract_text_content_drops_void_assistant(self):
        msgs = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content=None),
            Message(role="user", content="Tell me about this repo"),
        ]
        result = extract_text_content(msgs)
        assert all(m["role"] != "assistant" or m.get("content") for m in result)

    def test_void_drop_then_merge_consecutive_users(self):
        msgs = [
            Message(role="user", content="hello"),
            Message(role="assistant", content=None),
            Message(role="user", content="world"),
        ]
        result = extract_text_content(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "hello" in result[0]["content"]
        assert "world" in result[0]["content"]

    def test_multiple_void_assistants_merge_surrounding_users(self):
        msgs = [
            Message(role="user", content="a"),
            Message(role="assistant", content=None),
            Message(role="user", content="b"),
            Message(role="assistant", content="reply"),
            Message(role="user", content="c"),
            Message(role="assistant", content=None),
            Message(role="user", content="d"),
        ]
        result = extract_text_content(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert "a" in result[0]["content"] and "b" in result[0]["content"]
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "reply"
        assert result[2]["role"] == "user"
        assert "c" in result[2]["content"] and "d" in result[2]["content"]


class TestChatTemplateSupportsToolRole:

    def test_returns_true_when_has_tool_calling_set(self):
        class _Tok:
            has_tool_calling = True

        assert _chat_template_supports_tool_role(_Tok()) is True

    def test_returns_true_when_has_tool_calling_set_even_without_template(self):
        class _Tok:
            has_tool_calling = True
            chat_template = None

        assert _chat_template_supports_tool_role(_Tok()) is True

    def test_returns_true_for_template_with_tool_role_branch(self):
        template = (
            "{%- for msg in messages %}"
            '{%- if msg.role == "tool" %}<tool>{{ msg.content }}</tool>'
            '{%- elif msg.role == "assistant" and msg.tool_calls %}'
            "{%- for tc in msg.tool_calls %}<tool_call>{{ tc }}\n{%- endfor %}"
            "{%- endif %}"
            "{%- endfor %}"
        )

        class _Tok:
            has_tool_calling = False
            chat_template = template

        assert _chat_template_supports_tool_role(_Tok()) is True

    def test_returns_false_for_template_without_tool_role(self):
        template = (
            "{%- for msg in messages %}"
            '{%- if msg.role == "user" %}USER: {{ msg.content }}'
            '{%- elif msg.role == "assistant" %}AGENT: {{ msg.content }}'
            "{%- endif %}"
            "{%- endfor %}"
        )

        class _Tok:
            has_tool_calling = False
            chat_template = template

        assert _chat_template_supports_tool_role(_Tok()) is False

    def test_returns_false_when_only_tool_role_present(self):
        template = '{%- if msg.role == "tool" %}{{ msg.content }}{%- endif %}'

        class _Tok:
            has_tool_calling = False
            chat_template = template

        assert _chat_template_supports_tool_role(_Tok()) is False

    def test_returns_false_for_none_tokenizer(self):
        assert _chat_template_supports_tool_role(None) is False

    def test_returns_false_for_non_string_template(self):
        class _Tok:
            has_tool_calling = False
            chat_template = lambda *a, **k: ""  # noqa: E731

        assert _chat_template_supports_tool_role(_Tok()) is False


class TestToolResultWithToolAwareTokenizer:

    @staticmethod
    def _tool_aware_tokenizer():
        class _Tok:
            has_tool_calling = False
            chat_template = (
                '{%- if msg.role == "tool" %}{{ msg.content }}'
                "{%- elif msg.tool_calls %}{{ msg.tool_calls }}{%- endif %}"
            )

        return _Tok()

    def test_extract_text_content_preserves_tool_role(self):
        messages = [
            Message(
                role="tool",
                content='{"result": "ok"}',
                tool_call_id="call_xyz",
            )
        ]
        result = extract_text_content(messages, tokenizer=self._tool_aware_tokenizer())
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_xyz"
        assert result[0]["content"] == '{"result": "ok"}'

    def test_extract_multimodal_content_preserves_tool_role(self):
        messages = [
            Message(
                role="tool",
                content=[ContentPart(type="text", text='{"result": "ok"}')],
                tool_call_id="call_xyz",
            )
        ]
        result = extract_multimodal_content(
            messages, tokenizer=self._tool_aware_tokenizer()
        )
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_xyz"

    def test_assistant_tool_calls_kept_structured(self):
        messages = [
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "call_xyz",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Seoul"}',
                        },
                    }
                ],
            )
        ]
        result = extract_text_content(messages, tokenizer=self._tool_aware_tokenizer())
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result[0]["tool_calls"][0]["function"]["arguments"] == {"city": "Seoul"}
