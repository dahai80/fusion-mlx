# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from fusion_mlx.api.utils import extract_multimodal_content
from fusion_mlx.tool_parsers import (
    AutoToolParser,
    DeepSeekToolParser,
    DeepSeekV3ToolParser,
    DeepSeekV31ToolParser,
    FunctionaryToolParser,
    Gemma4ToolParser,
    Glm47ToolParser,
    GraniteToolParser,
    HarmonyToolParser,
    HermesToolParser,
    KimiToolParser,
    LlamaToolParser,
    MiniMaxToolParser,
    MistralToolParser,
    NemotronToolParser,
    Qwen3CoderToolParser,
    QwenToolParser,
    SeedOssToolParser,
    ToolParserManager,
    UiTarsToolParser,
    xLAMToolParser,
)

logger = logging.getLogger(__name__)


class TestNativeToolFormatCapability:

    def test_parsers_with_native_support(self):
        native_parsers = [
            MistralToolParser,
            LlamaToolParser,
            DeepSeekToolParser,
            DeepSeekV3ToolParser,
            DeepSeekV31ToolParser,
            GraniteToolParser,
            FunctionaryToolParser,
            KimiToolParser,
            HermesToolParser,
            HarmonyToolParser,
            Glm47ToolParser,
            Qwen3CoderToolParser,
            SeedOssToolParser,
        ]
        for parser_cls in native_parsers:
            assert (
                parser_cls.SUPPORTS_NATIVE_TOOL_FORMAT is True
            ), f"{parser_cls.__name__} should support native format"
            assert (
                parser_cls.supports_native_format() is True
            ), f"{parser_cls.__name__}.supports_native_format() should return True"

    def test_parsers_without_native_support(self):
        non_native_parsers = [
            QwenToolParser,
            NemotronToolParser,
            xLAMToolParser,
            AutoToolParser,
            Gemma4ToolParser,
            MiniMaxToolParser,
            UiTarsToolParser,
        ]
        for parser_cls in non_native_parsers:
            assert (
                parser_cls.SUPPORTS_NATIVE_TOOL_FORMAT is False
            ), f"{parser_cls.__name__} should not support native format"
            assert (
                parser_cls.supports_native_format() is False
            ), f"{parser_cls.__name__}.supports_native_format() should return False"

    def test_via_manager(self):
        for name in [
            "mistral",
            "llama",
            "deepseek",
            "deepseek_v3",
            "deepseek_v31",
            "granite",
            "functionary",
            "kimi",
            "hermes",
            "harmony",
            "glm47",
            "glm4",
            "qwen3_coder_xml",
            "seed_oss",
        ]:
            parser_cls = ToolParserManager.get_tool_parser(name)
            assert (
                parser_cls.supports_native_format() is True
            ), f"Parser '{name}' should support native format"

        for name in [
            "qwen",
            "nemotron",
            "xlam",
            "auto",
            "gemma4",
            "minimax",
            "ui_tars",
        ]:
            parser_cls = ToolParserManager.get_tool_parser(name)
            assert (
                parser_cls.supports_native_format() is False
            ), f"Parser '{name}' should not support native format"


class TestExtractMultimodalContentNativeFormat:

    @pytest.fixture
    def messages_with_tool_calls(self):
        return [
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "72°F and sunny",
            },
            {"role": "user", "content": "Thanks!"},
        ]

    def test_default_converts_to_text(self, messages_with_tool_calls):
        processed = extract_multimodal_content(messages_with_tool_calls)

        assert len(processed) == 4

        assert processed[0]["role"] == "user"
        assert processed[0]["content"] == "What is the weather in Paris?"

        assert processed[1]["role"] == "assistant"
        assert "[Calling tool: get_weather" in processed[1]["content"]
        assert "tool_calls" not in processed[1]

        assert processed[2]["role"] == "user"
        assert "[Tool Result (call_abc123)]" in processed[2]["content"]
        assert "72°F and sunny" in processed[2]["content"]

        assert processed[3]["role"] == "user"
        assert processed[3]["content"] == "Thanks!"

    @pytest.mark.skip(reason="preserve_native_format param not available in fusion_mlx")
    def test_preserve_native_format_true(self, messages_with_tool_calls):
        processed, images, videos = extract_multimodal_content(
            messages_with_tool_calls, preserve_native_format=True
        )

        assert len(processed) == 4

        assert processed[0]["role"] == "user"
        assert processed[0]["content"] == "What is the weather in Paris?"

        assert processed[1]["role"] == "assistant"
        assert "tool_calls" in processed[1]
        assert len(processed[1]["tool_calls"]) == 1
        assert processed[1]["tool_calls"][0]["id"] == "call_abc123"
        assert processed[1]["tool_calls"][0]["function"]["name"] == "get_weather"

        assert processed[2]["role"] == "tool"
        assert processed[2]["tool_call_id"] == "call_abc123"
        assert processed[2]["content"] == "72°F and sunny"

        assert processed[3]["role"] == "user"
        assert processed[3]["content"] == "Thanks!"

    def test_empty_tool_call_id(self):
        messages = [
            {"role": "tool", "content": "result without id"},
        ]

        processed = extract_multimodal_content(messages)
        assert processed[0]["role"] == "user"
        assert "[Tool Result ()]" in processed[0]["content"]

    @pytest.mark.skip(reason="preserve_native_format param not available in fusion_mlx")
    def test_empty_tool_call_id_native_mode(self):
        messages = [
            {"role": "tool", "content": "result without id"},
        ]
        processed, _, _ = extract_multimodal_content(
            messages, preserve_native_format=True
        )
        assert processed[0]["role"] == "tool"
        assert processed[0]["tool_call_id"] == ""
        assert processed[0]["content"] == "result without id"

    @pytest.mark.skip(reason="preserve_native_format param not available in fusion_mlx")
    def test_multiple_tool_calls(self):
        messages = [
            {"role": "user", "content": "Get weather and time"},
            {
                "role": "assistant",
                "content": "I'll check both.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
            {"role": "tool", "tool_call_id": "call_2", "content": "3:00 PM"},
        ]

        processed, _, _ = extract_multimodal_content(
            messages, preserve_native_format=True
        )

        assert len(processed) == 4
        assert len(processed[1]["tool_calls"]) == 2
        assert processed[2]["role"] == "tool"
        assert processed[2]["tool_call_id"] == "call_1"
        assert processed[3]["role"] == "tool"
        assert processed[3]["tool_call_id"] == "call_2"

    def test_mixed_content_preserved(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        processed = extract_multimodal_content(messages)

        assert len(processed) == 3
        assert processed[0]["role"] == "system"
        assert processed[1]["role"] == "user"
        assert processed[2]["role"] == "assistant"

    def test_assistant_with_content_and_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": "Let me check that for you.",
                "tool_calls": [
                    {
                        "id": "call_x",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "test"}'},
                    }
                ],
            }
        ]

        processed = extract_multimodal_content(messages)
        assert "Let me check that for you." in processed[0]["content"]
        assert "[Calling tool: search" in processed[0]["content"]


class TestEdgeCases:

    @pytest.mark.skip(reason="preserve_native_format param not available in fusion_mlx")
    def test_none_content_in_tool_message(self):
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": None},
        ]

        processed, _, _ = extract_multimodal_content(
            messages, preserve_native_format=True
        )
        assert processed[0]["content"] == ""

    @pytest.mark.skip(reason="preserve_native_format param not available in fusion_mlx")
    def test_pydantic_v2_model_tool_calls(self):
        class MockToolCallV2:
            def model_dump(self):
                return {
                    "id": "call_v2",
                    "type": "function",
                    "function": {"name": "v2_fn", "arguments": "{}"},
                }

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [MockToolCallV2()],
            }
        ]

        processed, _, _ = extract_multimodal_content(
            messages, preserve_native_format=True
        )
        assert processed[0]["tool_calls"][0]["id"] == "call_v2"
        assert processed[0]["tool_calls"][0]["function"]["name"] == "v2_fn"

    @pytest.mark.skip(reason="preserve_native_format param not available in fusion_mlx")
    def test_pydantic_v1_model_tool_calls(self):
        class MockToolCallV1:
            def dict(self):
                return {
                    "id": "call_v1",
                    "type": "function",
                    "function": {"name": "v1_fn", "arguments": "{}"},
                }

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [MockToolCallV1()],
            }
        ]

        processed, _, _ = extract_multimodal_content(
            messages, preserve_native_format=True
        )
        assert processed[0]["tool_calls"][0]["id"] == "call_v1"
        assert processed[0]["tool_calls"][0]["function"]["name"] == "v1_fn"

    @pytest.mark.skip(reason="preserve_native_format param not available in fusion_mlx")
    def test_images_and_videos_extracted_with_native_format(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://example.com/img.jpg"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Analysis result"},
        ]

        processed, images, videos = extract_multimodal_content(
            messages, preserve_native_format=True
        )

        assert len(images) == 1
        assert images[0] == "http://example.com/img.jpg"
        assert processed[1]["role"] == "tool"


class TestDecodeInlineToolCallArguments:

    @pytest.mark.skip(
        reason="decode_inline_tool_call_arguments not available in fusion_mlx.api.utils"
    )
    def test_string_arguments_decoded_to_dict(self):
        from fusion_mlx.api.utils import decode_inline_tool_call_arguments

        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris", "unit": "C"}',
                        },
                    }
                ],
            }
        ]
        decode_inline_tool_call_arguments(messages)
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == {
            "city": "Paris",
            "unit": "C",
        }

    @pytest.mark.skip(
        reason="decode_inline_tool_call_arguments not available in fusion_mlx.api.utils"
    )
    def test_dict_arguments_left_alone(self):
        from fusion_mlx.api.utils import decode_inline_tool_call_arguments

        original = {"city": "Paris"}
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "x", "arguments": original}}],
            }
        ]
        decode_inline_tool_call_arguments(messages)
        assert messages[0]["tool_calls"][0]["function"]["arguments"] is original

    @pytest.mark.skip(
        reason="decode_inline_tool_call_arguments not available in fusion_mlx.api.utils"
    )
    def test_malformed_json_left_as_string(self):
        from fusion_mlx.api.utils import decode_inline_tool_call_arguments

        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "x", "arguments": "not json"}}],
            }
        ]
        decode_inline_tool_call_arguments(messages)
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == "not json"

    @pytest.mark.skip(
        reason="decode_inline_tool_call_arguments not available in fusion_mlx.api.utils"
    )
    def test_messages_without_tool_calls_unchanged(self):
        from fusion_mlx.api.utils import decode_inline_tool_call_arguments

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "tool", "tool_call_id": "x", "content": "result"},
        ]
        decode_inline_tool_call_arguments(messages)
        assert messages == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "tool", "tool_call_id": "x", "content": "result"},
        ]

    @pytest.mark.skip(
        reason="decode_inline_tool_call_arguments not available in fusion_mlx.api.utils"
    )
    def test_multiple_tool_calls_each_decoded(self):
        from fusion_mlx.api.utils import decode_inline_tool_call_arguments

        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "a", "arguments": '{"x": 1}'}},
                    {"function": {"name": "b", "arguments": '{"y": 2}'}},
                ],
            }
        ]
        decode_inline_tool_call_arguments(messages)
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == {"x": 1}
        assert messages[0]["tool_calls"][1]["function"]["arguments"] == {"y": 2}
