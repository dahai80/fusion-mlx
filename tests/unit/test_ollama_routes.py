# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Ollama-compatible API routes."""

from fusion_mlx.api.ollama_routes import (
    OllamaChatMessage,
    OllamaChatRequest,
    OllamaGenerateRequest,
    _build_openai_messages_chat,
    _build_openai_messages_generate,
    _options_to_params,
)


class TestOptionsToParams:
    def test_empty_options(self):
        assert _options_to_params(None) == {}
        assert _options_to_params({}) == {}

    def test_temperature_mapping(self):
        assert _options_to_params({"temperature": 0.5}) == {"temperature": 0.5}

    def test_num_predict_to_max_tokens(self):
        assert _options_to_params({"num_predict": 512}) == {"max_tokens": 512}

    def test_repeat_penalty_mapping(self):
        assert _options_to_params({"repeat_penalty": 1.2}) == {
            "repetition_penalty": 1.2
        }

    def test_unknown_keys_ignored(self):
        assert _options_to_params({"unknown_key": 42}) == {}

    def test_none_values_skipped(self):
        assert _options_to_params({"temperature": None}) == {}


class TestBuildOpenAIMessagesGenerate:
    def test_prompt_only(self):
        msgs = _build_openai_messages_generate("hello")
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_prompt_with_system(self):
        msgs = _build_openai_messages_generate("hello", system="you are helpful")
        assert msgs == [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hello"},
        ]


class TestBuildOpenAIMessagesChat:
    def test_simple_messages(self):
        msgs = [
            OllamaChatMessage(role="user", content="hi"),
            OllamaChatMessage(role="assistant", content="hello"),
        ]
        result = _build_openai_messages_chat(msgs)
        assert result == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_images_generate_content_parts(self):
        msgs = [OllamaChatMessage(role="user", content="describe", images=["abc123"])]
        result = _build_openai_messages_chat(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "image_url"


class TestPydanticModels:
    def test_generate_request_defaults(self):
        req = OllamaGenerateRequest()
        assert req.model == "default"
        assert req.prompt == ""
        assert req.stream is True

    def test_chat_request_defaults(self):
        req = OllamaChatRequest(messages=[OllamaChatMessage(role="user", content="hi")])
        assert req.model == "default"
        assert req.stream is True

    def test_chat_message_defaults(self):
        msg = OllamaChatMessage()
        assert msg.role == "user"
        assert msg.content == ""
