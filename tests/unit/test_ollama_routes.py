# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Ollama-compatible API routes."""

from fusion_mlx.api.ollama_routes import (
    OllamaChatRequest,
    OllamaGenerateRequest,
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

    def test_none_values_passed_through(self):
        result = _options_to_params({"temperature": None})
        assert result == {"temperature": None}


class TestPydanticModels:
    def test_generate_request_defaults(self):
        req = OllamaGenerateRequest(model="test")
        assert req.model == "test"
        assert req.prompt == ""
        assert req.stream is True

    def test_chat_request_defaults(self):
        req = OllamaChatRequest(
            model="test", messages=[{"role": "user", "content": "hi"}]
        )
        assert req.model == "test"
        assert req.stream is True
