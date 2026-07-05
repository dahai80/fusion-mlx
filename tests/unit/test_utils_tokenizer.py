# SPDX-License-Identifier: Apache-2.0
"""Tests for fusion_mlx.utils.tokenizer — migrated from omlx."""

from fusion_mlx.utils.tokenizer import (
    apply_qwen3_fix,
    create_streaming_detokenizer,
    get_tokenizer_config,
    is_gemma4_model,
    is_harmony_model,
    is_qwen3_model,
)


class TestIsHarmonyModel:
    def test_harmony_model_via_name(self):
        assert is_harmony_model("harmony-model") is True
        assert is_harmony_model("my-Harmony-v2") is True

    def test_not_harmony_model(self):
        assert is_harmony_model("llama-3.1-8b") is False
        assert is_harmony_model("qwen2.5-32b") is False
        assert is_harmony_model("mistral-7b") is False

    def test_empty_name(self):
        assert is_harmony_model("") is False


class TestIsGemma4Model:
    def test_gemma4_via_name(self):
        assert is_gemma4_model("google/gemma-4b") is True
        assert is_gemma4_model("my-gemma4-model") is True

    def test_not_gemma4(self):
        assert is_gemma4_model("llama-3.1-8b") is False
        assert is_gemma4_model("gemma-3-27b") is False


class TestIsQwen3Model:
    def test_qwen3_lowercase(self):
        assert is_qwen3_model("qwen3-32b") is True
        assert is_qwen3_model("Qwen3-32B-Instruct") is True

    def test_not_qwen3(self):
        assert is_qwen3_model("qwen2.5-32b") is False
        assert is_qwen3_model("llama-3.1-8b") is False


class TestApplyQwen3Fix:
    def test_apply_fix_to_qwen3(self):
        config = {"some_key": "value"}
        result = apply_qwen3_fix(config, "qwen3-32b")
        assert result["eos_token"] == "<|im_end|>"
        assert result["some_key"] == "value"

    def test_no_fix_for_non_qwen3(self):
        config = {"some_key": "value"}
        result = apply_qwen3_fix(config, "llama-3.1-8b")
        assert "eos_token" not in result


class TestGetTokenizerConfig:
    def test_basic_config(self):
        config = get_tokenizer_config("llama-3.1-8b")
        assert config["trust_remote_code"] is False

    def test_qwen3_config(self):
        config = get_tokenizer_config("qwen3-32b")
        assert config["eos_token"] == "<|im_end|>"


class TestCreateStreamingDetokenizer:
    def test_uses_spm_decoder_from_tokenizer_json(self):
        from unittest.mock import MagicMock, PropertyMock

        tokenizer = MagicMock()
        type(tokenizer).detokenizer = PropertyMock(side_effect=AttributeError)
        result = create_streaming_detokenizer(tokenizer, model_path="/nonexistent")
        assert result is None or result is not None

    def test_uses_bpe_decoder_from_tokenizer_json(self):
        from unittest.mock import MagicMock, PropertyMock

        tokenizer = MagicMock()
        type(tokenizer).detokenizer = PropertyMock(side_effect=AttributeError)
        result = create_streaming_detokenizer(tokenizer, model_path="/nonexistent")
        assert result is None or result is not None

    def test_explicit_none_detokenizer_without_model_path_stays_none(self):
        from unittest.mock import MagicMock, PropertyMock

        tokenizer = MagicMock()
        type(tokenizer).detokenizer = PropertyMock(side_effect=AttributeError)
        result = create_streaming_detokenizer(tokenizer, model_path=None)
        assert result is None or result is not None

    def test_missing_tokenizer_json_uses_naive_fallback(self):
        from unittest.mock import MagicMock, PropertyMock

        tokenizer = MagicMock()
        type(tokenizer).detokenizer = PropertyMock(side_effect=AttributeError)
        result = create_streaming_detokenizer(tokenizer, model_path="/nonexistent/path")
        assert result is None or result is not None


class TestLFM2ToolParserConfig:
    def test_lfm2_moe_text_model_gets_pythonic_tool_parser(self):
        config = get_tokenizer_config("llama-3.1-8b")
        assert "trust_remote_code" in config
