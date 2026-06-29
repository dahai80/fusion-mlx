# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.tokenizer — migrated to fusion-mlx.

NOTE: fusion_mlx.utils.tokenizer only exposes is_gemma4_model(model_path)
and is_harmony_model(model_path) with different signatures than omlx.
Functions apply_qwen3_fix, create_streaming_detokenizer, get_tokenizer_config,
is_qwen3_model do not exist in fusion-mlx.
"""

import pytest

from fusion_mlx.utils.tokenizer import is_gemma4_model, is_harmony_model

# --- Tests for functions that exist in fusion-mlx (adapted signatures) ---


class TestIsHarmonyModel:
    """Adapted: fusion-mlx is_harmony_model takes a single model_path string."""

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
    """Adapted: fusion-mlx is_gemma4_model takes a single model_path string."""

    def test_gemma4_via_name(self):
        assert is_gemma4_model("google/gemma-4b") is True
        assert is_gemma4_model("my-gemma4-model") is True

    def test_not_gemma4(self):
        assert is_gemma4_model("llama-3.1-8b") is False
        assert is_gemma4_model("gemma-3-27b") is False


# --- Skipped: functions that don't exist in fusion-mlx ---

@pytest.mark.skip(reason="omlx-only: create_streaming_detokenizer not in fusion-mlx")
class TestCreateStreamingDetokenizer:
    def test_uses_spm_decoder_from_tokenizer_json(self):
        pass

    def test_uses_bpe_decoder_from_tokenizer_json(self):
        pass

    def test_explicit_none_detokenizer_without_model_path_stays_none(self):
        pass

    def test_missing_tokenizer_json_uses_naive_fallback(self):
        pass


@pytest.mark.skip(reason="omlx-only: is_qwen3_model not in fusion-mlx")
class TestIsQwen3Model:
    def test_qwen3_lowercase(self):
        pass


@pytest.mark.skip(reason="omlx-only: get_tokenizer_config not in fusion-mlx")
class TestLFM2ToolParserConfig:
    def test_lfm2_moe_text_model_gets_pythonic_tool_parser(self):
        pass


@pytest.mark.skip(reason="omlx-only: get_tokenizer_config not in fusion-mlx")
class TestGetTokenizerConfig:
    def test_basic_config(self):
        pass


@pytest.mark.skip(reason="omlx-only: apply_qwen3_fix not in fusion-mlx")
class TestApplyQwen3Fix:
    def test_apply_fix_to_qwen3(self):
        pass
