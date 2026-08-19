# SPDX-License-Identifier: Apache-2.0
"""Tests for BatchedEngine.count_chat_tokens().

Split out of test_context_window.py: that suite's 13 context-window-policy
tests pin an unported feature (validate_context_window / SamplingDefaults /
max_context_window_policy / model_context_length discovery) and stay
quarantined. These 2 count_chat_tokens tests exercise the live
engines.batched.BatchedEngine.count_chat_tokens method and are rescued.
"""

from unittest.mock import MagicMock


class TestCountChatTokens:
    def test_count_chat_tokens(self):
        from fusion_mlx.engines.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._loaded = True

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        engine._tokenizer = mock_tokenizer

        engine._model = MagicMock(spec=[])
        engine._enable_thinking = None
        engine._preserve_thinking = None

        messages = [{"role": "user", "content": "Hello"}]
        count = engine.count_chat_tokens(messages)

        assert count == 5
        mock_tokenizer.apply_chat_template.assert_called_once()
        mock_tokenizer.encode.assert_called_once_with("formatted prompt")

    def test_count_chat_tokens_with_tools(self):
        from fusion_mlx.engines.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._loaded = True

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt with tools"
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7]
        engine._tokenizer = mock_tokenizer

        engine._model = MagicMock(spec=[])
        engine._enable_thinking = None
        engine._preserve_thinking = None

        messages = [{"role": "user", "content": "Call a tool"}]
        tools = [{"type": "function", "function": {"name": "test"}}]
        count = engine.count_chat_tokens(messages, tools)

        assert count == 7
