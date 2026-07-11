"""Test that TypeError fallback in _apply_chat_template preserves tools.

The old code stripped tools/enable_thinking/preserve_thinking wholesale on
TypeError, which meant tool-calling requests silently lost their tools.
The fix does surgical fallback: tries removing only the problematic kwarg
first, preserving tools whenever possible.
"""

from __future__ import annotations

from unittest.mock import MagicMock


def _make_engine():
    from fusion_mlx.engines.batched import BatchedEngine

    engine = object.__new__(BatchedEngine)
    engine._enable_thinking = None
    engine._preserve_thinking = None
    return engine


def _fake_messages():
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ]


def _tools_spec():
    return [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]


class TestTypeErrorFallbackPreservesTools:

    def test_no_typeerror_no_fallback(self):
        engine = _make_engine()
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.return_value = "<tools>get_weather</tools>hi"
        engine._tokenizer = mock_tok

        result = engine._apply_chat_template(_fake_messages(), tools=_tools_spec())
        assert "<tools>" in result
        call_kwargs = mock_tok.apply_chat_template.call_args[1]
        assert "tools" in call_kwargs

    def test_typeerror_from_enable_thinking_preserves_tools(self):
        """enable_thinking=True raises TypeError, but tools should survive."""
        engine = _make_engine()
        engine._enable_thinking = True
        call_count = 0

        def _apply(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if "enable_thinking" in kwargs:
                raise TypeError("got an unexpected keyword argument 'enable_thinking'")
            if "tools" in kwargs:
                return "<tools>present</tools>hi"
            return "hi (no tools)"

        mock_tok = MagicMock()
        mock_tok.apply_chat_template.side_effect = _apply
        engine._tokenizer = mock_tok

        result = engine._apply_chat_template(_fake_messages(), tools=_tools_spec())
        assert "<tools>present</tools>" in result, f"Tools were stripped! Got: {result}"
        assert call_count == 2

    def test_typeerror_from_chat_template_kwargs_preserves_tools(self):
        """chat_template_kwargs key causes TypeError, tools survive."""
        engine = _make_engine()
        call_count = 0

        def _apply(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if "custom_param" in kwargs:
                raise TypeError("unexpected 'custom_param'")
            if "tools" in kwargs:
                return "<tools>present</tools>hi"
            return "hi (no tools)"

        mock_tok = MagicMock()
        mock_tok.apply_chat_template.side_effect = _apply
        engine._tokenizer = mock_tok

        result = engine._apply_chat_template(
            _fake_messages(),
            tools=_tools_spec(),
            chat_template_kwargs={"custom_param": "val"},
        )
        assert "<tools>present</tools>" in result

    def test_typeerror_from_tools_strips_tools_as_last_resort(self):
        """If tools itself causes TypeError, fallback strips it."""
        engine = _make_engine()
        call_count = 0

        def _apply(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if "tools" in kwargs:
                raise TypeError("unexpected 'tools'")
            return "hi (no tools)"

        mock_tok = MagicMock()
        mock_tok.apply_chat_template.side_effect = _apply
        engine._tokenizer = mock_tok

        result = engine._apply_chat_template(_fake_messages(), tools=_tools_spec())
        assert "no tools" in result
        assert call_count == 2

    def test_typeerror_from_multiple_kwargs_strips_incrementally(self):
        """Both enable_thinking and tools cause TypeError."""
        engine = _make_engine()
        engine._enable_thinking = True
        call_count = 0

        def _apply(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if "enable_thinking" in kwargs:
                raise TypeError("unexpected 'enable_thinking'")
            if "tools" in kwargs:
                raise TypeError("unexpected 'tools'")
            return "bare prompt"

        mock_tok = MagicMock()
        mock_tok.apply_chat_template.side_effect = _apply
        engine._tokenizer = mock_tok

        result = engine._apply_chat_template(_fake_messages(), tools=_tools_spec())
        assert result == "bare prompt"
        assert call_count == 3
