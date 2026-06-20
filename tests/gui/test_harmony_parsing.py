#!/usr/bin/env python3
"""
Test the Harmony response parsing for gpt-oss models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fusion_gui.server import _parse_harmony_response


def test_harmony_basic():
    """Test basic Harmony format parsing."""
    raw_response = '<|channel|>analysis<|message|>User says "sup homey". They are greeting. We respond casually but helpfully.<|end|><|start|>assistant<|channel|>final<|message|>Hey! What\'s up? Anything I can help you with today?'

    result = _parse_harmony_response(raw_response)

    print(f"Input: {raw_response[:100]}...")
    print(f"Output: {result}")

    assert "Hey! What's up?" in result
    assert "<|channel|>" not in result
    assert "<|message|>" not in result
    assert "analysis" not in result.lower() or "assist" in result.lower()  # 'analysis' shouldn't appear except maybe in context
    print("✅ Basic Harmony parsing test passed!")


def test_harmony_with_end_marker():
    """Test Harmony format with end marker in final channel."""
    raw_response = '<|channel|>analysis<|message|>Thinking about this...<|end|><|start|>assistant<|channel|>final<|message|>Here is my response.<|end|>'

    result = _parse_harmony_response(raw_response)

    print(f"Input: {raw_response}")
    print(f"Output: {result}")

    assert result == "Here is my response."
    print("✅ Harmony with end marker test passed!")


def test_non_harmony_passthrough():
    """Test that non-Harmony responses pass through unchanged."""
    regular_response = "Hello! How can I help you today?"

    result = _parse_harmony_response(regular_response)

    print(f"Input: {regular_response}")
    print(f"Output: {result}")

    assert result == regular_response
    print("✅ Non-Harmony passthrough test passed!")


def test_harmony_multiline():
    """Test Harmony format with multiline content."""
    raw_response = '''<|channel|>analysis<|message|>The user is asking about Python programming.
I should provide a clear explanation with examples.<|end|><|start|>assistant<|channel|>final<|message|>Python is a great language!

Here's why:
1. Easy to learn
2. Powerful libraries
3. Great community'''

    result = _parse_harmony_response(raw_response)

    print(f"Input: {raw_response[:80]}...")
    print(f"Output: {result}")

    assert "Python is a great language!" in result
    assert "<|channel|>" not in result
    assert "1. Easy to learn" in result
    print("✅ Multiline Harmony parsing test passed!")


def test_harmony_no_final_channel():
    """Test fallback when no final channel is found."""
    raw_response = '<|channel|>analysis<|message|>Some internal reasoning<|end|>'

    result = _parse_harmony_response(raw_response)

    print(f"Input: {raw_response}")
    print(f"Output: '{result}'")

    # Should strip tokens - result should be clean (internal reasoning stripped)
    assert "<|channel|>" not in result
    assert "<|message|>" not in result
    assert "<|end|>" not in result
    # The actual content "Some internal reasoning" should be extracted
    assert "internal reasoning" in result.lower() or result == ""
    print("✅ No final channel fallback test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Harmony Response Parsing")
    print("=" * 60)
    print()

    test_harmony_basic()
    print()

    test_harmony_with_end_marker()
    print()

    test_non_harmony_passthrough()
    print()

    test_harmony_multiline()
    print()

    test_harmony_no_final_channel()
    print()

    print("=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)


