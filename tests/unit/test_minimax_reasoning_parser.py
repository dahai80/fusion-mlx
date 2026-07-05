# SPDX-License-Identifier: Apache-2.0
import logging
from unittest.mock import MagicMock

import pytest

from fusion_mlx.reasoning.minimax_parser import MiniMaxReasoningParser

logger = logging.getLogger(__name__)


class TestMiniMaxReasoningParserInit:

    def test_default_init(self):
        parser = MiniMaxReasoningParser()
        assert parser._buffer == ""
        assert parser._decided is False
        assert parser._is_reasoning is False
        assert parser._transition_pos == 0

    def test_init_with_tokenizer(self):
        mock_tokenizer = MagicMock()
        parser = MiniMaxReasoningParser(tokenizer=mock_tokenizer)
        assert parser._buffer == ""
        assert parser._decided is False

    def test_reset_state(self):
        parser = MiniMaxReasoningParser()
        parser._buffer = "some text"
        parser._decided = True
        parser._is_reasoning = True
        parser._transition_pos = 100
        parser.reset_state()

        assert parser._buffer == ""
        assert parser._decided is False
        assert parser._is_reasoning is False
        assert parser._transition_pos == 0


class TestExtractReasoningExplicitTags:

    def test_explicit_standard_tags(self):
        parser = MiniMaxReasoningParser()
        output = "<think>Let me think about this.</think>Here is the answer."

        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "Let me think about this."
        assert content == "Here is the answer."

    def test_explicit_tag_no_content(self):
        parser = MiniMaxReasoningParser()
        output = "<think>Just thinking here.</think>"

        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "Just thinking here."
        assert content is None

    def test_explicit_tag_only_content_after(self):
        parser = MiniMaxReasoningParser()
        output = "<think></think>Some content"

        reasoning, content = parser.extract_reasoning(output)

        assert reasoning is None
        assert content == "Some content"

    def test_explicit_tag_with_whitespace(self):
        parser = MiniMaxReasoningParser()
        output = "<think>  Reasoning here.   </think>  Content here.  "

        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "Reasoning here."
        assert content == "Content here."


class TestDirectContentDetection:

    @pytest.mark.parametrize(
        "content",
        [
            "```python\nprint('hello')\n```",
            "<minimax:tool_call>some tool</minimax:tool_call>",
            "<tool_call>call>",
            "<invoke name='test'>",
            "# Heading",
            "## Subheading",
            '{"key": "value"}',
            '["item1", "item2"]',
        ],
    )
    def test_direct_content_patterns(self, content):
        parser = MiniMaxReasoningParser()
        reasoning, result_content = parser.extract_reasoning(content)

        assert reasoning is None
        assert result_content == content


class TestReasoningStartPatterns:

    @pytest.mark.parametrize(
        "text",
        [
            "The user asks for a poem.",
            "The user wants to know about AI.",
            "The user is asking for help.",
            "The user requests a summary.",
            "I need to analyze this.",
            "I should consider the options.",
            "I will provide the answer.",
            "I can help with that.",
            "I want to explain this.",
            "I have to make a decision.",
            "I must complete this task.",
            "I am going to solve this.",
            "Let me think about this.",
            "Let me check the details.",
            "Let me analyze the problem.",
            "Let me figure this out.",
            "Let me consider the options.",
            "Let me look at the data.",
            "Let me read the input.",
            "Let me review the context.",
            "Let me process this request.",
            "This is a complex question.",
            "This requires careful thought.",
            "This seems like a good approach.",
            "This looks like a request for help.",
            "This appears to be about code.",
            "First, I need to understand.",
            "First, let me explain.",
            "First, we should consider.",
            "So the user wants to know.",
            "Now I need to figure this out.",
            "OK let me think about this.",
            "Okay, I should help with this.",
            "Alright, let me proceed.",
            "Well, I need to think about this.",
            "what's worth storing in memory.",
            "Analyzing the request carefully.",
            "Thinking about the best approach.",
            "Processing the input data.",
            "Considering all options.",
            "Evaluating the requirements.",
            "Extracting the key information.",
            # Chinese reasoning patterns
            "用户想让我写一个测试文件。",
            "用户要知道这个函数的用法。",
            "用户需要帮助修复bug。",
            "用户问如何优化代码。",
            "用户请求一个总结。",
            "用户让我分析代码。",
            "我需要分析这段代码。",
            "我应该先看看文件内容。",
            "我要帮用户写测试。",
            "让我想想这个问题。",
            "让我看看代码结构。",
            "让我分析一下bug。",
            "这是一个复杂的问题。",
            "这需要仔细思考。",
            "首先，我需要理解代码。",
            "首先让我读取文件。",
            "好的，用户想让我写测试。",
            "那么我需要先分析代码。",
            "分析一下这段代码的问题。",
            "思考一下最佳方案。",
        ],
    )
    def test_reasoning_start_patterns(self, text):
        parser = MiniMaxReasoningParser()
        assert parser._REASONING_START_RE.match(text) is not None

    @pytest.mark.parametrize(
        "text",
        [
            "Hello, how are you?",
            "The weather is nice today.",
            "Welcome to the system.",
            "Python is a great language.",
        ],
    )
    def test_non_reasoning_start(self, text):
        parser = MiniMaxReasoningParser()
        reasoning, content = parser.extract_reasoning(text)

        assert reasoning is None
        assert content == text


class TestTransitionDetection:

    def test_transition_answer_is(self):
        parser = MiniMaxReasoningParser()
        output = "The user wants a poem.\n\nThe answer is: roses are red."

        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "The user wants a poem."
        assert content == "The answer is: roses are red."

    def test_transition_code_block(self):
        parser = MiniMaxReasoningParser()
        output = "Let me think about this.\n\n```python\nprint('hi')\n```"

        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "Let me think about this."
        assert content == "```python\nprint('hi')\n```"

    def test_transition_here_is(self):
        parser = MiniMaxReasoningParser()
        output = "I need to figure this out.\n\nHere is the solution."

        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "I need to figure this out."
        assert content == "Here is the solution."

    def test_transition_sure(self):
        parser = MiniMaxReasoningParser()
        output = "Let me analyze this.\n\nSure, here's the answer."

        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "Let me analyze this."
        assert content == "Sure, here's the answer."

    def test_transition_tool_call(self):
        parser = MiniMaxReasoningParser()
        output = "Let me think.\n<minimax:tool_call>call</minimax:tool_call>"

        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "Let me think."
        assert content is not None
        assert "<minimax:tool_call>" in content

    def test_transition_markdown_heading(self):
        parser = MiniMaxReasoningParser()
        output = "I need to analyze the code.\n\n## Analysis\n\nThe code has bugs."

        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "I need to analyze the code."
        assert "## Analysis" in content

    def test_transition_bold_text(self):
        parser = MiniMaxReasoningParser()
        output = "Let me review this.\n\n**Issue 1**: The code is wrong."

        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "Let me review this."
        assert "**Issue 1**" in content

    def test_transition_numbered_bold_list(self):
        parser = MiniMaxReasoningParser()
        output = "Analyzing the problem.\n\n1. **First** issue is here."

        reasoning, content = parser.extract_reasoning(output)

        assert reasoning is not None
        assert content is not None

    def test_short_reasoning_not_stripped(self):
        parser = MiniMaxReasoningParser()
        output = "I think\n\nHere is the answer."

        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == output

    def test_no_transition_double_newline_fallback(self):
        parser = MiniMaxReasoningParser()
        output = (
            "The user asks about Python.\n\nPython is a great language for beginners."
        )

        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "The user asks about Python."
        assert content == "Python is a great language for beginners."

    def test_no_transition_no_split(self):
        parser = MiniMaxReasoningParser()
        output = "The user asks about Python and it is great."

        reasoning, content = parser.extract_reasoning(output)

        assert reasoning is None
        assert content == output


class TestStreamingBasic:

    def test_direct_content_early_detection(self):
        parser = MiniMaxReasoningParser()

        result = parser.extract_reasoning_streaming("", "```", "```")
        assert result is not None
        assert result.content == "```"
        assert parser._decided is True
        assert parser._is_reasoning is False

    def test_direct_content_tool_call(self):
        parser = MiniMaxReasoningParser()

        result = parser.extract_reasoning_streaming(
            "", "<minimax:tool_call>", "<minimax:tool_call>"
        )
        assert result is not None
        assert result.content == "<minimax:tool_call>"

    def test_buffer_phase_returns_none(self):
        parser = MiniMaxReasoningParser()

        result = parser.extract_reasoning_streaming("", "The user", "The user")
        assert result is None
        assert parser._decided is False

    def test_buffer_accumulates(self):
        parser = MiniMaxReasoningParser()

        r1 = parser.extract_reasoning_streaming("", "The ", "The ")
        assert r1 is None

        r2 = parser.extract_reasoning_streaming("The ", "The user ", "user ")
        assert r2 is None

        assert parser._buffer == "The user "

    def test_reasoning_detected_after_buffer(self):
        parser = MiniMaxReasoningParser()

        text = "The user asks about a very complex topic that requires deep analysis and careful thinking to solve properly."
        current = ""
        last_result = None
        for ch in text:
            prev = current
            current += ch
            result = parser.extract_reasoning_streaming(prev, current, ch)
            if result is not None:
                last_result = result

        assert parser._decided is True
        assert parser._is_reasoning is True
        assert last_result is not None
        assert last_result.reasoning is not None

    def test_content_detected_after_buffer(self):
        parser = MiniMaxReasoningParser()

        text = "Hello! Welcome to our service. We are happy to help you with your question about programming today."
        current = ""
        last_result = None
        for ch in text:
            prev = current
            current += ch
            result = parser.extract_reasoning_streaming(prev, current, ch)
            if result is not None:
                last_result = result

        assert parser._decided is True
        assert parser._is_reasoning is False
        assert last_result is not None
        assert last_result.content is not None


class TestStreamingThinkTags:

    def test_think_tag_start(self):
        parser = MiniMaxReasoningParser()

        result = parser.extract_reasoning_streaming("", "<think>", "<think>")
        assert result is None
        assert parser._decided is True
        assert parser._is_reasoning is True

    def test_think_tag_with_content(self):
        parser = MiniMaxReasoningParser()

        result = parser.extract_reasoning_streaming(
            "", "<think>reasoning", "<think>reasoning"
        )
        assert result is not None
        assert result.reasoning == "reasoning"

    def test_think_tag_end(self):
        parser = MiniMaxReasoningParser()

        parser.extract_reasoning_streaming("", "<think>", "<think>")

        parser.extract_reasoning_streaming(
            "<think>", "<think>some reasoning", "some reasoning"
        )

        result = parser.extract_reasoning_streaming(
            "<think>some reasoning",
            "<think>some reasoning</think>content here",
            "</think>content here",
        )
        assert result is not None
        assert result.content == "content here"
        assert parser._is_reasoning is False

    def test_think_tag_end_no_content(self):
        parser = MiniMaxReasoningParser()

        parser.extract_reasoning_streaming("", "<think>", "<think>")
        result = parser.extract_reasoning_streaming(
            "<think>", "<think></think>", "</think>"
        )

        assert result is not None
        assert parser._is_reasoning is False


class TestStreamingTransition:

    def test_transition_during_stream(self):
        parser = MiniMaxReasoningParser()

        reasoning_text = "The user asks about a complex topic that requires very careful analysis and deep thinking to answer properly."
        current = ""
        for ch in reasoning_text:
            prev = current
            current += ch
            parser.extract_reasoning_streaming(prev, current, ch)

        assert parser._decided is True
        assert parser._is_reasoning is True

        prev = current
        transition = "\n\nHere is the answer."
        current += transition
        result = parser.extract_reasoning_streaming(prev, current, transition)

        assert result is not None
        if result.content:
            assert "Here is" in result.content

    def test_post_decision_content_passthrough(self):
        parser = MiniMaxReasoningParser()

        parser.extract_reasoning_streaming("", "```python", "```python")

        result = parser.extract_reasoning_streaming(
            "```python", "```python\nprint", "\nprint"
        )
        assert result is not None
        assert result.content == "\nprint"


class TestFinalizeStreaming:

    def test_finalize_undecided_with_text(self):
        parser = MiniMaxReasoningParser()

        parser.extract_reasoning_streaming("", "Short", "Short")
        assert parser._decided is False

        result = parser.finalize_streaming("Short")
        assert result is not None
        assert result.content == "Short"

    def test_finalize_undecided_empty(self):
        parser = MiniMaxReasoningParser()

        result = parser.finalize_streaming("")
        assert result is None

    def test_finalize_decided_content(self):
        parser = MiniMaxReasoningParser()
        parser._decided = True
        parser._is_reasoning = False

        result = parser.finalize_streaming("some content")
        assert result is None

    def test_finalize_all_reasoning_reclassify(self):
        parser = MiniMaxReasoningParser()
        parser._decided = True
        parser._is_reasoning = True

        result = parser.finalize_streaming("The user asks about Python.")
        assert result is not None
        assert result.content is not None


class TestEdgeCases:

    def test_empty_input_extract(self):
        parser = MiniMaxReasoningParser()
        reasoning, content = parser.extract_reasoning("")

        assert reasoning is None
        assert content == ""

    def test_whitespace_only_extract(self):
        parser = MiniMaxReasoningParser()
        reasoning, content = parser.extract_reasoning("   \n\n   ")

        assert reasoning is None
        assert content == "   \n\n   "

    def test_very_long_reasoning_no_transition(self):
        parser = MiniMaxReasoningParser()
        output = "The user asks " + "word " * 200

        reasoning, content = parser.extract_reasoning(output)
        assert content is not None

    def test_streaming_single_char_at_a_time(self):
        parser = MiniMaxReasoningParser()
        text = "```python\nprint('hello')\n```"

        current = ""
        results = []
        for ch in text:
            prev = current
            current += ch
            result = parser.extract_reasoning_streaming(prev, current, ch)
            if result is not None:
                results.append(result)

        assert len(results) >= 1
        for r in results:
            assert r.reasoning is None

    def test_reset_between_streams(self):
        parser = MiniMaxReasoningParser()

        parser.extract_reasoning_streaming(
            "", "<think>stuff</think>content", "<think>stuff</think>content"
        )
        assert parser._decided is True

        parser.reset_state()
        assert parser._decided is False
        assert parser._buffer == ""

        result = parser.extract_reasoning_streaming("", "```code", "```code")
        assert result is not None
        assert result.content == "```code"

    def test_thus_answer_transition(self):
        parser = MiniMaxReasoningParser()
        output = "I need to analyze this carefully.\nThus answer: The result is 42."

        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "I need to analyze this carefully."
        assert content == "Thus answer: The result is 42."

    def test_chinese_reasoning_then_content(self):
        parser = MiniMaxReasoningParser()
        output = "用户想让我写一个快速排序算法。\n\n```python\ndef quicksort(arr):\n    pass\n```"

        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "用户想让我写一个快速排序算法。"
        assert "```python" in content

    def test_chinese_reasoning_streaming(self):
        parser = MiniMaxReasoningParser()

        text = "用户想让我分析这段代码并找出所有的bug。这个代码有很多潜在的安全问题和性能瓶颈需要仔细检查，包括SQL注入攻击、线程安全竞态条件和内存泄漏等方面的问题，还需要考虑错误处理和边界情况。"
        current = ""
        last_result = None
        for ch in text:
            prev = current
            current += ch
            result = parser.extract_reasoning_streaming(prev, current, ch)
            if result is not None:
                last_result = result

        assert parser._decided is True
        assert parser._is_reasoning is True
        assert last_result is not None
        assert last_result.reasoning is not None

    def test_chinese_transition_pattern(self):
        parser = MiniMaxReasoningParser()
        output = "我需要分析这段代码。\n\n以下是修复后的代码。"

        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "我需要分析这段代码。"
        assert "以下是" in content
