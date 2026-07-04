import logging

import pytest

from fusion_mlx.reasoning import (
    DeltaMessage,
    ReasoningParser,
    get_parser,
    list_parsers,
    register_parser,
)
from fusion_mlx.api.models import AssistantMessage, ChatCompletionChunkDelta

logger = logging.getLogger(__name__)


class TestParserRegistry:

    def test_list_parsers_includes_builtin(self):
        parsers = list_parsers()
        assert "qwen3" in parsers
        assert "deepseek_r1" in parsers

    def test_get_parser_qwen3(self):
        parser_cls = get_parser("qwen3")
        parser = parser_cls()
        assert isinstance(parser, ReasoningParser)

    def test_get_parser_deepseek(self):
        parser_cls = get_parser("deepseek_r1")
        parser = parser_cls()
        assert isinstance(parser, ReasoningParser)

    def test_get_unknown_parser_raises(self):
        with pytest.raises(KeyError) as exc_info:
            get_parser("unknown_parser")
        assert "unknown_parser" in str(exc_info.value)
        assert "Available parsers" in str(exc_info.value)

    def test_register_custom_parser(self):

        class CustomParser(ReasoningParser):
            def extract_reasoning(self, model_output):
                return None, model_output

            def extract_reasoning_streaming(self, prev, curr, delta):
                return DeltaMessage(content=delta)

        register_parser("custom_test", CustomParser)
        assert "custom_test" in list_parsers()

        parser = get_parser("custom_test")()
        assert isinstance(parser, CustomParser)


class TestQwen3Parser:

    @pytest.fixture
    def parser(self):
        return get_parser("qwen3")()

    def test_extract_with_both_tags(self, parser):
        output = "<think>Let me analyze this problem</think>The answer is 42."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Let me analyze this problem"
        assert content == "The answer is 42."

    def test_extract_only_reasoning(self, parser):
        output = "<think>Just thinking out loud</think>"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Just thinking out loud"
        assert content is None

    def test_extract_multiline_reasoning(self, parser):
        output = (
            "<think>Step 1: Analyze\nStep 2: Solve\nStep 3: Verify</think>"
            "Result: 42"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert "Step 1" in reasoning
        assert "Step 2" in reasoning
        assert "Step 3" in reasoning
        assert content == "Result: 42"

    def test_no_tags_returns_content_only(self, parser):
        output = "Just a regular response without thinking."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == output

    def test_only_start_tag_truncated(self, parser):
        output = "<think>Started thinking but never finished"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Started thinking but never finished"
        assert content is None

    def test_only_end_tag_implicit_mode(self, parser):
        output = "Some text</think>more text"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Some text"
        assert content == "more text"

    def test_streaming_simple_flow(self, parser):
        parser.reset_state()

        deltas = ["<think>", "think", "ing", "</think>", "answer"]
        accumulated = ""
        results = []

        for delta in deltas:
            prev = accumulated
            accumulated += delta
            result = parser.extract_reasoning_streaming(prev, accumulated, delta)
            if result:
                results.append(result)

        reasoning_parts = [r.reasoning for r in results if r.reasoning]
        content_parts = [r.content for r in results if r.content]

        assert "".join(reasoning_parts) == "thinking"
        assert "".join(content_parts) == "answer"

    def test_streaming_skip_tags(self, parser):
        parser.reset_state()

        result = parser.extract_reasoning_streaming("", "<think>", "<think>")
        assert result is None

        result = parser.extract_reasoning_streaming(
            "<think>reasoning", "<think>reasoning</think>", "</think>"
        )
        assert result is None

    def test_streaming_transition_chunk(self, parser):
        parser.reset_state()

        prev = "<think>reasoning"
        delta = " more</think>content here"
        curr = prev + delta

        result = parser.extract_reasoning_streaming(prev, curr, delta)

        assert result is not None
        assert result.reasoning == " more"
        assert result.content == "content here"


class TestDeepSeekR1Parser:

    @pytest.fixture
    def parser(self):
        return get_parser("deepseek_r1")()

    def test_extract_with_both_tags(self, parser):
        output = "<think>Step by step analysis</think>Final answer: 42"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Step by step analysis"
        assert content == "Final answer: 42"

    def test_extract_implicit_start_tag(self, parser):
        output = "Implicit reasoning content</think>The answer"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Implicit reasoning content"
        assert content == "The answer"

    def test_extract_no_tags_pure_content(self, parser):
        output = "Just a regular response."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == output

    def test_extract_multiline_reasoning(self, parser):
        output = "<think>Line 1\nLine 2\nLine 3</think>Result"
        reasoning, content = parser.extract_reasoning(output)
        assert "Line 1" in reasoning
        assert "Line 2" in reasoning
        assert "Line 3" in reasoning
        assert content == "Result"

    def test_streaming_simple_flow(self, parser):
        parser.reset_state()

        deltas = ["<think>", "think", "ing", "</think>", "answer"]
        accumulated = ""
        results = []

        for delta in deltas:
            prev = accumulated
            accumulated += delta
            result = parser.extract_reasoning_streaming(prev, accumulated, delta)
            if result:
                results.append(result)

        reasoning_parts = [r.reasoning for r in results if r.reasoning]
        content_parts = [r.content for r in results if r.content]

        assert "".join(reasoning_parts) == "thinking"
        assert "".join(content_parts) == "answer"


class TestDeltaMessage:

    def test_reasoning_content_alias(self):
        msg = DeltaMessage(reasoning="test reasoning")
        assert msg.reasoning == "test reasoning"
        assert msg.reasoning_content == "test reasoning"

    def test_content_only(self):
        msg = DeltaMessage(content="just content")
        assert msg.content == "just content"
        assert msg.reasoning is None
        assert msg.reasoning_content is None

    def test_both_fields(self):
        msg = DeltaMessage(reasoning="ending", content="starting")
        assert msg.reasoning == "ending"
        assert msg.content == "starting"


class TestEdgeCases:

    @pytest.fixture(params=["qwen3", "deepseek_r1"])
    def parser(self, request):
        return get_parser(request.param)()

    def test_empty_output(self, parser):
        reasoning, content = parser.extract_reasoning("")
        assert reasoning is None or reasoning == ""

    def test_whitespace_only_reasoning(self, parser):
        output = "<think>    </think>content"
        reasoning, content = parser.extract_reasoning(output)
        if reasoning is not None:
            assert reasoning.strip() == "" or reasoning is None

    def test_nested_tags_not_supported(self, parser):
        output = "<think>outer<think>inner</think>still outer</think>content"
        reasoning, content = parser.extract_reasoning(output)

    def test_streaming_reset_state(self, parser):
        parser.reset_state()
        parser.extract_reasoning_streaming("", "<think>", "<think>")

        parser.reset_state()

        result = parser.extract_reasoning_streaming("", "content", "content")
        assert result is not None


class TestRealisticStreaming:

    @pytest.fixture(params=["qwen3", "deepseek_r1"])
    def parser(self, request):
        return get_parser(request.param)()

    def test_token_by_token_streaming(self, parser):
        tokens = [
            "<",
            "think",
            ">",
            "Let",
            " me",
            " analyze",
            " this",
            ".",
            "\n",
            "Step",
            " 1",
            ":",
            " check",
            " input",
            "\n",
            "Step",
            " 2",
            ":",
            " compute",
            "</",
            "think",
            ">",
            "The",
            " answer",
            " is",
            " 42",
            ".",
        ]

        parser.reset_state()
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        full_reasoning = "".join(reasoning_parts)
        full_content = "".join(content_parts)

        assert "Let me analyze" in full_reasoning
        assert "Step 1" in full_reasoning
        assert "Step 2" in full_reasoning

        assert "The answer is 42" in full_content

    def test_long_reasoning_streaming(self, parser):
        reasoning_text = """
        First, I need to understand the problem.
        The user is asking about quantum computing.

        Let me break this down:
        1. Quantum bits (qubits) can be in superposition
        2. Entanglement allows correlated states
        3. Quantum gates perform operations

        After careful analysis, I can provide an answer.
        """

        output = f"<think>{reasoning_text}</think>Quantum computing uses qubits."

        parser.reset_state()
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for char in output:
            prev = accumulated
            accumulated += char
            result = parser.extract_reasoning_streaming(prev, accumulated, char)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        full_reasoning = "".join(reasoning_parts)
        full_content = "".join(content_parts)

        assert "quantum computing" in full_reasoning.lower()
        assert "qubits" in full_reasoning.lower()
        assert "Quantum computing uses qubits" in full_content

    def test_streaming_no_content_after_reasoning(self, parser):
        tokens = ["<think>", "just", " thinking", "</think>"]

        parser.reset_state()
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        assert "just thinking" in "".join(reasoning_parts)
        assert len(content_parts) == 0 or "".join(content_parts).strip() == ""


class TestUnicodeAndSpecialCharacters:

    @pytest.fixture(params=["qwen3", "deepseek_r1"])
    def parser(self, request):
        return get_parser(request.param)()

    def test_unicode_reasoning(self, parser):
        output = "<think>分析这个问题：日本語テスト émojis: 🤔💭</think>答案是42"
        reasoning, content = parser.extract_reasoning(output)
        assert "分析" in reasoning
        assert "日本語" in reasoning
        assert "🤔" in reasoning
        assert "42" in content

    def test_code_in_reasoning(self, parser):
        output = (
            "<think>\n"
            "Let me analyze the code:\n"
            "```python\n"
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n-1)\n"
            "```\n"
            "This is a recursive implementation.\n"
            "</think>The factorial function uses recursion."
        )

        reasoning, content = parser.extract_reasoning(output)
        assert "def factorial" in reasoning
        assert "recursive" in reasoning
        assert "uses recursion" in content

    def test_html_like_content(self, parser):
        output = "<think>The user mentioned <div> and <span> tags</think>Use CSS for styling."
        reasoning, content = parser.extract_reasoning(output)
        assert "<div>" in reasoning
        assert "<span>" in reasoning
        assert "CSS" in content

    def test_math_expressions(self, parser):
        output = "<think>Given: x² + 2x + 1 = 0, so (x+1)² = 0, x = -1</think>x = -1"
        reasoning, content = parser.extract_reasoning(output)
        assert "x²" in reasoning
        assert "(x+1)²" in reasoning
        assert "-1" in content


class TestAPIModelsIntegration:

    def test_assistant_message_with_reasoning(self):
        msg = AssistantMessage(
            content="The answer is 42.",
            reasoning_content="Let me think step by step...",
        )
        assert msg.content == "The answer is 42."
        assert msg.reasoning_content == "Let me think step by step..."
        assert msg.role == "assistant"

    def test_assistant_message_reasoning_none(self):
        msg = AssistantMessage(content="Simple response without reasoning.")
        assert msg.content == "Simple response without reasoning."
        assert msg.reasoning_content is None

    def test_chat_completion_chunk_delta_with_reasoning(self):
        delta = ChatCompletionChunkDelta(reasoning_content="thinking...")
        assert delta.reasoning_content == "thinking..."
        assert delta.content is None

        delta2 = ChatCompletionChunkDelta(content="response text")
        assert delta2.content == "response text"
        assert delta2.reasoning_content is None

    def test_delta_transition(self):
        delta = ChatCompletionChunkDelta(
            reasoning_content="final thought", content="starting answer"
        )
        assert delta.reasoning_content == "final thought"
        assert delta.content == "starting answer"


class TestParserPerformance:

    @pytest.fixture(params=["qwen3", "deepseek_r1"])
    def parser(self, request):
        return get_parser(request.param)()

    def test_large_output_extraction(self, parser):
        reasoning_lines = [f"Step {i}: processing data chunk {i}" for i in range(100)]
        reasoning_text = "\n".join(reasoning_lines)
        output = f"<think>{reasoning_text}</think>Processing complete."

        reasoning, content = parser.extract_reasoning(output)

        assert reasoning is not None
        assert "Step 0" in reasoning
        assert "Step 99" in reasoning
        assert content == "Processing complete."

    def test_streaming_many_chunks(self, parser):
        parser.reset_state()

        base_output = "<think>A" * 100 + "</think>" + "B" * 50
        accumulated = ""
        chunk_count = 0

        for char in base_output:
            prev = accumulated
            accumulated += char
            result = parser.extract_reasoning_streaming(prev, accumulated, char)
            if result:
                chunk_count += 1

        assert chunk_count > 0

    def test_repeated_parsing(self, parser):
        output = "<think>Quick thought</think>Quick answer"

        for _ in range(100):
            reasoning, content = parser.extract_reasoning(output)
            assert reasoning == "Quick thought"
            assert content == "Quick answer"


class TestDeepSeekSpecificCases:

    @pytest.fixture
    def parser(self):
        return get_parser("deepseek_r1")()

    def test_implicit_reasoning_streaming(self, parser):
        tokens = ["reasoning", " text", " here", "</think>", "answer"]

        parser.reset_state()
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        all_parts = reasoning_parts + content_parts
        assert len(all_parts) > 0

    def test_deepseek_long_implicit_reasoning(self, parser):
        output = (
            "Let me think about this problem carefully.\n"
            "\n"
            "First, I need to consider the constraints.\n"
            "Then, I'll apply the algorithm.\n"
            "Finally, I'll verify the result.</think>The answer is 42."
        )

        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is not None
        assert "think about this problem" in reasoning
        assert "42" in content


class TestQwen3SpecificCases:

    @pytest.fixture
    def parser(self):
        return get_parser("qwen3")()

    def test_qwen3_implicit_mode_support(self, parser):
        output1 = "some text</think>more text"
        reasoning, content = parser.extract_reasoning(output1)
        assert reasoning == "some text"
        assert content == "more text"

        output2 = "<think>incomplete reasoning"
        reasoning, content = parser.extract_reasoning(output2)
        assert reasoning == "incomplete reasoning"
        assert content is None

    def test_qwen3_empty_think_tags(self, parser):
        output = "<think></think>Just the answer."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None or reasoning.strip() == ""
        assert content == "Just the answer."

    def test_qwen3_whitespace_between_tags(self, parser):
        test_cases = [
            ("<think>  </think>answer", None, "answer"),
            ("<think>\n\n</think>answer", None, "answer"),
            ("<think>\t\t</think>answer", None, "answer"),
        ]

        for output, expected_reasoning, expected_content in test_cases:
            reasoning, content = parser.extract_reasoning(output)
            if expected_reasoning is None:
                assert reasoning is None or reasoning.strip() == ""
            assert expected_content in (content or "")


class TestGptOssParser:

    @pytest.fixture
    def parser(self):
        return get_parser("gpt_oss")()

    def test_extract_both_channels(self, parser):
        output = (
            "<|channel|>analysis<|message|>Let me think step by step"
            "<|start|>assistant<|channel|>final<|message|>The answer is 42<|return|>"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Let me think step by step"
        assert content == "The answer is 42"

    def test_extract_only_final(self, parser):
        output = "<|channel|>final<|message|>Just the answer<|return|>"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == "Just the answer"

    def test_extract_only_analysis(self, parser):
        output = "<|channel|>analysis<|message|>Just thinking out loud"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Just thinking out loud"
        assert content is None

    def test_no_channel_tokens_fallback(self, parser):
        output = "Just a regular response."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == output

    def test_empty_analysis_channel(self, parser):
        output = (
            "<|channel|>analysis<|message|>"
            "<|start|>assistant<|channel|>final<|message|>Content here<|return|>"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == "Content here"

    def test_multiline_analysis(self, parser):
        output = (
            "<|channel|>analysis<|message|>Step 1: Analyze\nStep 2: Solve\nStep 3: Verify"
            "<|start|>assistant<|channel|>final<|message|>Result: 42<|return|>"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert "Step 1" in reasoning
        assert "Step 2" in reasoning
        assert "Step 3" in reasoning
        assert content == "Result: 42"

    def test_no_return_token(self, parser):
        output = (
            "<|channel|>analysis<|message|>Thinking"
            "<|start|>assistant<|channel|>final<|message|>Answer"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Thinking"
        assert content == "Answer"

    def test_streaming_full_flow(self, parser):
        parser.reset_state()

        tokens = [
            "<|channel|>",
            "analysis",
            "<|message|>",
            "Let me ",
            "think",
            "<|start|>",
            "assistant",
            "<|channel|>",
            "final",
            "<|message|>",
            "The answer",
            " is 42",
            "<|return|>",
        ]

        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        full_reasoning = "".join(reasoning_parts)
        full_content = "".join(content_parts)

        assert "Let me think" in full_reasoning
        assert "The answer is 42" in full_content

    def test_streaming_only_final(self, parser):
        parser.reset_state()

        tokens = [
            "<|channel|>",
            "final",
            "<|message|>",
            "Direct ",
            "answer",
            "<|return|>",
        ]

        accumulated = ""
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result and result.content:
                content_parts.append(result.content)

        assert "Direct answer" in "".join(content_parts)

    def test_streaming_suppresses_structural_tokens(self, parser):
        parser.reset_state()

        tokens = [
            "<|channel|>analysis<|message|>",
            "thinking",
            "<|start|>",
            "assistant",
            "<|channel|>final<|message|>",
            "answer",
            "<|return|>",
        ]

        accumulated = ""
        all_output = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.reasoning:
                    all_output.append(result.reasoning)
                if result.content:
                    all_output.append(result.content)

        combined = "".join(all_output)
        assert "<|" not in combined

    def test_registry_includes_gpt_oss(self):
        assert "gpt_oss" in list_parsers()

    def test_extract_constrain_format(self, parser):
        output = (
            "<|channel|>analysis<|message|>We need to output JSON"
            "<|end|><|channel|>final <|constrain|>JSON<|message|>"
            '{"hello":"world"}<|return|>'
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "We need to output JSON"
        assert content == '{"hello":"world"}'

    def test_extract_constrain_no_analysis(self, parser):
        output = (
            '<|channel|>final <|constrain|>JSON<|message|>{"key":"value"}<|return|>'
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == '{"key":"value"}'

    def test_streaming_constrain_format(self, parser):
        parser.reset_state()

        tokens = [
            "<|channel|>analysis<|message|>",
            "Thinking...",
            "<|end|>",
            "<|channel|>final <|constrain|>JSON<|message|>",
            '{"result":',
            '"ok"}',
            "<|return|>",
        ]

        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        full_reasoning = "".join(reasoning_parts)
        full_content = "".join(content_parts)

        assert "Thinking" in full_reasoning
        assert '{"result":"ok"}' in full_content
        assert "<|constrain|>" not in full_content

    def test_constrain_tokens_stripped(self, parser):
        output = (
            '<|channel|>final <|constrain|>JSON<|message|>{"hello":"world"}<|return|>'
        )
        reasoning, content = parser.extract_reasoning(output)
        assert "<|constrain|>" not in (content or "")
        assert "<|channel|>" not in (content or "")


class TestDeepSeekNoTagThreshold:

    @pytest.fixture
    def parser(self):
        return get_parser("deepseek_r1")()

    def test_no_tag_long_output_becomes_content(self, parser):
        parser.reset_state()

        text = "This is a regular response without any thinking tags. " * 3
        assert len(text) > parser.NO_TAG_CONTENT_THRESHOLD
        accumulated = ""
        content_parts = []
        reasoning_parts = []

        for char in text:
            prev = accumulated
            accumulated += char
            result = parser.extract_reasoning_streaming(prev, accumulated, char)
            if result:
                if result.content:
                    content_parts.append(result.content)
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)

        full_content = "".join(content_parts)
        assert len(full_content) > 0, "Long no-tag output should have content"

    def test_with_tags_still_separates_correctly(self, parser):
        parser.reset_state()

        tokens = ["<think>", "reasoning here", "</think>", "content here"]
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        assert "reasoning here" in "".join(reasoning_parts)
        assert "content here" in "".join(content_parts)

    def test_finalize_corrects_short_no_tag_output(self, parser):
        parser.reset_state()

        text = "Short answer."
        accumulated = ""

        for char in text:
            prev = accumulated
            accumulated += char
            parser.extract_reasoning_streaming(prev, accumulated, char)

        correction = parser.finalize_streaming(accumulated)
        assert correction is not None
        assert correction.content == text
        assert correction.reasoning is None

    def test_finalize_no_correction_with_tags(self, parser):
        parser.reset_state()

        text = "<think>thinking</think>answer"
        accumulated = ""

        for char in text:
            prev = accumulated
            accumulated += char
            parser.extract_reasoning_streaming(prev, accumulated, char)

        correction = parser.finalize_streaming(accumulated)
        assert correction is None

    def test_finalize_no_correction_for_long_no_tag(self, parser):
        parser.reset_state()

        text = "A" * (parser.NO_TAG_CONTENT_THRESHOLD + 50)
        accumulated = ""

        for char in text:
            prev = accumulated
            accumulated += char
            parser.extract_reasoning_streaming(prev, accumulated, char)

        correction = parser.finalize_streaming(accumulated)
        assert correction is None

    def test_saw_any_tag_flag_persists(self, parser):
        parser.reset_state()
        assert not parser._saw_any_tag

        text = "<think>test</think>done"
        accumulated = ""
        for char in text:
            prev = accumulated
            accumulated += char
            parser.extract_reasoning_streaming(prev, accumulated, char)

        assert parser._saw_any_tag

        parser.reset_state()
        assert not parser._saw_any_tag


class TestGlm4Parser:

    @pytest.fixture
    def parser(self):
        return get_parser("glm4")()

    def test_registry_includes_glm4(self):
        assert "glm4" in list_parsers()

    def test_extract_with_both_tags(self, parser):
        output = "<think>Let me analyze this</think>The answer is 42."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Let me analyze this"
        assert content == "The answer is 42."

    def test_no_tags_returns_content(self, parser):
        output = "Just a regular response."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == output

    def test_implicit_mode_only_closing_tag(self, parser):
        output = "reasoning text</think>content text"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "reasoning text"
        assert content == "content text"

    def test_strips_box_tags_pure_content(self, parser):
        output = "<|box_begin|>Paris<|box_end|>"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == "Paris"

    def test_strips_box_tags_with_thinking(self, parser):
        output = (
            "<think><|box_begin|>analysis<|box_end|></think>"
            "<|box_begin|>answer<|box_end|>"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "analysis"
        assert content == "answer"

    def test_streaming_no_tags_emits_content(self, parser):
        parser.reset_state()
        result = parser.extract_reasoning_streaming("", "Hello", "Hello")
        assert result is not None
        assert result.content == "Hello"
        assert result.reasoning is None

    def test_streaming_with_thinking(self, parser):
        parser.reset_state()
        tokens = ["<think>", "analyze", "</think>", "answer"]
        accumulated = ""
        reasoning_parts = []
        content_parts = []
        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result is None:
                continue
            if result.reasoning:
                reasoning_parts.append(result.reasoning)
            if result.content:
                content_parts.append(result.content)
        assert "analyze" in "".join(reasoning_parts)
        assert "answer" in "".join(content_parts)

    def test_streaming_strips_box_tags(self, parser):
        parser.reset_state()
        tokens = ["<|box_begin|>", "Paris", "<|box_end|>"]
        accumulated = ""
        content_parts = []
        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result is not None and result.content:
                content_parts.append(result.content)
        full = "".join(content_parts)
        assert "Paris" in full
        assert "<|box_begin|>" not in full
        assert "<|box_end|>" not in full

    def test_streaming_pure_box_tag_delta_returns_none(self, parser):
        parser.reset_state()
        result = parser.extract_reasoning_streaming(
            "", "<|box_begin|>", "<|box_begin|>"
        )
        assert result is None

    def test_streaming_state_resets(self, parser):
        parser.reset_state()
        parser.extract_reasoning_streaming("", "<think>x", "<think>x")
        assert parser._saw_any_tag
        parser.reset_state()
        assert not parser._saw_any_tag
        result = parser.extract_reasoning_streaming("", "fresh", "fresh")
        assert result is not None and result.content == "fresh"
