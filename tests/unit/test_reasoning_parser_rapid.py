# SPDX-License-Identifier: Apache-2.0
import logging

import pytest

from fusion_mlx.reasoning import (
    DeltaMessage,
    ReasoningParser,
    get_parser,
    list_parsers,
    register_parser,
)

logger = logging.getLogger(__name__)

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


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
        output = (
            THINK_OPEN
            + "Let me analyze this problem"
            + THINK_CLOSE
            + "The answer is 42."
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Let me analyze this problem"
        assert content == "The answer is 42."

    def test_extract_only_reasoning(self, parser):
        output = THINK_OPEN + "Just thinking out loud" + THINK_CLOSE
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Just thinking out loud"
        assert content is None

    def test_no_tags_returns_content_only(self, parser):
        output = "Just a regular response without thinking."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == output

    def test_only_start_tag_truncated(self, parser):
        output = THINK_OPEN + "Started thinking but never finished"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Started thinking but never finished"
        assert content is None

    def test_streaming_simple_flow(self, parser):
        parser.reset_state()
        deltas = [THINK_OPEN, "think", "ing", THINK_CLOSE, "answer"]
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


class TestDeepSeekR1Parser:
    @pytest.fixture
    def parser(self):
        return get_parser("deepseek_r1")()

    def test_extract_with_both_tags(self, parser):
        output = THINK_OPEN + "Step by step analysis" + THINK_CLOSE + "Final answer: 42"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Step by step analysis"
        assert content == "Final answer: 42"

    def test_extract_no_tags_pure_content(self, parser):
        output = "Just a regular response."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == output

    def test_streaming_simple_flow(self, parser):
        parser.reset_state()
        deltas = [THINK_OPEN, "think", "ing", THINK_CLOSE, "answer"]
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

    def test_content_only(self):
        msg = DeltaMessage(content="just content")
        assert msg.content == "just content"
        assert msg.reasoning is None

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

    def test_streaming_reset_state(self, parser):
        parser.reset_state()
        parser.extract_reasoning_streaming("", THINK_OPEN, THINK_OPEN)
        parser.reset_state()
        result = parser.extract_reasoning_streaming("", "content", "content")
        assert result is not None


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
        for char in text:
            prev = accumulated
            accumulated += char
            result = parser.extract_reasoning_streaming(prev, accumulated, char)
            if result and result.content:
                content_parts.append(result.content)
        full_content = "".join(content_parts)
        assert len(full_content) > 0

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

    def test_saw_any_tag_flag_persists(self, parser):
        parser.reset_state()
        assert not parser._saw_any_tag
        text = THINK_OPEN + "test" + THINK_CLOSE + "done"
        accumulated = ""
        for char in text:
            prev = accumulated
            accumulated += char
            parser.extract_reasoning_streaming(prev, accumulated, char)
        assert parser._saw_any_tag
        parser.reset_state()
        assert not parser._saw_any_tag


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

    def test_no_channel_tokens_fallback(self, parser):
        output = "Just a regular response."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == output

    def test_registry_includes_gpt_oss(self):
        assert "gpt_oss" in list_parsers()


class TestGlm4Parser:
    @pytest.fixture
    def parser(self):
        return get_parser("glm4")()

    def test_registry_includes_glm4(self):
        assert "glm4" in list_parsers()

    def test_extract_with_both_tags(self, parser):
        output = THINK_OPEN + "Let me analyze this" + THINK_CLOSE + "The answer is 42."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Let me analyze this"
        assert content == "The answer is 42."

    def test_no_tags_returns_content(self, parser):
        output = "Just a regular response."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == output

    def test_streaming_no_tags_emits_content(self, parser):
        parser.reset_state()
        result = parser.extract_reasoning_streaming("", "Hello", "Hello")
        assert result is not None
        assert result.content == "Hello"
        assert result.reasoning is None
