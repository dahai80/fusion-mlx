# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.eval benchmark series (BBQ/HellaSwag/JMMLU/MathQA/MMLU-Pro/SafetyBench/WinoGrande).

All 7 share the BaseBenchmark contract: load_dataset normalizes bundled JSONL,
format_prompt renders multiple-choice, extract_answer scans response for valid
labels, check_answer compares equality, get_category returns None or a topic.
Mocks load_jsonl to avoid needing the bundled data files. Aims at ≥90% line
coverage of each benchmark module.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fusion_mlx.eval import (
    bbq,
    hellaswag,
    jmmlu,
    mathqa,
    mmlu_pro,
    safetybench,
    winogrande,
)

# Benchmarks whose format_prompt/check_answer use the shared question/choices/labels/answer
# schema. WinoGrande (sentence/option1/option2) and HellaSwag (context/endings/answer-letter)
# have non-standard schemas and are covered by their dedicated Special classes below —
# exclude them from the parameterized TestContract so the shared-item fixture fits.
BENCHES = [
    (bbq, "BBQBenchmark", "bbq", 300, ["A", "B", "C"]),
    (jmmlu, "JMMLUBenchmark", "jmmlu", 300, ["A", "B", "C", "D"]),
    (mathqa, "MathQABenchmark", "mathqa", 300, ["A", "B", "C", "D", "E"]),
    (
        mmlu_pro,
        "MMLUProBenchmark",
        "mmlu_pro",
        300,
        ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    ),
    (safetybench, "SafetyBenchBenchmark", "safetybench", 300, ["A", "B", "C", "D"]),
]


@pytest.fixture(params=BENCHES, ids=[b[1] for b in BENCHES])
def bench_ctx(request):
    module, cls_name, name, qs, valid = request.param
    cls = getattr(module, cls_name)
    return {
        "module": module,
        "cls": cls,
        "name": name,
        "quick_size": qs,
        "valid": valid,
        "bench": cls(),
    }


class TestContract:
    def test_name(self, bench_ctx):
        assert bench_ctx["bench"].name == bench_ctx["name"]

    def test_quick_size(self, bench_ctx):
        assert bench_ctx["bench"].quick_size == bench_ctx["quick_size"]

    @pytest.mark.asyncio
    async def test_load_dataset_no_sample_returns_all(self, bench_ctx):
        # Include context/subject so BBQ (needs context) + JMMLU (needs subject)
        # load_dataset normalization passes.
        items = [
            {
                "context": "ctx",
                "subject": "math",
                "question": "q",
                "choices": ["a", "b"],
                "labels": ["A", "B"],
                "answer": "A",
            }
        ]
        with patch.object(bench_ctx["module"], "load_jsonl", return_value=items):
            result = await bench_ctx["bench"].load_dataset()
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_load_dataset_sample_size(self, bench_ctx):
        items = [
            {
                "context": "ctx",
                "subject": "math",
                "question": f"q{i}",
                "choices": ["a", "b"],
                "labels": ["A", "B"],
                "answer": "A",
            }
            for i in range(20)
        ]
        with patch.object(bench_ctx["module"], "load_jsonl", return_value=items):
            result = await bench_ctx["bench"].load_dataset(sample_size=5)
            assert len(result) == 5

    def test_format_prompt_returns_user_msg(self, bench_ctx):
        # BBQ needs context+question+choices+labels; JMMLU needs subject+question+choices+labels;
        # MathQA/MMLU-Pro/SafetyBench need question+choices+labels. Include all fields so the
        # parameterized fixture fits every shared-schema bench.
        item = {
            "context": "ctx",
            "question": "q?",
            "subject": "math",
            "choices": ["a", "b"],
            "labels": ["A", "B"],
            "answer": "A",
        }
        msgs = bench_ctx["bench"].format_prompt(item)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert isinstance(msgs[0]["content"], str)

    def test_extract_answer_scans_response(self, bench_ctx):
        item = {
            "question": "q",
            "choices": ["a", "b"],
            "labels": ["A", "B"],
            "answer": "A",
        }
        result = bench_ctx["bench"].extract_answer("The answer is A", item)
        assert result in bench_ctx["valid"]

    def test_check_answer_match(self, bench_ctx):
        item = {"answer": "A"}
        assert bench_ctx["bench"].check_answer("A", item) is True

    def test_check_answer_mismatch(self, bench_ctx):
        item = {"answer": "A"}
        assert bench_ctx["bench"].check_answer("B", item) is False

    def test_get_category_returns_none_or_str(self, bench_ctx):
        result = bench_ctx["bench"].get_category({})
        assert result is None or isinstance(result, str)


class TestWinograndeSpecial:
    """WinoGrande uses sentence + option1/option2 (no underscore), not question/choices."""

    def test_format_prompt_uses_sentence(self):
        b = winogrande.WinograndeBenchmark()
        item = {
            "sentence": "The cat ___ on the mat.",
            "option1": "sat",
            "option2": "ran",
            "answer": "1",
        }
        msgs = b.format_prompt(item)
        content = msgs[0]["content"]
        assert "sat" in content
        assert "ran" in content

    def test_extract_answer_uses_1_2_labels(self):
        b = winogrande.WinograndeBenchmark()
        assert b.extract_answer("I choose 1", {}) in ("1", "2")

    def test_check_answer_uses_string_answer(self):
        b = winogrande.WinograndeBenchmark()
        assert b.check_answer("1", {"answer": "1"}) is True


class TestJMMLUSpecial:
    """JMMLU is Japanese — format_prompt needs subject + ends with '答え:'."""

    def test_format_prompt_japanese_suffix(self):
        b = jmmlu.JMMLUBenchmark()
        item = {
            "question": "日本語",
            "choices": ["a", "b"],
            "labels": ["A", "B"],
            "answer": "A",
            "subject": "math",
        }
        content = b.format_prompt(item)[0]["content"]
        assert "答え:" in content

    def test_get_category_returns_subject(self):
        b = jmmlu.JMMLUBenchmark()
        assert b.get_category({"subject": "math"}) == "math"


class TestHellaSwagSpecial:
    """HellaSwag uses context + endings + label (int 0-3), not choices/labels."""

    def test_format_prompt_uses_endings(self):
        b = hellaswag.HellaSwagBenchmark()
        item = {
            "context": "Context here.",
            "endings": ["e1", "e2", "e3", "e4"],
            "label": 2,
        }
        msgs = b.format_prompt(item)
        content = msgs[0]["content"]
        assert "Context here." in content
        assert "e1" in content
        assert "e4" in content

    def test_check_answer_against_expected_letter(self):
        b = hellaswag.HellaSwagBenchmark()
        # check_answer does ANSWER_MAP.get(item["answer"]) where answer is the
        # raw int label (0-3), not the letter. label=2 → expected letter "C".
        item = {"answer": 2}
        assert b.check_answer("C", item) is True
        assert b.check_answer("A", item) is False
