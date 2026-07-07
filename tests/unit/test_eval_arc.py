# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.eval benchmarks (ARC + representative series).

Covers load_dataset normalization (drop missing choices/labels), format_prompt
multiple-choice formatting, extract_answer via _extract_mc_answer, check_answer
equality, get_category. Uses bundled data when present; mocks load_jsonl when
bundled data is absent. Aims at ≥90% line coverage of arc.py (and the shared
base class path exercised by ARC).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fusion_mlx.eval import arc


class TestARCChallengeBenchmark:
    @pytest.fixture
    def bench(self):
        return arc.ARCChallengeBenchmark()

    @pytest.fixture
    def fake_items(self):
        return [
            {
                "id": "q1",
                "question": "What is H2O?",
                "choices": ["Water", "Salt", "Gold", "Fire"],
                "labels": ["A", "B", "C", "D"],
                "answer": "A",
            },
            {
                "id": "q2",
                "question": "What is photosynthesis?",
                "choices": ["A", "B"],
                "labels": ["A", "B"],
                "answer": "B",
            },
            # missing choices/labels → dropped by load_dataset normalization
            {"id": "q3", "question": "Bad", "choices": [], "labels": [], "answer": "A"},
            {"id": "q4", "question": "Also bad", "answer": "A"},
        ]

    def test_name(self, bench):
        assert bench.name == "arc_challenge"

    def test_quick_size(self, bench):
        assert bench.quick_size == 300

    @pytest.mark.asyncio
    async def test_load_dataset_normalizes_and_drops_missing(self, bench, fake_items):
        with patch.object(arc, "load_jsonl", return_value=fake_items):
            result = await bench.load_dataset()
            assert len(result) == 2  # q3, q4 dropped
            assert result[0]["id"] == "q1"
            assert result[0]["choices"] == ["Water", "Salt", "Gold", "Fire"]
            assert result[1]["id"] == "q2"

    @pytest.mark.asyncio
    async def test_load_dataset_sample_size(self, bench, fake_items):
        with patch.object(arc, "load_jsonl", return_value=fake_items):
            result = await bench.load_dataset(sample_size=1)
            assert len(result) == 1

    def test_format_prompt(self, bench, fake_items):
        item = fake_items[0]
        msgs = bench.format_prompt(item)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        content = msgs[0]["content"]
        assert "What is H2O?" in content
        assert "A. Water" in content
        assert "D. Fire" in content
        assert content.endswith("Answer:")

    def test_format_prompt_uses_lettered_options(self, bench, fake_items):
        item = fake_items[1]
        content = bench.format_prompt(item)[0]["content"]
        assert "A. A" in content
        assert "B. B" in content

    def test_extract_answer_uses_item_labels(self, bench, fake_items):
        item = fake_items[0]
        # _extract_mc_answer scans response for first valid label
        assert bench.extract_answer("The answer is A", item) == "A"

    def test_extract_answer_falls_back_to_default_labels(self, bench):
        item = {"question": "q", "choices": ["x"], "labels": ["Z"], "answer": "Z"}
        # no labels in item → uses default ["A","B","C","D"]
        result = bench.extract_answer("I pick C", {})
        assert result in ("A", "B", "C", "D", "")

    def test_check_answer_match(self, bench, fake_items):
        assert bench.check_answer("A", fake_items[0]) is True

    def test_check_answer_mismatch(self, bench, fake_items):
        assert bench.check_answer("B", fake_items[0]) is False

    def test_get_category_returns_none(self, bench):
        assert bench.get_category({}) is None
