# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.eval.datasets — JSONL loader + sampling helpers."""

from __future__ import annotations

from fusion_mlx.eval.datasets import (
    SAMPLE_SEED,
    deterministic_sample,
    load_jsonl,
    stratified_sample,
)


class TestLoadJsonl:
    def test_loads_valid_jsonl(self, tmp_path):
        f = tmp_path / "x.jsonl"
        f.write_text('{"a":1}\n{"b":2}\n')
        assert load_jsonl(f) == [{"a": 1}, {"b": 2}]

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "x.jsonl"
        f.write_text('{"a":1}\n\n  \n{"b":2}\n')
        assert load_jsonl(f) == [{"a": 1}, {"b": 2}]

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "x.jsonl"
        f.write_text("")
        assert load_jsonl(f) == []


class TestDeterministicSample:
    def test_n_ge_len_returns_all(self):
        items = [{"i": i} for i in range(5)]
        assert deterministic_sample(items, 10) == items
        assert deterministic_sample(items, 5) == items

    def test_deterministic_same_subset_same_input(self):
        items = [{"i": i} for i in range(100)]
        a = deterministic_sample(items, 10)
        b = deterministic_sample(items, 10)
        assert a == b  # fixed seed → reproducible

    def test_subset_size_n(self):
        items = [{"i": i} for i in range(100)]
        assert len(deterministic_sample(items, 10)) == 10


class TestStratifiedSample:
    def test_n_ge_len_returns_all(self):
        items = [{"c": "a"}, {"c": "b"}]
        assert stratified_sample(items, 10, "c") == items

    def test_stratified_reproducible(self):
        items = [{"c": cat, "i": i} for i in range(20) for cat in ("a", "b")]
        a = stratified_sample(items, 5, "c")
        b = stratified_sample(items, 5, "c")
        assert a == b

    def test_missing_key_grouped_as_unknown(self):
        items = [{"i": 1}, {"c": "a", "i": 2}]
        result = stratified_sample(items, 2, "c")
        assert len(result) <= 2

    def test_subset_size_le_n(self):
        items = [{"c": cat, "i": i} for i in range(50) for cat in ("a", "b", "c")]
        result = stratified_sample(items, 5, "c")
        assert len(result) <= 5


class TestConstants:
    def test_sample_seed_is_42(self):
        assert SAMPLE_SEED == 42
