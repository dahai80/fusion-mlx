# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.eval.__init__ — BENCHMARKS registry."""

from __future__ import annotations

from fusion_mlx.eval import BENCHMARKS


class TestBenchmarksRegistry:
    def test_registry_is_dict(self):
        assert isinstance(BENCHMARKS, dict)

    def test_expected_keys_present(self):
        expected = [
            "mmlu",
            "mmlu_pro",
            "kmmlu",
            "cmmlu",
            "jmmlu",
            "hellaswag",
            "truthfulqa",
            "arc_challenge",
            "winogrande",
            "gsm8k",
            "mathqa",
            "humaneval",
            "mbpp",
            "livecodebench",
            "bbq",
            "safetybench",
        ]
        for k in expected:
            assert k in BENCHMARKS, f"missing benchmark key {k}"

    def test_all_values_are_classes(self):
        for name, cls in BENCHMARKS.items():
            assert isinstance(cls, type), f"{name} value is not a class"

    def test_registry_size(self):
        assert len(BENCHMARKS) == 16
