# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.bench stubs (__init__.py + tier_runner.py)."""

from __future__ import annotations

import pytest

from fusion_mlx.bench import run_benchmark, tier_runner


class TestRunBenchmark:
    def test_returns_tokens_per_second_zero(self):
        result = run_benchmark("test-model")
        assert result == {"tokens_per_second": 0}

    def test_ignores_kwargs(self):
        result = run_benchmark("x", batch=4, warmup=2)
        assert result == {"tokens_per_second": 0}

    def test_empty_model_name(self):
        assert run_benchmark("") == {"tokens_per_second": 0}


class TestRunTier:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="not available"):
            tier_runner.run_tier()

    def test_raises_with_args(self):
        with pytest.raises(NotImplementedError):
            tier_runner.run_tier("arg", kw="val")

    def test_logger_defined(self):
        assert tier_runner.logger is not None
