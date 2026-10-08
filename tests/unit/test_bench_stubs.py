# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.bench stubs (__init__.py + tier_runner.py)."""

from __future__ import annotations

import pytest

from fusion_mlx.bench import BenchmarkRunnerUnavailable, run_benchmark, tier_runner


class TestRunBenchmark:
    # #1012: run_benchmark was returning {"tokens_per_second": 0} fake data.
    # Now raises BenchmarkRunnerUnavailable (fail-visible, matches tier_runner).

    def test_raises_unavailable(self):
        with pytest.raises(BenchmarkRunnerUnavailable, match="no runner"):
            run_benchmark("test-model")

    def test_raises_with_kwargs(self):
        with pytest.raises(BenchmarkRunnerUnavailable):
            run_benchmark("x", batch=4, warmup=2)

    def test_raises_empty_model_name(self):
        with pytest.raises(BenchmarkRunnerUnavailable):
            run_benchmark("")


class TestRunTier:
    def test_raises_not_implemented(self):
        with pytest.raises(tier_runner.TierRunnerUnavailable, match="not implemented"):
            tier_runner.run_tier()

    def test_raises_with_args(self):
        with pytest.raises(tier_runner.TierRunnerUnavailable):
            tier_runner.run_tier("arg", kw="val")

    def test_logger_defined(self):
        assert tier_runner.logger is not None
