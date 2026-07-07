# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.community_bench stubs (hardware/runner/submission)."""

from __future__ import annotations

import pytest

from fusion_mlx.community_bench import hardware, runner, submission


class TestHardwareStub:
    def test_detect_hardware_returns_empty_dict(self):
        assert hardware.detect_hardware() == {}

    def test_collect_returns_empty_dict(self):
        assert hardware.collect() == {}

    def test_is_apple_silicon_returns_true(self):
        assert hardware.is_apple_silicon() is True

    def test_logger_defined(self):
        assert hardware.logger is not None


class TestRunnerStub:
    def test_run_benchmark_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="not available"):
            runner.run_benchmark()

    def test_run_benchmark_raises_with_args(self):
        with pytest.raises(NotImplementedError):
            runner.run_benchmark("arg", kw="val")

    def test_run_standardized_bench_raises(self):
        with pytest.raises(NotImplementedError, match="not available"):
            runner.run_standardized_bench()

    def test_logger_defined(self):
        assert runner.logger is not None


class TestSubmissionStub:
    def test_submit_benchmark_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="not available"):
            submission.submit_benchmark()

    def test_submit_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="not available"):
            submission.submit()

    def test_submit_raises_with_args(self):
        with pytest.raises(NotImplementedError):
            submission.submit("arg", kw="val")

    def test_logger_defined(self):
        assert submission.logger is not None
