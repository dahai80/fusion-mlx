# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.community_bench (hardware/runner/submission).

The runner/submission surfaces evolved from NotImplementedError stubs into
real implementations. hardware.detect_hardware() returns a dict with real
system info; runner.run_standardized_bench is async (needs engine+tokenizer);
submission is no-op (local-only, no network).
"""

from __future__ import annotations

from fusion_mlx.community_bench import hardware, runner, submission


class TestHardware:
    def test_detect_hardware_returns_dict(self):
        result = hardware.detect_hardware()
        assert isinstance(result, dict)
        assert "chip" in result
        assert "ram_gb" in result
        assert "gpu_cores" in result

    def test_collect_returns_tuple(self):
        result = hardware.collect()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_is_apple_silicon_returns_bool(self):
        assert isinstance(hardware.is_apple_silicon(), bool)

    def test_logger_defined(self):
        assert hardware.logger is not None


class TestRunner:
    def test_run_standardized_bench_is_async(self):
        import inspect

        assert inspect.iscoroutinefunction(runner.run_standardized_bench)

    def test_bench_result_dataclass_exists(self):
        assert hasattr(runner, "BenchResult")
        assert hasattr(runner, "BucketResult")
        assert hasattr(runner, "StatResult")

    def test_logger_defined(self):
        assert runner.logger is not None


class TestSubmission:
    def test_submit_benchmark_returns_dict(self):
        result = submission.submit_benchmark()
        assert isinstance(result, dict)

    def test_submit_returns_dict(self):
        result = submission.submit()
        assert isinstance(result, dict)

    def test_submit_accepts_args(self):
        result = submission.submit("arg", kw="val")
        assert isinstance(result, dict)

    def test_logger_defined(self):
        assert submission.logger is not None
