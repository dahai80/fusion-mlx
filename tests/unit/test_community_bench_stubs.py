# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.community_bench stubs (hardware/runner/submission).

The runner/submission surfaces evolved from NotImplementedError stubs into
local-only real implementations (no network, no model load) so callers that
import the community_bench surface get a usable result instead of a crash.
These tests pin the local-only contract: every entrypoint returns a dict
tagged with ``source="local"`` (runner) / a no-op descriptor (submission),
and accepts arbitrary args/kwargs without raising.
"""

from __future__ import annotations

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
    def test_run_benchmark_returns_local_dict(self):
        result = runner.run_benchmark()
        assert isinstance(result, dict)
        assert result["source"] == "local"

    def test_run_benchmark_accepts_args(self):
        result = runner.run_benchmark("arg", kw="val")
        assert isinstance(result, dict)
        assert result["source"] == "local"

    def test_run_standardized_bench_returns_local_dict(self):
        result = runner.run_standardized_bench()
        assert isinstance(result, dict)
        assert result["source"] == "local"
        assert result["standardized"] is True

    def test_logger_defined(self):
        assert runner.logger is not None


class TestSubmissionStub:
    def test_submit_benchmark_returns_noop_descriptor(self):
        result = submission.submit_benchmark()
        assert isinstance(result, dict)

    def test_submit_returns_noop_descriptor(self):
        result = submission.submit()
        assert isinstance(result, dict)

    def test_submit_accepts_args(self):
        result = submission.submit("arg", kw="val")
        assert isinstance(result, dict)

    def test_logger_defined(self):
        assert submission.logger is not None
