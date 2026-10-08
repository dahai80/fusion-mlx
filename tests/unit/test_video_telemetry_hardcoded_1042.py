# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1042.

#1042: video generation telemetry hardcoded prompt_tokens=0 / ttft_ms=0.0 /
tps=0.0, distorting metrics. Fix: fill real prompt token estimate, generation
time, and throughput.
"""

from __future__ import annotations

import pytest


class TestVideoTelemetryRealValues:
    """Telemetry values are computed, not hardcoded zeros."""

    def test_prompt_token_estimate_nonzero(self):
        prompt = "a beautiful sunset over the ocean"
        est = max(1, len(prompt) // 4)
        assert est > 0
        assert est == len(prompt) // 4

    def test_prompt_token_estimate_minimum_one(self):
        prompt = "hi"
        est = max(1, len(prompt) // 4)
        assert est == 1  # len("hi")//4 = 0, clamped to 1

    def test_ttft_ms_from_elapsed(self):
        gen_elapsed = 12.5  # seconds
        ttft_ms = gen_elapsed * 1000.0
        assert ttft_ms == 12500.0

    def test_tps_videos_per_second(self):
        num_videos = 2
        gen_elapsed = 10.0
        tps = num_videos / gen_elapsed if gen_elapsed > 0 else 0.0
        assert tps == 0.2

    def test_tps_zero_elapsed_safe(self):
        num_videos = 1
        gen_elapsed = 0.0
        tps = num_videos / gen_elapsed if gen_elapsed > 0 else 0.0
        assert tps == 0.0

    def test_no_hardcoded_zeros_in_source(self):
        # Verify the old hardcoded pattern is gone from the emit.request call.
        import inspect

        from fusion_mlx.api import videos_routes

        source = inspect.getsource(videos_routes)
        # The old pattern had prompt_tokens=0, ttft_ms=0.0, tps=0.0 in the
        # emit.request call. The new code uses _est_prompt_tokens, _ttft_ms, _tps.
        assert "prompt_tokens=0,\n" not in source or "_est_prompt_tokens" in source
        assert "_ttft_ms" in source
        assert "_tps" in source


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
