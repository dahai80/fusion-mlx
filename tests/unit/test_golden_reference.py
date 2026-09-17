# SPDX-License-Identifier: Apache-2.0
"""Tests for the Golden Reference alignment harness (PR-F, v2 doc §7 L2/L5)."""

from __future__ import annotations

import math
import os

import mlx.core as mx
import numpy as np
import pytest

from fusion_mlx.eval.golden_reference import (
    DEFAULT_KL_TOL,
    MemoryGrowthTracker,
    assert_logits_aligned,
    assert_long_text_stable,
    check_long_text_stability,
    kl_divergence,
    logits_kl,
)


class TestKLDivergence:
    def test_identical_distributions_zero(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert kl_divergence(p, p) == 0.0

    def test_known_kl_value(self):
        # P = [0.5, 0.5], Q = [0.9, 0.1]
        # KL = 0.5*ln(0.5/0.9) + 0.5*ln(0.5/0.1)
        p = np.array([0.5, 0.5])
        q = np.array([0.9, 0.1])
        expected = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)
        assert abs(kl_divergence(p, q) - expected) < 1e-12

    def test_nonnegative(self):
        p = np.array([0.1, 0.2, 0.7])
        q = np.array([0.3, 0.3, 0.4])
        assert kl_divergence(p, q) >= 0.0

    def test_handles_zeros_in_p(self):
        # p_i = 0 terms contribute 0 (0 * log = 0).
        p = np.array([0.0, 1.0])
        q = np.array([0.5, 0.5])
        kl = kl_divergence(p, q)
        assert kl == math.log(1.0 / 0.5)

    def test_unnormalized_inputs_normalized(self):
        p = np.array([1.0, 3.0])  # sums to 4
        q = np.array([1.0, 1.0])  # sums to 2
        kl = kl_divergence(p, q)
        # Should match normalized: [0.25,0.75] vs [0.5,0.5]
        expected = 0.25 * math.log(0.25 / 0.5) + 0.75 * math.log(0.75 / 0.5)
        assert abs(kl - expected) < 1e-12


class TestLogitsKL:
    def test_identical_logits_zero(self):
        logits = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
        assert logits_kl(logits, logits) == 0.0

    def test_small_perturbation_small_kl(self):
        ref = mx.array([2.0, 1.0, 0.5, 0.1], dtype=mx.float32)
        test = ref + mx.array([1e-8, 0.0, 0.0, 0.0], dtype=mx.float32)
        kl = logits_kl(ref, test)
        assert kl < 1e-6

    def test_2d_batch_mean(self):
        ref = mx.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=mx.float32)
        kl = logits_kl(ref, ref)
        assert kl == 0.0

    def test_assert_aligned_passes_identical(self):
        logits = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
        assert_logits_aligned(logits, logits, label="identical")

    def test_assert_aligned_raises_on_divergence(self):
        ref = mx.array([0.0, 10.0, 0.0, 0.0], dtype=mx.float32)
        test = mx.array([10.0, 0.0, 0.0, 0.0], dtype=mx.float32)
        with pytest.raises(AssertionError, match="KL divergence"):
            assert_logits_aligned(ref, test)

    def test_default_tol_is_1e_minus_6(self):
        assert DEFAULT_KL_TOL == 1e-6

    def test_custom_tol(self):
        ref = mx.array([0.0, 10.0, 0.0], dtype=mx.float32)
        test = mx.array([0.0, 9.0, 0.0], dtype=mx.float32)
        kl = logits_kl(ref, test)
        # Should pass with a loose tol, fail with tight.
        assert_logits_aligned(ref, test, tol=kl + 0.1)
        with pytest.raises(AssertionError):
            assert_logits_aligned(ref, test, tol=kl - 0.1)


class TestMemoryGrowthTracker:
    def test_sample_appends(self):
        tracker = MemoryGrowthTracker(warmup_tokens=10)
        s0 = tracker.sample(0)
        s1 = tracker.sample(100)
        assert len(tracker._samples) == 2
        assert s0.tokens == 0
        assert s1.tokens == 100

    def test_report_needs_two_samples(self):
        tracker = MemoryGrowthTracker()
        tracker.sample(0)
        with pytest.raises(ValueError, match="2 samples"):
            tracker.report()

    def test_report_flat_growth(self, monkeypatch):
        tracker = MemoryGrowthTracker(warmup_tokens=10)
        # Mock RSS to be flat — no growth.
        monkeypatch.setattr(tracker, "_rss", lambda: 100_000_000)
        for i in range(20):
            tracker.sample(i * 100)
        rep = tracker.report()
        assert rep.net_growth_bytes == 0
        assert rep.post_warmup_slope_bytes_per_token == 0.0
        tracker.assert_no_leak(rep)

    def test_report_detects_leak(self, monkeypatch):
        tracker = MemoryGrowthTracker(warmup_tokens=100, max_post_warmup_slope=100.0)

        # RSS grows 1 MB per 100 tokens after warmup.
        def rss_gen():
            calls = {"i": 0}

            def _rss():
                t = calls["i"] * 100
                calls["i"] += 1
                if t < 100:
                    return 100_000_000
                return 100_000_000 + (t - 100) * 10_485  # ~10KB/token

            return _rss

        monkeypatch.setattr(tracker, "_rss", rss_gen())
        for i in range(20):
            tracker.sample(i * 100)
        rep = tracker.report()
        assert rep.post_warmup_slope_bytes_per_token > 100.0
        with pytest.raises(AssertionError, match="Memory leak"):
            tracker.assert_no_leak(rep)

    def test_warmup_growth_separated(self, monkeypatch):
        tracker = MemoryGrowthTracker(warmup_tokens=500)
        # Big growth during warmup, flat after.
        seq = [100_000_000 + i * 1_000_000 for i in range(5)] + [105_000_000] * 15

        def _rss():
            return seq[len(tracker._samples)]

        monkeypatch.setattr(tracker, "_rss", _rss)
        for i in range(20):
            tracker.sample(i * 100)
        rep = tracker.report()
        assert rep.warmup_net_growth_bytes > 0
        assert rep.post_warmup_slope_bytes_per_token <= 1.0
        tracker.assert_no_leak(rep)

    def test_report_to_dict(self, monkeypatch):
        tracker = MemoryGrowthTracker()
        monkeypatch.setattr(tracker, "_rss", lambda: 50_000_000)
        tracker.sample(0)
        tracker.sample(100)
        d = tracker.report().to_dict()
        assert "net_growth_mb" in d
        assert "post_warmup_slope_b_per_tok" in d


class TestLongTextStability:
    def test_normal_stream(self):
        ids = list(range(256)) * 4  # 1024 tokens, no long streak
        rep = check_long_text_stability(ids, chunk_size=256)
        assert rep.total_tokens == 1024
        assert rep.had_nan_logits is False
        assert rep.repeated_token_streak_max == 1

    def test_detects_repeated_stall(self):
        ids = [0] * 100 + [1, 2, 3]
        with pytest.raises(AssertionError, match="repeated-token streak"):
            assert_long_text_stable(ids, max_repeated_streak=32)

    def test_allows_short_repeats(self):
        ids = [0] * 10 + [1] * 10 + [2] * 10
        rep = assert_long_text_stable(ids, max_repeated_streak=32)
        assert rep.repeated_token_streak_max == 10

    def test_detects_nan_sentinel(self):
        ids = [1, 2, 3, -1, 5]
        with pytest.raises(AssertionError, match="NaN/sentinel"):
            assert_long_text_stable(ids)

    def test_chunk_counts(self):
        ids = list(range(700))
        rep = check_long_text_stability(ids, chunk_size=256)
        assert rep.n_chunks == 3
        assert rep.per_chunk_tokens == [256, 256, 188]

    def test_report_to_dict(self):
        ids = list(range(100))
        rep = check_long_text_stability(ids)
        d = rep.to_dict()
        assert d["total_tokens"] == 100
        assert d["had_nan_logits"] is False


# ---------------------------------------------------------------------------
# Real-model tests — gated by FUSION_MLX_REAL_MODEL_TESTS + running server.
# These exercise the harness against a live fusion-mlx server: generate a
# long stream, check stability + memory net-growth.
# ---------------------------------------------------------------------------

_REAL = os.environ.get("FUSION_MLX_REAL_MODEL_TESTS", "0") == "1"
_HOST = os.environ.get("FUSION_HOST", "127.0.0.1:11434")
_API_KEY = os.environ.get("FUSION_MLX_API_KEY", "fg-admin-key")
_TEST_MODEL = os.environ.get("FUSION_GOLDEN_MODEL", "")


def _server_up() -> bool:
    try:
        import httpx

        r = httpx.get(f"http://{_HOST}/v1/models", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.real_model
@pytest.mark.skipif(
    not (_REAL and _server_up() and _TEST_MODEL),
    reason="needs FUSION_MLX_REAL_MODEL_TESTS=1 + running server + FUSION_GOLDEN_MODEL",
)
class TestRealModelGoldenReference:
    def _generate_stream(self, prompt: str, max_tokens: int) -> list[int]:
        import httpx

        ids: list[int] = []
        with httpx.stream(
            "POST",
            f"http://{_HOST}/v1/completions",
            headers={"Authorization": f"Bearer {_API_KEY}"},
            json={
                "model": _TEST_MODEL,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": True,
            },
            timeout=300.0,
        ) as resp:
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                import json

                chunk = json.loads(line[6:])
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                # Non-streaming-shape token ids not exposed by /completions;
                # use the text length as a proxy stability signal. A real
                # token-id stream would need /v1/completions logprobs.
                text = choices[0].get("text", "")
                ids.extend([ord(c) for c in text if ord(c) >= 0])
        return ids

    def test_long_text_stability_4k(self):
        ids = self._generate_stream(
            "Write a long detailed essay about the history of computing.",
            max_tokens=4096,
        )
        assert len(ids) > 100, "server produced too few tokens"
        rep = assert_long_text_stable(ids, max_repeated_streak=64)
        assert rep.had_nan_logits is False

    def test_memory_growth_no_leak(self):
        tracker = MemoryGrowthTracker(warmup_tokens=512, max_post_warmup_slope=2048.0)
        # Sample in chunks by re-issuing short generations.
        for i in range(10):
            self._generate_stream("Continue: ", max_tokens=512)
            tracker.sample((i + 1) * 512)
        rep = tracker.report()
        # Log the report for visibility; assert no leak.
        tracker.assert_no_leak(rep)
