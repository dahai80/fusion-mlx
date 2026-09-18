# SPDX-License-Identifier: Apache-2.0
"""Golden Reference alignment harness (PR-F, v2 doc §7 L2/L5).

Provides deterministic numeric primitives for verifying that an enhanced
code path (Shim kernels, fused RoPE/RMSNorm, quantized KV, spec decode)
produces logits aligned with the stock MLX reference path:

  - kl_divergence(p, q)    — KL(P‖Q) in nats, FP32-stable
  - logits_kl(ref, test)   — softmax both, return KL
  - assert_logits_aligned  — raise if KL > tol (default 1e-6)

Memory net-growth detection (L5 gate) — sample RSS across a long generation
run (100k tokens) and verify post-warmup growth slope is bounded:

  - MemoryGrowthTracker    — psutil-backed RSS sampler, slope + net-growth

Long-text stability (L2) — 32k-token generation harness records per-chunk
token counts + timing, flags divergence (NaN logits, repeated-token stalls).

All primitives are deterministic and unit-testable without a real model.
Real-model verification lives in tests/unit/test_golden_reference.py under
the ``real_model`` marker (FUSION_MLX_REAL_MODEL_TESTS=1).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

# Default KL tolerance (v2 doc §7 L2): enhanced path must match reference
# logits to within 1e-6 nats. Tighter than fp16 epsilon (~1e-3) because the
# harness runs in fp32; fp16-path comparisons raise the tol at the call site.
DEFAULT_KL_TOL = 1e-6

# Numerical stability floor for log(p/q).
_EPS = 1e-30


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P‖Q) in nats. Both arrays are probability vectors (sum to 1).

    Uses FP64 internally for stability. Returns 0.0 for identical inputs.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + _EPS)
    q = q / (q.sum() + _EPS)
    # KL = Σ p_i log(p_i / q_i); skip p_i == 0 terms (0 * log = 0).
    mask = p > _EPS
    ratio = p[mask] / (q[mask] + _EPS)
    return float(np.sum(p[mask] * np.log(ratio)))


def logits_kl(ref_logits: mx.array, test_logits: mx.array) -> float:
    """Softmax both logits, return KL(ref‖test) in nats.

    Logits may be 1D (vocab,) or 2D (batch, vocab); for 2D the mean KL
    across the batch is returned.
    """
    ref_f = ref_logits.astype(mx.float32)
    test_f = test_logits.astype(mx.float32)
    ref_p = mx.softmax(ref_f, axis=-1)
    test_p = mx.softmax(test_f, axis=-1)
    ref_np = np.asarray(ref_p, dtype=np.float64)
    test_np = np.asarray(test_p, dtype=np.float64)
    if ref_np.ndim == 1:
        return kl_divergence(ref_np, test_np)
    # 2D: mean KL across batch rows.
    kls = [kl_divergence(ref_np[i], test_np[i]) for i in range(ref_np.shape[0])]
    return float(np.mean(kls))


def assert_logits_aligned(
    ref_logits: mx.array,
    test_logits: mx.array,
    tol: float = DEFAULT_KL_TOL,
    label: str = "",
) -> None:
    """Raise AssertionError if KL(ref‖test) > tol."""
    kl = logits_kl(ref_logits, test_logits)
    if kl > tol:
        raise AssertionError(
            f"Logits KL divergence {kl:.3e} exceeds tol {tol:.3e}"
            + (f" [{label}]" if label else "")
        )
    logger.debug("Logits aligned: KL=%.3e tol=%.3e %s", kl, tol, label)


# ---------------------------------------------------------------------------
# Memory net-growth tracker (v2 doc §7 L5).
# ---------------------------------------------------------------------------


@dataclass
class MemorySample:
    step: int  # token / chunk index
    tokens: int  # cumulative tokens generated
    rss_bytes: int  # process RSS at this sample
    elapsed_s: float  # wall-clock since tracker start


@dataclass
class GrowthReport:
    baseline_rss: int
    final_rss: int
    peak_rss: int
    net_growth_bytes: int  # final - baseline
    warmup_net_growth_bytes: int  # growth during warmup phase
    post_warmup_slope_bytes_per_token: float
    n_samples: int
    samples: list[MemorySample] = field(default_factory=list)

    @property
    def net_growth_mb(self) -> float:
        return self.net_growth_bytes / (1024.0 * 1024.0)

    def to_dict(self) -> dict:
        return {
            "baseline_rss_mb": round(self.baseline_rss / 1048576.0, 1),
            "final_rss_mb": round(self.final_rss / 1048576.0, 1),
            "peak_rss_mb": round(self.peak_rss / 1048576.0, 1),
            "net_growth_mb": round(self.net_growth_mb, 2),
            "warmup_net_growth_mb": round(self.warmup_net_growth_bytes / 1048576.0, 2),
            "post_warmup_slope_b_per_tok": round(
                self.post_warmup_slope_bytes_per_token, 2
            ),
            "n_samples": self.n_samples,
        }


class MemoryGrowthTracker:
    """Sample process RSS across a generation run, detect unbounded growth.

    Usage:
        tracker = MemoryGrowthTracker(warmup_tokens=2048)
        for chunk in generate(...):
            tracker.sample(cumulative_tokens)
        report = tracker.report()
        assert report.post_warmup_slope_bytes_per_token < max_slope

    The L5 gate: after a warmup phase (KV cache fills, allocator stabilizes),
    RSS growth per token must be ~0 — a positive slope signals a leak
    (KV cache not trimmed, prefix cache unbounded, activation retained).
    """

    def __init__(
        self,
        warmup_tokens: int = 2048,
        max_post_warmup_slope: float = 1024.0,
    ):
        self.warmup_tokens = warmup_tokens
        self.max_post_warmup_slope = max_post_warmup_slope
        self._samples: list[MemorySample] = []
        self._t0 = time.monotonic()
        self._pid = None
        try:
            import os

            self._pid = os.getpid()
        except Exception:
            self._pid = None

    def _rss(self) -> int:
        try:
            import psutil

            return int(psutil.Process(self._pid).memory_info().rss)
        except Exception as exc:
            logger.debug("MemoryGrowthTracker: psutil unavailable: %s", exc)
            return 0

    def sample(self, cumulative_tokens: int, step: int | None = None) -> MemorySample:
        s = MemorySample(
            step=step if step is not None else len(self._samples),
            tokens=cumulative_tokens,
            rss_bytes=self._rss(),
            elapsed_s=time.monotonic() - self._t0,
        )
        self._samples.append(s)
        return s

    def report(self) -> GrowthReport:
        if len(self._samples) < 2:
            raise ValueError("need >= 2 samples for a growth report")
        if all(s.rss_bytes == 0 for s in self._samples):
            # Fail loudly: an all-zero RSS series means RSS sampling never
            # worked (psutil missing or failed) — the slope would be 0.0
            # and the leak gate would pass vacuously on such hosts.
            raise ValueError(
                "MemoryGrowthTracker: RSS sampling unavailable (psutil "
                "missing or failed) — leak gate cannot be evaluated"
            )
        samples = self._samples
        baseline = samples[0].rss_bytes
        final = samples[-1].rss_bytes
        peak = max(s.rss_bytes for s in samples)
        # Warmup boundary: first sample at/after warmup_tokens.
        warmup_idx = 0
        for i, s in enumerate(samples):
            if s.tokens >= self.warmup_tokens:
                warmup_idx = i
                break
        warmup_rss = samples[warmup_idx].rss_bytes
        warmup_growth = warmup_rss - baseline
        # Post-warmup slope: least-squares fit of RSS vs tokens.
        post = samples[warmup_idx:]
        if len(post) >= 2:
            xs = np.array([s.tokens for s in post], dtype=np.float64)
            ys = np.array([s.rss_bytes for s in post], dtype=np.float64)
            x_mean = xs.mean()
            y_mean = ys.mean()
            denom = np.sum((xs - x_mean) ** 2)
            if denom > 0:
                slope = float(np.sum((xs - x_mean) * (ys - y_mean)) / denom)
            else:
                slope = 0.0
        else:
            slope = 0.0
        return GrowthReport(
            baseline_rss=baseline,
            final_rss=final,
            peak_rss=peak,
            net_growth_bytes=final - baseline,
            warmup_net_growth_bytes=warmup_growth,
            post_warmup_slope_bytes_per_token=slope,
            n_samples=len(samples),
            samples=samples,
        )

    def assert_no_leak(self, report: GrowthReport | None = None) -> None:
        """Raise if post-warmup slope exceeds the configured max."""
        rep = report or self.report()
        if rep.post_warmup_slope_bytes_per_token > self.max_post_warmup_slope:
            raise AssertionError(
                f"Memory leak: post-warmup slope "
                f"{rep.post_warmup_slope_bytes_per_token:.1f} B/token > "
                f"max {self.max_post_warmup_slope:.1f} B/token. "
                f"Net growth {rep.net_growth_mb:.1f} MB over {rep.n_samples} "
                f"samples."
            )


# ---------------------------------------------------------------------------
# Long-text stability harness (v2 doc §7 L2).
# ---------------------------------------------------------------------------


@dataclass
class LongTextReport:
    total_tokens: int
    n_chunks: int
    chunk_size: int
    elapsed_s: float
    tokens_per_s: float
    repeated_token_streak_max: int
    had_nan_logits: bool
    per_chunk_tokens: list[int] = field(default_factory=list)
    per_chunk_latency_ms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "n_chunks": self.n_chunks,
            "tokens_per_s": round(self.tokens_per_s, 1),
            "repeated_token_streak_max": self.repeated_token_streak_max,
            "had_nan_logits": self.had_nan_logits,
        }


def check_long_text_stability(
    token_ids: list[int],
    chunk_size: int = 256,
    max_repeated_streak: int = 32,
) -> LongTextReport:
    """Analyze a generated token stream for stability anomalies.

    Flags:
      - NaN logits (caller must inject had_nan; here we infer from token ids
        being all-zero or sentinel -1).
      - Repeated-token stalls: a run of the same token longer than
        ``max_repeated_streak`` indicates a sampling/degenerate loop.
      - Per-chunk throughput + total tokens.

    Does NOT generate tokens — it analyzes a stream the caller produced
    (real-model test drives generation, passes the id list here).
    """
    n = len(token_ids)
    n_chunks = max(1, (n + chunk_size - 1) // chunk_size)
    # Repeated-token streak detection.
    max_streak = 1
    cur = 1
    for i in range(1, n):
        if token_ids[i] == token_ids[i - 1] and token_ids[i] >= 0:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 1
    had_nan = any(t < 0 for t in token_ids)
    per_chunk = []
    for c in range(n_chunks):
        per_chunk.append(min(chunk_size, n - c * chunk_size))
    return LongTextReport(
        total_tokens=n,
        n_chunks=n_chunks,
        chunk_size=chunk_size,
        elapsed_s=0.0,
        tokens_per_s=0.0,
        repeated_token_streak_max=max_streak,
        had_nan_logits=had_nan,
        per_chunk_tokens=per_chunk,
        per_chunk_latency_ms=[],
    )


def assert_long_text_stable(
    token_ids: list[int],
    chunk_size: int = 256,
    max_repeated_streak: int = 32,
) -> LongTextReport:
    """Run check + raise on anomalies (NaN logits or repeated-token stall)."""
    rep = check_long_text_stability(
        token_ids, chunk_size=chunk_size, max_repeated_streak=max_repeated_streak
    )
    if rep.had_nan_logits:
        raise AssertionError(
            f"Long-text instability: NaN/sentinel tokens in stream "
            f"({sum(1 for t in token_ids if t < 0)} of {rep.total_tokens})"
        )
    if rep.repeated_token_streak_max > max_repeated_streak:
        raise AssertionError(
            f"Long-text instability: repeated-token streak "
            f"{rep.repeated_token_streak_max} > max {max_repeated_streak} "
            f"(degenerate sampling loop)"
        )
    return rep
