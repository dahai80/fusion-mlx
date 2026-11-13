# SPDX-License-Identifier: Apache-2.0
"""N11 hybrid drafting — n-gram prefix chains into DSpine drafts.

Design: fusion-research/docs/n11-hybrid-drafting-1113.zh-CN.md (Hybrid
Spark, arXiv:2611.05611). Per-step serial chaining, NOT parallel dual-tree:
each decode step first queries a zero-forward-cost n-gram predictor; a hit
of >= min_prefix_tokens continuations builds the draft head (capped at
ngram_draft_tokens), the DSpine drafter continues the chain after it (total
capped at max_draft_total), and the existing T>0 sampled rejection-verify
path consumes the single chain unchanged.

Why serial chaining and not dual drafters: on Apple b1 the verify forward
is bandwidth-bound, so widening the verify tree costs even when drafts are
wrong. A n-gram hit costs O(1) table lookups and only deepens the chain
when it exists; a miss degrades to pure DSpine with zero added GPU work.

Lossless: n-gram head tokens enter the rejection verifier as point-mass
proposals (one-hot q rows). Accept coin < p(token), rejection resamples
from (p - q)+ — the emitted marginal equals the target distribution for
any proposal source, so T>0 output stays distribution-identical and the
greedy (T=0) path is untouched entirely.

Config follows the scheduler-side DSpine knob convention (env vars read
once at state construction, e.g. FUSION_DSPINE_PREFILL_BOOTSTRAP). Default
OFF: the GPU experiment gate in the design doc (math >= 1.46x, repeat
>= 1.55x, agent >= 1.3x, greedy diff = 0) has not run yet — enable with
FUSION_DSPINE_HYBRID=1 and flip the default only after the gate passes.
"""

import logging
import os
from dataclasses import dataclass

from ..ngram_predictor import NGramPredictor

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        val = int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r, using default %d", name, raw, default)
        return default
    if val < lo or val > hi:
        logger.warning(
            "%s=%d out of range [%d, %d], using default %d",
            name,
            val,
            lo,
            hi,
            default,
        )
        return default
    return val


def _env_float(name: str, default: float, lo: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        val = float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r, using default %s", name, raw, default)
        return default
    if val < lo:
        logger.warning("%s=%s below %s, using default %s", name, raw, lo, default)
        return default
    return val


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "off")


@dataclass(frozen=True)
class HybridDraftConfig:
    """N11 hybrid drafting knobs (all env-overridable, see from_env)."""

    enabled: bool = False
    ngram_draft_tokens: int = 4
    min_prefix_tokens: int = 2
    max_draft_total: int = 8
    degrade_window: int = 32
    degrade_min_accept: float = 1.2

    @classmethod
    def from_env(cls) -> "HybridDraftConfig":
        cfg = cls(
            enabled=_env_bool("FUSION_DSPINE_HYBRID", False),
            ngram_draft_tokens=_env_int("FUSION_DSPINE_HYBRID_NGRAM_TOKENS", 4, 1, 16),
            min_prefix_tokens=_env_int("FUSION_DSPINE_HYBRID_MIN_PREFIX", 2, 1, 16),
            max_draft_total=_env_int("FUSION_DSPINE_HYBRID_MAX_TOTAL", 8, 2, 64),
            degrade_window=_env_int("FUSION_DSPINE_HYBRID_DEGRADE_WINDOW", 32, 1, 4096),
            degrade_min_accept=_env_float(
                "FUSION_DSPINE_HYBRID_DEGRADE_MIN_ACCEPT", 1.2, 0.0
            ),
        )
        if cfg.ngram_draft_tokens >= cfg.max_draft_total:
            # The head must leave room for at least one drafter suffix
            # token, otherwise the "hybrid" step is a pure n-gram draft.
            cfg = cls(
                enabled=cfg.enabled,
                ngram_draft_tokens=cfg.max_draft_total - 1,
                min_prefix_tokens=min(cfg.min_prefix_tokens, cfg.max_draft_total - 1),
                max_draft_total=cfg.max_draft_total,
                degrade_window=cfg.degrade_window,
                degrade_min_accept=cfg.degrade_min_accept,
            )
            logger.warning(
                "FUSION_DSPINE_HYBRID_NGRAM_TOKENS >= MAX_TOTAL; head cap "
                "clamped to %d",
                cfg.ngram_draft_tokens,
            )
        return cfg


class HybridDraftState:
    """Per-scheduler N11 bookkeeping: n-gram predictor + source counters +
    accepted-from-n-gram degradation window.

    Counters are cumulative (lifecycle) like the DSpineSpecState total_*
    fields; the n-gram predictor resets per request (same convention as
    NGramSpecState). The degradation window is workload-level and persists
    across requests.
    """

    def __init__(self, cfg: HybridDraftConfig | None = None):
        self.cfg = cfg if cfg is not None else HybridDraftConfig.from_env()
        self.predictor = NGramPredictor(num_draft=self.cfg.ngram_draft_tokens)
        self.total_hybrid_steps = 0
        self.total_ngram_tokens = 0
        self.total_drafter_tokens = 0
        self.total_accepted_from_ngram = 0
        self.degraded = False
        self._acc_window: list[int] = []

    def reset_request(self):
        # Predictor history is per-request (continuous batching must not
        # leak request A's n-grams into request B — see NGramPredictor.reset).
        self.predictor.reset()

    def add_token(self, token: int):
        if not self.cfg.enabled:
            return
        self.predictor.add_token(token)

    def add_tokens(self, tokens: list[int]):
        if not self.cfg.enabled:
            return
        for t in tokens:
            self.predictor.add_token(t)

    def prefix_for_step(self) -> list[int]:
        """Draft-head candidates for this step, [] when pure DSpine."""
        if self.degraded or not self.cfg.enabled:
            return []
        drafts = self.predictor.predict()
        if len(drafts) < self.cfg.min_prefix_tokens:
            return []
        return drafts[: self.cfg.ngram_draft_tokens]

    def record_round(self, ngram_len: int, drafter_len: int, n_accepted: int):
        """Record one hybrid step's source split for telemetry + fallback.

        accepted_from_ngram = min(n_accepted, ngram_len): rejection
        sampling accepts a prefix of the chain, so accepted positions map
        onto the head before the drafter suffix.
        """
        self.total_hybrid_steps += 1
        self.total_ngram_tokens += ngram_len
        self.total_drafter_tokens += drafter_len
        acc_ngram = min(n_accepted, ngram_len)
        self.total_accepted_from_ngram += acc_ngram
        self._acc_window.append(acc_ngram)
        if len(self._acc_window) > self.cfg.degrade_window:
            self._acc_window.pop(0)
        if not self.degraded and len(self._acc_window) == self.cfg.degrade_window:
            mean = sum(self._acc_window) / len(self._acc_window)
            if mean < self.cfg.degrade_min_accept:
                self.degraded = True
                logger.warning(
                    "dspine hybrid: n-gram head accepted length mean %.2f < "
                    "%.2f over %d hybrid steps — degrading to pure DSpine "
                    "(telemetry key hybrid_degraded; auto_router can consume "
                    "it once bucket-acceptance hysteresis lands)",
                    mean,
                    self.cfg.degrade_min_accept,
                    self.cfg.degrade_window,
                )

    def get_stats(self) -> dict:
        window = self._acc_window
        return {
            "hybrid_enabled": self.cfg.enabled,
            "hybrid_degraded": self.degraded,
            "ngram_prefix_hit": self.total_hybrid_steps,
            "ngram_len": self.total_ngram_tokens,
            "drafter_len": self.total_drafter_tokens,
            "accepted_from_ngram": self.total_accepted_from_ngram,
            "hybrid_window_mean": (sum(window) / len(window) if window else 0.0),
            "hybrid_window_filled": len(window),
        }


def hybrid_suffix_len(cfg: HybridDraftConfig, prefix_len: int, block_size: int) -> int:
    """Drafter continuation length for a hybrid step.

    Total chain (prefix + suffix) is capped at min(max_draft_total,
    block_size - 1); at least one suffix token so the step still exercises
    the drafter.
    """
    cap_total = min(int(cfg.max_draft_total), max(1, int(block_size) - 1))
    suffix = cap_total - int(prefix_len)
    if suffix < 1:
        suffix = 1
    return suffix


def plan_hybrid_step(state: HybridDraftState, block_size: int) -> tuple[list[int], int]:
    """Decide this step's draft shape: (prefix_tokens, n_draft_tokens).

    No hit / disabled / degraded returns ([], max(1, block_size - 1)) —
    byte-identical to the pure-DSpine drafting shape the pre-N11 path used,
    which is the no-hit regression guarantee.
    """
    prefix = state.prefix_for_step()
    if not prefix:
        return [], max(1, int(block_size) - 1)
    # A head needs at least one suffix row inside the block: block layout is
    # [anchor, *prefix, mask * suffix], so prefix + 1 must fit the block.
    if len(prefix) + 1 > int(block_size) - 1:
        logger.debug(
            "dspine hybrid: prefix %d does not fit block_size %d, pure DSpine",
            len(prefix),
            block_size,
        )
        return [], max(1, int(block_size) - 1)
    return prefix, hybrid_suffix_len(state.cfg, len(prefix), block_size)
