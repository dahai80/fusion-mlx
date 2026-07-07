# SPDX-License-Identifier: Apache-2.0
"""Auto-router for speculative-decoding method selection (FR-005 ``--spec-route auto``).

fusion-mlx ships several spec-decode algorithms — n-gram/suffix, DFlash,
MTP, DSpark — but until now selection was manual. ``SpecAutoRouter`` picks a
method at request setup time from cheap signals (prompt length, model MTP
capability, the previous request's acceptance rate) and defers *runtime*
tuning to the existing per-method pause/resume hysteresis in
``scheduler.spec_decode`` (which already pauses a method whose acceptance
drops below ``SPEC_MIN_ACCEPT_RATE``).

This is a SETUP-time choice, not a mid-decode hot-swap: no cache rebuild,
no draft-model reload. The router is a pure decision function — same inputs
always yield the same method — so the entire decision table is unit-testable
and no model forward pass is ever invoked.

Decision order (see ``decide``):
  1. Abandon a clearly-failing current method (acceptance < abandon_accept)
     and exclude it from immediate re-selection.
  2. Hysteresis: keep the current method if it is working (acceptance >=
     keep_accept) to avoid thrashing between requests.
  3. Fresh selection — long-context / RAG prompts route to DFlash (diffusion
     blocks exploit repetition in long documents).
  4. Model-native MTP when the model exposes MTP heads (no extra draft-model
     load, good quality).
  5. n-gram as the cheapest default; its D1-match gate already self-disables
     on hostile input, so it never regresses below baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Canonical method names — must mirror speculative/registry.py ``method=`` fields,
# NOT the aliases (dflash/ngram are aliases; available_methods() yields canonical
# names so the router compares against ddtree/suffix here).
METHOD_NGRAM = "suffix"
METHOD_DFLASH = "ddtree"
METHOD_MTP = "mtp"
METHOD_DSPARK = "dspark"
DEFAULT_AVAILABLE: frozenset[str] = frozenset(
    {METHOD_NGRAM, METHOD_DFLASH, METHOD_MTP, METHOD_DSPARK}
)


@dataclass(frozen=True)
class RouteSignals:
    """Inputs to a single routing decision.

    ``recent_accept_rate`` is the acceptance rate observed by ``current_method``
    on the *previous* request (None on the first request). ``available`` is the
    set of registry methods usable for this model (defaults to all four).
    """

    prompt_token_count: int
    has_mtp: bool = False
    recent_accept_rate: float | None = None
    current_method: str | None = None
    available: frozenset[str] = field(default_factory=lambda: DEFAULT_AVAILABLE)


@dataclass
class SpecAutoRouter:
    """Deterministic spec-decode method router with configurable thresholds.

    Thresholds are public dataclass fields so callers (tests, admin panel,
    per-model settings) can tune them without touching the decision code.
    """

    long_doc_threshold: int = 4096
    abandon_accept: float = 0.20
    keep_accept: float = 0.40

    def decide(self, signals: RouteSignals) -> str:
        """Return the spec-decode method name to use for this request."""
        avail = signals.available or DEFAULT_AVAILABLE
        cur = signals.current_method
        rate = signals.recent_accept_rate
        excluded: set[str] = set()

        if cur is not None and rate is not None and rate < self.abandon_accept:
            # Current method is clearly failing — don't keep it (hysteresis)
            # and don't immediately re-pick it (fresh selection).
            excluded.add(cur)
            cur = None
            logger.info(
                "spec-route: abandoning %s (acceptance %.1f%% < %.1f%%)",
                signals.current_method,
                rate * 100,
                self.abandon_accept * 100,
            )

        # Hysteresis: a working current method stays put.
        if (
            cur is not None
            and cur in avail
            and rate is not None
            and rate >= self.keep_accept
        ):
            return cur

        candidates = avail - excluded

        # Fresh selection — ordered by expected payoff for the signal.
        if (
            signals.prompt_token_count >= self.long_doc_threshold
            and METHOD_DFLASH in candidates
        ):
            return METHOD_DFLASH
        if signals.has_mtp and METHOD_MTP in candidates:
            return METHOD_MTP
        if METHOD_NGRAM in candidates:
            return METHOD_NGRAM
        # Degenerate fallback: anything still available, else n-gram sentinel.
        return next(iter(sorted(candidates)), METHOD_NGRAM)


_DEFAULT_ROUTER = SpecAutoRouter()


def auto_route(signals: RouteSignals, router: SpecAutoRouter | None = None) -> str:
    """Convenience wrapper around the default ``SpecAutoRouter``."""
    return (router or _DEFAULT_ROUTER).decide(signals)


def available_methods() -> frozenset[str]:
    """Methods registered AND config-enabled in the spec-decode registry.

    The wiring layer builds ``RouteSignals.available`` from this so the router
    never recommends a method the registry doesn't actually provide.
    """
    from .registry import iter_spec_decoders

    return frozenset(p.method for p in iter_spec_decoders() if p.config_enabled)
