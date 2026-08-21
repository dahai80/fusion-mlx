# SPDX-License-Identifier: Apache-2.0
"""Auto-router for speculative-decoding method selection (FR-005 ``--spec-route auto``).

fusion-mlx ships several spec-decode algorithms — n-gram/suffix, DFlash,
MTP, DSpark, DFly — but until now selection was manual. ``SpecAutoRouter``
picks a method at request setup time from cheap signals (prompt length, model
MTP capability, the previous request's acceptance rate, model family) and
defers *runtime* tuning to the existing per-method pause/resume hysteresis
in ``scheduler.spec_decode`` (which already pauses a method whose acceptance
drops below ``SPEC_MIN_ACCEPT_RATE``).

This is a SETUP-time choice, not a mid-decode hot-swap: no cache rebuild,
no draft-model reload. The router is a pure decision function — same inputs
always yield the same method — so the entire decision table is unit-testable
and no model forward pass is ever invoked.

Phase 2 — Intelligent Auto-Router:
  The routing table (``_SPEC_ROUTING_TABLE``) replaces hardcoded if/else
  chains with declarative ``SpecRouteEntry`` rows. Each entry maps a
  ``model_family`` to a priority-ordered list of spec methods, with optional
  constraint gates (not_moe, min_bits, not_recurrent). New methods only need
  a table entry, not code changes to decide().

Decision order (see ``decide``):
  1. Abandon a clearly-failing current method (acceptance < abandon_accept)
     and exclude it from immediate re-selection.
  2. Hysteresis: keep the current method if it is working (acceptance >=
     keep_accept) to avoid thrashing between requests.
  3. Table-driven selection — match model_family against the routing table,
     evaluate constraints, pick the first available method.
  4. Fallback — long-doc DFlash, MTP, n-gram (legacy path for signals
     without model_family).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

METHOD_NGRAM = "suffix"
METHOD_EAGLE3 = "eagle3"
METHOD_DFLASH = "ddtree"
METHOD_DFLASH2 = "dflash2"
METHOD_MTP = "mtp"
METHOD_DSPARK = "dspark"
METHOD_DFLY = "dfly"
DEFAULT_AVAILABLE: frozenset[str] = frozenset(
    {
        METHOD_NGRAM,
        METHOD_EAGLE3,
        METHOD_DFLASH,
        METHOD_DFLASH2,
        METHOD_MTP,
        METHOD_DSPARK,
        METHOD_DFLY,
    }
)


@dataclass(frozen=True)
class SpecRouteEntry:
    """Declarative routing entry: model_family -> ordered spec methods + constraints.

    ``family`` is matched against ``RouteSignals.model_family`` (exact match),
    or "*" for the wildcard/default entry. ``methods`` lists method names in
    priority order — the first one also present in ``signals.available`` wins.
    ``constraints`` are gates that must ALL pass for the entry to match; if any
    gate fails the whole entry is skipped and the router falls through to the
    next one.
    """

    family: str
    methods: tuple[str, ...]
    constraints: tuple[str, ...] = ()
    drafter_map: dict[str, str] | None = None

    def evaluate_constraints(self, signals: RouteSignals) -> bool:
        for c in self.constraints:
            if c == "not_moe" and signals.is_moe:
                return False
            if c == "not_recurrent" and signals.is_recurrent:
                return False
            if c.startswith("min_bits:"):
                threshold = int(c.split(":")[1])
                if signals.quant_bits is not None and signals.quant_bits < threshold:
                    return False
        return True


_SPEC_ROUTING_TABLE: list[SpecRouteEntry] = [
    SpecRouteEntry(
        family="hunyuan",
        methods=(METHOD_DFLY,),
        constraints=("not_moe", "min_bits:8"),
        drafter_map={"dfly": "AngelSlim/Hy3-DFly-Block8"},
    ),
    SpecRouteEntry(
        family="hunyuan",
        methods=(METHOD_MTP,),
        constraints=(),
    ),
    SpecRouteEntry(
        family="hunyuan",
        methods=(METHOD_NGRAM,),
        constraints=(),
    ),
    SpecRouteEntry(
        family="qwen3_5",
        methods=(METHOD_DFLASH,),
        constraints=("not_moe",),
    ),
    SpecRouteEntry(
        family="qwen3_5",
        methods=(METHOD_MTP,),
        constraints=(),
    ),
    SpecRouteEntry(
        family="qwen3_5",
        methods=(METHOD_NGRAM,),
        constraints=(),
    ),
    # DFlash2: block-diffusion speculative decode for Qwen3.8 dense
    # targets (z-lab DFlash2DraftModel, official dflash pip pkg). MoE
    # excluded — the draft reads target hidden states through a dense
    # GroupedDynamicCausalConv that assumes single-expert activation.
    SpecRouteEntry(
        family="qwen3_8",
        methods=(METHOD_DFLASH2,),
        constraints=("not_moe",),
    ),
    SpecRouteEntry(
        family="qwen3_8",
        methods=(METHOD_NGRAM,),
        constraints=(),
    ),
    SpecRouteEntry(
        family="qwen3",
        methods=(METHOD_MTP,),
        constraints=(),
    ),
    SpecRouteEntry(
        family="qwen3",
        methods=(METHOD_NGRAM,),
        constraints=(),
    ),
    SpecRouteEntry(
        family="deepseek",
        methods=(METHOD_DFLASH,),
        constraints=("not_moe",),
    ),
    SpecRouteEntry(
        family="deepseek",
        methods=(METHOD_NGRAM,),
        constraints=(),
    ),
    SpecRouteEntry(
        family="llama3",
        methods=(METHOD_EAGLE3,),
        constraints=("not_recurrent",),
    ),
    SpecRouteEntry(
        family="llama3",
        methods=(METHOD_NGRAM,),
        constraints=(),
    ),
    SpecRouteEntry(
        family="*",
        methods=(METHOD_NGRAM,),
        constraints=(),
    ),
]


@dataclass(frozen=True)
class RouteSignals:
    """Inputs to a single routing decision.

    ``recent_accept_rate`` is the acceptance rate observed by ``current_method``
    on the *previous* request (None on the first request). ``available`` is the
    set of registry methods usable for this model. Extended in Phase 2 with
    model_family, is_moe, quant_bits, is_recurrent for table-driven routing.
    """

    prompt_token_count: int
    has_mtp: bool = False
    recent_accept_rate: float | None = None
    current_method: str | None = None
    available: frozenset[str] = field(default_factory=lambda: DEFAULT_AVAILABLE)
    model_family: str | None = None
    is_moe: bool = False
    quant_bits: int | None = None
    is_recurrent: bool = False


@dataclass
class SpecAutoRouter:
    """Deterministic spec-decode method router with configurable thresholds.

    Phase 2: table-driven routing via _SPEC_ROUTING_TABLE. When
    RouteSignals.model_family is set, the table is consulted first; the
    legacy long_doc/mtp/ngram fallback only fires when model_family is None.
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
            excluded.add(cur)
            cur = None
            logger.info(
                "spec-route: abandoning %s (acceptance %.1f%% < %.1f%%)",
                signals.current_method,
                rate * 100,
                self.abandon_accept * 100,
            )

        if (
            cur is not None
            and cur in avail
            and rate is not None
            and rate >= self.keep_accept
        ):
            return cur

        candidates = avail - excluded

        if signals.model_family is not None:
            result = self._table_route(signals, candidates)
            if result is not None:
                return result

        if (
            signals.prompt_token_count >= self.long_doc_threshold
            and METHOD_DFLASH in candidates
        ):
            return METHOD_DFLASH
        if signals.has_mtp and METHOD_MTP in candidates:
            return METHOD_MTP
        if METHOD_NGRAM in candidates:
            return METHOD_NGRAM
        return next(iter(sorted(candidates)), METHOD_NGRAM)

    def _method_usable(self, method: str, signals: RouteSignals) -> bool:
        if method == METHOD_MTP and not signals.has_mtp:
            return False
        return True

    def _table_route(
        self, signals: RouteSignals, candidates: frozenset[str]
    ) -> str | None:
        family = signals.model_family
        for entry in _SPEC_ROUTING_TABLE:
            if entry.family != "*" and entry.family != family:
                continue
            if not entry.evaluate_constraints(signals):
                logger.debug(
                    "spec-route: entry %s skipped (constraints blocked for family=%s)",
                    entry.family,
                    family,
                )
                continue
            for method in entry.methods:
                if method in candidates and self._method_usable(method, signals):
                    logger.info(
                        "spec-route: table hit family=%s -> %s (entry=%s)",
                        family,
                        method,
                        entry.family,
                    )
                    return method
        return None


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


def routing_table() -> list[SpecRouteEntry]:
    """Return a shallow copy of the routing table for inspection / API."""
    return list(_SPEC_ROUTING_TABLE)


def drafter_for(family: str, method: str) -> str | None:
    """Look up the default drafter HF path for a family+method combo.

    Returns None when no drafter_map is defined for the entry.
    """
    for entry in _SPEC_ROUTING_TABLE:
        if entry.family in (family, "*") and entry.drafter_map:
            return entry.drafter_map.get(method)
    return None
