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

import hashlib
import json
import logging
import os
import time
from array import array
from dataclasses import dataclass, field
from hashlib import sha1 as _sha1
from pathlib import Path

logger = logging.getLogger(__name__)

METHOD_NGRAM = "suffix"
METHOD_EAGLE3 = "eagle3"
METHOD_DFLASH = "ddtree"
METHOD_DFLASH2 = "dflash2"
METHOD_MTP = "mtp"
METHOD_MEDUSA = "medusa"


# D2.6/G10/L13: spec-route decision jsonl logger. Append-only, size-guarded
# (500 MiB), test-disable switch, I/O errors NEVER break inference.
_SPEC_ROUTE_LOG_PATH = os.environ.get(
    "FUSION_MLX_SPEC_ROUTE_LOG_PATH",
    str(Path.home() / ".fusion-mlx" / "spec_route_decisions.jsonl"),
)
_SPEC_ROUTE_LOG_MAX_BYTES = 500 * 1024 * 1024  # 500 MiB size guard


def _spec_route_log_enabled() -> bool:
    # D2.6: test-disable switch. FUSION_MLX_SPEC_ROUTE_LOG=0 disables
    # (default ON). Tests set =0 to avoid writing during unit runs.
    val = os.environ.get("FUSION_MLX_SPEC_ROUTE_LOG", "1").strip().lower()
    return val not in ("0", "false", "off")


def _append_spec_route_decision(decision: dict) -> None:
    # D2.6: append a routing decision to jsonl. Three guards:
    # 1. test-disable switch (FUSION_MLX_SPEC_ROUTE_LOG=0)
    # 2. 500 MiB size guard (stop logging when file exceeds cap)
    # 3. I/O errors NEVER break inference (try/except + debug log)
    if not _spec_route_log_enabled():
        return
    try:
        p = Path(_SPEC_ROUTE_LOG_PATH)
        try:
            if p.exists() and p.stat().st_size >= _SPEC_ROUTE_LOG_MAX_BYTES:
                logger.debug(
                    "spec-route log: size guard hit (%d bytes), skipping append",
                    p.stat().st_size,
                )
                return
        except OSError:
            pass
        p.parent.mkdir(parents=True, exist_ok=True)
        decision["ts"] = time.time()
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(decision, ensure_ascii=False) + "\n")
    except OSError as e:
        # I/O error never breaks inference — routing decision already made.
        logger.debug("spec-route log: append failed (non-fatal): %s", e)


def append_spec_request_end(record: dict) -> None:
    """Append a request-end shape record to the same spec-route jsonl.

    #1549 follow-up (shape telemetry): realized request shape — output
    length vs the max_tokens the router saw at route time, plus the spec
    method that actually served the request. Same guards as route records
    (disable switch, size cap, I/O errors never break inference).
    """
    _append_spec_route_decision({"event": "request_end", **record})


def prompt_shape_hashes(prompt: str | list[int] | None) -> dict:
    # PIC sizing telemetry (r6-pic-segment-index-sizing-1112): deterministic
    # prompt-shape digests for reuse-pattern analysis. No raw content leaves
    # the process — only truncated SHA-1 digests. prompt_hash identifies
    # exact-repeat traffic; prompt_head_hash covers the first 256 token ids
    # (or 1024 chars of a text prompt) so requests sharing a template
    # skeleton (RAG system + instructions) cluster under one sig even when
    # the retrieved middle content differs. Failures are non-fatal and
    # yield an empty dict.
    try:
        if isinstance(prompt, list) and prompt:
            head_ids = prompt[:256]
            full = _sha1(array("q", prompt).tobytes()).hexdigest()[:16]
            head = _sha1(array("q", head_ids).tobytes()).hexdigest()[:16]
            return {"prompt_hash": full, "prompt_head_hash": head}
        if isinstance(prompt, str) and prompt:
            encoded = prompt.encode("utf-8", "ignore")
            return {
                "prompt_hash": hashlib.sha1(encoded).hexdigest()[:16],
                "prompt_head_hash": hashlib.sha1(encoded[:1024]).hexdigest()[:16],
            }
    except Exception as e:
        logger.debug("prompt_shape_hashes failed (non-fatal): %s", e)
    return {}


METHOD_DSPARK = "dspark"
METHOD_DSPINE = "dspine"
METHOD_DFLY = "dfly"
DEFAULT_AVAILABLE: frozenset[str] = frozenset(
    {
        METHOD_NGRAM,
        METHOD_EAGLE3,
        METHOD_DFLASH,
        METHOD_DFLASH2,
        METHOD_MTP,
        METHOD_MEDUSA,
        METHOD_DSPARK,
        METHOD_DSPINE,
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
            if c.startswith("max_prompt:"):
                threshold = int(c.split(":")[1])
                if signals.prompt_token_count > threshold:
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
    # max_prompt:32768 — in-target verify replays the block through the
    # scheduler's model; very long prompts leave little KV headroom for
    # the verify forward, and the draft's hidden-state context degrades.
    SpecRouteEntry(
        family="qwen3_8",
        methods=(METHOD_DFLASH2,),
        constraints=("not_moe", "max_prompt:32768"),
    ),
    SpecRouteEntry(
        family="qwen3_8",
        methods=(METHOD_NGRAM,),
        constraints=(),
    ),
    # DSpine: parallel drafter over target intermediate-layer hiddens
    # (#1549). Loaded only when --dspine-drafter-path is set and the
    # eligibility gates passed, so the not_moe constraint here mostly
    # documents intent; greedy (T=0) requests only — the dispatch step
    # skips sampled requests.
    SpecRouteEntry(
        family="qwen3",
        methods=(METHOD_DSPINE,),
        constraints=("not_moe",),
    ),
    SpecRouteEntry(
        family="qwen3",
        methods=(METHOD_MTP,),
        constraints=(),
    ),
    SpecRouteEntry(
        family="qwen3",
        methods=(METHOD_MEDUSA,),
        constraints=("not_recurrent",),
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
    max_tokens: int | None = None
    has_mtp: bool = False
    recent_accept_rate: float | None = None
    current_method: str | None = None
    available: frozenset[str] = field(default_factory=lambda: DEFAULT_AVAILABLE)
    model_family: str | None = None
    is_moe: bool = False
    quant_bits: int | None = None
    is_recurrent: bool = False
    # #1610: bucketed acceptance for current_method in the request's
    # output-length bucket (None when the (method, bucket) cell is empty —
    # hysteresis then falls back to the scalar recent_accept_rate).
    bucket_accept_rate: float | None = None
    output_bucket: int | None = None


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
    min_output_tokens: int = 32
    abandon_cooldown: int = 8

    def _record(
        self,
        method: str,
        reason: str,
        signals: RouteSignals,
        candidates: set[str] | frozenset[str],
        rate: float | None,
        rate_source: str = "scalar",
    ) -> None:
        # Audit 1019 L30: six hand-built copies of this dict meant any new
        # signal had to be added six times, and every copy could drift.
        _append_spec_route_decision(
            {
                "event": "route",
                "method": method,
                "reason": reason,
                "candidates": sorted(candidates),
                "model_family": signals.model_family,
                "prompt_tokens": signals.prompt_token_count,
                "max_tokens": signals.max_tokens,
                "accept_rate": rate,
                "rate_source": rate_source,
                "output_bucket": signals.output_bucket,
                "kv_pressure": getattr(signals, "kv_pressure", None),
            }
        )

    def decide(self, signals: RouteSignals) -> str:
        """Return the spec-decode method name to use for this request."""
        avail = signals.available or DEFAULT_AVAILABLE
        cur = signals.current_method
        rate = signals.recent_accept_rate
        rate_source = "scalar"
        # #1610: bucketed acceptance takes precedence over the scalar rate
        # when the (current_method, bucket) cell has data — a method can be
        # great on short chat and useless on 8k+ generation, which the scalar
        # average hides. Empty cell -> None -> scalar hysteresis fallback.
        if signals.bucket_accept_rate is not None and cur is not None:
            rate = signals.bucket_accept_rate
            rate_source = f"bucket:{signals.output_bucket}"
        excluded: set[str] = set()

        # Abandon cooldown: once a method is abandoned for low acceptance,
        # skip it for the next N routing decisions. Without this the router
        # oscillates: abandon(mtp) -> cur=None -> has_mtp re-selects mtp ->
        # bad request -> abandon(mtp) ... — every other request pays the
        # draft warmup on a method we already measured as failing
        # (production: 2,713 mtp requests at 0.05 acceptance).
        cooldowns = getattr(self, "_abandon_cooldowns", None)
        if cooldowns is None:
            cooldowns = self._abandon_cooldowns = {}
        if self.abandon_cooldown > 0 and cooldowns:
            for m in list(cooldowns):
                cooldowns[m] -= 1
                if cooldowns[m] <= 0:
                    del cooldowns[m]
            excluded.update(cooldowns)

        # Output-length gate (#1549 follow-up): draft warmup (2 steps) plus
        # the first 16-token draft round never amortize on very short
        # outputs. 4-bit targets make this worse (quantized AR is fast
        # enough that even 256-token open-ended chat loses to plain AR).
        # Deterministic floor; None max_tokens (unknown) never gates.
        if (
            signals.max_tokens is not None
            and signals.max_tokens < self.min_output_tokens
        ):
            if METHOD_NGRAM in avail:
                self._record(
                    METHOD_NGRAM,
                    "short_output floor",
                    signals,
                    avail,
                    rate,
                    rate_source,
                )
                return METHOD_NGRAM
            self._record("", "short_output floor", signals, avail, rate, rate_source)
            return ""

        if cur is not None and rate is not None and rate < self.abandon_accept:
            excluded.add(cur)
            if self.abandon_cooldown > 0:
                cooldowns[signals.current_method] = self.abandon_cooldown
            cur = None
            logger.info(
                "spec-route: abandoning %s (acceptance %.1f%% < %.1f%%, "
                "source %s, cooldown %d requests)",
                signals.current_method,
                rate * 100,
                self.abandon_accept * 100,
                rate_source,
                self.abandon_cooldown,
            )

        if (
            cur is not None
            and cur in avail
            and rate is not None
            and rate >= self.keep_accept
        ):
            self._record(
                cur,
                "keep_accept (rate >= threshold)",
                signals,
                avail,
                rate,
                rate_source,
            )
            return cur

        candidates = avail - excluded

        if signals.model_family is not None:
            result = self._table_route(signals, candidates)
            if result is not None:
                self._record(
                    result, "table_route hit", signals, candidates, rate, rate_source
                )
                return result

        if (
            signals.prompt_token_count >= self.long_doc_threshold
            and METHOD_DFLASH in candidates
        ):
            self._record(
                METHOD_DFLASH,
                "long_doc threshold",
                signals,
                candidates,
                rate,
                rate_source,
            )
            return METHOD_DFLASH
        if signals.has_mtp and METHOD_MTP in candidates:
            self._record(
                METHOD_MTP, "mtp available", signals, candidates, rate, rate_source
            )
            return METHOD_MTP
        if METHOD_NGRAM in candidates:
            self._record(
                METHOD_NGRAM, "ngram fallback", signals, candidates, rate, rate_source
            )
            return METHOD_NGRAM
        fallback = next(iter(sorted(candidates)), METHOD_NGRAM)
        self._record(
            fallback, "sorted fallback", signals, candidates, rate, rate_source
        )
        return fallback

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
