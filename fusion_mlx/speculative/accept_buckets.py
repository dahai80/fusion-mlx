# SPDX-License-Identifier: Apache-2.0
#
# Bucketed spec-decode acceptance telemetry (#1610). The scalar
# recent_accept_rate feedback averages acceptance across bimodal output-length
# traffic (42% <512 tokens, 49% in 8k-32k), hiding methods that are excellent
# in one bucket and useless in another. This module keeps a bounded rolling
# window of (accepted, proposed) per (method, output-length bucket) cell and
# exposes:
#   - record(): called from the scheduler request-end feedback loop
#   - route_rate(): expected-bucket (or last-bucket) rate for the auto-router
#   - export(): compact dict for the /stats spec_decode surface
#
# Pure telemetry + routing feature — never changes draft-method behavior.
from __future__ import annotations

import logging
import threading
from collections import deque

logger = logging.getLogger(__name__)

DEFAULT_BUCKET_EDGES = (512, 2048, 8192)
DEFAULT_WINDOW = 64
BUCKET_LABELS = ("<512", "512-2k", "2k-8k", "8k+")
# #1638 (TraceSpec V1): decision-type axis. Tool-call requests exercise the
# drafter on grammar-constrained continuations; plain requests do not. The
# aggregate rate hides a drafter that is fine on prose and bad on tool args.
DTYPES = ("plain", "tool")
DEFAULT_DTYPE = "plain"


def normalize_dtype(dtype: str | None) -> str:
    # Best-effort telemetry: unknown values collapse onto the default so a
    # future axis can never silently fork the window space.
    return dtype if dtype in DTYPES else DEFAULT_DTYPE


def bucket_index(output_tokens: int, edges: tuple[int, ...]) -> int:
    for i, edge in enumerate(edges):
        if output_tokens < edge:
            return i
    return len(edges)


def bucket_label(index: int) -> str:
    if 0 <= index < len(BUCKET_LABELS):
        return BUCKET_LABELS[index]
    return f"bucket{index}"


class SpecAcceptBuckets:
    def __init__(
        self,
        edges: tuple[int, ...] | None = None,
        window: int = DEFAULT_WINDOW,
    ):
        self.edges = tuple(edges) if edges is not None else DEFAULT_BUCKET_EDGES
        if not self.edges or any(e <= 0 for e in self.edges):
            raise ValueError(f"bucket edges must be positive ints, got {self.edges}")
        if any(self.edges[i] >= self.edges[i + 1] for i in range(len(self.edges) - 1)):
            raise ValueError(
                f"bucket edges must be strictly increasing, got {self.edges}"
            )
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window
        self._cells: dict[tuple[str, int, str], deque] = {}
        self._last_bucket: int | None = None
        self._lock = threading.Lock()

    def record(
        self,
        method: str | None,
        output_tokens: int | None,
        proposed: int,
        accepted: int,
        dtype: str | None = DEFAULT_DTYPE,
    ) -> None:
        # Best-effort telemetry: invalid inputs are dropped, never raised.
        if not method or output_tokens is None or output_tokens < 0 or proposed <= 0:
            return
        idx = bucket_index(int(output_tokens), self.edges)
        dt = normalize_dtype(dtype)
        with self._lock:
            key = (method, idx, dt)
            cell = self._cells.get(key)
            if cell is None:
                cell = deque(maxlen=self.window)
                self._cells[key] = cell
            cell.append((int(accepted), int(proposed)))
            self._last_bucket = idx
        logger.debug(
            "spec accept bucket: method=%s bucket=%s dtype=%s accepted=%d proposed=%d",
            method,
            bucket_label(idx),
            dt,
            accepted,
            proposed,
        )

    def rate(
        self,
        method: str | None,
        bucket: int | None,
        dtype: str | None = DEFAULT_DTYPE,
    ) -> float | None:
        # Token-weighted acceptance over the rolling window (sum/sum), matching
        # the scalar feedback's accepted/proposed semantics.
        if not method or bucket is None:
            return None
        with self._lock:
            cell = self._cells.get((method, int(bucket), normalize_dtype(dtype)))
            samples = list(cell) if cell else []
        if not samples:
            return None
        accepted = sum(a for a, _ in samples)
        proposed = sum(p for _, p in samples)
        if proposed <= 0:
            return None
        return accepted / proposed

    def route_bucket(self, max_tokens: int | None) -> int | None:
        # Expected output bucket when max_tokens is known, else the last
        # finished request's realized bucket.
        if max_tokens is not None and max_tokens > 0:
            return bucket_index(int(max_tokens), self.edges)
        return self._last_bucket

    def route_rate(
        self, method: str | None, max_tokens: int | None
    ) -> tuple[float | None, int | None]:
        idx = self.route_bucket(max_tokens)
        return self.rate(method, idx), idx

    def export(self) -> dict:
        with self._lock:
            # #1638: cells nest as buckets[method][bucket_label][dtype] so the
            # tool/plain split is a first-class axis; plain-only windows look
            # like a single-dtype level, matching the pre-#1638 consumers.
            buckets: dict[str, dict[str, dict[str, dict]]] = {}
            for (method, idx, dt), cell in sorted(self._cells.items()):
                samples = list(cell)
                if not samples:
                    continue
                accepted = sum(a for a, _ in samples)
                proposed = sum(p for _, p in samples)
                buckets.setdefault(method, {}).setdefault(bucket_label(idx), {})[dt] = {
                    "accepted": accepted,
                    "proposed": proposed,
                    "samples": len(samples),
                    "rate": round(accepted / proposed, 6) if proposed > 0 else None,
                }
            return {
                "edges": list(self.edges),
                "window": self.window,
                "last_bucket": (
                    bucket_label(self._last_bucket)
                    if self._last_bucket is not None
                    else None
                ),
                "buckets": buckets,
            }


_GLOBAL_BUCKETS = SpecAcceptBuckets()


def get_accept_buckets() -> SpecAcceptBuckets:
    return _GLOBAL_BUCKETS
