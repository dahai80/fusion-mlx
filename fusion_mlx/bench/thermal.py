# SPDX-License-Identifier: Apache-2.0
"""Thermal-aware benchmark protocol helpers (#1588).

Apple Silicon throttles silently: a constant-workload decode loop drifts
7.2 -> 14.5 ms/step over 10 sustained minutes and fully recovers after
~10 min idle (measured on the #1582 investigation chassis). Any comparative
bench that mixes a cool run with a heat-soaked run is invalid.

This module gives bench runners the two protocol primitives:

- ``record_idle`` — pre-run GPU idle window with a log line stating the
  duration, so the artifact records time-since-idle.
- ``ThermalTracker`` — samples the scheduler's rolling decode ms/step
  during a run and summarizes first/last window plus drift ratio, so
  within-run throttle drift is visible in the output.

Recommended idle window for comparative runs: 600 s (``THERMAL_IDLE_RECOMMENDED_S``).

Clock-phase extension (2026-11-13, P2-1 follow-up): single-stream decode
on M5 Max walks a full curve — idle-ramp (6.5-7.0 ms/step) -> narrow boost
window (5.92-6.07 ms/step, ~30-90 s of GPU work) -> throttle decay ->
sustained floor (~106 tok/s on the 4B). Point-in-time tok/s numbers taken
at different phases are not comparable, even with identical idle windows.
``record_warmup`` standardizes the phase before measurement, and
``ThermalTracker.summary`` tags the observed phase so peak and sustained
numbers never get mixed in one comparison table.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

THERMAL_IDLE_RECOMMENDED_S = 600
# Phase-tag thresholds on the drift ratio (last / first sample). The
# measured curves give wide margins: ramp moves >=1.2x within a window,
# throttle decay >=1.25x, boost/steady stays within a few percent.
PHASE_RAMP_THRESHOLD = 0.94
PHASE_THROTTLE_THRESHOLD = 1.08


async def record_idle(
    seconds: float, log: Callable[[str], None] | None = None
) -> float:
    """Idle the GPU for ``seconds`` before a comparative bench run.

    Returns the actual idle duration. Logs the window so bench logs carry
    the idle-before metadata the protocol requires.
    """
    if seconds <= 0:
        return 0.0
    emit = log or (lambda msg: logger.info(msg))
    emit(f"thermal protocol: idling GPU for {seconds:.0f}s before bench run")
    t0 = time.perf_counter()
    await asyncio.sleep(seconds)
    elapsed = time.perf_counter() - t0
    emit(f"thermal protocol: idle done ({elapsed:.0f}s)")
    return elapsed


async def record_warmup(
    seconds: float,
    work: Callable[[], Awaitable[Any]],
    log: Callable[[str], None] | None = None,
) -> float:
    """Run a repeated GPU workload for ``seconds`` before a bench run.

    ``work`` is a short async unit (e.g. one small generation) invoked in
    a loop until the deadline elapses. This standardizes the clock phase:
    a bench that starts immediately after idle lands in the ramp phase
    (141-153 tok/s on the 4B) while the same engine lands at 165-169 in
    the boost window — the phase, not the engine, decides the number.

    Returns the actual warm-up duration (0 when ``seconds`` <= 0).
    """
    if seconds <= 0:
        return 0.0
    emit = log or (lambda msg: logger.info(msg))
    emit(f"thermal protocol: warming up GPU for {seconds:.0f}s before bench run")
    t0 = time.perf_counter()
    rounds = 0
    while time.perf_counter() - t0 < seconds:
        await work()
        rounds += 1
    elapsed = time.perf_counter() - t0
    emit(f"thermal protocol: warmup done ({elapsed:.0f}s, {rounds} work rounds)")
    return elapsed


def classify_phase(samples: list[float]) -> str:
    """Deterministic clock-phase tag from decode ms/step samples.

    ``ramping`` — ms/step still falling (clocks climbing toward boost);
    ``throttling`` — ms/step rising (thermal decay underway);
    ``steady`` — flat within thresholds. Fewer than 2 samples -> ``unknown``.
    """
    if len(samples) < 2:
        return "unknown"
    first = samples[0]
    last = samples[-1]
    if first <= 0:
        return "unknown"
    drift = last / first
    if drift <= PHASE_RAMP_THRESHOLD:
        return "ramping"
    if drift >= PHASE_THROTTLE_THRESHOLD:
        return "throttling"
    return "steady"


class ThermalTracker:
    """Samples rolling decode ms/step during a bench run (#1588).

    ``get_step_ms`` returns the scheduler's current rolling decode
    ms/step (``scheduler.get_stats().get("decode_step_time_ms")``) or
    None when no decode steps have been observed yet. Sampling runs as a
    background task while the bench coroutine executes; ``summary()``
    yields the thermal_state block for the result artifact.
    """

    def __init__(
        self, get_step_ms: Callable[[], float | None], interval_s: float = 2.0
    ):
        self._get_step_ms = get_step_ms
        self._interval_s = max(0.2, interval_s)
        self._samples: list[float] = []
        self._sample_times: list[float] = []
        self.idle_before_s: float = 0.0
        self.warmup_before_s: float = 0.0
        self._task: asyncio.Task | None = None

    def note_idle(self, seconds: float) -> None:
        self.idle_before_s = seconds

    def note_warmup(self, seconds: float) -> None:
        self.warmup_before_s = seconds

    def _sample_now(self) -> None:
        try:
            ms = self._get_step_ms()
        except Exception:
            logger.debug("thermal tracker sampler read failed", exc_info=True)
            return
        if ms is not None and ms > 0:
            self._samples.append(float(ms))
            self._sample_times.append(time.monotonic())

    async def _sampler(self) -> None:
        while True:
            self._sample_now()
            # First poll usually lands before any decode step exists
            # (stats still empty) — retry sooner so short runs still
            # produce a first window.
            await asyncio.sleep(
                self._interval_s if self._samples else min(self._interval_s, 0.5)
            )

    async def run_with_sampling(self, bench_coro: Any) -> Any:
        """Run ``bench_coro`` to completion while sampling in the background."""
        self._task = asyncio.ensure_future(self._sampler())
        try:
            return await bench_coro
        finally:
            # Post-run sample: the scheduler's rolling window still holds
            # the last decode steps even when the run finished between
            # polls — guarantees last_window for short runs.
            self._sample_now()
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def summary(self) -> dict[str, Any]:
        """Thermal-state block: idle/warmup metadata + windows + phase tag."""
        out: dict[str, Any] = {
            "idle_before_s": round(self.idle_before_s, 1),
            "warmup_before_s": round(self.warmup_before_s, 1),
            "first_window_ms_per_step": None,
            "last_window_ms_per_step": None,
            "drift_ratio": None,
            "phase": "unknown",
            "near_best": None,
        }
        if not self._samples:
            return out
        out["first_window_ms_per_step"] = round(self._samples[0], 2)
        out["last_window_ms_per_step"] = round(self._samples[-1], 2)
        first = self._samples[0]
        best = min(self._samples)
        if first > 0:
            out["drift_ratio"] = round(self._samples[-1] / first, 3)
        out["phase"] = classify_phase(self._samples)
        # ``near_best`` marks a run whose last window sits within 5% of the
        # best observed window — the boost-window signature. Peak-vs-sustained
        # comparisons must only ever match on this flag.
        out["near_best"] = bool(best > 0 and self._samples[-1] <= best * 1.05)
        return out

    def format_summary(self) -> str:
        s = self.summary()
        return (
            f"  Thermal state: idle_before={s['idle_before_s']:.0f}s"
            f", warmup_before={s['warmup_before_s']:.0f}s"
            f", first_window={s['first_window_ms_per_step']}"
            f" ms/step, last_window={s['last_window_ms_per_step']}"
            f" ms/step, drift_ratio={s['drift_ratio']}"
            f", phase={s['phase']}"
            f", near_best={s['near_best']}"
        )
