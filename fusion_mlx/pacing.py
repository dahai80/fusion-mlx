# SPDX-License-Identifier: Apache-2.0
"""#1197: Metal submission pacing gate (per-class token bucket) + dynamic
TDM-style throttle.

Every heavy-workload submission to a Metal-bound executor passes a
per-workload-class token bucket BEFORE the executor submit. The gate is
Python-level (no Metal queue handles — a classic TDM bus scheduler is not
measurable/controllable from Python; this is the redefined PRD F4 shape).

Safety rules (hard-won, see scheduler/sched_batch.py #1171 note):
- acquire() is NEVER called from inside an executor worker — only from the
  async route/loop side, so a bounded wait can never self-deadlock a
  max_workers=1 pool.
- Disabled (FUSION_MLX_PACING=off / rate<=0) -> every acquire succeeds
  instantly; the gate can never become a new hang path.
- Stage 2 controller: AIMD (additive increase / multiplicative decrease)
  with hysteresis + cooldown — converges without oscillation under load
  changes. Victim quality signals (LLM TPS drop, Diffusion FPS drop)
  shrink the AGGRESSOR classes' budgets, never the victim's own.
"""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

CLASSES = ("llm", "image", "video", "audio", "laya", "training")

# Static defaults: submissions per second per class. Generous enough that
# solo workloads never notice the gate (llm decode bursts ~20-60 submits/s;
# laya decide probes must keep P50 < 15ms under load).
_DEFAULTS = {
    "llm": (200.0, 40.0),
    "image": (8.0, 4.0),
    "video": (4.0, 2.0),
    "audio": (10.0, 5.0),
    "laya": (50.0, 20.0),
    "training": (1.0, 1.0),
}

# AIMD bounds: a class's dynamic rate never leaves [floor, max]. Floors
# keep decode-critical classes alive under any controller action.
_FLOOR_FRACTION = 0.25
_DECREASE_FACTOR = 0.8
_INCREASE_ADDEND = 1.0
# Hysteresis: a victim signal must be bad for this many consecutive
# updates before budgets shrink (single-sample noise never throttles).
_BAD_STREAK_THRESHOLD = 3
_BAD_RATIO = 0.85
_CONTROLLER_COOLDOWN_S = 10.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except ValueError:
        logger.warning("pacing: invalid %s=%r, using %.1f", name, raw, default)
        return default
    if val < 0:
        logger.warning("pacing: %s=%r < 0, using %.1f", name, raw, default)
        return default
    return val


class TokenBucket:
    def __init__(self, rate_per_s: float, burst: float):
        self.rate = float(rate_per_s)
        self.burst = max(1.0, float(burst))
        self.tokens = self.burst
        self.last = time.monotonic()
        self._lock = threading.Lock()

    def _refill_locked(self, now: float) -> None:
        elapsed = now - self.last
        if elapsed > 0:
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last = now

    def try_acquire(self, n: float = 1.0) -> bool:
        if self.rate <= 0:
            return True
        with self._lock:
            self._refill_locked(time.monotonic())
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False

    def try_acquire_with_wait(self, n: float, timeout_s: float) -> bool:
        # Bounded wait. Caller MUST be on an async/event-loop thread, never
        # inside an executor worker (see module docstring).
        if self.rate <= 0:
            return True
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            with self._lock:
                self._refill_locked(time.monotonic())
                if self.tokens >= n:
                    self.tokens -= n
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.005, max(0.0, deadline - time.monotonic())))

    def set_rate(self, rate_per_s: float) -> None:
        with self._lock:
            self.rate = float(rate_per_s)

    def snapshot(self) -> dict:
        with self._lock:
            self._refill_locked(time.monotonic())
            return {
                "rate_per_s": self.rate,
                "burst": self.burst,
                "tokens": round(self.tokens, 3),
            }


class PacingGate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[str, TokenBucket] = {}
        self._initial_rate: dict[str, float] = {}
        self._submissions: dict[str, int] = {}
        self._throttled: dict[str, int] = {}
        # Controller state (stage 2).
        self._bad_streak: dict[str, int] = {}
        self._last_adjust = 0.0
        self._enabled = os.environ.get("FUSION_MLX_PACING", "").strip().lower() not in (
            "off",
            "0",
            "false",
        )
        # Literal env names (no f-string composition): the routing-shape
        # scanner in test_no_out_of_band_routing requires single string
        # literals for every FUSION_MLX_* key.
        _env_names = {
            "llm": ("FUSION_MLX_PACE_LLM_RATE", "FUSION_MLX_PACE_LLM_BURST"),
            "image": ("FUSION_MLX_PACE_IMAGE_RATE", "FUSION_MLX_PACE_IMAGE_BURST"),
            "video": ("FUSION_MLX_PACE_VIDEO_RATE", "FUSION_MLX_PACE_VIDEO_BURST"),
            "audio": ("FUSION_MLX_PACE_AUDIO_RATE", "FUSION_MLX_PACE_AUDIO_BURST"),
            "laya": ("FUSION_MLX_PACE_LAYA_RATE", "FUSION_MLX_PACE_LAYA_BURST"),
            "training": (
                "FUSION_MLX_PACE_TRAINING_RATE",
                "FUSION_MLX_PACE_TRAINING_BURST",
            ),
        }
        for cls in CLASSES:
            default_rate, default_burst = _DEFAULTS[cls]
            rate_name, burst_name = _env_names[cls]
            rate = _env_float(rate_name, default_rate)
            burst = _env_float(burst_name, default_burst)
            self._buckets[cls] = TokenBucket(rate, burst)
            self._initial_rate[cls] = rate
            self._submissions[cls] = 0
            self._throttled[cls] = 0
            self._bad_streak[cls] = 0

    def enabled(self) -> bool:
        return self._enabled

    def bucket(self, cls: str) -> TokenBucket:
        return self._buckets.get(cls) or self._buckets["llm"]

    def try_acquire_counted(self, cls: str) -> bool:
        # Non-blocking fast path (event-loop safe, no thread hop).
        if not self._enabled:
            with self._lock:
                self._submissions[cls] = self._submissions.get(cls, 0) + 1
            return True
        bucket = self._buckets.get(cls)
        if bucket is None:
            with self._lock:
                self._submissions[cls] = self._submissions.get(cls, 0) + 1
            return True
        if bucket.try_acquire(1.0):
            with self._lock:
                self._submissions[cls] = self._submissions.get(cls, 0) + 1
            return True
        return False

    def acquire(self, cls: str, timeout_s: float = 0.05) -> bool:
        # Gate a submission. Returns True when the submission may proceed.
        # A False return means THROTTLED: the caller must surface an
        # explicit error (route maps to 503) — never silently drop.
        if not self._enabled:
            with self._lock:
                self._submissions[cls] = self._submissions.get(cls, 0) + 1
            return True
        bucket = self._buckets.get(cls)
        if bucket is None:
            with self._lock:
                self._submissions[cls] = self._submissions.get(cls, 0) + 1
            return True
        if bucket.try_acquire_with_wait(1.0, timeout_s):
            with self._lock:
                self._submissions[cls] = self._submissions.get(cls, 0) + 1
            return True
        with self._lock:
            self._throttled[cls] = self._throttled.get(cls, 0) + 1
        logger.warning(
            "pacing: %s submissions throttled (rate=%.1f/s) — surface an "
            "explicit error to the caller, never drop silently",
            cls,
            bucket.rate,
        )
        return False

    def record_throttled(self, cls: str) -> None:
        with self._lock:
            self._throttled[cls] = self._throttled.get(cls, 0) + 1
        logger.warning(
            "pacing: %s submissions throttled — surface an explicit error "
            "to the caller, never drop silently",
            cls,
        )

    # -- stage 2: dynamic TDM throttle (AIMD) --------------------------

    def update_signal(self, victim_class: str, quality_ratio: float) -> None:
        # A victim (llm/video/...) reports its current quality vs solo
        # baseline. Sustained bad ratios shrink every OTHER class's budget
        # (aggressors); recovery grows them back additively. The victim's
        # own budget is never touched — it is already the starved party.
        if not self._enabled:
            return
        now = time.monotonic()
        with self._lock:
            streak = self._bad_streak.get(victim_class, 0)
            if quality_ratio < _BAD_RATIO:
                streak += 1
                self._bad_streak[victim_class] = streak
                if streak < _BAD_STREAK_THRESHOLD:
                    return
                if now - self._last_adjust < _CONTROLLER_COOLDOWN_S:
                    return
                self._last_adjust = now
                for cls in self._buckets:
                    if cls == victim_class:
                        continue
                    bucket = self._buckets[cls]
                    floor = self._initial_rate[cls] * _FLOOR_FRACTION
                    new_rate = max(floor, bucket.rate * _DECREASE_FACTOR)
                    if new_rate != bucket.rate:
                        bucket.set_rate(new_rate)
                        logger.info(
                            "pacing controller: %s quality %.2f sustained — "
                            "%s budget %.1f -> %.1f/s",
                            victim_class,
                            quality_ratio,
                            cls,
                            bucket.rate,
                            new_rate,
                        )
            else:
                if streak:
                    self._bad_streak[victim_class] = 0
                if now - self._last_adjust < _CONTROLLER_COOLDOWN_S:
                    return
                self._last_adjust = now
                grew = False
                for cls, bucket in self._buckets.items():
                    if cls == victim_class:
                        continue
                    cap = self._initial_rate[cls]
                    new_rate = min(cap, bucket.rate + _INCREASE_ADDEND)
                    if new_rate != bucket.rate:
                        bucket.set_rate(new_rate)
                        grew = True
                if grew:
                    logger.debug("pacing controller: signals healthy — budgets +1/s")

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "enabled": self._enabled,
                "classes": {
                    cls: {
                        **bucket.snapshot(),
                        "submissions": self._submissions.get(cls, 0),
                        "throttled": self._throttled.get(cls, 0),
                        "initial_rate": self._initial_rate[cls],
                    }
                    for cls, bucket in self._buckets.items()
                },
            }

    def reset_dynamic(self) -> None:
        # Test/ops hook: restore static budgets.
        with self._lock:
            for cls, bucket in self._buckets.items():
                bucket.set_rate(self._initial_rate[cls])
            self._bad_streak = {c: 0 for c in CLASSES}
            self._last_adjust = 0.0


_gate: PacingGate | None = None
_gate_lock = threading.Lock()


def get_pacing_gate() -> PacingGate:
    global _gate
    if _gate is None:
        with _gate_lock:
            if _gate is None:
                _gate = PacingGate()
    return _gate


def reset_pacing_gate() -> None:
    global _gate
    with _gate_lock:
        _gate = None


async def pace_acquire_async(cls: str, timeout_s: float = 0.05) -> bool:
    # Async-friendly wrapper: runs the bounded bucket wait in a thread so
    # the event loop is never blocked by the (<=5ms) sleep loop inside.
    import asyncio

    gate = get_pacing_gate()
    # Fast path: tokens available -> no thread hop on the event loop.
    if gate.try_acquire_counted(cls):
        return True
    if timeout_s <= 0:
        gate.record_throttled(cls)
        return False
    return await asyncio.to_thread(gate.acquire, cls, timeout_s)


def record_pacing_metrics() -> dict:
    # Rendered by the metrics route: per-class rate + drop signals.
    try:
        return get_pacing_gate().snapshot()
    except Exception:
        logger.exception("pacing metrics render failed")
        return {}
