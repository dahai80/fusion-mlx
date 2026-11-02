# SPDX-License-Identifier: Apache-2.0
# #1480 item 2b: cooperative training-job abort. When the memory enforcer
# is under sustained pressure, has already evicted (or cannot find) any
# inference victim, and an in-process training job is the dominant memory
# consumer, evicting models cannot relieve the pressure — the training job
# itself must stop before the host dies (the #1473 panic class).
#
# Mechanics: each training service registers its active job before the
# train loop and deregisters in a finally. The loop calls check_abort()
# between iterations — Python cannot cancel a running thread, so the abort
# is cooperative and bounds the blast radius to one train step. The
# enforcer calls abort_all_active() (via request_abort_all) when its soft
# pressure persists past the #1473 hard-tier threshold with no victim at
# all. Opt-out knob: FUSION_MLX_ENFORCER_ABORT_TRAINING=0.

# #1483 item 2b: check_abort() also enforces the training-side RSS hard
# stop. The enforcer abort reacts to pool-level pressure, but the training
# process itself can balloon (activation headroom underestimate, co-resident
# growth after admission) faster than the enforcer's poll cadence — the
# #1473 panic class died mid-flight with admission passed. Knob
# FUSION_MLX_TRAIN_HARD_STOP_RSS_GB: when process RSS stays above it for
# _RSS_STRIKES consecutive between-iteration checks, the job is aborted
# (FAILED with reason) before the enforcer's process-level fatal-exit and
# long before the host watchdog. 0/unset = disabled.

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
# job_id -> {"service": <label>, "reason": <str | None>}
_active: dict[str, dict] = {}

# #1483: consecutive over-limit observations required before the hard stop
# fires. One train step apart each — a single noisy reading must not kill a
# healthy job.
_RSS_STRIKES = 3
# job_id -> consecutive over-limit count
_rss_strikes: dict[str, int] = {}


class TrainingAbortedError(RuntimeError):
    # Raised by check_abort() inside the train loop. Subclasses RuntimeError
    # so the services' generic `except Exception` in _run_job records it as
    # a FAILED job with the reason as the message.
    pass


def abort_enabled() -> bool:
    raw = os.environ.get("FUSION_MLX_ENFORCER_ABORT_TRAINING", "1")
    return raw.strip().lower() not in ("0", "false", "off", "no")


def register_job(job_id: str, service: str) -> None:
    with _lock:
        _active[job_id] = {"service": service, "reason": None}
    logger.debug(
        "training abort registry: +%s (%s), active=%d", job_id, service, len(_active)
    )


def deregister_job(job_id: str) -> None:
    with _lock:
        entry = _active.pop(job_id, None)
        _rss_strikes.pop(job_id, None)
    if entry is not None:
        logger.debug(
            "training abort registry: -%s (%s), active=%d",
            job_id,
            entry["service"],
            len(_active),
        )


def request_abort(job_id: str, reason: str) -> bool:
    with _lock:
        entry = _active.get(job_id)
        if entry is None or entry["reason"] is not None:
            return False
        entry["reason"] = reason
    logger.warning(
        "training job %s (%s) abort requested: %s",
        job_id,
        _active_label(job_id),
        reason,
    )
    return True


def abort_all_active(reason: str) -> int:
    with _lock:
        pending = [jid for jid, e in _active.items() if e["reason"] is None]
        for jid in pending:
            _active[jid]["reason"] = reason
        total = len(_active)
    if pending:
        logger.warning(
            "training abort: requested for %d job(s) %s (%d registered) — reason: %s",
            len(pending),
            pending,
            total,
            reason,
        )
    return len(pending)


def abort_reason(job_id: str) -> str | None:
    with _lock:
        entry = _active.get(job_id)
        return entry["reason"] if entry is not None else None


def active_jobs() -> list[tuple[str, str]]:
    with _lock:
        return [(jid, e["service"]) for jid, e in _active.items()]


def check_abort(job_id: str) -> None:
    # Called between train iterations. Raises TrainingAbortedError when an
    # abort was requested for this job, or when the #1483 RSS hard stop
    # trips; no-op otherwise (including when the job is not registered).
    check_rss_hard_stop(job_id)
    with _lock:
        entry = _active.get(job_id)
        reason = entry["reason"] if entry is not None else None
    if reason is not None:
        raise TrainingAbortedError(reason)


def hard_stop_limit_gb() -> float:
    # #1483: FUSION_MLX_TRAIN_HARD_STOP_RSS_GB, parsed per call (cheap) so
    # tests and ops can flip it without an import-cycle dance. 0/unset =
    # disabled. Invalid values fail loudly and disable — a mis-typed knob
    # must never kill every job on the box.
    raw = os.environ.get("FUSION_MLX_TRAIN_HARD_STOP_RSS_GB")
    if not raw:
        return 0.0
    try:
        val = float(raw)
    except ValueError:
        logger.warning(
            "Invalid FUSION_MLX_TRAIN_HARD_STOP_RSS_GB=%r — RSS hard stop "
            "disabled (#1483)",
            raw,
        )
        return 0.0
    if val <= 0:
        return 0.0
    return val


def _current_rss_gb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (2**30)
    except Exception:
        logger.debug("RSS read unavailable — hard-stop check skipped (#1483)")
        return None


def check_rss_hard_stop(job_id: str) -> None:
    # #1483: sustained-RSS gate. Called from check_abort between train
    # iterations. Requires _RSS_STRIKES consecutive over-limit readings so a
    # transient spike (one big prefill on a co-resident engine) cannot kill
    # a healthy job; resets to zero the first time RSS drops back under.
    limit = hard_stop_limit_gb()
    if limit <= 0:
        with _lock:
            _rss_strikes.pop(job_id, None)
        return
    rss = _current_rss_gb()
    if rss is None:
        return
    if rss < limit:
        with _lock:
            _rss_strikes[job_id] = 0
        return
    with _lock:
        strikes = _rss_strikes.get(job_id, 0) + 1
        _rss_strikes[job_id] = strikes
    logger.warning(
        "training RSS %.1fGB over hard-stop limit %.1fGB (job %s, %d/%d "
        "consecutive, #1483)",
        rss,
        limit,
        job_id,
        strikes,
        _RSS_STRIKES,
    )
    if strikes >= _RSS_STRIKES:
        raise TrainingAbortedError(
            f"RSS hard stop (#1483): process RSS {rss:.1f}GB exceeded "
            f"FUSION_MLX_TRAIN_HARD_STOP_RSS_GB={limit:.1f}GB for "
            f"{strikes} consecutive checks — job aborted before host-level "
            f"damage. Partial adapter state (if saved) is resumable."
        )


def _active_label(job_id: str) -> str:
    with _lock:
        entry = _active.get(job_id)
        return entry["service"] if entry is not None else "?"
