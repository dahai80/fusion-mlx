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

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
# job_id -> {"service": <label>, "reason": <str | None>}
_active: dict[str, dict] = {}


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
    # abort was requested for this job; no-op otherwise (including when the
    # job is not registered).
    with _lock:
        entry = _active.get(job_id)
        reason = entry["reason"] if entry is not None else None
    if reason is not None:
        raise TrainingAbortedError(reason)


def _active_label(job_id: str) -> str:
    with _lock:
        entry = _active.get(job_id)
        return entry["service"] if entry is not None else "?"
