# SPDX-License-Identifier: Apache-2.0
"""#1187: unified admission gate for heavy workloads (training / image / video).

The LLM scheduler already has admission (sched_admission.py) and image gen
has a per-engine byte gate (image_gen._admit_generation). Training and video
start without any memory admission — #1184/#1186 showed a fine-tune starting
while serving weights are resident saturates the GPU for hours. This module
is the single gate all heavy workloads pass before starting compute:

    resident usage + task budget <= physical * safety line (default 0.85)

- training: deferred while the serving window holds a pause (#1187 pause
  endpoint), rejected when the line is exceeded (clear FAILED error).
- image/video: rejected (routes map to 507 via existing exception mapping).

Importers/callers: training services (_process_queue), image_gen
(_admit_generation), engines/video.py (VideoGenEngine.generate), and the
/v1/admission routes (pause/resume/status).
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Safety line: heavy workloads may start only while
# resident usage + task budget <= physical * line. Calibrated on 128GB:
# observed fine-tune peak 64.9GB (#1186), resident serving set + 60%-of-RAM
# training budget must stay under the line (see PR #1227 audit note).
DEFAULT_SAFETY_LINE = 0.85
# Training budget when a job/config carries no explicit memory_limit_gb —
# matches training/proc.py's child budget default (#1186 calibration:
# observed training peak 64.9GB on 128GB = ~50%; 60% leaves serving headroom).
DEFAULT_TRAINING_BUDGET_FRACTION = 0.60
# While paused, training jobs stay queued and poll for the window to end.
ADMISSION_RECHECK_SECONDS = 30.0

ENV_SAFETY_LINE = "FUSION_MLX_HEAVY_ADMISSION_LINE"
ENV_DISABLE = "FUSION_MLX_HEAVY_ADMISSION"


@dataclass
class HeavyAdmissionDecision:
    admitted: bool
    deferred: bool = False
    reason: str = ""
    detail: dict = field(default_factory=dict)


class HeavyAdmissionGate:
    def __init__(self):
        self._lock = threading.Lock()
        self._paused = False
        self._paused_at: float | None = None
        self._safety_line = DEFAULT_SAFETY_LINE
        self._enabled = True
        self._recheck_targets: weakref.WeakSet = weakref.WeakSet()

    # -- configuration ------------------------------------------------

    def configure(
        self,
        safety_line_fraction: float | None = None,
        enabled: bool | None = None,
    ) -> None:
        with self._lock:
            if safety_line_fraction is not None:
                self._safety_line = max(0.5, min(0.99, float(safety_line_fraction)))
            if enabled is not None:
                self._enabled = bool(enabled)
        logger.info(
            "heavy admission gate configured: line=%.2f enabled=%s",
            self._safety_line,
            self._enabled,
        )

    def _resolve_safety_line(self) -> float:
        import os

        raw = os.environ.get(ENV_SAFETY_LINE, "").strip()
        if raw:
            try:
                return max(0.5, min(0.99, float(raw)))
            except ValueError:
                logger.warning("invalid %s=%r; using default", ENV_SAFETY_LINE, raw)
        return self._safety_line

    def _resolve_enabled(self) -> bool:
        import os

        if os.environ.get(ENV_DISABLE, "").strip().lower() in ("off", "0", "false"):
            return False
        return self._enabled

    # -- budget -------------------------------------------------------

    def training_budget_bytes(self, memory_limit_gb: float = 0.0) -> int:
        # 0 = auto: default fraction of physical memory (#1186 calibration).
        if memory_limit_gb and memory_limit_gb > 0:
            return int(memory_limit_gb * (2**30))
        return int(self.physical_bytes() * DEFAULT_TRAINING_BUDGET_FRACTION)

    # -- memory sources -----------------------------------------------

    def physical_bytes(self) -> int:
        try:
            from .pool.settings import get_system_memory

            return int(get_system_memory())
        except Exception:
            logger.exception("heavy admission: physical memory unavailable")
            return 0

    def current_usage_bytes(self) -> int:
        # Prefer the enforcer's jetsam-consistent sample (active-memory vs
        # phys_footprint logic, ANE/IOSurface adjustments); fall back to the
        # kernel phys_footprint when the enforcer is not wired (tests).
        try:
            from .server import _server_state

            enforcer = getattr(
                _server_state.engine_pool, "process_memory_enforcer", None
            )
            if enforcer is not None:
                return int(enforcer._current_usage_bytes())
        except Exception:
            pass
        try:
            from .utils.proc_memory import get_phys_footprint

            return int(get_phys_footprint())
        except Exception:
            logger.exception("heavy admission: usage unavailable")
            return 0

    # -- admission ----------------------------------------------------

    def admit(self, kind: str, budget_bytes: int) -> HeavyAdmissionDecision:
        started = time.monotonic()
        if not self._resolve_enabled():
            logger.debug("heavy admission disabled; admitting %s", kind)
            return HeavyAdmissionDecision(admitted=True, reason="gate disabled")
        with self._lock:
            paused = self._paused
        if paused and kind == "training":
            self._record(kind, admitted=False, deferred=True)
            return HeavyAdmissionDecision(
                admitted=False,
                deferred=True,
                reason=(
                    "training paused by serving window "
                    "(/v1/admission/resume to end the window)"
                ),
            )
        physical = self.physical_bytes()
        usage = self.current_usage_bytes()
        # Audit L2: resolve the fraction once — the message used to re-read
        # the env var and re-parse it, so a concurrent env change could make
        # the reported line disagree with the one actually enforced.
        line_fraction = self._resolve_safety_line()
        line = int(physical * line_fraction)
        projected = usage + max(0, int(budget_bytes))
        detail = {
            "kind": kind,
            "usage_bytes": usage,
            "budget_bytes": max(0, int(budget_bytes)),
            "physical_bytes": physical,
            "safety_line_bytes": line,
            "projected_bytes": projected,
        }
        if physical <= 0 or projected > line:
            self._record(kind, admitted=False, deferred=False)
            reason = (
                f"heavy workload '{kind}' rejected by admission gate: "
                f"projected {projected / 2**30:.1f}GB exceeds safety line "
                f"{line / 2**30:.1f}GB "
                f"(resident {usage / 2**30:.1f}GB + budget "
                f"{max(0, int(budget_bytes)) / 2**30:.1f}GB, line = "
                f"{physical / 2**30:.1f}GB x {line_fraction:.2f})"
            )
            logger.warning(
                "admission rejected %s in %.1fms: %s",
                kind,
                (time.monotonic() - started) * 1000,
                reason,
            )
            return HeavyAdmissionDecision(
                admitted=False, deferred=False, reason=reason, detail=detail
            )
        self._record(kind, admitted=True, deferred=False)
        logger.info(
            "admission admitted %s: projected %.1fGB <= line %.1fGB (%.1fms)",
            kind,
            projected / 2**30,
            line / 2**30,
            (time.monotonic() - started) * 1000,
        )
        return HeavyAdmissionDecision(admitted=True, detail=detail)

    def admit_training(self, budget_bytes: int | float = 0) -> HeavyAdmissionDecision:
        return self.admit("training", int(budget_bytes))

    def admit_media(self, kind: str, budget_bytes: int) -> HeavyAdmissionDecision:
        # #1452: stack outstanding media reservations from the enforcer onto
        # the budget. admit() computes usage + budget against the safety line;
        # concurrent media jobs each passed that check because the others'
        # admitted-but-not-yet-resident budgets were invisible — N jobs each
        # admitted at usage + own_budget. The pool gate (admit_media_job)
        # stacks the same reservations for its own ceiling check.
        outstanding = 0
        try:
            from .server import _server_state

            enforcer = getattr(
                _server_state.engine_pool, "process_memory_enforcer", None
            )
            if enforcer is not None:
                outstanding = int(enforcer.get_media_reservation_bytes())
        except Exception:
            outstanding = 0
        return self.admit(kind, int(budget_bytes) + outstanding)

    # -- pause / resume -------------------------------------------------

    def pause(self, scope: str = "training") -> dict:
        # Scope is training-only for now; media kinds stay admissible so a
        # serving window can still render images/videos while training yields.
        if scope != "training":
            raise ValueError(f"unsupported pause scope: {scope}")
        with self._lock:
            self._paused = True
            self._paused_at = time.time()
        logger.warning("heavy admission: training PAUSED by serving window")
        return self.snapshot()

    def resume(self) -> dict:
        with self._lock:
            was = self._paused
            self._paused = False
            self._paused_at = None
        if was:
            logger.warning("heavy admission: training RESUMED")
        for target in list(self._recheck_targets):
            try:
                target._process_queue()
            except Exception:
                logger.exception("admission resume recheck failed")
        return self.snapshot()

    def add_recheck_target(self, target) -> None:
        # WeakSet: dead services drop out automatically.
        self._recheck_targets.add(target)

    # -- observability --------------------------------------------------

    def _record(self, kind: str, admitted: bool, deferred: bool) -> None:
        try:
            from .server_metrics import get_server_metrics

            get_server_metrics().record_admission(
                kind, admitted=admitted, deferred=deferred
            )
        except Exception:
            logger.exception("admission metric record failed")

    def snapshot(self) -> dict:
        with self._lock:
            paused = self._paused
            paused_at = self._paused_at
        physical = self.physical_bytes()
        line_fraction = self._resolve_safety_line()
        line = int(physical * line_fraction)
        return {
            "enabled": self._resolve_enabled(),
            "paused": paused,
            "paused_at": paused_at,
            "safety_line_fraction": line_fraction,
            "physical_bytes": physical,
            "safety_line_bytes": line,
            "usage_bytes": self.current_usage_bytes(),
            "training_budget_default_bytes": self.training_budget_bytes(0.0),
            "recheck_seconds": ADMISSION_RECHECK_SECONDS,
        }


_gate: HeavyAdmissionGate | None = None
_gate_lock = threading.Lock()


def get_heavy_admission_gate() -> HeavyAdmissionGate:
    global _gate
    if _gate is None:
        with _gate_lock:
            if _gate is None:
                _gate = HeavyAdmissionGate()
    return _gate


def reset_heavy_admission_gate() -> None:
    global _gate
    with _gate_lock:
        _gate = None


def schedule_recheck(owner, delay: float = ADMISSION_RECHECK_SECONDS) -> None:
    # One pending recheck per owner; services reset the flag when the
    # recheck fires and _process_queue re-runs the gate.
    if getattr(owner, "_admission_recheck_scheduled", False):
        return
    loop = getattr(owner, "_loop", None)
    if loop is None or loop.is_closed():
        return
    owner._admission_recheck_scheduled = True

    def _fire():
        owner._admission_recheck_scheduled = False
        try:
            owner._process_queue()
        except Exception:
            logger.exception("admission recheck failed for %s", type(owner).__name__)

    try:
        loop.call_later(delay, _fire)
    except RuntimeError:
        owner._admission_recheck_scheduled = False
