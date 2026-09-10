"""G4 Agent governance — bounds graph execution resource use.

Enforces four guardrails on agent graph runs (A7/R7 audit):
  1. Step-count cap — max LLM calls per graph execution
  2. Wall-clock timeout — overall time limit
  3. Token budget — cumulative token spend across all steps
  4. Kill switch — running graph can be cancelled via API

Only active in ``full`` profile (route itself is profile-gated).
Env-configurable for power users; set to 0 to disable a guardrail.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

_MAX_STEPS = int(os.environ.get("FUSION_AGENT_MAX_STEPS", "20") or 20)
_TIMEOUT_S = float(os.environ.get("FUSION_AGENT_TIMEOUT_S", "300") or 300)
_TOKEN_BUDGET = int(os.environ.get("FUSION_AGENT_TOKEN_BUDGET", "32768") or 32768)


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    STEPS_EXCEEDED = "steps_exceeded"
    TIMEOUT = "timeout"
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"
    FAILED = "failed"


@dataclass
class GraphRun:
    run_id: str
    graph_id: str
    started_at: float = field(default_factory=time.time)
    steps: int = 0
    tokens_used: int = 0
    status: RunStatus = RunStatus.RUNNING
    cancel_requested: bool = False
    error: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def elapsed(self) -> float:
        return time.time() - self.started_at

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "graph_id": self.graph_id,
            "status": self.status.value,
            "steps": self.steps,
            "tokens_used": self.tokens_used,
            "elapsed_s": round(self.elapsed(), 2),
            "cancel_requested": self.cancel_requested,
            "error": self.error,
        }


class GovernorLimitExceeded(Exception):
    def __init__(self, run_id: str, reason: str, status: RunStatus):
        self.run_id = run_id
        self.reason = reason
        self.status = status
        super().__init__(f"Agent graph run {run_id} terminated: {reason}")


class AgentGovernor:
    """Tracks active graph executions and enforces guardrails."""

    def __init__(
        self,
        max_steps: int = _MAX_STEPS,
        timeout_s: float = _TIMEOUT_S,
        token_budget: int = _TOKEN_BUDGET,
    ):
        self.max_steps = max_steps
        self.timeout_s = timeout_s
        self.token_budget = token_budget
        self._runs: dict[str, GraphRun] = {}
        self._lock = threading.Lock()
        logger.info(
            "G4 agent governor: max_steps=%d timeout=%.0fs token_budget=%d",
            max_steps,
            timeout_s,
            token_budget,
        )

    def start_run(self, graph_id: str) -> GraphRun:
        run_id = uuid.uuid4().hex[:16]
        run = GraphRun(run_id=run_id, graph_id=graph_id)
        with self._lock:
            self._runs[run_id] = run
        logger.info("G4: started graph run %s for graph %s", run_id, graph_id)
        return run

    def get_run(self, run_id: str) -> GraphRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self) -> list[dict]:
        with self._lock:
            return [r.to_dict() for r in self._runs.values()]

    def cancel(self, run_id: str) -> bool:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            return False
        with run._lock:
            if run.status != RunStatus.RUNNING:
                return False
            run.cancel_requested = True
            run.status = RunStatus.CANCELLED
        logger.warning("G4: kill switch activated for run %s", run_id)
        return True

    def check_step(self, run: GraphRun) -> None:
        if self.max_steps > 0 and run.steps >= self.max_steps:
            run.status = RunStatus.STEPS_EXCEEDED
            raise GovernorLimitExceeded(
                run.run_id,
                f"step count {run.steps} >= max_steps {self.max_steps}",
                RunStatus.STEPS_EXCEEDED,
            )
        if self.timeout_s > 0 and run.elapsed() > self.timeout_s:
            run.status = RunStatus.TIMEOUT
            raise GovernorLimitExceeded(
                run.run_id,
                f"elapsed {run.elapsed():.1f}s > timeout {self.timeout_s}s",
                RunStatus.TIMEOUT,
            )
        if run.cancel_requested:
            raise GovernorLimitExceeded(
                run.run_id, "cancelled via kill switch", RunStatus.CANCELLED
            )

    def record_usage(self, run: GraphRun, usage: dict | None) -> None:
        if not usage:
            return
        tokens = usage.get("total_tokens") or 0
        if tokens == 0:
            tokens = (usage.get("prompt_tokens") or 0) + (
                usage.get("completion_tokens") or 0
            )
        with run._lock:
            run.tokens_used += tokens
            run.steps += 1
        if self.token_budget > 0 and run.tokens_used > self.token_budget:
            run.status = RunStatus.TOKEN_BUDGET_EXCEEDED
            raise GovernorLimitExceeded(
                run.run_id,
                f"tokens {run.tokens_used} > budget {self.token_budget}",
                RunStatus.TOKEN_BUDGET_EXCEEDED,
            )

    def finish(self, run: GraphRun, error: str = "") -> None:
        with run._lock:
            if run.status == RunStatus.RUNNING:
                run.status = RunStatus.FAILED if error else RunStatus.COMPLETED
                run.error = error

    def purge(self, max_age_s: float = 3600) -> int:
        now = time.time()
        purged = 0
        with self._lock:
            stale = [
                rid
                for rid, r in self._runs.items()
                if r.status != RunStatus.RUNNING and (now - r.started_at) > max_age_s
            ]
            for rid in stale:
                del self._runs[rid]
                purged += 1
        if purged:
            logger.debug("G4: purged %d stale graph run records", purged)
        return purged


_governor: AgentGovernor | None = None


def get_governor() -> AgentGovernor:
    global _governor
    if _governor is None:
        _governor = AgentGovernor()
    return _governor
