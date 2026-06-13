# SPDX-License-Identifier: Apache-2.0
"""PriorityScheduler - Metal multi-queue priority scheduling.

Sits on top of the existing Scheduler (scheduler.py) and adds:
- 3 priority queues: REALTIME, BATCH, BACKGROUND
- Metal stream mapping per priority (separate command queues)
- Preemption: REALTIME arrival can pause BATCH prefill
- Starvation prevention: BACKGROUND gets at least 10% capacity
- UnifiedMemoryPool integration for preemption memory cleanup

Does NOT modify the base Scheduler. Composes via wrapper pattern.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, AsyncIterator, Optional
from typing import Protocol

import mlx.core as mx

from ..request import Request, RequestOutput, SamplingParams
from ..router.smart_router import TaskPriority

logger = logging.getLogger(__name__)


class SchedulerProtocol(Protocol):
    """Minimal interface required from the base scheduler.

    Replaces runtime hasattr checks with a static contract.
    If the base Scheduler changes internal names, this Protocol
    will fail at type-check time rather than silently at runtime.
    """

    running: dict[str, Any]
    waiting: dict[str, Any]

    def add_request(self, request: Request) -> None: ...
    def abort_request(self, request_id: str) -> bool: ...
    def has_requests(self) -> bool: ...
    def step(self) -> Any: ...
    def get_request(self, request_id: str) -> Optional[Request]: ...


class PriorityLevel(IntEnum):
    """Priority levels, lower number = higher priority.

    Maps to TaskPriority from smart_router but uses IntEnum for
    natural sorting (lower = more urgent).
    """
    REALTIME = 0        # Claude Code, interactive - Metal queue 0
    BATCH = 1             # OpenClaw, agents - Metal queue 1
    BACKGROUND = 2        # Embedding, offline - Metal queue 2


@dataclass
class PriorityRequest:
    """Wrapper around Request with priority metadata."""
    request: Request
    priority: PriorityLevel
    queued_at: float = field(default_factory=time.time)
    deadline: float | None = None      # Optional deadline (unix timestamp)
    source_tag: str = ""                # "claude_code", "openclaw", etc.

    @property
    def is_expired(self) -> bool:
        if self.deadline is None:
            return False
        return time.time() > self.deadline


@dataclass
class PrioritySchedulerConfig:
    """Configuration for priority scheduling."""

    # Max concurrent requests per priority level
    max_realtime_seqs: int = 8
    max_batch_seqs: int = 16
    max_background_seqs: int = 4

    # Minimum capacity reservation for lower priorities (0.0-1.0)
    min_background_share: float = 0.10
    min_batch_share: float = 0.20

    # Preemption settings
    preempt_batch_for_realtime: bool = True
    preempt_background_for_any: bool = True

    # Metal stream configuration
    use_separate_streams: bool = True

    # Starvation detection (steps without service -> force-schedule)
    starvation_threshold_steps: int = 100

    # Chunked prefill for soft-preemption (tokens per chunk)
    prefill_chunk_size: int = 512

    # Metal command queue priority (0 = highest, used for MTLCommandQueue)
    metal_queue_priority_realtime: int = 0
    metal_queue_priority_batch: int = 1
    metal_queue_priority_background: int = 2

    # Software time-slicing: max BATCH requests to schedule per step
    # when REALTIME queue is non-empty (prevents BATCH from hogging)
    max_batch_per_step_when_rt_waiting: int = 1
    max_background_per_step_when_rt_waiting: int = 0


@dataclass
class ScheduleDecision:
    """Result of one scheduling cycle."""
    scheduled_request_ids: list[str]
    preempted_request_ids: list[str]
    rejected_outputs: list[RequestOutput]
    active_streams: list[str]


class PriorityScheduler:
    """Priority-aware request scheduler with Metal multi-queue support.

    Architecture:
         - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         |  PriorityScheduler (this module)                   |
         |  +- - - - - -+- - - - - -+- - - - - - - - - - - -+ |
         |  |REALTIME   |  BATCH    |    BACKGROUND        | |
         |  |queue 0    |  queue 1  |    queue 2           | |
         |  +- - +- - - -+- - +- - - -+- - - - - +- - - - + |
         |        |           |                 |               |
         - - - - - -+- - - - - - - - - - - - - - - - - - - - - - - - -
                         |           |                 |
         - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         |  Base Scheduler (scheduler.py)                        |
         |  +- - - - - - - - - - - - - - - - - - - - - - - - -+ |
         |  |  Single BatchGenerator, single step()             | |
         |  +- - - - - - - - - - - - - - - - - - - - - - - - -+ |
         - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    Key behaviors:
         1. Requests enter priority queues based on TaskPriority tag
         2. _schedule_waiting drains REALTIME first, then BATCH, then BACKGROUND
         3. Each priority gets its own Metal stream (command queue)
         4. When REALTIME queue is not empty and preempt_batch_for_realtime=True,
            in-progress BATCH prefill can be paused (via admission pause signal)
         5. BACKGROUND always gets min_background_share of capacity
         6. Preemption coordinates with UnifiedMemoryPool to free KV cache
    """

    def __init__(
        self,
        base_scheduler: SchedulerProtocol,
        config: PrioritySchedulerConfig | None = None,
        memory_pool: Any | None = None,
         ):
        self.base = base_scheduler
        self.config = config or PrioritySchedulerConfig()
        self.memory_pool = memory_pool
        self._lock = threading.Lock()

        # Priority queues (deque per level)
        self._queues: dict[PriorityLevel, deque[PriorityRequest]] = {
            PriorityLevel.REALTIME: deque(),
            PriorityLevel.BATCH: deque(),
            PriorityLevel.BACKGROUND: deque(),
         }

        # Metal streams per priority (command queues)
        self._streams: dict[PriorityLevel, Any] = {}
        if self.config.use_separate_streams:
            for pl in PriorityLevel:
                self._streams[pl] = mx.new_stream(mx.default_device())

        # Track which priority level each running request belongs to
        self._request_priorities: dict[str, PriorityLevel] = {}

        # Starvation tracking: steps since last service per priority
        self._steps_since_served: dict[PriorityLevel, int] = {
            PriorityLevel.REALTIME: 0,
            PriorityLevel.BATCH: 0,
            PriorityLevel.BACKGROUND: 0,
         }

        # Stats
        self._total_preemptions = 0
        self._total_starvation_rescues = 0
        self._steps = 0

    def submit(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        priority: TaskPriority | PriorityLevel = TaskPriority.BATCH,
        source_tag: str = "",
        deadline: float | None = None,
         **kwargs,
         ) -> str:
        if isinstance(priority, TaskPriority):
            priority = self._task_to_priority(priority)

        if request_id is None:
            request_id = uuid.uuid4().hex[:16]

        if sampling_params is None:
            sampling_params = SamplingParams()

        req = Request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
         )

        pri_req = PriorityRequest(
            request=req,
            priority=priority,
            source_tag=source_tag,
            deadline=deadline,
         )

        with self._lock:
            self._queues[priority].append(pri_req)
            depth = self._queues[priority].__len__()
        logger.debug(
            f"[PriorityScheduler] queued {request_id} @ {priority.name}, "
            f"queue_depth={depth}"
         )
        return request_id

    def submit_to_base(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        priority: TaskPriority | PriorityLevel = TaskPriority.BATCH,
         **kwargs,
         ) -> str:
        rid = self.submit(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            priority=priority,
             **kwargs,
         )
        self._drain_to_base(max_requests=1, priority_filter=priority if isinstance(priority, PriorityLevel) else self._task_to_priority(priority))
        return rid

    def step(self) -> Any:
        """Execute one scheduling step with priority awareness."""
        with self._lock:
            self._steps += 1

        with self._lock:
            rt_waiting = bool(self._queues[PriorityLevel.REALTIME])
        if self.config.preempt_batch_for_realtime and rt_waiting:
            self._maybe_preempt()

        self._check_chunked_prefill()

        # Software time-slicing: cap BATCH/BG when RT is waiting
        scheduled_this_step = 0
        reservations = self._get_reserved_slots()
        for pl in PriorityLevel:
            max_req = min(self._get_max_for_level(pl), reservations.get(pl, 0) + 1)
            if rt_waiting and pl != PriorityLevel.REALTIME:
                max_req = min(
                    max_req,
                    self.config.max_batch_per_step_when_rt_waiting if pl == PriorityLevel.BATCH
                    else self.config.max_background_per_step_when_rt_waiting,
                )
            n = self._drain_to_base(
                max_requests=max_req,
                priority_filter=pl,
             )
            scheduled_this_step += n
            with self._lock:
                if n > 0:
                    self._steps_since_served[pl] = 0
                else:
                    self._steps_since_served[pl] += 1

        self._check_starvation()
        return self.base.step()

    def abort_request(self, request_id: str) -> bool:
        return self.base.abort_request(request_id)

    def has_requests(self) -> bool:
        with self._lock:
            q_active = any(q for q in self._queues.values())
        return self.base.has_requests() or q_active

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                 "queue_depths": {pl.name: len(q) for pl, q in self._queues.items()},
                 "running_count": len(self.base.running),
                 "steps": self._steps,
                 "total_preemptions": self._total_preemptions,
                 "total_starvation_rescues": self._total_starvation_rescues,
                 "steps_since_served": {pl.name: c for pl, c in self._steps_since_served.items()},
                 "streams": {pl.name: str(s) for pl, s in self._streams.items()},
              }

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Internal - queue draining, preemption, starvation
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def _drain_to_base(self, max_requests: int = 1, priority_filter: PriorityLevel | None = None) -> int:
        if not hasattr(self.base, "add_request"):
            return 0

        scheduled = 0
        levels_to_check = [priority_filter] if priority_filter else list(PriorityLevel)

        for pl in levels_to_check:
            if scheduled >= max_requests:
                break
            while True:
                pri_req = None
                with self._lock:
                    q = self._queues[pl]
                    if not q:
                        break
                    pri_req = q[0]
                    max_seqs = self._get_max_for_level(pl)
                    running_at_level = sum(
                          1 for p in self._request_priorities.values() if p == pl
                        )
                    if running_at_level >= max_seqs:
                        break
                    pri_req = q.popleft()
                    self._request_priorities[pri_req.request.request_id] = pl
                    scheduled += 1
                if pri_req is None:
                    break
                # Set stream and submit to base outside lock (may block)
            self._set_stream_for_request(pri_req.request, pl)
            try:
                self.base.add_request(pri_req.request)
            except Exception as e:
                logger.warning(f"[PriorityScheduler] failed to schedule {pri_req.request.request_id}: {e}")
                with self._lock:
                    self._request_priorities.pop(pri_req.request.request_id, None)

        return scheduled

    def _maybe_preempt(self) -> None:
        """Preempt low-priority requests when REALTIME queue is waiting."""
        with self._lock:
            running_counts = {pl: 0 for pl in PriorityLevel}
            for rid, pl in self._request_priorities.items():
                running_counts[pl] += 1
            rt_has = bool(self._queues[PriorityLevel.REALTIME])
            batch_has = bool(self._queues[PriorityLevel.BATCH])

        if (
            self.config.preempt_background_for_any
            and running_counts[PriorityLevel.BACKGROUND] > 0
            and (rt_has or batch_has)
           ):
            self._preempt_at_level(PriorityLevel.BACKGROUND)

        if (
            self.config.preempt_batch_for_realtime
            and running_counts[PriorityLevel.BATCH] > 0
            and rt_has
           ):
            if len(self._queues[PriorityLevel.REALTIME]) >= 2:
                self._preempt_at_level(PriorityLevel.BATCH)

    def _preempt_at_level(self, level: PriorityLevel) -> None:
        """Abort all running requests at a given priority level.

        Coordinates with UnifiedMemoryPool to free KV cache buffers,
        preventing memory deadlocks where a REALTIME request can't start
        because a preempted request still holds GPU memory.
        """
        with self._lock:
            to_preempt = [
                rid for rid, pl in self._request_priorities.items()
                if pl == level
              ]
            for rid in to_preempt:
                del self._request_priorities[rid]
                self._total_preemptions += 1

        for rid in to_preempt:
            # Free KV cache in memory pool before aborting
            self._evict_kv_for_request(rid)
            self.base.abort_request(rid)

        if to_preempt:
            logger.warning(
                f"[PriorityScheduler] preempted {len(to_preempt)} "
                f"{level.name} requests for higher priority"
                  )

    def _evict_kv_for_request(self, request_id: str) -> None:
        """Coordinate with UnifiedMemoryPool to free KV cache on preemption."""
        if self.memory_pool is None:
            return
        try:
            self.memory_pool.bridge.evict_or_swap_active_kv(request_id)
        except Exception as e:
            logger.error(f"[PriorityScheduler] KV evict failed for {request_id}: {e}")

    def _check_chunked_prefill(self) -> None:
        """Check if any BATCH prefill should be chunked for preemption safety.

        Metal doesn't support hardware-level command buffer preemption.
        Instead, we break long prefills into chunks of prefill_chunk_size tokens.
        Between chunks, we check if REALTIME requests are waiting.

        This is a 'soft preemption' - the current chunk finishes, then we yield.
        """
        with self._lock:
            if not self._queues[PriorityLevel.REALTIME]:
                return
            rids = list(self._request_priorities.keys())

        if not hasattr(self.base, "running"):
            return
        if not hasattr(self.base, "get_request"):
            return

        for rid in rids:
            with self._lock:
                pl = self._request_priorities.get(rid)
            if pl != PriorityLevel.BATCH:
                continue
            try:
                req = self.base.get_request(rid)
                if req is None:
                    continue
                prompt = getattr(req, "prompt", "")
                if isinstance(prompt, str):
                    token_est = len(prompt) // 4
                else:
                    token_est = len(prompt) if isinstance(prompt, list) else 0
                if token_est > self.config.prefill_chunk_size:
                    logger.debug(
                        f"[PriorityScheduler] BATCH request {rid} has {token_est} tokens, "
                        f"chunked prefill active (chunk={self.config.prefill_chunk_size})"
                     )
                    if hasattr(self.base, "set_max_tokens_per_step"):
                        self.base.set_max_tokens_per_step(self.config.prefill_chunk_size)
            except Exception:
                pass

    def _set_metal_queue_priority(self, stream: Any, level: PriorityLevel) -> None:
        """Set native Metal command queue priority for a stream.

        Best-effort - depends on MLX exposing MTLCommandQueue priority.
        If MLX doesn't support this, the call is silently ignored.
        The real priority enforcement happens in software via time-slicing.
        """
        priority_map = {
            PriorityLevel.REALTIME: self.config.metal_queue_priority_realtime,
            PriorityLevel.BATCH: self.config.metal_queue_priority_batch,
            PriorityLevel.BACKGROUND: self.config.metal_queue_priority_background,
         }
        pri = priority_map.get(level)
        if pri is not None and hasattr(stream, "set_priority"):
            try:
                stream.set_priority(pri)
            except Exception:
                pass

    def _check_starvation(self) -> None:
        """Force-schedule starved low-priority requests."""
        for pl in [PriorityLevel.BACKGROUND, PriorityLevel.BATCH]:
            with self._lock:
                starved = self._steps_since_served[pl] >= self.config.starvation_threshold_steps
                has_q = bool(self._queues[pl])
                steps = self._steps_since_served[pl]
            if starved and has_q:
                n = self._drain_to_base(max_requests=1, priority_filter=pl)
                if n > 0:
                    with self._lock:
                        self._total_starvation_rescues += 1
                    logger.info(
                        f"[PriorityScheduler] starvation rescue: "
                        f"scheduled 1 {pl.name} request after {steps} steps"
                          )

    def _get_reserved_slots(self) -> dict[PriorityLevel, int]:
        """Calculate minimum slot reservation per priority based on config shares."""
        with self._lock:
            running = len(self._request_priorities)
            rt_running = sum(1 for p in self._request_priorities.values() if p == PriorityLevel.REALTIME)
            bt_running = sum(1 for p in self._request_priorities.values() if p == PriorityLevel.BATCH)
            bg_running = sum(1 for p in self._request_priorities.values() if p == PriorityLevel.BACKGROUND)
        total_capacity = self.config.max_realtime_seqs + self.config.max_batch_seqs + self.config.max_background_seqs
        available = max(0, total_capacity - running)
        bg_reserve = max(0, int(available * self.config.min_background_share) - bg_running)
        bt_reserve = max(0, int(available * self.config.min_batch_share) - bt_running)
        rt_available = max(0, available - bg_reserve - bt_reserve)
        return {
            PriorityLevel.REALTIME: max(0, min(
                self.config.max_realtime_seqs - rt_running, rt_available + self.config.max_realtime_seqs)),
            PriorityLevel.BATCH: max(0, min(
                self.config.max_batch_seqs - bt_running, bt_reserve + self.config.max_batch_seqs)),
            PriorityLevel.BACKGROUND: max(0, min(
                self.config.max_background_seqs - bg_running, bg_reserve + self.config.max_background_seqs)),
          }

    def _get_max_for_level(self, level: PriorityLevel) -> int:
        if level == PriorityLevel.REALTIME:
            return self.config.max_realtime_seqs
        if level == PriorityLevel.BATCH:
            return self.config.max_batch_seqs
        return self.config.max_background_seqs

    def _set_stream_for_request(self, request: Request, level: PriorityLevel) -> None:
        if not self.config.use_separate_streams:
            return
        stream = self._streams.get(level)
        if stream and hasattr(request, "stream"):
            request.stream = stream
            self._set_metal_queue_priority(stream, level)

    @staticmethod
    def _task_to_priority(task: TaskPriority) -> PriorityLevel:
        mapping = {
            TaskPriority.REALTIME: PriorityLevel.REALTIME,
            TaskPriority.BATCH: PriorityLevel.BATCH,
            TaskPriority.BACKGROUND: PriorityLevel.BACKGROUND,
         }
        return mapping.get(task, PriorityLevel.BATCH)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Cleanup
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def cleanup_finished(self) -> None:
        """Clean up priority tracking for finished requests."""
        active_ids = set(self.base.running.keys())
        to_remove = [
            rid for rid in self._request_priorities
            if rid not in active_ids
         ]
        for rid in to_remove:
            del self._request_priorities[rid]
