from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from fusion_mlx.request import (
    Request, RequestOutput, RequestStatus, SamplingParams,
)
from fusion_mlx.scheduler.config import SchedulerConfig
from fusion_mlx.scheduler.types import (
    _StoreCacheGate, _PrefillAbortedError, _PrefillState,
)


class TestSoftPreemption:
    """Test _reschedule_running_requests soft preemption path."""

    def _make_scheduler(self):
        from fusion_mlx.scheduler.sched_response import _reschedule_running_requests
        s = MagicMock()
        s.running = {}
        s.waiting = deque()
        s.requests = {}
        return s, _reschedule_running_requests

    def test_reschedule_moves_to_waiting(self):
        s, fn = self._make_scheduler()
        req = Request("r1", "hi", SamplingParams())
        req.status = RequestStatus.RUNNING
        s.running["r1"] = req
        s.requests["r1"] = req

        failed = fn(s)
        assert failed == []
        assert "r1" not in s.running
        assert s.waiting[0].request_id == "r1"
        assert req.status == RequestStatus.WAITING

    def test_reschedule_resets_state(self):
        s, fn = self._make_scheduler()
        req = Request("r1", "hi", SamplingParams())
        req.status = RequestStatus.RUNNING
        req.output_token_ids = [1, 2]
        req.num_computed_tokens = 5
        req.cached_tokens = 10
        req.prompt_cache = ["cache"]
        req.remaining_tokens = [99]
        s.running["r1"] = req
        s.requests["r1"] = req

        fn(s)
        assert req.output_token_ids == []
        assert req.num_computed_tokens == 0
        assert req.cached_tokens == 0
        assert req.prompt_cache is None
        assert req.remaining_tokens == req.prompt_token_ids

    def test_corruption_retry_limit(self):
        s, fn = self._make_scheduler()
        req = Request("r1", "hi", SamplingParams())
        req.status = RequestStatus.RUNNING
        req.cache_corruption_retries = 3
        s.running["r1"] = req
        s.requests["r1"] = req

        failed = fn(s, is_corruption=True, max_corruption_retries=3)
        assert "r1" in failed
        assert "r1" not in s.running

    def test_corruption_below_limit_reschedules(self):
        s, fn = self._make_scheduler()
        req = Request("r1", "hi", SamplingParams())
        req.status = RequestStatus.RUNNING
        req.cache_corruption_retries = 1
        s.running["r1"] = req
        s.requests["r1"] = req

        failed = fn(s, is_corruption=True, max_corruption_retries=3)
        assert failed == []
        assert "r1" not in s.running
        assert s.waiting[0].request_id == "r1"

    def test_multiple_requests_reschedule(self):
        s, fn = self._make_scheduler()
        for i in range(3):
            req = Request(f"r{i}", "hi", SamplingParams())
            req.status = RequestStatus.RUNNING
            s.running[f"r{i}"] = req
            s.requests[f"r{i}"] = req

        failed = fn(s)
        assert failed == []
        assert len(s.waiting) == 3


class TestChunkedPrefillOrdering:
    """Test that chunked prefills are processed in FIFO order."""

    def test_prefill_fifo_order(self):
        from fusion_mlx.scheduler.sched_batch import _advance_chunked_prefills

        s = MagicMock()
        req_a = Request("ra", "prompt a", SamplingParams(max_tokens=64))
        req_b = Request("rb", "prompt b", SamplingParams(max_tokens=64))
        s.prefilling = deque([req_a, req_b])

        state_a = MagicMock()
        state_a.request = req_a
        state_b = MagicMock()
        state_b.request = req_b
        s._prefill_states = {"ra": state_a, "rb": state_b}
        s.config = SchedulerConfig(chunked_prefill=True, prefill_step_size=256)

        call_order = []

        def mock_step(state):
            call_order.append(state.request.request_id)
            return True

        s._step_prefill_chunk = mock_step
        s.batch_generator = MagicMock()
        s.batch_generator.insert = MagicMock(return_value=[1])
        s._ensure_batch_generator = MagicMock()
        s._emit_final_boundary_if_needed = MagicMock()
        s.running = {}
        s.requests = {}
        s.request_id_to_uid = {}
        s.uid_to_request_id = {}
        s.total_prompt_tokens = 0

        scheduled = []
        rejected = []
        _advance_chunked_prefills(s, scheduled, rejected)

        assert call_order == ["ra", "rb"]

    def test_chunked_prefill_partial_done(self):
        from fusion_mlx.scheduler.sched_batch import _advance_chunked_prefills

        s = MagicMock()
        req_a = Request("ra", "prompt a", SamplingParams())
        req_b = Request("rb", "prompt b", SamplingParams())
        s.prefilling = deque([req_a, req_b])

        state_a = MagicMock()
        state_a.request = req_a
        state_b = MagicMock()
        state_b.request = req_b
        s._prefill_states = {"ra": state_a, "rb": state_b}
        s.config = SchedulerConfig(chunked_prefill=True)

        call_count = [0]

        def mock_step(state):
            call_count[0] += 1
            return state.request.request_id == "ra"

        s._step_prefill_chunk = mock_step
        s.batch_generator = MagicMock()
        s.batch_generator.insert = MagicMock(return_value=[1])
        s._ensure_batch_generator = MagicMock()
        s._emit_final_boundary_if_needed = MagicMock()
        s.running = {}
        s.requests = {}
        s.request_id_to_uid = {}
        s.uid_to_request_id = {}
        s.total_prompt_tokens = 0

        scheduled = []
        rejected = []
        _advance_chunked_prefills(s, scheduled, rejected)

        assert len(s.prefilling) == 1
        assert s.prefilling[0].request_id == "rb"
        assert "ra" in s.running

    def test_chunked_prefill_empty_skip(self):
        from fusion_mlx.scheduler.sched_batch import _advance_chunked_prefills

        s = MagicMock()
        s.prefilling = deque()
        scheduled = []
        rejected = []
        _advance_chunked_prefills(s, scheduled, rejected)
        assert scheduled == []
        assert rejected == []

    def test_chunked_prefill_missing_state_skipped(self):
        from fusion_mlx.scheduler.sched_batch import _advance_chunked_prefills

        s = MagicMock()
        req = Request("r1", "hi", SamplingParams())
        s.prefilling = deque([req])
        s._prefill_states = {}  # state was cleaned up by abort
        s.running = {}
        s.requests = {}

        scheduled = []
        rejected = []
        _advance_chunked_prefills(s, scheduled, rejected)

        assert s.prefilling == deque()
        assert rejected == []


class TestAdmissionBackpressure:
    """Test admission controls during memory pressure."""

    def test_store_cache_gate_blocks_admission(self):
        from fusion_mlx.scheduler.types import _StoreCacheGate

        gate = _StoreCacheGate(cap=2)
        gate.note_submitted()
        gate.note_submitted()
        assert gate.has_capacity is False
        gate.note_done()
        assert gate.has_capacity is True

    def test_store_cache_cap_adjustment(self):
        from fusion_mlx.scheduler.sched_misc import adjust_store_cache_cap

        s = MagicMock()
        gate = _StoreCacheGate(cap=5)
        s._store_cache_gate = gate
        s.config = SchedulerConfig(max_num_seqs=10)

        adjust_store_cache_cap(s, "soft")
        assert gate.cap == 4

        adjust_store_cache_cap(s, "hard")
        assert gate.cap == 3

        adjust_store_cache_cap(s, "ok")
        assert gate.cap == 4

    def test_store_cache_cap_floor_one(self):
        from fusion_mlx.scheduler.sched_misc import adjust_store_cache_cap

        s = MagicMock()
        gate = _StoreCacheGate(cap=1)
        s._store_cache_gate = gate
        s.config = SchedulerConfig(max_num_seqs=10)

        adjust_store_cache_cap(s, "hard")
        assert gate.cap == 1

    def test_prefill_aborted_error_propagation(self):
        err = _PrefillAbortedError(aborted_uids=[1, 2], processed_tokens=50)
        assert err.aborted_uids == [1, 2]
        assert err.processed_tokens == 50
        assert "UIDs" in str(err)

    def test_admission_paused_break(self):
        from fusion_mlx.scheduler.sched_schedule import _schedule_waiting

        s = MagicMock()
        s.waiting = deque()
        s.running = {"r1": MagicMock()}
        s._admission_paused = True
        s.config = SchedulerConfig()
        s._store_cache_gate = None
        s._prefill_memory_guard = False

        scheduled, rejected = _schedule_waiting(s)
        assert scheduled == []
        assert rejected == []

    def test_store_cache_backpressure_break(self):
        from fusion_mlx.scheduler.sched_schedule import _schedule_waiting

        s = MagicMock()
        s.waiting = deque()
        s.running = {"r1": MagicMock()}
        s._admission_paused = False
        s.config = SchedulerConfig()
        gate = _StoreCacheGate(cap=1)
        gate.note_submitted()
        s._store_cache_gate = gate
        s._prefill_memory_guard = False

        scheduled, rejected = _schedule_waiting(s)
        assert scheduled == []
