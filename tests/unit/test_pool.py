# SPDX-License-Identifier: Apache-2.0
"""Unit tests for pool modules — PriorityScheduler, EnginePool, MemoryEnforcer."""

from unittest.mock import MagicMock, patch

from fusion_mlx.pool.priority_scheduler import (
    PriorityLevel,
    PriorityRequest,
    PriorityScheduler,
    PrioritySchedulerConfig,
)


class TestPrioritySchedulerSubmit:

    def _make_scheduler(self):
        base = MagicMock()
        base.has_requests = MagicMock(return_value=False)
        return PriorityScheduler(base, PrioritySchedulerConfig(use_separate_streams=False))

    def test_submit_to_realtime_queue(self):
        sched = self._make_scheduler()
        rid = sched.submit("hello", priority=PriorityLevel.REALTIME)
        assert len(sched._queues[PriorityLevel.REALTIME]) == 1
        assert rid == sched._queues[PriorityLevel.REALTIME][0].request.request_id

    def test_submit_to_batch_queue(self):
        sched = self._make_scheduler()
        rid = sched.submit("hello", priority=PriorityLevel.BATCH)
        assert len(sched._queues[PriorityLevel.BATCH]) == 1

    def test_submit_generates_request_id(self):
        sched = self._make_scheduler()
        rid = sched.submit("hello")
        assert isinstance(rid, str)
        assert len(rid) == 16

    def test_submit_with_custom_id(self):
        sched = self._make_scheduler()
        rid = sched.submit("hello", request_id="custom-id-123")
        assert rid == "custom-id-123"

    def test_submit_with_task_priority(self):
        from fusion_mlx.router.smart_router import TaskPriority
        sched = self._make_scheduler()
        rid = sched.submit("hello", priority=TaskPriority.REALTIME)
        assert len(sched._queues[PriorityLevel.REALTIME]) == 1


class TestPrioritySchedulerReservedSlots:

    def _make_scheduler(self, config=None):
        base = MagicMock()
        base.has_requests = MagicMock(return_value=False)
        return PriorityScheduler(base, config or PrioritySchedulerConfig(use_separate_streams=False))

    def test_reserved_slots_non_negative(self):
        sched = self._make_scheduler()
        slots = sched._get_reserved_slots()
        for pl, count in slots.items():
            assert count >= 0, f"{pl.name} has negative reserved slots: {count}"

    def test_reserved_slots_respects_max_seqs(self):
        config = PrioritySchedulerConfig(
            max_realtime_seqs=4, max_batch_seqs=8, max_background_seqs=2,
            use_separate_streams=False,
          )
        sched = self._make_scheduler(config)
        slots = sched._get_reserved_slots()
        assert slots[PriorityLevel.REALTIME] >= 4
        assert slots[PriorityLevel.BATCH] >= 8
        assert slots[PriorityLevel.BACKGROUND] >= 2

    def test_reserved_slots_with_running_requests(self):
        sched = self._make_scheduler()
        sched._request_priorities["r1"] = PriorityLevel.REALTIME
        sched._request_priorities["r2"] = PriorityLevel.REALTIME
        sched._request_priorities["r3"] = PriorityLevel.REALTIME
        slots = sched._get_reserved_slots()
        assert all(v >= 0 for v in slots.values()), "Reserved slots should never be negative"


class TestPriorityRequest:

    def test_is_expired_false_when_no_deadline(self):
        req = PriorityRequest(
            request=MagicMock(), priority=PriorityLevel.BATCH, deadline=None,
          )
        assert req.is_expired is False

    def test_is_expired_true_when_past_deadline(self):
        req = PriorityRequest(
            request=MagicMock(), priority=PriorityLevel.BATCH, deadline=1000.0,
          )
        with patch("time.time", return_value=1001.0):
            assert req.is_expired is True

    def test_is_expired_false_when_not_yet(self):
        req = PriorityRequest(
            request=MagicMock(), priority=PriorityLevel.BATCH, deadline=2000.0,
          )
        with patch("time.time", return_value=1500.0):
            assert req.is_expired is False


class TestPriorityLevel:

    def test_priority_ordering(self):
        assert PriorityLevel.REALTIME < PriorityLevel.BATCH < PriorityLevel.BACKGROUND

    def test_priority_names(self):
        assert PriorityLevel.REALTIME.name == "REALTIME"
        assert PriorityLevel.BATCH.name == "BATCH"
        assert PriorityLevel.BACKGROUND.name == "BACKGROUND"
