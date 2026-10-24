# SPDX-License-Identifier: Apache-2.0
# H4 (#1354 audit, 1019 report): abort_all_requests must mutate the scheduler
# waiting/running deques on the same thread that runs step() — the engine's
# single-worker MLX executor — never on the event-loop thread. deque
# popleft/append are not atomic across threads (the P0-3 race documented in
# add_request's rollback path).

import concurrent.futures
import threading

import pytest

from fusion_mlx.engine_core import EngineCore


class _FakeCollector:
    def __init__(self):
        self.outputs = []

    def put(self, output):
        self.outputs.append(output)


class _FakeEvent:
    def __init__(self):
        self._set = False

    def set(self):
        self._set = True

    def is_set(self):
        return self._set


class _FakeCtx:
    def __init__(self):
        self.collector = _FakeCollector()
        self.finished_event = _FakeEvent()


class _RecordingScheduler:
    def __init__(self):
        self.thread_idents = []
        self.aborted = []

    def abort_request(self, request_id):
        self.thread_idents.append(threading.get_ident())
        self.aborted.append(request_id)
        return True


def _make_engine(scheduler, request_ids, executor):
    engine = EngineCore.__new__(EngineCore)
    engine.scheduler = scheduler
    engine._mlx_executor = executor
    engine._active_contexts = {rid: _FakeCtx() for rid in request_ids}
    engine._finished_at = {}
    engine._closed = False
    return engine


@pytest.mark.asyncio
async def test_abort_all_requests_mutates_scheduler_on_mlx_executor_thread():
    loop_ident = threading.get_ident()
    request_ids = ["req-1", "req-2", "req-3"]
    scheduler = _RecordingScheduler()
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="test-mlx-h4"
    )
    engine = _make_engine(scheduler, request_ids, executor)
    try:
        aborted = await engine.abort_all_requests()
    finally:
        executor.shutdown(wait=True)

    assert aborted == len(request_ids)
    assert scheduler.aborted == request_ids
    assert scheduler.thread_idents, "scheduler.abort_request was never called"
    assert all(
        tid != loop_ident for tid in scheduler.thread_idents
    ), "abort_all_requests mutated scheduler queues on the event-loop thread"
    assert (
        len(set(scheduler.thread_idents)) == 1
    ), "scheduler aborts were split across threads; step() owns one executor"

    for rid in request_ids:
        ctx = engine._active_contexts[rid]
        assert ctx.finished_event.is_set()
        assert rid in engine._finished_at
        assert ctx.collector.outputs, f"no terminal pushed for {rid}"
        terminal = ctx.collector.outputs[-1]
        assert terminal.finished is True
        assert terminal.finish_reason == "error"
        assert "memory" in terminal.error.lower()


@pytest.mark.asyncio
async def test_abort_all_requests_empty_returns_zero_without_executor():
    scheduler = _RecordingScheduler()
    engine = _make_engine(scheduler, [], None)

    assert await engine.abort_all_requests() == 0
    assert scheduler.aborted == []
