# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1056.

#1056: abort_request returned False on _closed without putting a terminal
output or marking finished -> ctx lingered in _active_contexts until close().
Also re-abort of a finished request pushed a duplicate "abort" terminal
(P2-02 double-terminal window).
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from fusion_mlx.engine_core import EngineCore, RequestContext
from fusion_mlx.request import RequestOutput


def _make_core(closed=False, scheduler=None):
    core = EngineCore.__new__(EngineCore)
    core._closed = closed
    core.scheduler = scheduler
    core._active_contexts = {}
    core._finished_at = {}
    return core


def _add_ctx(core, rid, finished=False):
    collector = MagicMock()
    finished_event = asyncio.Event()
    if finished:
        finished_event.set()
    ctx = RequestContext(
        collector=collector,
        stream_state=MagicMock(),
        finished_event=finished_event,
    )
    core._active_contexts[rid] = ctx
    return ctx, collector


class TestAbortCloseRace1056:
    """_closed path still terminals; idempotent on finished requests."""

    @pytest.mark.asyncio
    async def test_closed_puts_terminal_and_marks_finished(self):
        sched = MagicMock()
        sched.abort_request = MagicMock(return_value=True)
        core = _make_core(closed=True, scheduler=sched)
        ctx, collector = _add_ctx(core, "req-1", finished=False)

        result = await core.abort_request("req-1")

        assert result is False
        collector.put.assert_called_once()
        out = collector.put.call_args.args[0]
        assert isinstance(out, RequestOutput)
        assert out.finished is True
        assert out.finish_reason == "abort"
        assert ctx.finished_event.is_set()
        assert "req-1" in core._finished_at

    @pytest.mark.asyncio
    async def test_closed_no_ctx_returns_false_clean(self):
        core = _make_core(closed=True, scheduler=MagicMock())
        result = await core.abort_request("ghost-req")
        assert result is False

    @pytest.mark.asyncio
    async def test_idempotent_no_duplicate_terminal(self):
        sched = MagicMock()
        sched.abort_request = MagicMock(return_value=True)
        core = _make_core(closed=False, scheduler=sched)
        ctx, collector = _add_ctx(core, "req-2", finished=True)

        result = await core.abort_request("req-2")

        assert result is True
        collector.put.assert_not_called()

    @pytest.mark.asyncio
    async def test_open_path_terminals_and_aborts(self):
        sched = MagicMock()
        sched.abort_request = MagicMock(return_value=True)
        core = _make_core(closed=False, scheduler=sched)
        ctx, collector = _add_ctx(core, "req-3", finished=False)

        result = await core.abort_request("req-3")

        assert result is True
        collector.put.assert_called_once()
        assert ctx.finished_event.is_set()
        sched.abort_request.assert_called_once_with("req-3")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
