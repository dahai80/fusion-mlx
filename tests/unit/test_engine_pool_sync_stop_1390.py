# SPDX-License-Identifier: Apache-2.0
# #1390: the sync pool unload paths called the async engine.stop() without
# awaiting it, dropping the coroutine (RuntimeWarning: never awaited) and
# skipping the teardown.

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from fusion_mlx.pool.engine_pool import EnginePool


def _pool_with_engine(engine):
    pool = EnginePool.__new__(EnginePool)
    pool._entries = {}
    entry = MagicMock()
    entry.engine = engine
    pool._entries["m"] = entry
    return pool, entry


def test_stop_runs_to_completion_without_a_running_loop():
    engine = MagicMock()
    engine.stop = AsyncMock()
    pool, entry = _pool_with_engine(engine)
    pool._stop_engine_now(entry, "m")
    engine.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_is_scheduled_when_a_loop_is_running():
    engine = MagicMock()
    stop_event = asyncio.Event()

    async def _stop():
        stop_event.set()

    engine.stop = MagicMock(side_effect=_stop)
    pool, entry = _pool_with_engine(engine)
    pool._stop_engine_now(entry, "m")
    assert not stop_event.is_set()
    await asyncio.wait_for(stop_event.wait(), timeout=1.0)


def test_sync_stop_engine_is_ignored():
    engine = MagicMock()
    engine.stop = MagicMock(return_value=None)
    pool, entry = _pool_with_engine(engine)
    pool._stop_engine_now(entry, "m")
    engine.stop.assert_called_once()


def test_stop_failure_is_logged_not_raised():
    engine = MagicMock()
    engine.stop = MagicMock(side_effect=RuntimeError("boom"))
    pool, entry = _pool_with_engine(engine)
    pool._stop_engine_now(entry, "m")
    assert entry.engine is engine


def test_missing_engine_is_a_noop():
    pool, entry = _pool_with_engine(None)
    pool._stop_engine_now(entry, "m")
