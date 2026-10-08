# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1067.

#1067: ClusterHealthMonitor.stop used `except (CancelledError, Exception):
pass` — swallowed every exception silently (fake-success on the shutdown
path). Fix: CancelledError is expected (silent); other exceptions are logged
at warning level with exc_info.
"""

from __future__ import annotations

import asyncio
import inspect
import logging

import pytest

from fusion_mlx.cluster.registry import (
    ClusterHealthMonitor,
    ClusterNode,
    NodeRegistry,
)


def _node(node_id: str = "a") -> ClusterNode:
    return ClusterNode(node_id=node_id, host="127.0.0.1", port=8000)


class TestHealthMonitorStop1067:
    """stop logs non-cancellation exceptions instead of swallowing them."""

    def test_source_no_broad_swallow(self):
        source = inspect.getsource(ClusterHealthMonitor.stop)
        # pre-fix: `except (asyncio.CancelledError, Exception): pass`
        assert "except (asyncio.CancelledError, Exception)" not in source
        # post-fix: separate CancelledError (silent) + Exception (logged)
        assert "except asyncio.CancelledError" in source
        assert "except Exception" in source
        assert "logger.warning" in source

    @pytest.mark.asyncio
    async def test_stop_logs_non_cancellation_error(self, caplog):
        registry = NodeRegistry()

        async def boom(node):
            return True

        monitor = ClusterHealthMonitor(registry, boom, interval=999, max_missed=3)
        # A done Future carrying an exception: cancel() is a no-op on it and
        # `await self._task` re-raises — exercises the non-CancelledError branch
        # without racing the event loop's task scheduling.
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        fut.set_exception(RuntimeError("loop blew up"))
        monitor._task = fut  # type: ignore[assignment]
        with caplog.at_level(logging.WARNING, logger="fusion_mlx.cluster.registry"):
            await monitor.stop()
        assert any(
            "ended with error on stop" in r.message and "loop blew up" in r.message
            for r in caplog.records
        ), [r.message for r in caplog.records]

    @pytest.mark.asyncio
    async def test_stop_silent_on_cancellation(self, caplog):
        registry = NodeRegistry()

        async def beat(node):
            return True

        monitor = ClusterHealthMonitor(registry, beat, interval=999, max_missed=3)
        await monitor.start()
        # INFO level so the "stopped" line is captured too.
        with caplog.at_level(logging.INFO, logger="fusion_mlx.cluster.registry"):
            await monitor.stop()
        # only the info "stopped" line — no warning
        assert not any("ended with error on stop" in r.message for r in caplog.records)
        assert any(
            "cluster health monitor stopped" in r.message for r in caplog.records
        )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
