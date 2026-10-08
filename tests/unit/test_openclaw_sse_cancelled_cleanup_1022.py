# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1022.

#1022: openclaw_routes SSE generator had `except asyncio.CancelledError: pass`
— swallowed disconnect silently, no logging, no re-raise (breaks cancellation
chains). Also session["active"] set True on turn start but never reset.

Fix (#1022):
- SSE CancelledError: log + re-raise (align with anthropic_routes pattern).
- execute_turn: finally block resets session["active"] = False.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from fastapi import HTTPException

from fusion_mlx.api.openclaw_routes import (
    TurnRequest,
    _sessions,
    execute_turn,
    stream_events,
)

_LOGGER = "fusion_mlx.api.openclaw_routes"


class TestSSECancelledReRaise:
    """SSE generator: CancelledError logged + re-raised, not swallowed."""

    @pytest.mark.asyncio
    async def test_cancelled_error_logged_and_reraised(self, caplog):
        _sessions["test-sse-1022"] = {
            "turn_count": 0,
            "active": False,
            "messages": [],
            "last_accessed": 0.0,
        }
        agen = None
        try:
            response = await stream_events("test-sse-1022")
            agen = response.body_iterator

            # Iterate via async for (mirrors Starlette StreamingResponse) inside
            # a task. Cancel the task to simulate client disconnect — the
            # except CancelledError block must log + re-raise.
            async def _consume():
                async for _chunk in agen:
                    pass

            task = asyncio.create_task(_consume())
            await asyncio.sleep(0.05)  # let it reach asyncio.sleep(30)

            with caplog.at_level(logging.INFO, logger=_LOGGER):
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

            assert any(
                "SSE client disconnected" in r.message for r in caplog.records
            ), f"expected disconnect log, got {[r.message for r in caplog.records]}"
        finally:
            _sessions.pop("test-sse-1022", None)
            if agen is not None:
                await agen.aclose()


class TestActiveFlagReset:
    """execute_turn: session['active'] reset to False after turn completes."""

    @pytest.mark.asyncio
    async def test_active_reset_on_failure(self):
        _sessions["test-active-ok"] = {
            "turn_count": 0,
            "active": False,
            "messages": [],
            "last_accessed": 0.0,
        }
        try:

            class FakePool:
                async def get_engine(self, name):
                    return None

            execute_turn._pool = FakePool()
            session = _sessions["test-active-ok"]

            req = TurnRequest(
                messages=[{"role": "user", "content": "hi"}],
                auto_execute=False,
                max_auto_iterations=1,
            )

            # Turn fails (no engine) but finally must reset active.
            with pytest.raises(HTTPException):
                await execute_turn("test-active-ok", req)

            assert session["active"] is False, "active flag not reset after failed turn"
        finally:
            _sessions.pop("test-active-ok", None)
            if hasattr(execute_turn, "_pool"):
                del execute_turn._pool

    @pytest.mark.asyncio
    async def test_active_reset_on_exception(self):
        _sessions["test-active-exc"] = {
            "turn_count": 0,
            "active": False,
            "messages": [],
            "last_accessed": 0.0,
        }
        try:

            class FakePool:
                async def get_engine(self, name):
                    raise RuntimeError("pool crash")

            execute_turn._pool = FakePool()
            session = _sessions["test-active-exc"]

            req = TurnRequest(
                messages=[{"role": "user", "content": "hi"}],
                auto_execute=False,
                max_auto_iterations=1,
            )

            with pytest.raises(Exception):
                await execute_turn("test-active-exc", req)

            assert session["active"] is False, "active flag not reset after exception"
        finally:
            _sessions.pop("test-active-exc", None)
            if hasattr(execute_turn, "_pool"):
                del execute_turn._pool


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
