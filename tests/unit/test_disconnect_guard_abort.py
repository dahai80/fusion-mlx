# SPDX-License-Identifier: Apache-2.0
"""Unit tests for _disconnect_guard abort-surface probing (#903)."""

import asyncio

import pytest

from fusion_mlx.api._disconnect_guard import handle_disconnect


class _EngineWithPerRequestAbort:
    def __init__(self):
        self.aborted = []

    async def abort_request(self, request_id):
        self.aborted.append(request_id)
        return True


class _EngineWithAbortAllOnly:
    """Pre-#903 VLMBatchedEngine shape: only abort_all_requests."""

    def __init__(self):
        self.abort_all_calls = 0

    async def abort_all_requests(self):
        self.abort_all_calls += 1
        return 1


class _EngineWithNoAbort:
    pass


@pytest.mark.asyncio
async def test_handle_disconnect_uses_per_request_abort():
    e = _EngineWithPerRequestAbort()
    handle_disconnect("req-1", e)
    await asyncio.sleep(0.05)
    assert e.aborted == ["req-1"]


@pytest.mark.asyncio
async def test_handle_disconnect_falls_back_to_abort_all():
    e = _EngineWithAbortAllOnly()
    handle_disconnect("req-1", e)
    await asyncio.sleep(0.05)
    assert e.abort_all_calls == 1


@pytest.mark.asyncio
async def test_handle_disconnect_no_abort_surface_no_crash():
    e = _EngineWithNoAbort()
    handle_disconnect("req-1", e)  # must not raise
    await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_handle_disconnect_abort_raising_is_contained(caplog):
    class _Bad:
        async def abort_request(self, request_id):
            raise RuntimeError("boom")

    handle_disconnect("req-1", _Bad())
    await asyncio.sleep(0.05)
    assert any("abort_request failed" in r.message for r in caplog.records)
