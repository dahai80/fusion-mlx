# SPDX-License-Identifier: Apache-2.0
# #1294: this file used to import `_with_sse_keepalive` / `_resolve_keepalive`
# / `_KEEPALIVE_*` from fusion_mlx.server. Those symbols never existed on
# main — the module-level keepalive wrapper was replaced by the SSEKeepalive
# class plus stream_with_keepalive in fusion_mlx/api/utils.py, so all 13 tests
# here failed at import. The suite now pins the current API.

import asyncio

from fusion_mlx.api.utils import KEEPALIVE_PING, SSEKeepalive, stream_with_keepalive


class TestSSEKeepalive:
    def test_no_ping_before_interval(self):
        keepalive = SSEKeepalive(interval_seconds=60.0)
        keepalive.reset()
        assert keepalive.maybe_ping() is None

    def test_ping_after_interval(self, monkeypatch):
        clock = {"now": 1000.0}
        monkeypatch.setattr("fusion_mlx.api.utils.time.monotonic", lambda: clock["now"])
        keepalive = SSEKeepalive(interval_seconds=5.0)
        keepalive.reset()
        clock["now"] += 5.5
        assert keepalive.maybe_ping() == ": keepalive\n\n"

    def test_ping_resets_timer(self, monkeypatch):
        clock = {"now": 1000.0}
        monkeypatch.setattr("fusion_mlx.api.utils.time.monotonic", lambda: clock["now"])
        keepalive = SSEKeepalive(interval_seconds=5.0)
        keepalive.reset()
        clock["now"] += 6.0
        assert keepalive.maybe_ping() is not None
        assert keepalive.maybe_ping() is None

    def test_disabled_interval_never_pings(self):
        keepalive = SSEKeepalive(interval_seconds=0)
        assert keepalive.maybe_ping() is None


async def _agen(items):
    for item in items:
        yield item


class TestStreamWithKeepalive:
    async def test_passes_items_through(self):
        out = []
        async for item in stream_with_keepalive(
            _agen(["a", "b"]), interval_seconds=5.0
        ):
            out.append(item)
        assert out == ["a", "b"]

    async def test_empty_stream_yields_nothing(self):
        out = []
        async for item in stream_with_keepalive(_agen([]), interval_seconds=0.01):
            out.append(item)
        assert out == []

    async def test_ping_injected_during_long_gap(self):
        async def slow():
            await asyncio.sleep(0.12)
            yield "late"

        out = []
        async for item in stream_with_keepalive(slow(), interval_seconds=0.02):
            out.append(item)
        assert out.count(KEEPALIVE_PING) >= 2
        assert out[-1] == "late"

    async def test_cancellation_cancels_inner_generator(self):
        cancelled = {"value": False}

        async def stalled():
            try:
                await asyncio.sleep(60)
                yield "never"
            except asyncio.CancelledError:
                cancelled["value"] = True
                raise

        gen = stream_with_keepalive(stalled(), interval_seconds=0.01)
        first = await gen.__anext__()
        assert first is KEEPALIVE_PING
        await gen.aclose()
        await asyncio.sleep(0.05)
        assert cancelled["value"], "inner generator was not cancelled"
