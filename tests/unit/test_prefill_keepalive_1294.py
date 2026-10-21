import asyncio

import pytest

from fusion_mlx.api.utils import KEEPALIVE_PING, stream_with_keepalive


async def _agen(items, delays=None):
    for i, item in enumerate(items):
        if delays and delays[i]:
            await asyncio.sleep(delays[i])
        yield item


async def _collect(agen, max_items=50):
    out = []
    async for x in agen:
        out.append(x)
        if len(out) >= max_items:
            break
    return out


async def test_no_ping_when_chunks_arrive_fast():
    got = await _collect(stream_with_keepalive(_agen(["a", "b", "c"]), 5.0))
    assert got == ["a", "b", "c"]
    assert KEEPALIVE_PING not in got


async def test_ping_injected_during_long_gap():
    got = await _collect(
        stream_with_keepalive(_agen(["first", "second"], delays=[0.0, 0.5]), 0.1)
    )
    pings = [x for x in got if x is KEEPALIVE_PING]
    assert got[-1] == "second"
    assert len(pings) >= 2
    assert got[0] == "first"


async def test_empty_stream_yields_no_ping_forever_short_run():
    got = await _collect(stream_with_keepalive(_agen([]), 0.05), max_items=3)
    assert got == []


async def test_client_cancel_cancels_pending_anext():
    started = asyncio.Event()

    async def _slow():
        started.set()
        await asyncio.sleep(30)
        yield "late"

    agen = stream_with_keepalive(_slow(), 0.05)
    task = asyncio.ensure_future(_collect(agen, max_items=10))
    await started.wait()
    await asyncio.sleep(0.15)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
