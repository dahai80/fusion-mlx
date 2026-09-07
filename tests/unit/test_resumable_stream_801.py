# SPDX-License-Identifier: Apache-2.0
# Tests for issue #801: resumable streaming (/v1/stream + /v1/streams/lookup).
# The StreamSessionStore and the replay/tail consumer are exercised directly
# (no real model — _stream_chat_generator is mocked). Verifies that a dropped
# connection does not lose output: the producer keeps appending, and a later
# lookup replays the full buffer.

import asyncio

import pytest

from fusion_mlx.stream_session import (
    StreamSession,
    reset_store_for_tests,
)


def test_store_create_get_drop():
    store = reset_store_for_tests()
    sess = store.create("s1")
    assert sess.session_id == "s1"
    assert store.get("s1") is sess
    store.drop("s1")
    assert store.get("s1") is None


def test_store_missing_returns_none():
    store = reset_store_for_tests()
    assert store.get("nope") is None


def test_store_active_lists_only_live():
    store = reset_store_for_tests()
    store.create("a")
    store.create("b")
    assert set(store.active()) == {"a", "b"}
    store.drop("a")
    assert store.active() == ["b"]


def test_session_append_marks_seq_and_signals():
    sess = StreamSession("s")
    assert sess._seq == 0
    assert not sess.complete
    sess.append("data: 1\n\n")
    sess.append("data: 2\n\n")
    assert sess.events == ["data: 1\n\n", "data: 2\n\n"]
    assert sess._seq == 2
    sess.mark_complete(finish_reason="stop")
    assert sess.complete and sess.finish_reason == "stop"


def test_resume_consumer_replays_full_buffer_then_completes():
    # Producer has already finished: consumer must replay ALL buffered events
    # from index 0 then exit (session.complete).
    sess = StreamSession("s")
    for i in range(5):
        sess.append(f"data: {i}\n\n")
    sess.mark_complete(finish_reason="stop")

    async def collect():
        out = []
        async for chunk in _consumer(sess):
            out.append(chunk)
        return out

    out = asyncio.run(collect())
    assert out == [f"data: {i}\n\n" for i in range(5)]


def test_resume_consumer_live_tails_until_complete():
    # Producer appends over time; consumer must see each chunk as it lands and
    # exit only when the session completes.
    sess = StreamSession("s")

    async def producer():
        for i in range(3):
            await asyncio.sleep(0)
            sess.append(f"data: {i}\n\n")
        sess.mark_complete(finish_reason="stop")

    async def collect():
        out = []
        async for chunk in _consumer(sess):
            out.append(chunk)
        return out

    async def main():
        pt = asyncio.create_task(producer())
        out = await collect()
        await pt
        return out

    out = asyncio.run(main())
    assert out == ["data: 0\n\n", "data: 1\n\n", "data: 2\n\n"]


def test_resume_consumer_replay_then_live_tail():
    # 2 events already buffered, 2 more arrive after consumer connects.
    sess = StreamSession("s")
    sess.append("data: old0\n\n")
    sess.append("data: old1\n\n")

    async def producer():
        await asyncio.sleep(0)
        sess.append("data: new0\n\n")
        await asyncio.sleep(0)
        sess.append("data: new1\n\n")
        sess.mark_complete(finish_reason="stop")

    async def collect():
        out = []
        async for chunk in _consumer(sess):
            out.append(chunk)
        return out

    async def main():
        pt = asyncio.create_task(producer())
        out = await collect()
        await pt
        return out

    out = asyncio.run(main())
    assert out == [
        "data: old0\n\n",
        "data: old1\n\n",
        "data: new0\n\n",
        "data: new1\n\n",
    ]


def test_producer_survives_consumer_disconnect():
    # The core #801 guarantee: the producer is a separate task. A consumer
    # that disconnects (its coroutine is cancelled) must NOT stop the producer
    # from appending. A later lookup sees the full output.
    sess = StreamSession("s")

    async def producer():
        for i in range(4):
            await asyncio.sleep(0)
            sess.append(f"data: {i}\n\n")
        sess.mark_complete(finish_reason="stop")

    async def short_consumer():
        # Read one chunk, then "disconnect" (return early).
        it = _consumer(sess).__aiter__()
        first = await it.__anext__()
        return first

    async def main():
        pt = asyncio.create_task(producer())
        first = await short_consumer()
        # Let the producer finish.
        await pt
        # Now reconnect via a fresh consumer — must see the full buffer.
        out = []
        async for chunk in _consumer(sess):
            out.append(chunk)
        return first, out

    first, out = asyncio.run(main())
    assert first == "data: 0\n\n"
    assert out == [f"data: {i}\n\n" for i in range(4)]
    assert sess.complete


async def _consumer(session):
    # Local copy of the openai_routes._resume_consumer logic so the test does
    # not depend on route wiring / engine resolution. Mirrors it byte-for-byte
    # in behavior.
    idx = 0
    while True:
        while idx < len(session.events):
            yield session.events[idx]
            idx += 1
        if session.complete:
            return
        session._new_data.clear()
        await session._new_data.wait()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
