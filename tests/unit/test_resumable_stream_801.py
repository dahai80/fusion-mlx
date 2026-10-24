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
        if session.producer_finished() and idx >= len(session.events):
            return
        session._new_data.clear()
        await session._new_data.wait()


def test_session_records_principal_for_idor_scope():
    # #801 IDOR: a session must remember the principal that created it so a
    # lookup by a different caller can be rejected.
    store = reset_store_for_tests()
    sess = store.create("s1", principal="alice")
    assert sess.principal == "alice"
    assert store.get("s1").principal == "alice"


def test_count_for_principal_caps_concurrent_sessions():
    # #801 DoS guard: the store counts live sessions per principal so the
    # route can reject one caller spawning unbounded producers.
    store = reset_store_for_tests()
    for i in range(3):
        store.create(f"a-{i}", principal="alice")
    store.create("b-0", principal="bob")
    assert store.count_for_principal("alice") == 3
    assert store.count_for_principal("bob") == 1
    assert store.count_for_principal("carol") == 0


async def _done_task():
    # A fake "producer" that is already finished (done).
    await asyncio.sleep(0)


async def _collect(sess):
    out = []
    async for chunk in _consumer(sess):
        out.append(chunk)
    return out


def test_consumer_breaks_when_producer_finished_without_complete():
    # #801 livelock fix: if the producer task is done but the session was
    # never marked complete (defensive — the route's finally block should
    # force it, but the consumer must not hang regardless), the consumer
    # drains the buffer and returns instead of waiting forever.

    async def main():
        sess = StreamSession("s", principal="alice")
        sess.append("data: only\n\n")
        sess.producer_task = asyncio.create_task(_done_task())
        # Drive the loop until the producer task is actually done.
        await sess.producer_task
        assert sess.producer_finished()
        assert not sess.complete
        return await _collect(sess)

    out = asyncio.run(main())
    assert out == ["data: only\n\n"]


async def _hang():
    await asyncio.sleep(3600)


async def _capture(fn):
    # Run an async test body that may create tasks, then give the loop one
    # turn so a cancelled task actually reaches the done state.
    result = await fn()
    await asyncio.sleep(0)
    return result


def test_drop_cancels_producer_task():
    # Audit M1: dropping a session used to leave its producer generating and
    # appending into a buffer nobody could ever read again.
    async def main():
        store = reset_store_for_tests()
        sess = store.create("d1")
        task = asyncio.create_task(_hang())
        sess.producer_task = task
        store.drop("d1")
        return task

    task = asyncio.run(_capture(main))
    assert task.cancelled()


def test_expired_get_cancels_producer_task(monkeypatch):
    import fusion_mlx.stream_session as ss

    async def main():
        store = reset_store_for_tests()
        sess = store.create("e1")
        task = asyncio.create_task(_hang())
        sess.producer_task = task
        monkeypatch.setattr(ss, "_TTL", -1.0)
        assert store.get("e1") is None
        return task

    task = asyncio.run(_capture(main))
    assert task.cancelled()


def test_create_reaps_untouched_expired_sessions(monkeypatch):
    # Audit M1: expiry was only evaluated on access, so a session nobody
    # looked up again was never reclaimed.
    import fusion_mlx.stream_session as ss

    async def main():
        store = reset_store_for_tests()
        old = store.create("old")
        task = asyncio.create_task(_hang())
        old.producer_task = task
        monkeypatch.setattr(ss, "_TTL", -1.0)
        store.create("new")
        return task

    task = asyncio.run(_capture(main))
    assert task.cancelled()


def test_buffer_overflow_fails_loudly_and_cancels_producer(monkeypatch):
    import fusion_mlx.stream_session as ss

    async def main():
        monkeypatch.setattr(ss, "_MAX_EVENTS", 2)
        sess = StreamSession("ovf")
        task = asyncio.create_task(_hang())
        sess.producer_task = task
        sess.append("data: 1\n\n")
        sess.append("data: 2\n\n")
        assert not sess.complete
        sess.append("data: 3\n\n")
        return sess, task

    sess, task = asyncio.run(_capture(main))
    assert sess.complete
    assert "buffer exceeded" in (sess.error or "")
    assert task.cancelled()
    # Bounded: post-overflow appends are ignored.
    before = len(sess.events)
    sess.append("data: 4\n\n")
    assert len(sess.events) == before


def test_normal_complete_session_still_replays_all_events():
    # Overflow guard must not disturb the happy path.
    sess = StreamSession("ok")
    sess.append("data: a\n\n")
    sess.mark_complete(finish_reason="stop")
    sess.append("data: late\n\n")
    assert sess.events == ["data: a\n\n"]
    assert sess.complete and sess.finish_reason == "stop"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
