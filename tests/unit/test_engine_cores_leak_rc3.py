from __future__ import annotations

import asyncio

import pytest

from fusion_mlx.pool.engine_pool import EngineEntry, EnginePool


class _FakeEngine:
    """Minimal duck-typed engine for the detach path."""

    def __init__(self):
        self.stopped = False

    async def stop(self):
        self.stopped = True


def _make_entry():
    # EngineEntry requires model_id/model_path/model_type/engine_type/
    # estimated_size positionally; only engine matters for the detach path.
    return EngineEntry(
        model_id="m",
        model_path="/tmp/m",
        model_type="llm",
        engine_type="batched",
        estimated_size=0,
        engine=_FakeEngine(),
    )


@pytest.mark.asyncio
async def test_detach_engine_fires_on_engine_detached_callback():
    # RC-3 (#811 audit 0906): _detach_engine MUST fire the registered callback
    # on every unload path so the Server can drop its engine_cores reference.
    pool = EnginePool()
    fired: list[str] = []
    pool._on_engine_detached = lambda mid: fired.append(mid)

    entry = _make_entry()
    pool._entries["model-x"] = entry

    await pool._detach_engine("model-x")

    assert fired == ["model-x"]
    assert entry.engine is None


@pytest.mark.asyncio
async def test_detach_engine_no_callback_stays_silent():
    # No callback registered (pre-wire) — detach still works, nothing breaks.
    pool = EnginePool()
    entry = _make_entry()
    pool._entries["model-y"] = entry

    await pool._detach_engine("model-y")

    assert entry.engine is None


@pytest.mark.asyncio
async def test_detach_engine_callback_exception_swallowed():
    # A buggy callback must not abort the detach (engine still set to None).
    pool = EnginePool()

    def boom(_mid):
        raise RuntimeError("bad callback")

    pool._on_engine_detached = boom
    entry = _make_entry()
    pool._entries["model-z"] = entry

    await pool._detach_engine("model-z")

    assert entry.engine is None


@pytest.mark.asyncio
async def test_server_drop_engine_core_pops_idempotent():
    # Mirror the Server._drop_engine_core wiring: pop must be idempotent so
    # the double-touch (unload_model pops, then callback pops again) is safe.
    engine_cores: dict[str, object] = {"m1": object(), "m2": object()}

    def on_detached(mid):
        engine_cores.pop(mid, None)

    pool = EnginePool()
    pool._on_engine_detached = on_detached

    for mid in ("m1", "m1", "m2"):
        entry = _make_entry()
        pool._entries[mid] = entry
        await pool._detach_engine(mid)

    # m1 popped twice (idempotent), m2 once — both gone, no KeyError.
    assert engine_cores == {}
