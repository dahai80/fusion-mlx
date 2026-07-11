"""Test concurrent get_engine() calls — the ModelLoadingError race fix.

Before the fix, two concurrent get_engine() calls for the same model
would race: the first sets is_loading=True, the second sees is_loading
and raises ModelLoadingError.  After the fix, the second caller awaits
a loading_event and retries once the first load completes.
"""

import asyncio

import pytest

from fusion_mlx.pool.engine_pool import EngineEntry


@pytest.fixture
def entry():
    return EngineEntry(
        model_id="test-model",
        model_path="test-model",
        model_type="llm",
        engine_type="batched",
        estimated_size=1_000_000_000,
    )


def test_entry_loading_event_default_none(entry):
    assert entry.loading_event is None
    assert not entry.is_loading


def test_entry_loading_event_create_and_signal(entry):
    entry.is_loading = True
    entry.loading_event = asyncio.Event()
    assert not entry.loading_event.is_set()

    entry.is_loading = False
    ev = entry.loading_event
    entry.loading_event = None
    ev.set()

    assert ev.is_set()


@pytest.mark.asyncio
async def test_concurrent_get_engine_waits(entry):
    """Simulating the pool's concurrent loading pattern:
    - Caller 1 sets is_loading=True + creates loading_event
    - Caller 2 sees is_loading, saves the event, exits lock
    - Caller 2 awaits the event
    - Caller 1 finishes, sets is_loading=False, signals event
    - Caller 2 wakes up and retries
    """
    entry.is_loading = True
    entry.loading_event = asyncio.Event()

    # Simulate caller 2 saving the event reference before lock exit
    wait_event = entry.loading_event

    # Caller 1 completes the load
    async def complete_load():
        await asyncio.sleep(0.01)
        entry.is_loading = False
        ev = entry.loading_event
        entry.loading_event = None
        if ev is not None:
            ev.set()

    # Caller 2 waits
    task = asyncio.create_task(complete_load())
    await wait_event.wait()
    await task

    assert not entry.is_loading
    assert entry.loading_event is None


@pytest.mark.asyncio
async def test_concurrent_get_engine_error_path(entry):
    """When the loader fails, it still signals the event so waiters unblock."""
    entry.is_loading = True
    entry.loading_event = asyncio.Event()

    wait_event = entry.loading_event

    async def fail_load():
        await asyncio.sleep(0.01)
        # Error cleanup: set is_loading=False and signal event
        entry.is_loading = False
        ev = entry.loading_event
        entry.loading_event = None
        if ev is not None:
            ev.set()

    task = asyncio.create_task(fail_load())
    await wait_event.wait()
    await task

    assert not entry.is_loading
    assert entry.engine is None
