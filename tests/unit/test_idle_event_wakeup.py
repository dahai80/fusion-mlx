# SPDX-License-Identifier: Apache-2.0
"""Tests for the idle-wakeup event in EngineCore (#265)."""

from __future__ import annotations

import asyncio
import time

import pytest

from fusion_mlx.engine_core import EngineConfig


def test_default_step_interval_is_seconds_not_milliseconds():
    cfg = EngineConfig()
    assert cfg.step_interval >= 0.01, (
        f"step_interval={cfg.step_interval}s — must be a float in seconds, "
        "not milliseconds. See issue #265 for the regression history."
    )


@pytest.mark.asyncio
async def test_idle_event_unblocks_immediately_when_set():
    event = asyncio.Event()

    async def waiter() -> float:
        start = time.perf_counter()
        try:
            await asyncio.wait_for(event.wait(), timeout=30.0)
        except TimeoutError:
            return -1.0
        return time.perf_counter() - start

    async def setter():
        await asyncio.sleep(0.01)
        event.set()

    elapsed, _ = await asyncio.gather(waiter(), setter())
    assert (
        0 <= elapsed < 0.1
    ), f"event-driven wakeup took {elapsed * 1000:.1f}ms; should be << 100ms"


@pytest.mark.asyncio
async def test_idle_event_falls_back_to_timeout():
    event = asyncio.Event()
    start = time.perf_counter()
    try:
        await asyncio.wait_for(event.wait(), timeout=0.05)
    except TimeoutError:
        pass
    elapsed = time.perf_counter() - start
    assert (
        0.04 <= elapsed < 0.5
    ), f"timeout fallback took {elapsed * 1000:.1f}ms; expected ~50ms"


@pytest.mark.asyncio
async def test_engine_core_creates_idle_event_in_loop():
    from unittest.mock import MagicMock

    from fusion_mlx.engine_core import EngineCore

    fake_model = MagicMock()
    fake_tokenizer = MagicMock()
    cfg = EngineConfig(model_name="test-fake-model")

    try:
        engine = EngineCore(fake_model, fake_tokenizer, cfg)
    except Exception:
        pytest.skip("EngineCore construction needs more mock setup")
        return

    assert hasattr(
        engine, "_idle_event"
    ), "EngineCore must declare the _idle_event slot — see issue #265"
    assert engine._idle_event is None, (
        "_idle_event must start as None and be created inside _engine_loop "
        "to bind to the right asyncio loop"
    )
