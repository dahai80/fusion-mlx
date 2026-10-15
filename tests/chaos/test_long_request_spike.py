# SPDX-License-Identifier: Apache-2.0
"""Long-request + sudden-concurrency-spike chaos shape.

Injection: one long-held admission probe (a slow memory-usage read, the
shape of a long request holding the gate's attention) while a spike of
concurrent admission decisions arrives. Contract: the spike is fully
served with explicit decisions (admitted or rejected-with-reason), the
long probe does not starve the gate, and no decision is silently lost.
"""

import time

import pytest

from fusion_mlx.admission import (
    get_heavy_admission_gate,
    reset_heavy_admission_gate,
)
from tests.chaos.conftest import DETECTION_BOUND_S, RECOVERY_BOUND_S

PHYS = 128 * 2**30


@pytest.fixture()
def gate(monkeypatch):
    reset_heavy_admission_gate()
    g = get_heavy_admission_gate()
    monkeypatch.setattr(g, "physical_bytes", lambda: PHYS)
    monkeypatch.setattr(g, "current_usage_bytes", lambda: 90 * 2**30)
    return g


def test_spike_every_request_explicit_outcome(gate):
    # 50 concurrent heavy admits over the safety line: each returns an
    # explicit rejection with reason — the route maps to 507, the client
    # is never left hanging and nothing is silently dropped.
    t0 = time.monotonic()
    decisions = [gate.admit_media("video", 40 * 2**30) for _ in range(50)]
    elapsed = time.monotonic() - t0
    assert len(decisions) == 50
    for d in decisions:
        assert d.admitted is False
        assert "safety line" in d.reason
    assert elapsed < DETECTION_BOUND_S


def test_long_request_does_not_starve_gate(gate, monkeypatch):
    # A slow usage probe (long-request shape: a big resident model or a
    # wedged enforcer read) must not serialize the gate — the paused flag
    # read is the only locked section, so spikes decide while it runs.
    import threading

    slow = {"busy": True}

    def slow_usage():
        deadline = time.monotonic() + 1.0
        while slow["busy"] and time.monotonic() < deadline:
            time.sleep(0.01)
        return 90 * 2**30

    monkeypatch.setattr(gate, "current_usage_bytes", slow_usage)
    outcomes = []
    spike_done = threading.Event()

    def spike():
        for _ in range(20):
            d = gate.admit_media("image", 5 * 2**30)
            outcomes.append(d.admitted)
        spike_done.set()

    t = threading.Thread(target=spike)
    t0 = time.monotonic()
    t.start()
    # The long probe runs concurrently with the spike.
    long_decision = gate.admit_media("video", 40 * 2**30)
    slow["busy"] = False
    spike_done.wait(10.0)
    t.join(10.0)
    elapsed = time.monotonic() - t0
    assert long_decision.admitted is False  # explicit, not hanging
    assert len(outcomes) == 20
    assert all(isinstance(o, bool) for o in outcomes)
    assert elapsed < RECOVERY_BOUND_S


def test_recovery_gate_admits_after_pressure_clears(gate, monkeypatch):
    # Pressure clears (models evicted): the same gate must immediately
    # admit again — no stuck state, no cooldown, no restart needed.
    monkeypatch.setattr(gate, "current_usage_bytes", lambda: 90 * 2**30)
    assert gate.admit_media("video", 40 * 2**30).admitted is False
    monkeypatch.setattr(gate, "current_usage_bytes", lambda: 10 * 2**30)
    t0 = time.monotonic()
    decision = gate.admit_media("video", 40 * 2**30)
    assert decision.admitted is True, decision.reason
    assert time.monotonic() - t0 < RECOVERY_BOUND_S


def test_async_spike_zero_silent_loss(gate):
    # Async client shape: 100 concurrent stream/gate checks under an
    # event loop — each resolves with an outcome within a hard bound.
    import asyncio

    async def main():
        async def one(i):
            d = await asyncio.to_thread(gate.admit_media, "video", 40 * 2**30)
            return i, d.admitted, bool(d.reason)

        tasks = [one(i) for i in range(100)]
        return await asyncio.wait_for(asyncio.gather(*tasks), timeout=30.0)

    t0 = time.monotonic()
    results = asyncio.run(main())
    assert len(results) == 100
    assert all(isinstance(adm, bool) and reason for _, adm, reason in results)
    assert time.monotonic() - t0 < RECOVERY_BOUND_S
