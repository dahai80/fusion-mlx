# SPDX-License-Identifier: Apache-2.0
"""#1722: soft-tier busy-victim escalation must not abort+unload the ONLY
loaded non-pinned model.

Observed 2026-11-26 (131K RULER run, single serialized client): soft
pressure (usage 121.7 GB, target ~108 GB, ceiling 127.5 GB) escalated
every poll to busy-victim eviction of the sole serving engine — aborting
the 131K prefill mid-flight. The client retry re-created the identical
transient, giving 10+ consecutive aborts, and the post-abort unload
turned the loop into a 503 admission churn until restart. Hard pressure
still reaches the hard-tier busy-victim path, so a genuinely over-ceiling
process is still shed.
"""

from __future__ import annotations

import logging

import pytest

from tests.unit.test_soft_busy_evict_1473 import (
    _FakeEntry,
    _FakePool,
    _import_memory_enforcer,
    _make_enforcer,
    _run_check,
)


class _SingleModelPool(_FakePool):
    def __init__(self):
        super().__init__()
        self._entries = {"solo": _FakeEntry("solo")}


@pytest.fixture()
def soft_busy_env(monkeypatch):
    me = _import_memory_enforcer()
    monkeypatch.setattr(me, "_SOFT_BUSY_EVICT_POLLS", 2)
    return me


class TestSoftSingleModelHold:
    def test_no_abort_no_unload_on_sole_model(self, soft_busy_env, caplog):
        me = soft_busy_env
        pool = _SingleModelPool()
        enforcer = _make_enforcer(me, pool)
        for _ in range(5):
            _run_check(enforcer)
        engine = pool._entries["solo"].engine
        assert engine.aborted == 0
        assert pool.calls == []
        assert not pool._entries["solo"].abort_loading

    def test_hold_warning_carries_governing_numbers(self, soft_busy_env, caplog):
        me = soft_busy_env
        pool = _SingleModelPool()
        enforcer = _make_enforcer(me, pool)
        with caplog.at_level(logging.WARNING, logger="fusion_mlx.pool.memory_enforcer"):
            # 3 polls: escalation arms on poll 2, hold log fires on poll 3.
            _run_check(enforcer)
            _run_check(enforcer)
            _run_check(enforcer)
        texts = [r.getMessage() for r in caplog.records]
        holds = [t for t in texts if "only loaded non-pinned" in t]
        assert holds, texts
        assert "usage 90.0 GB" in holds[0]
        assert "ceiling 100.0 GB" in holds[0]

    def test_hold_warning_throttled_after_first(self, soft_busy_env):
        me = soft_busy_env
        pool = _SingleModelPool()
        enforcer = _make_enforcer(me, pool)

        from unittest.mock import MagicMock

        import fusion_mlx.pool.memory_enforcer as me_mod

        fake = MagicMock()
        orig = me_mod.logger
        me_mod.logger = fake
        try:
            for _ in range(5):
                _run_check(enforcer)
        finally:
            me_mod.logger = orig
        holds = [
            str(c.args[0])
            for c in fake.warning.call_args_list
            if "only loaded non-pinned" in str(c.args[0])
        ]
        assert len(holds) == 1

    def test_two_busy_models_still_escalate(self, soft_busy_env):
        me = soft_busy_env
        pool = _FakePool()
        enforcer = _make_enforcer(me, pool)
        for _ in range(3):
            _run_check(enforcer)
        engine = pool._entries["busy-a"].engine
        assert engine.aborted >= 1
        assert any(c[0] == "mark_pending" for c in pool.calls)

    def test_hold_state_reset_tracking(self, soft_busy_env):
        me = soft_busy_env
        pool = _SingleModelPool()
        enforcer = _make_enforcer(me, pool)
        # Escalation arms after _SOFT_BUSY_EVICT_POLLS=2 consecutive polls;
        # the third poll enters the hold branch (first cycle for the victim).
        _run_check(enforcer)
        _run_check(enforcer)
        _run_check(enforcer)
        assert enforcer._soft_busy_victim == "solo"
        assert enforcer._soft_busy_single_cycles == 1
        assert enforcer._soft_busy_cycles >= 1
