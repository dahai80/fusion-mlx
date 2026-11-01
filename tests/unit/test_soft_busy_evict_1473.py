# SPDX-License-Identifier: Apache-2.0
"""#1473 item (2): memory_enforcer hard-eviction tier under sustained
soft pressure.

Soft pressure with no idle evictable model used to loop forever on the
waiting-warning while the process kept growing (72.8GB RSS, 5 host
watchdog panics in 12h). After _SOFT_BUSY_EVICT_POLLS consecutive polls
the enforcer must escalate to busy-victim eviction — abort in-flight
requests, mark pending unload — instead of waiting for a model to go
idle.
"""

from __future__ import annotations

import asyncio
import inspect
import textwrap
import threading

import pytest


def _import_memory_enforcer():
    from fusion_mlx.pool import memory_enforcer as me

    return me


class _FakeEngine:
    def __init__(self):
        self.aborted = 0

    async def abort_all_requests(self):
        self.aborted += 1
        return 3

    def has_active_requests(self):
        return True


class _FakeEntry:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.engine = _FakeEngine()
        self.is_pinned = False
        self.is_loading = False
        self.in_use = 1
        self.last_access = 1.0
        self.abort_loading = False
        self.ane_pinned = False
        self.is_unloading = False


class _FakePool:
    def __init__(self):
        self._lock = threading.Lock()
        self._entries = {"busy-a": _FakeEntry("busy-a"), "busy-b": _FakeEntry("busy-b")}
        self.calls = []

    def _find_lru_victim(self):
        return None

    def _find_pending_unload_ready_locked(self):
        return None

    def _find_lru_busy_non_pinned_victim_locked(self):
        # Delegate to the real implementation on the fake entries.
        candidates = [
            (getattr(e, "last_access", 0.0), mid)
            for mid, e in self._entries.items()
            if e.engine is not None and not e.is_pinned and not e.is_loading
        ]
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    def _entry_has_active_requests(self, entry):
        return True

    def _mark_pending_unload_locked(self, mid, reason, abort_requested=False):
        self.calls.append(("mark_pending", mid, reason, abort_requested))

    async def _unload_pending_if_idle_locked(self, mid):
        self.calls.append(("unload_pending", mid))


def _run_check(enforcer):
    # One enforcement poll. The real pool lock is an asyncio.Lock bound
    # to the running loop, so create a fresh one per poll (a lock reused
    # across asyncio.run loops is bound to a dead loop).
    async def _go():
        enforcer._engine_pool._lock = asyncio.Lock()
        await enforcer._check_and_enforce()

    asyncio.run(_go())


def _make_enforcer(me, pool):
    enforcer = me.ProcessMemoryEnforcer.__new__(me.ProcessMemoryEnforcer)
    enforcer._engine_pool = pool
    enforcer._soft_threshold = 0.85
    enforcer._hard_threshold = 0.95
    enforcer._pressure_level = "ok"
    enforcer._over_ceiling_polls = 0
    enforcer._unrecoverable_polls = 0
    enforcer._no_models_loaded_cycles = 0
    enforcer._soft_busy_cycles = 0
    enforcer._hard_busy_victim = None
    enforcer._hard_busy_cycles = 0
    enforcer._eviction_marked = set()
    enforcer._deferred_victim_pick = False
    enforcer._state_lock = threading.Lock()
    enforcer._last_evicted_at = {}
    enforcer._drain_flipped = False
    enforcer._propagate_memory_limit = lambda: None
    enforcer._walk_store_cache_caps = lambda: None
    enforcer._shrink_hot_cache_for_pressure = lambda current, target: 0
    enforcer.get_media_reservation_bytes = lambda: 0
    # 100GB ceiling, usage 90GB: soft (85-95GB), never hard/emergency.
    enforcer._get_hard_limit_bytes = lambda: 100 * 1024**3
    enforcer._current_usage_bytes = lambda: 90 * 1024**3
    return enforcer


@pytest.fixture()
def soft_busy_env(monkeypatch):
    me = _import_memory_enforcer()
    monkeypatch.setattr(me, "_SOFT_BUSY_EVICT_POLLS", 2)
    return me


def test_soft_busy_escalation_evicts_busy_victim(soft_busy_env):
    # Acceptance (#1473 item 2): after the configured poll count, the
    # enforcer evicts a non-pinned model WITH active requests instead of
    # looping the waiting-warning.
    me = soft_busy_env
    pool = _FakePool()
    enforcer = _make_enforcer(me, pool)

    # Poll 1: patience — warning only, no eviction.
    _run_check(enforcer)
    assert enforcer._soft_busy_cycles == 1
    assert pool.calls == []
    assert pool._entries["busy-a"].engine.aborted == 0

    # Poll 2: escalation — abort + pending unload of the LRU busy victim.
    _run_check(enforcer)
    marks = [c for c in pool.calls if c[0] == "mark_pending"]
    unloads = [c for c in pool.calls if c[0] == "unload_pending"]
    assert len(marks) == 1, f"expected one busy-victim eviction, got {pool.calls}"
    assert marks[0][1] == "busy-a"
    assert "#1473" in marks[0][2]
    assert marks[0][3] is True
    assert len(unloads) == 1 and unloads[0][1] == "busy-a"
    assert pool._entries["busy-a"].engine.aborted == 1
    # Counter reset after escalation.
    assert enforcer._soft_busy_cycles == 0


def test_soft_busy_disabled_keeps_legacy_warn_wait(monkeypatch):
    # _SOFT_BUSY_EVICT_POLLS=0: legacy behavior — warn and wait forever.
    me = _import_memory_enforcer()
    monkeypatch.setattr(me, "_SOFT_BUSY_EVICT_POLLS", 0)
    pool = _FakePool()
    enforcer = _make_enforcer(me, pool)
    for _ in range(5):
        _run_check(enforcer)
    assert pool.calls == []
    assert pool._entries["busy-a"].engine.aborted == 0


def test_soft_busy_counter_resets_when_pressure_clears(soft_busy_env):
    # One poll of soft pressure, then usage drops to ok: counter resets,
    # so a later soft episode starts patience from scratch.
    me = soft_busy_env
    pool = _FakePool()
    enforcer = _make_enforcer(me, pool)
    _run_check(enforcer)
    assert enforcer._soft_busy_cycles == 1
    enforcer._current_usage_bytes = lambda: 10 * 1024**3
    _run_check(enforcer)
    assert enforcer._soft_busy_cycles == 0
    assert pool.calls == []


def test_soft_busy_escalation_guard_precedes_abort(soft_busy_env):
    me = soft_busy_env
    pool = _FakePool()
    enforcer = _make_enforcer(me, pool)
    # First poll never aborts even with the threshold at 1 — verify the
    # guard order in source: the escalation condition must gate the
    # abort call inside the soft branch.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(me, "_SOFT_BUSY_EVICT_POLLS", 1)
    try:
        _run_check(enforcer)
        assert pool._entries["busy-a"].engine.aborted == 1
    finally:
        monkeypatch.undo()


def test_check_and_enforce_source_orders_guard_before_abort():
    # AST guard (mirrors the #1174 suite): in the soft no-victim branch,
    # abort_all_requests may only run behind the escalation counter
    # guard — never unconditionally.
    me = _import_memory_enforcer()
    src = textwrap.dedent(
        inspect.getsource(me.ProcessMemoryEnforcer._check_and_enforce)
    )
    idx_soft = src.find('new_level != "hard"')
    assert idx_soft != -1
    idx_escalation = src.find("_soft_busy_cycles >= _SOFT_BUSY_EVICT_POLLS", idx_soft)
    idx_abort = src.find("abort_all_requests", idx_soft)
    assert idx_escalation != -1, (
        "#1473: soft no-victim branch must gate busy eviction on the "
        "sustained-poll counter"
    )
    assert idx_abort != -1, (
        "#1473: soft no-victim branch must reach abort_all_requests "
        "behind the escalation guard"
    )
    assert idx_escalation < idx_abort, (
        "#1174/#1473: abort_all_requests in the soft branch must come "
        "after the escalation guard, never before it"
    )
