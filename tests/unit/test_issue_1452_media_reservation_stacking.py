# SPDX-License-Identifier: Apache-2.0
"""#1452: concurrent media jobs must stack in admission.

Prod evidence (2026-10-30): 3 concurrent FLUX.1-schnell jobs each admitted
with the identical line

    image admission ok: projected=116.93GB ceiling=120.00GB
    (current=59.87GB required=57.06GB evicted=0)

— each admission was blind to the others' 57.1GB reservations (~171GB
admitted against 120GB), and all three died in Metal GPU-timeout. The
reservation was also a last-writer-wins scalar: job 2's register clobbered
job 1's, and job 1's unregister zeroed it while job 2 still ran.
"""

import threading

import pytest

from fusion_mlx.pool.engine_pool import EnginePool
from fusion_mlx.pool.memory_enforcer import ProcessMemoryEnforcer

_GB = 1024**3


@pytest.fixture()
def enforcer():
    enf = ProcessMemoryEnforcer.__new__(ProcessMemoryEnforcer)
    enf._state_lock = threading.Lock()
    enf._media_reservations = {}
    return enf


class TestKeyedReservationRegistry:
    def test_concurrent_registers_stack(self, enforcer):
        k1 = enforcer.register_media_reservation(8 * _GB)
        k2 = enforcer.register_media_reservation(4 * _GB)
        assert k1 != k2
        assert enforcer.get_media_reservation_bytes() == 12 * _GB

    def test_unregister_by_key_keeps_others(self, enforcer):
        k1 = enforcer.register_media_reservation(8 * _GB)
        k2 = enforcer.register_media_reservation(4 * _GB)
        enforcer.unregister_media_reservation(k1)
        # Job 1's release must not drop job 2's reservation (the scalar bug).
        assert enforcer.get_media_reservation_bytes() == 4 * _GB
        enforcer.unregister_media_reservation(k2)
        assert enforcer.get_media_reservation_bytes() == 0

    def test_unregister_unknown_key_is_noop(self, enforcer):
        enforcer.register_media_reservation(8 * _GB)
        enforcer.unregister_media_reservation("no-such-key")
        assert enforcer.get_media_reservation_bytes() == 8 * _GB

    def test_keyless_unregister_clears_all_legacy(self, enforcer):
        enforcer.register_media_reservation(8 * _GB)
        enforcer.register_media_reservation(4 * _GB)
        enforcer.unregister_media_reservation()
        assert enforcer.get_media_reservation_bytes() == 0

    def test_explicit_key_roundtrip(self, enforcer):
        enforcer.register_media_reservation(8 * _GB, key="job-a")
        assert enforcer.get_media_reservation_bytes() == 8 * _GB
        enforcer.unregister_media_reservation("job-a")
        assert enforcer.get_media_reservation_bytes() == 0


class _ReservationEnforcer:
    def __init__(self, reserved):
        self._reserved = reserved

    def get_media_reservation_bytes(self):
        return self._reserved


class _FakePool:
    """Minimal stand-in exposing only what admit_media_job calls."""

    def __init__(self, ceiling, usage, reserved=0, victim=None):
        self._ceiling = ceiling
        self._usage = usage
        self._victim = victim
        self._process_memory_enforcer = _ReservationEnforcer(reserved)
        self.unloaded = []
        self.evictions = []
        self._unloaded_victims = set()

    def _current_ceiling(self):
        return self._ceiling

    def _admission_current_usage(self, *, exclude_entry_key=None):
        return self._usage

    async def _claim_lru_victim(self):
        if self._unloaded_victims:
            return None
        return self._victim

    def _record_eviction(self, reason):
        self.evictions.append(reason)

    def _purge_pure_caches(self):
        return 0

    def _admission_breakdown(self):
        return ""

    async def unload_engine_async(self, mid, **kwargs):
        self.unloaded.append(mid)
        self._unloaded_victims.add(mid)


class TestAdmissionStacksReservations:
    @pytest.mark.asyncio
    async def test_second_concurrent_job_rejected(self):
        # The prod scenario: 1st FLUX job admitted+registered 57GB; the 2nd
        # must NOT also pass at the same 59.87GB current.
        pool = _FakePool(
            ceiling=120 * _GB, usage=59.87 * _GB, reserved=57.06 * _GB, victim=None
        )
        ok = await EnginePool.admit_media_job(pool, required_bytes=57.06 * _GB)
        assert ok is False
        assert pool.unloaded == []

    @pytest.mark.asyncio
    async def test_second_job_fits_when_headroom_exists(self):
        pool = _FakePool(
            ceiling=120 * _GB, usage=2 * _GB, reserved=57.06 * _GB, victim=None
        )
        ok = await EnginePool.admit_media_job(pool, required_bytes=57.06 * _GB)
        assert ok is True

    @pytest.mark.asyncio
    async def test_reservation_release_frees_slot_for_next_job(self):
        pool = _FakePool(
            ceiling=120 * _GB, usage=59.87 * _GB, reserved=57.06 * _GB, victim=None
        )
        assert await EnginePool.admit_media_job(pool, 57.06 * _GB) is False
        pool._process_memory_enforcer._reserved = 0
        assert await EnginePool.admit_media_job(pool, 57.06 * _GB) is True

    @pytest.mark.asyncio
    async def test_ane_admission_ignores_media_reservations(self):
        # Media jobs draw from the GPU process ceiling, not the separate ANE
        # budget — an image job's reservation must not block an ANE load.
        class _AneEnforcer(_ReservationEnforcer):
            def __init__(self, reserved, ane_resident, ane_budget):
                super().__init__(reserved)
                self._ane_resident = ane_resident
                self._ane_budget = ane_budget

            def get_ane_resident_bytes(self):
                return self._ane_resident

            def get_ane_memory_budget(self):
                return self._ane_budget

        pool = _FakePool(ceiling=64 * _GB, usage=0, reserved=50 * _GB, victim=None)
        pool._process_memory_enforcer = _AneEnforcer(50 * _GB, 2 * _GB, 16 * _GB)
        ok = await EnginePool.admit_media_job(pool, 2 * _GB, kind="ane")
        # ane current = usage(0) + ane_resident(2) = 2; required 2 → 4 <= 16
        assert ok is True


class TestHeavyGateStacksReservations:
    def test_admit_media_stacks_outstanding(self, monkeypatch):
        from fusion_mlx import admission as admission_mod

        gate = admission_mod.HeavyAdmissionGate()

        class _State:
            engine_pool = type(
                "P",
                (),
                {"process_memory_enforcer": _ReservationEnforcer(20 * _GB)},
            )()

        monkeypatch.setattr(admission_mod, "reset_heavy_admission_gate", lambda: None)
        monkeypatch.setattr(
            "fusion_mlx.server._server_state",
            _state_for_gate(_ReservationEnforcer(20 * _GB)),
        )
        # physical 128GB, line 0.85 → 108.8GB. usage 70GB + budget 10GB +
        # outstanding 20GB = 100GB <= 108.8 → admitted.
        monkeypatch.setattr(gate, "physical_bytes", lambda: 128 * _GB)
        monkeypatch.setattr(gate, "current_usage_bytes", lambda: 70 * _GB)
        decision = gate.admit_media("image", 10 * _GB)
        assert decision.admitted is True

        # usage 70 + budget 20 + outstanding 20 = 110 > 108.8 → rejected.
        # Pre-#1452 the outstanding term was missing and this passed.
        decision = gate.admit_media("image", 20 * _GB)
        assert decision.admitted is False
        assert "rejected" in decision.reason

    def test_admit_media_no_state_admits_without_stacking(self, monkeypatch):
        import fusion_mlx.server as server_mod

        class _BrokenState:
            @property
            def engine_pool(self):
                raise RuntimeError("no pool")

        monkeypatch.setattr(server_mod, "_server_state", _BrokenState())
        from fusion_mlx import admission as admission_mod

        gate = admission_mod.HeavyAdmissionGate()
        monkeypatch.setattr(gate, "physical_bytes", lambda: 128 * _GB)
        monkeypatch.setattr(gate, "current_usage_bytes", lambda: 10 * _GB)
        decision = gate.admit_media("image", 10 * _GB)
        assert decision.admitted is True


def _state_for_gate(enforcer):
    class _State:
        engine_pool = type("P", (), {"process_memory_enforcer": enforcer})()

    return _State()
