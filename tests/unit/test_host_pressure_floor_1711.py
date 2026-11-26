import asyncio
import logging
from unittest.mock import Mock

from fusion_mlx.pool import memory_enforcer as me
from fusion_mlx.pool.memory_enforcer import ProcessMemoryEnforcer


def _make_enforcer(**kwargs):
    pool = Mock()
    pool.loaded_model_bytes = 0
    pool.get_loaded_model_bytes = Mock(return_value=0)
    pool._entries = {}
    pool._lock = asyncio.Lock()
    return ProcessMemoryEnforcer(engine_pool=pool, **kwargs)


def _clear_env(monkeypatch):
    monkeypatch.delenv("FUSION_MLX_HOST_MIN_FREE_GB", raising=False)


class TestHostPressureFloorLevel:
    def test_disabled_by_env_zero(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("FUSION_MLX_HOST_MIN_FREE_GB", "0")
        enforcer = _make_enforcer()
        with patch_attr(enforcer, "_host_available_bytes", lambda: 0):
            assert enforcer._host_pressure_floor_level() is None

    def test_none_when_host_has_room(self, monkeypatch):
        _clear_env(monkeypatch)
        enforcer = _make_enforcer()
        # Default reserve 8 GiB; host has 16 GiB available.
        with patch_attr(enforcer, "_host_available_bytes", lambda: 16 * 1024**3):
            assert enforcer._host_pressure_floor_level() is None

    def test_soft_when_under_reserve(self, monkeypatch):
        _clear_env(monkeypatch)
        enforcer = _make_enforcer()
        with patch_attr(enforcer, "_host_available_bytes", lambda: 6 * 1024**3):
            assert enforcer._host_pressure_floor_level() == "soft"

    def test_hard_when_under_half_reserve(self, monkeypatch):
        _clear_env(monkeypatch)
        enforcer = _make_enforcer()
        with patch_attr(enforcer, "_host_available_bytes", lambda: 2 * 1024**3):
            assert enforcer._host_pressure_floor_level() == "hard"

    def test_none_when_telemetry_unavailable(self, monkeypatch):
        _clear_env(monkeypatch)
        enforcer = _make_enforcer()
        with patch_attr(enforcer, "_host_available_bytes", lambda: 0):
            assert enforcer._host_pressure_floor_level() is None

    def test_invalid_env_uses_default_and_warns_once(self, monkeypatch, caplog):
        _clear_env(monkeypatch)
        monkeypatch.setenv("FUSION_MLX_HOST_MIN_FREE_GB", "abc")
        enforcer = _make_enforcer()
        with caplog.at_level(logging.WARNING, logger="fusion_mlx.pool.memory_enforcer"):
            first = enforcer._host_min_free_bytes()
            second = enforcer._host_min_free_bytes()
        assert first == second == int(8.0 * 1024**3)
        assert caplog.text.count("FUSION_MLX_HOST_MIN_FREE_GB") == 1


class TestCheckAndEnforceEscalation:
    def _make_escalation_enforcer(self, monkeypatch):
        # #1711 escalation shape with ALL ambient inputs pinned: the ceiling
        # must come from the patch below, not settings.json — CI runners have
        # no custom ceiling configured and a small default ceiling would send
        # _check_and_enforce into the eviction loop against the Mock pool.
        _clear_env(monkeypatch)
        monkeypatch.setenv("FUSION_MLX_HOST_MIN_FREE_GB", "8")
        enforcer = _make_enforcer()
        gb = 1024**3
        enforcer._get_hard_limit_bytes = Mock(return_value=120 * gb)
        enforcer._current_usage_bytes = Mock(return_value=40 * gb)
        return enforcer

    def test_footprint_ok_but_host_low_escalates_to_soft(self, monkeypatch):
        # #1711 production shape: process flat under its ceiling (current
        # < soft), host avail 6.6% -> the enforcer must still shed load.
        enforcer = self._make_escalation_enforcer(monkeypatch)
        gb = 1024**3
        with (
            patch_attr(enforcer, "_host_available_bytes", lambda: 6 * gb),
            patch_attr(me, "get_phys_footprint", lambda: 40 * gb),
        ):
            asyncio.run(enforcer._check_and_enforce())
        assert enforcer._pressure_level == "soft"

    def test_host_room_keeps_level_ok(self, monkeypatch):
        enforcer = self._make_escalation_enforcer(monkeypatch)
        gb = 1024**3
        with (
            patch_attr(enforcer, "_host_available_bytes", lambda: 60 * gb),
            patch_attr(me, "get_phys_footprint", lambda: 40 * gb),
        ):
            asyncio.run(enforcer._check_and_enforce())
        assert enforcer._pressure_level == "ok"


class _patch_attr:
    def __init__(self, obj, name, value):
        self._obj = obj
        self._name = name
        self._value = value
        self._old = None

    def __enter__(self):
        self._old = getattr(self._obj, self._name)
        setattr(self._obj, self._name, self._value)
        return self

    def __exit__(self, *exc):
        setattr(self._obj, self._name, self._old)


def patch_attr(obj, name, value):
    return _patch_attr(obj, name, value)
