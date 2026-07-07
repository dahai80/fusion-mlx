# SPDX-License-Identifier: Apache-2.0
"""Tests for runtime LoRA adapter hot-swap (B.2 path A')."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from fusion_mlx.exceptions import ModelNotFoundError
from fusion_mlx.pool.engine_pool import EngineEntry, EnginePool


def _make_pool(ceiling: int = 0, **kwargs) -> EnginePool:
    pool = EnginePool(**kwargs)
    pool._get_final_ceiling = lambda c=int(ceiling): c
    # Avoid real mlx stop/settle paths in unit tests
    pool._unload_engine = AsyncMock()
    pool._unload_pending_if_idle_locked = AsyncMock()
    pool._wake_process_memory_enforcer = MagicMock()
    return pool


def _base_entry(model_id: str = "base", path: str = "/m") -> EngineEntry:
    return EngineEntry(
        model_id=model_id,
        model_path=path,
        model_type="llm",
        engine_type="batched",
        estimated_size=1024,
    )


def _stub_engine() -> MagicMock:
    eng = MagicMock()
    eng.start = AsyncMock()
    return eng


def _install_load(pool: EnginePool) -> None:
    # Fake _load_engine: stamp a stub engine onto the entry so get_engine's
    # fast path returns it on subsequent calls.
    async def fake_load(model_id, **kw):
        entry = pool._entries[model_id]
        entry.engine = _stub_engine()
        entry.last_access = 1.0
        pool._current_model_memory += entry.estimated_size

    pool._load_engine = AsyncMock(side_effect=fake_load)


class TestAdapterKey:
    def test_no_adapter_returns_model_id(self):
        pool = _make_pool()
        assert pool._adapter_key("m", None) == "m"
        assert pool._adapter_key("m", "") == "m"

    def test_adapter_returns_combo_key(self):
        pool = _make_pool()
        assert pool._adapter_key("m", "/a") == "m::lora::/a"


class TestMakeAdapterEntry:
    def test_derived_entry_fields(self):
        pool = _make_pool()
        base = _base_entry("base", "/m")
        entry = pool._make_adapter_entry(base, "/lora", "base::lora::/lora")
        assert entry.model_id == "base::lora::/lora"
        assert entry.model_path == "/m"
        assert entry.adapter_path == "/lora"
        assert entry.base_model_id == "base"
        assert entry.source_type == "lora_adapter"
        assert entry.estimated_size == 1024
        assert entry.engine_type == "batched"


class TestGetEngineAdapter:
    async def test_creates_derived_entry_lazy(self):
        pool = _make_pool()
        pool._entries["base"] = _base_entry()
        _install_load(pool)
        result = await pool.get_engine("base", _lease=True, adapter_path="/lora")
        assert result is not None
        assert "base::lora::/lora" in pool._entries
        derived = pool._entries["base::lora::/lora"]
        assert derived.adapter_path == "/lora"
        assert derived.base_model_id == "base"
        assert derived.in_use == 1
        await pool.release_engine("base", adapter_path="/lora")
        assert derived.in_use == 0

    async def test_no_adapter_uses_base_entry(self):
        pool = _make_pool()
        pool._entries["base"] = _base_entry()
        _install_load(pool)
        result = await pool.get_engine("base", _lease=True)
        assert result is not None
        assert "base::lora::/lora" not in pool._entries
        await pool.release_engine("base")

    async def test_missing_base_with_adapter_raises(self):
        pool = _make_pool()
        with pytest.raises(ModelNotFoundError):
            await pool.get_engine("ghost", adapter_path="/lora")

    async def test_release_without_adapter_does_not_touch_derived(self):
        pool = _make_pool()
        pool._entries["base"] = _base_entry()
        _install_load(pool)
        await pool.get_engine("base", _lease=True, adapter_path="/lora")
        derived = pool._entries["base::lora::/lora"]
        derived.in_use = 1
        # release on the base key (no adapter) must not decrement derived lease
        await pool.release_engine("base")
        assert derived.in_use == 1
        await pool.release_engine("base", adapter_path="/lora")
        assert derived.in_use == 0

    async def test_second_call_reuses_loaded_derived(self):
        pool = _make_pool()
        pool._entries["base"] = _base_entry()
        _install_load(pool)
        first = await pool.get_engine("base", _lease=True, adapter_path="/lora")
        await pool.release_engine("base", adapter_path="/lora")
        second = await pool.get_engine("base", _lease=True, adapter_path="/lora")
        assert first is second
        pool._load_engine.assert_called_once()


class TestAdapterCap:
    def test_cap_zero_disables(self):
        pool = _make_pool()
        pool._max_adapter_engines = 0
        assert pool._select_adapter_cap_victims("new") == []

    def test_cap_evicts_oldest_idle(self):
        pool = _make_pool()
        pool._max_adapter_engines = 2
        for i, k in enumerate(["a1", "a2", "a3"]):
            e = _base_entry(k, "/m")
            e.source_type = "lora_adapter"
            e.engine = _stub_engine()
            e.last_access = float(i)
            pool._entries[k] = e
        # 3 loaded + 1 new = 4, cap 2 → evict 2 oldest (a1, a2)
        victims = pool._select_adapter_cap_victims("new")
        assert victims == ["a1", "a2"]

    def test_cap_skips_in_use(self):
        pool = _make_pool()
        pool._max_adapter_engines = 1
        e = _base_entry("a1", "/m")
        e.source_type = "lora_adapter"
        e.engine = _stub_engine()
        e.in_use = 1
        e.last_access = 0.0
        pool._entries["a1"] = e
        assert pool._select_adapter_cap_victims("new") == []

    def test_cap_skips_unloaded(self):
        pool = _make_pool()
        pool._max_adapter_engines = 1
        e = _base_entry("a1", "/m")
        e.source_type = "lora_adapter"
        e.engine = None
        e.last_access = 0.0
        pool._entries["a1"] = e
        assert pool._select_adapter_cap_victims("new") == []

    async def test_cap_eviction_triggers_unload(self):
        pool = _make_pool()
        pool._max_adapter_engines = 1
        # one loaded idle adapter already present
        e = _base_entry("a1", "/m")
        e.source_type = "lora_adapter"
        e.engine = _stub_engine()
        e.last_access = 0.0
        pool._entries["a1"] = e
        pool._entries["base"] = _base_entry()
        _install_load(pool)
        await pool.get_engine("base", _lease=True, adapter_path="/lora2")
        # a1 should have been unloaded to make room for the new derived entry
        unloaded_keys = [c.args[0] for c in pool._unload_engine.await_args_list]
        assert "a1" in unloaded_keys
