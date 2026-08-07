# SPDX-License-Identifier: Apache-2.0
"""Tests for in-place LoRA swap (issue #389)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_mlx.pool.engine_pool import EngineEntry, EnginePool


def _pool(ceiling: int | None = None, **kwargs) -> EnginePool:
    pool = EnginePool(**kwargs)
    if ceiling is None or ceiling <= 0:
        pool._get_final_ceiling = lambda: 0
    else:
        pool._get_final_ceiling = lambda c=int(ceiling): c
    return pool


def _base_entry(model_id: str = "qwen-base") -> EngineEntry:
    entry = EngineEntry(
        model_id=model_id,
        model_path=f"/models/{model_id}",
        model_type="llm",
        engine_type="batched",
        estimated_size=1000,
    )
    engine = MagicMock()
    engine._model = MagicMock(name="model")
    engine.stop = AsyncMock()
    engine.safe_evict = AsyncMock()
    engine.has_active_requests = MagicMock(return_value=False)
    entry.engine = engine
    return entry


class TestInPlaceSwapFlag:
    def test_default_off(self):
        assert _pool()._inplace_swap is False

    def test_flag_on(self, monkeypatch):
        monkeypatch.setenv("FUSION_LORA_INPLACE_SWAP", "1")
        assert _pool()._inplace_swap is True


class TestSwapHelpers:
    @pytest.mark.asyncio
    async def test_acquire_applies_and_increments_in_use(self, monkeypatch):
        monkeypatch.setenv("FUSION_LORA_INPLACE_SWAP", "1")
        pool = _pool()
        entry = _base_entry()
        pool._entries["qwen-base"] = entry

        swap = MagicMock()
        swap.apply = MagicMock(return_value=0.007)
        with patch(
            "fusion_mlx.adapter.weight_swap.InPlaceLoRASwap",
            return_value=swap,
        ):
            eng = await pool._acquire_inplace_adapter("qwen-base", "/adapters/fixA")

        assert eng is entry.engine
        swap.apply.assert_called_once()
        assert pool._active_swap["qwen-base"] is swap
        assert entry.in_use == 1

    @pytest.mark.asyncio
    async def test_acquire_restores_prior_swap_first(self, monkeypatch):
        monkeypatch.setenv("FUSION_LORA_INPLACE_SWAP", "1")
        pool = _pool()
        entry = _base_entry()
        pool._entries["qwen-base"] = entry

        prior = MagicMock()
        prior.restore = MagicMock()
        pool._active_swap["qwen-base"] = prior

        swap = MagicMock()
        with patch(
            "fusion_mlx.adapter.weight_swap.InPlaceLoRASwap",
            return_value=swap,
        ):
            await pool._acquire_inplace_adapter("qwen-base", "/adapters/fixB")

        prior.restore.assert_called_once()
        assert pool._active_swap["qwen-base"] is swap
        assert entry.in_use == 1

    @pytest.mark.asyncio
    async def test_acquire_missing_base_raises(self, monkeypatch):
        monkeypatch.setenv("FUSION_LORA_INPLACE_SWAP", "1")
        pool = _pool()
        from fusion_mlx.exceptions import ModelNotFoundError

        with pytest.raises(ModelNotFoundError):
            await pool._acquire_inplace_adapter("nope", "/adapters/fixA")

    @pytest.mark.asyncio
    async def test_acquire_engine_without_model_raises(self, monkeypatch):
        monkeypatch.setenv("FUSION_LORA_INPLACE_SWAP", "1")
        pool = _pool()
        entry = _base_entry()
        entry.engine._model = None
        pool._entries["qwen-base"] = entry

        with pytest.raises(RuntimeError):
            await pool._acquire_inplace_adapter("qwen-base", "/adapters/fixA")

    @pytest.mark.asyncio
    async def test_release_restores_and_decrements(self, monkeypatch):
        monkeypatch.setenv("FUSION_LORA_INPLACE_SWAP", "1")
        pool = _pool()
        entry = _base_entry()
        pool._entries["qwen-base"] = entry

        swap = MagicMock()
        swap.restore = MagicMock()
        pool._active_swap["qwen-base"] = swap
        entry.in_use = 1

        await pool._release_inplace_adapter("qwen-base", "/adapters/fixA")

        swap.restore.assert_called_once()
        assert "qwen-base" not in pool._active_swap
        assert entry.in_use == 0

    @pytest.mark.asyncio
    async def test_release_missing_swap_is_safe(self, monkeypatch):
        monkeypatch.setenv("FUSION_LORA_INPLACE_SWAP", "1")
        pool = _pool()
        entry = _base_entry()
        pool._entries["qwen-base"] = entry

        await pool._release_inplace_adapter("qwen-base", "/adapters/fixA")
        assert entry.in_use == 0

    @pytest.mark.asyncio
    async def test_release_restore_failure_keeps_in_use_decremented(self, monkeypatch):
        monkeypatch.setenv("FUSION_LORA_INPLACE_SWAP", "1")
        pool = _pool()
        entry = _base_entry()
        pool._entries["qwen-base"] = entry

        swap = MagicMock()
        swap.restore = MagicMock(side_effect=RuntimeError("boom"))
        pool._active_swap["qwen-base"] = swap
        entry.in_use = 1

        await pool._release_inplace_adapter("qwen-base", "/adapters/fixA")
        assert entry.in_use == 0


class TestGetEngineGating:
    @pytest.mark.asyncio
    async def test_flag_off_never_uses_swap_path(self, monkeypatch):
        pool = _pool()
        assert pool._inplace_swap is False

        acquire = AsyncMock(
            side_effect=AssertionError("swap path must not run when flag off")
        )
        monkeypatch.setattr(pool, "_acquire_inplace_adapter", acquire)
        monkeypatch.setattr(pool, "_validate_adapter_path", lambda p: p)
        # Pre-register a derived adapter entry that is already "loaded" so the
        # get_engine fast-path returns it without touching the real load path.
        # This isolates the test to: flag off -> early-return guard skipped.
        base = _base_entry()
        adapter_key = pool._adapter_key("qwen-base", "/adapters/fixA")
        adapter_entry = EngineEntry(
            model_id=adapter_key,
            model_path=base.model_path,
            model_type="llm",
            engine_type="batched",
            estimated_size=base.estimated_size,
        )
        adapter_entry.engine = base.engine
        pool._entries["qwen-base"] = base
        pool._entries[adapter_key] = adapter_entry

        await pool.get_engine("qwen-base", adapter_path="/adapters/fixA")
        acquire.assert_not_called()

    @pytest.mark.asyncio
    async def test_flag_on_with_loaded_base_routes_to_swap(self, monkeypatch):
        monkeypatch.setenv("FUSION_LORA_INPLACE_SWAP", "1")
        pool = _pool()
        entry = _base_entry()
        pool._entries["qwen-base"] = entry

        acquire = AsyncMock(return_value=entry.engine)
        monkeypatch.setattr(pool, "_acquire_inplace_adapter", acquire)
        monkeypatch.setattr(pool, "_validate_adapter_path", lambda p: p)

        eng = await pool.get_engine("qwen-base", adapter_path="/adapters/fixA")
        assert eng is entry.engine
        acquire.assert_awaited_once_with("qwen-base", "/adapters/fixA")

    @pytest.mark.asyncio
    async def test_flag_on_no_adapter_waits_for_active_swap(self, monkeypatch):
        monkeypatch.setenv("FUSION_LORA_INPLACE_SWAP", "1")
        pool = _pool()
        entry = _base_entry()
        pool._entries["qwen-base"] = entry

        pool._active_swap["qwen-base"] = MagicMock()

        waited = {"hit": False}

        class _Lock:
            async def __aenter__(self):
                waited["hit"] = True
                return self

            async def __aexit__(self, *a):
                return False

        monkeypatch.setattr(pool, "_swap_lock", lambda mid: _Lock())

        # Base is already loaded -> get_engine fast-path returns it. We only
        # assert the bare-base path waited for the in-flight swap first.
        await pool.get_engine("qwen-base")
        assert waited["hit"] is True


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_serialized_acquire_release(self, monkeypatch):
        monkeypatch.setenv("FUSION_LORA_INPLACE_SWAP", "1")
        pool = _pool()
        entry = _base_entry()
        pool._entries["qwen-base"] = entry

        order = []

        async def run(name):
            swap = MagicMock()
            swap.apply = MagicMock(side_effect=lambda: order.append(f"apply-{name}"))
            swap.restore = MagicMock(
                side_effect=lambda: order.append(f"restore-{name}")
            )
            with patch(
                "fusion_mlx.adapter.weight_swap.InPlaceLoRASwap",
                return_value=swap,
            ):
                await pool._acquire_inplace_adapter("qwen-base", f"/adapters/{name}")
                await asyncio.sleep(0.01)
                await pool._release_inplace_adapter("qwen-base", f"/adapters/{name}")

        await asyncio.gather(run("A"), run("B"))

        a_idx = order.index("apply-A")
        r_idx = order.index("restore-A")
        b_idx = order.index("apply-B")
        assert a_idx < r_idx < b_idx
