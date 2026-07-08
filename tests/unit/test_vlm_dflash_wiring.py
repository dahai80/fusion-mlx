# SPDX-License-Identifier: Apache-2.0
# Regression: the VLM engine never loaded _dflash_runtime (only the text
# engine batched.py did), so the per-request router found no loaded method and
# VLM decode never speculated via DFlash. _apply_dflash() ports the text
# engine's loader. DFlash is VLM-safe: dflash_spec_step drafts from
# current_token (a generated text token) and verifies via gen.model +
# gen.prompt_cache, which for a VLM already hold the vision features computed
# at prefill - decode-phase verify is identical to text. (DSpark is NOT ported:
# it reloads its own text-only target and re-feeds the full multimodal prompt.)
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from fusion_mlx.engines.vlm import VLMBatchedEngine


def _make_engine(model_settings, scheduler=None, scheduler_config=None):
    engine = VLMBatchedEngine.__new__(VLMBatchedEngine)
    engine._model_settings = model_settings
    engine._model_name = "test_vlm"
    engine._scheduler_config = scheduler_config
    sched = scheduler if scheduler is not None else SimpleNamespace()
    engine._engine = SimpleNamespace(engine=SimpleNamespace(scheduler=sched))
    return engine


class _FakeDFlashRuntime:
    def __init__(self, kind="dflash"):
        self.kind = kind
        self.drafter = SimpleNamespace()


def test_enabled_loads_dflash_runtime_onto_scheduler():
    sched = SimpleNamespace()
    engine = _make_engine(
        SimpleNamespace(dflash_drafter_path="org/drafter"),
        scheduler=sched,
    )
    fake = _FakeDFlashRuntime()

    with patch(
        "fusion_mlx.speculative.dflash.load_runtime", return_value=fake
    ):
        asyncio.run(engine._apply_dflash())

    assert sched._dflash_runtime is fake


def test_scheduler_config_fallback_path_used():
    # No per-model override -> fall back to scheduler_config.dflash_drafter_path.
    sched = SimpleNamespace()
    engine = _make_engine(
        SimpleNamespace(dflash_drafter_path=None),
        scheduler=sched,
        scheduler_config=SimpleNamespace(dflash_drafter_path="org/cfg-drafter"),
    )
    captured = {}

    def fake_load(path, kind="dflash"):
        captured["path"] = path
        return _FakeDFlashRuntime()

    with patch("fusion_mlx.speculative.dflash.load_runtime", side_effect=fake_load):
        asyncio.run(engine._apply_dflash())

    assert captured["path"] == "org/cfg-drafter"
    assert sched._dflash_runtime is not None


def test_no_path_is_noop():
    # No drafter path anywhere -> no load, scheduler attr untouched.
    sched = SimpleNamespace()
    engine = _make_engine(
        SimpleNamespace(dflash_drafter_path=None),
        scheduler=sched,
        scheduler_config=SimpleNamespace(dflash_drafter_path=""),
    )

    with patch(
        "fusion_mlx.speculative.dflash.load_runtime", return_value=_FakeDFlashRuntime()
    ) as m:
        asyncio.run(engine._apply_dflash())

    m.assert_not_called()
    assert not hasattr(sched, "_dflash_runtime")


def test_none_model_settings_falls_back_to_config():
    sched = SimpleNamespace()
    engine = _make_engine(
        None,
        scheduler=sched,
        scheduler_config=SimpleNamespace(dflash_drafter_path="org/cfg"),
    )
    fake = _FakeDFlashRuntime()

    with patch(
        "fusion_mlx.speculative.dflash.load_runtime", return_value=fake
    ):
        asyncio.run(engine._apply_dflash())

    assert sched._dflash_runtime is fake


def test_load_failure_is_swallowed_and_logged(caplog):
    sched = SimpleNamespace()
    engine = _make_engine(
        SimpleNamespace(dflash_drafter_path="org/bad"),
        scheduler=sched,
    )

    with patch(
        "fusion_mlx.speculative.dflash.load_runtime",
        side_effect=RuntimeError("drafter boom"),
    ):
        asyncio.run(engine._apply_dflash())

    assert not hasattr(sched, "_dflash_runtime")
    assert any("DFlash drafter load failed" in r.message for r in caplog.records)


def test_router_assigns_dflash_to_request_without_vlm_guard():
    # The dispatch path is engine-type-agnostic: given dflash is the only loaded
    # method, the router must return METHOD_DFLASH for ANY request (including a
    # VLM-shaped one). No is_vlm guard exists - that is what makes VLM DFlash
    # spec decode work once _dflash_runtime is loaded.
    from fusion_mlx.speculative.per_request_route import (
        loaded_methods,
        select_active_method,
    )

    loaded = loaded_methods(dflash=True)
    method = select_active_method(
        prompt_token_count=128,
        loaded=loaded,
        has_mtp=False,
    )
    from fusion_mlx.speculative.auto_router import METHOD_DFLASH

    assert method == METHOD_DFLASH
