# SPDX-License-Identifier: Apache-2.0
# Tests for the grad_checkpoint idempotency guard (#1647).

import importlib.util
import sys

import pytest

from fusion_mlx.training import gc_guard


class FakeBlock:
    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return x + 1


def fake_upstream(layer):
    # Mimics mlx_lm.tuner.trainer.grad_checkpoint: re-wraps the current
    # __call__ every time it is invoked.
    fn = type(layer).__call__

    def checkpointed(model, *args, **kwargs):
        return fn(model, *args, **kwargs)

    type(layer).__call__ = checkpointed


@pytest.fixture(autouse=True)
def _restore_patched_sites():
    # Save upstream attrs before the test, restore after, and drop the
    # guard's wrapped-class bookkeeping so tests stay independent.
    saved = []
    for mod_name, attr in gc_guard._PATCH_TARGETS:
        try:
            mod = sys.modules.get(mod_name)
            if mod is None:
                import importlib

                mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        saved.append((mod, attr, getattr(mod, attr, None)))
    for ref_name in gc_guard._REFERENCING_MODULES:
        mod = sys.modules.get(ref_name)
        if mod is not None:
            saved.append(
                (mod, "grad_checkpoint", getattr(mod, "grad_checkpoint", None))
            )
    yield
    for mod, attr, original in saved:
        setattr(mod, attr, original)
    gc_guard._WRAPPED_CLASSES.clear()


def test_guarded_call_is_idempotent():
    guarded = gc_guard._make_guarded(fake_upstream, "test")
    b = FakeBlock()
    guarded(b)
    first = type(b).__call__
    guarded(b)
    second = type(b).__call__
    assert first is second
    assert gc_guard._is_wrapped(FakeBlock)


def test_guarded_call_preserves_behavior():
    guarded = gc_guard._make_guarded(fake_upstream, "test")
    b = FakeBlock()
    guarded(b)
    assert b(1) == 2
    assert b.calls == 1


def test_distinct_classes_wrapped_independently():
    guarded = gc_guard._make_guarded(fake_upstream, "test")

    class OtherBlock(FakeBlock):
        pass

    b1 = FakeBlock()
    b2 = OtherBlock()
    guarded(b1)
    guarded(b2)
    assert gc_guard._is_wrapped(FakeBlock)
    assert gc_guard._is_wrapped(OtherBlock)


def test_marker_survives_weakset_clear():
    guarded = gc_guard._make_guarded(fake_upstream, "test")
    b = FakeBlock()
    guarded(b)
    gc_guard._WRAPPED_CLASSES.clear()
    first = type(b).__call__
    guarded(b)
    assert type(b).__call__ is first


def test_install_patches_mlx_lm_site():
    import mlx_lm.tuner.trainer as trainer_mod

    original = trainer_mod.grad_checkpoint
    try:
        gc_guard.install()
        guarded = trainer_mod.grad_checkpoint
        assert getattr(guarded, "_fusion_gc_guard", False)
        # Second install is a no-op for the already-guarded site.
        gc_guard.install()
        assert trainer_mod.grad_checkpoint is guarded
    finally:
        trainer_mod.grad_checkpoint = original


def test_install_guarded_entry_wraps_once_with_real_upstream():
    # End-to-end: guarded entry point over the REAL mlx_lm grad_checkpoint.
    # Wrap-time only defines closures, so no MLX compute is needed here.
    import mlx_lm.tuner.trainer as trainer_mod

    original = trainer_mod.grad_checkpoint
    try:
        trainer_mod.grad_checkpoint = gc_guard._make_guarded(
            original, "mlx_lm.tuner.trainer"
        )
        b = FakeBlock()
        trainer_mod.grad_checkpoint(b)
        first = type(b).__call__
        trainer_mod.grad_checkpoint(b)
        assert type(b).__call__ is first
        assert b(1) == 2
    finally:
        trainer_mod.grad_checkpoint = original
    gc_guard._WRAPPED_CLASSES.clear()


@pytest.mark.skipif(
    importlib.util.find_spec("mlx_vlm") is None, reason="mlx_vlm not installed"
)
def test_install_patches_mlx_vlm_sites():
    import mlx_vlm.trainer.utils as vlm_utils

    original = vlm_utils.grad_checkpoint
    try:
        gc_guard.install()
        assert getattr(vlm_utils.grad_checkpoint, "_fusion_gc_guard", False)
        assert vlm_utils.grad_checkpoint is not original or getattr(
            original, "_fusion_gc_guard", False
        )
        # Referencing modules already in sys.modules must be re-patched too.
        sft = sys.modules.get("mlx_vlm.trainer.sft_trainer")
        if sft is not None:
            assert getattr(sft.grad_checkpoint, "_fusion_gc_guard", False)
    finally:
        vlm_utils.grad_checkpoint = original
