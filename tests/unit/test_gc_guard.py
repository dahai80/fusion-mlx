"""#1647: gradient checkpointing idempotency guard tests."""

import sys

import mlx.nn as nn
import pytest
from mlx_lm.tuner import trainer as tuner_trainer

from fusion_mlx.training import gc_guard
from fusion_mlx.training.gc_guard import ensure_grad_checkpoint_guard


@pytest.fixture(autouse=True)
def _reset_gc_guard_state():
    # Guard markers live on layer classes process-globally (mlx-lm patches
    # type(layer).__call__, so e.g. nn.Linear is shared across tests).
    # Reset the bookkeeping so each test observes a fresh wrap.
    def _clear():
        for cls in list(gc_guard._MARKED_CLASSES):
            try:
                delattr(cls, gc_guard._GUARD_ATTR)
            except AttributeError:
                pass
        gc_guard._MARKED_CLASSES.clear()

    _clear()
    yield
    _clear()


class _Block(nn.Module):
    def __call__(self, x):
        return x


def test_guard_installed_and_idempotent():
    ensure_grad_checkpoint_guard()
    ensure_grad_checkpoint_guard()

    original = tuner_trainer.grad_checkpoint
    blk = _Block()
    fn_before = type(blk).__call__

    original(blk)
    fn_after_first = type(blk).__call__
    original(blk)
    fn_after_second = type(blk).__call__

    assert fn_before is not fn_after_first
    assert fn_after_first is fn_after_second


def test_guard_prevents_nested_wrap():
    ensure_grad_checkpoint_guard()
    tuner_trainer.grad_checkpoint(_Block())

    class _Other(nn.Module):
        def __call__(self, x):
            return x

    blk = _Other()
    tuner_trainer.grad_checkpoint(blk)
    fn_once = type(blk).__call__
    tuner_trainer.grad_checkpoint(blk)
    assert type(blk).__call__ is fn_once


def test_guard_per_class_keying():
    ensure_grad_checkpoint_guard()

    class _A(nn.Module):
        def __call__(self, x):
            return x

    class _B(nn.Module):
        def __call__(self, x):
            return x

    a, b = _A(), _B()
    tuner_trainer.grad_checkpoint(a)
    assert getattr(_A, "_fusion_mlx_gc_wrapped", False)
    assert not getattr(_B, "_fusion_mlx_gc_wrapped", False)
    tuner_trainer.grad_checkpoint(b)
    assert getattr(_B, "_fusion_mlx_gc_wrapped", False)


def test_text_only_call_does_not_patch_mlx_vlm():
    ensure_grad_checkpoint_guard()
    vlm_utils = sys.modules.get("mlx_vlm.trainer.utils")
    if vlm_utils is None:
        pytest.skip("mlx_vlm not yet imported in this process")
    assert not getattr(vlm_utils.grad_checkpoint, "_fusion_mlx_gc_guard", False)


def test_vlm_guard_installed_on_include_vlm():
    pytest.importorskip("mlx_vlm")
    from mlx_vlm.trainer import utils as vlm_utils

    ensure_grad_checkpoint_guard(include_vlm=True)
    assert getattr(vlm_utils.grad_checkpoint, "_fusion_mlx_gc_guard", False)

    blk = _Block()
    vlm_utils.grad_checkpoint(blk)
    fn_once = type(blk).__call__
    vlm_utils.grad_checkpoint(blk)
    assert type(blk).__call__ is fn_once


def test_vlm_patched_after_text_only_first_call(monkeypatch):
    pytest.importorskip("mlx_vlm")
    from mlx_vlm.trainer import utils as vlm_utils

    # Simulate a fresh process where only the text-only install ran:
    # restore an unguarded upstream-shaped entry point first.
    def plain_grad_checkpoint(layer):
        pass

    monkeypatch.setattr(vlm_utils, "grad_checkpoint", plain_grad_checkpoint)
    ensure_grad_checkpoint_guard()
    assert not getattr(vlm_utils.grad_checkpoint, "_fusion_mlx_gc_guard", False)
    ensure_grad_checkpoint_guard(include_vlm=True)
    assert getattr(vlm_utils.grad_checkpoint, "_fusion_mlx_gc_guard", False)


def test_vlm_sft_trainer_binds_guarded_entry_point():
    pytest.importorskip("mlx_vlm")
    import sys

    ensure_grad_checkpoint_guard(include_vlm=True)
    import mlx_vlm.trainer.sft_trainer as sft_trainer

    assert getattr(sft_trainer.grad_checkpoint, "_fusion_mlx_gc_guard", False)
    # Modules already in sys.modules must also be re-patched in place.
    assert "mlx_vlm.trainer.sft_trainer" in sys.modules
