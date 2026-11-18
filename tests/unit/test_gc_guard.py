"""#1647: gradient checkpointing idempotency guard tests."""

import mlx.nn as nn
from mlx_lm.tuner import trainer as tuner_trainer

from fusion_mlx.training.gc_guard import ensure_grad_checkpoint_guard


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
