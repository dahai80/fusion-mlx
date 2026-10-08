# SPDX-License-Identifier: Apache-2.0
"""#1075: DFlyAcceptCounter must be isolated per DFlyDrafter instance so
multi-model EnginePool stats don't mix across engines."""

from fusion_mlx.speculative.dfly.drafter import DFlyDrafter


def test_drafter_has_own_counter_instance():
    d1 = DFlyDrafter(model_path="/fake/a")
    d2 = DFlyDrafter(model_path="/fake/b")
    assert d1.accept_counter is not None
    assert d2.accept_counter is not None
    assert d1.accept_counter is not d2.accept_counter


def test_per_drafter_counters_isolated():
    d1 = DFlyDrafter(model_path="/fake/a")
    d2 = DFlyDrafter(model_path="/fake/b")
    d1.accept_counter.record(accepted=5, drafted=10)
    d2.accept_counter.record(accepted=1, drafted=4)

    assert d1.accept_counter.snapshot().accepted == 5
    assert d1.accept_counter.snapshot().drafted == 10
    assert d2.accept_counter.snapshot().accepted == 1
    assert d2.accept_counter.snapshot().drafted == 4


def test_drafter_counter_independent_of_global():
    d = DFlyDrafter(model_path="/fake/a")
    d.accept_counter.record(accepted=7, drafted=7)
    # The global is a separate instance — recording on the drafter must
    # not leak into it.
    from fusion_mlx.speculative.dfly.accept_counter import (
        get_global_counter,
        reset_global_counter_for_tests,
    )

    reset_global_counter_for_tests()
    g = get_global_counter()
    assert g.snapshot().accepted == 0
    assert g.snapshot().drafted == 0
    assert d.accept_counter.snapshot().accepted == 7


def test_counter_reset_clears_only_owner():
    d1 = DFlyDrafter(model_path="/fake/a")
    d2 = DFlyDrafter(model_path="/fake/b")
    d1.accept_counter.record(accepted=3, drafted=6)
    d2.accept_counter.record(accepted=2, drafted=8)
    d1.accept_counter.reset()
    assert d1.accept_counter.snapshot().accepted == 0
    assert d2.accept_counter.snapshot().accepted == 2
