# SPDX-License-Identifier: Apache-2.0

import pytest

from fusion_mlx.scheduler import helpers
from fusion_mlx.scheduler.helpers import _current_fragmentation_ratio


@pytest.fixture
def fake_memory(monkeypatch):
    def _set(phys, active, cache):
        monkeypatch.setattr(
            "fusion_mlx.utils.proc_memory.get_phys_footprint", lambda: phys
        )
        monkeypatch.setattr(helpers.mx, "get_active_memory", lambda: active)
        monkeypatch.setattr(helpers.mx, "get_cache_memory", lambda: cache)

    return _set


def test_ratio_matches_metric_formula(fake_memory):
    fake_memory(phys=1000, active=200, cache=300)

    assert _current_fragmentation_ratio() == pytest.approx(0.5)


def test_zero_footprint_is_not_fragmented(fake_memory):
    fake_memory(phys=0, active=0, cache=0)

    assert _current_fragmentation_ratio() == 0.0


def test_allocator_claiming_more_than_footprint_clamps_to_zero(fake_memory):
    fake_memory(phys=1000, active=900, cache=600)

    assert _current_fragmentation_ratio() == 0.0


def test_probe_failure_returns_zero(monkeypatch):
    def _boom():
        raise RuntimeError("task_info unavailable")

    monkeypatch.setattr("fusion_mlx.utils.proc_memory.get_phys_footprint", _boom)

    assert _current_fragmentation_ratio() == 0.0


def test_scheduler_reads_the_live_ratio():
    import inspect

    from fusion_mlx.scheduler import sched_step

    source = inspect.getsource(sched_step)
    assert "_fragmentation_ratio" not in source.replace(
        "_current_fragmentation_ratio", ""
    )
    assert source.count("_current_fragmentation_ratio()") == 2
