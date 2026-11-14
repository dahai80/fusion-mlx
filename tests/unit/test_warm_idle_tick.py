"""Engine idle warm tick: queueing rules and busy-traffic priority.

Covers Lever-3 stage 2b: warm replay requests are queued only after a
full idle window, one head per tick, never while real traffic exists,
and warm state resets on the first busy tick.
"""

from types import SimpleNamespace

import pytest

from fusion_mlx.cache import warm_list
from fusion_mlx.engine_core import EngineCore
from fusion_mlx.request import Request


@pytest.fixture
def wl_path(tmp_path, monkeypatch):
    path = tmp_path / "warm_list.json"
    monkeypatch.setenv("FUSION_WARMLIST_DATA_DIR", str(tmp_path))
    wl = warm_list.WarmList(str(path))
    wl.record("h1", "system head one", 32)
    wl.record("h2", "system head two", 32)
    wl.flush()
    warm_list._default = wl
    yield wl
    warm_list._default = None


def make_fake(scheduler):
    return SimpleNamespace(
        _warm_idle_since=None,
        _warm_pending=[],
        _warm_seq=0,
        scheduler=scheduler,
    )


class FakeSched:
    def __init__(self):
        self.added = []
        self.busy = False

    def has_requests(self):
        return self.busy

    def add_request(self, req):
        self.added.append(req)


def test_disabled_env_never_queues(wl_path, monkeypatch):
    monkeypatch.setenv("FUSION_WARMLIST_DISABLE", "1")
    sched = FakeSched()
    fake = make_fake(sched)
    EngineCore._warm_tick(fake, 100.0, False)
    EngineCore._warm_tick(fake, 100.0 + warm_list.warm_idle_threshold_s() + 1, False)
    EngineCore._warm_tick(fake, 100.0 + warm_list.warm_idle_threshold_s() + 2, False)
    assert not sched.added
    warm_list._default = None


def test_busy_resets_idle_window(wl_path):
    sched = FakeSched()
    fake = make_fake(sched)
    EngineCore._warm_tick(fake, 100.0, False)
    assert fake._warm_idle_since == 100.0
    EngineCore._warm_tick(fake, 110.0, True)
    assert fake._warm_idle_since is None
    assert not fake._warm_pending


def test_idle_below_threshold_no_queue(wl_path):
    sched = FakeSched()
    fake = make_fake(sched)
    EngineCore._warm_tick(fake, 100.0, False)
    EngineCore._warm_tick(fake, 100.0 + warm_list.warm_idle_threshold_s() - 1, False)
    assert not fake._warm_pending
    assert not sched.added


def test_idle_past_threshold_queues_then_pops_one_per_tick(wl_path):
    sched = FakeSched()
    fake = make_fake(sched)
    t0 = 100.0
    EngineCore._warm_tick(fake, t0, False)
    EngineCore._warm_tick(fake, t0 + warm_list.warm_idle_threshold_s() + 1, False)
    assert len(fake._warm_pending) == 2
    assert not sched.added
    EngineCore._warm_tick(fake, t0 + warm_list.warm_idle_threshold_s() + 2, False)
    assert len(sched.added) == 1
    req = sched.added[0]
    assert isinstance(req, Request)
    assert req.request_id == "warm-1"
    assert req._warm_request is True
    assert req.sampling_params.max_tokens == 1
    assert req.sampling_params.temperature == 0.0
    EngineCore._warm_tick(fake, t0 + warm_list.warm_idle_threshold_s() + 3, False)
    assert len(sched.added) == 2
    assert sched.added[1].request_id == "warm-2"


def test_real_traffic_blocks_warm_pop(wl_path):
    sched = FakeSched()
    sched.busy = True
    fake = make_fake(sched)
    t0 = 100.0
    EngineCore._warm_tick(fake, t0, False)
    EngineCore._warm_tick(fake, t0 + warm_list.warm_idle_threshold_s() + 1, False)
    assert fake._warm_pending
    EngineCore._warm_tick(fake, t0 + warm_list.warm_idle_threshold_s() + 2, False)
    assert not sched.added
    assert fake._warm_pending


def test_drained_queue_restarts_idle_clock(wl_path):
    sched = FakeSched()
    fake = make_fake(sched)
    t0 = 100.0
    thr = warm_list.warm_idle_threshold_s()
    EngineCore._warm_tick(fake, t0, False)
    EngineCore._warm_tick(fake, t0 + thr + 1, False)
    EngineCore._warm_tick(fake, t0 + thr + 2, False)
    EngineCore._warm_tick(fake, t0 + thr + 3, False)
    assert len(sched.added) == 2
    assert fake._warm_idle_since == t0 + thr + 3
    assert not fake._warm_pending


def test_empty_warm_list_never_queues(tmp_path, monkeypatch):
    monkeypatch.setenv("FUSION_WARMLIST_DATA_DIR", str(tmp_path))
    warm_list._default = warm_list.WarmList(str(tmp_path / "empty.json"))
    sched = FakeSched()
    fake = make_fake(sched)
    t0 = 100.0
    EngineCore._warm_tick(fake, t0, False)
    EngineCore._warm_tick(fake, t0 + warm_list.warm_idle_threshold_s() + 1, False)
    EngineCore._warm_tick(fake, t0 + warm_list.warm_idle_threshold_s() + 2, False)
    assert not sched.added
    warm_list._default = None
