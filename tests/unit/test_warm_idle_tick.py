"""Engine idle warm tick: queueing rules and busy-traffic priority.

Covers Lever-3 stage 2b: warm replay requests are queued only after a
full idle window, one head per tick, never while real traffic exists.
Real traffic resets the idle window; warm replays do not (they would
otherwise requeue themselves forever while the server sits idle).
"""

from types import SimpleNamespace

import pytest

from fusion_mlx.cache import warm_list
from fusion_mlx.engine_core import EngineCore
from fusion_mlx.request import Request


@pytest.fixture(autouse=True)
def _no_min_tokens(monkeypatch):
    # Suite exercises queueing mechanics with tiny recorded prompts; the
    # size filter is pinned separately in test_warm_list.py and below.
    monkeypatch.setenv("FUSION_WARMLIST_MIN_TOKENS", "0")


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
    fake = SimpleNamespace(
        _warm_idle_since=None,
        _warm_pending=[],
        _warm_seq=0,
        _warm_round_done=False,
        scheduler=scheduler,
    )
    fake._warm_has_real_traffic = lambda: EngineCore._warm_has_real_traffic(fake)
    return fake


def _req(warm):
    return SimpleNamespace(_warm_request=warm)


class FakeSched:
    def __init__(self):
        self.added = []
        self.busy = False
        self.waiting = []
        self.prefilling = []
        self.running = []

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


def test_busy_real_traffic_resets_idle_window(wl_path):
    sched = FakeSched()
    sched.running = [_req(warm=False)]
    fake = make_fake(sched)
    EngineCore._warm_tick(fake, 100.0, False)
    assert fake._warm_idle_since == 100.0
    EngineCore._warm_tick(fake, 110.0, True)
    assert fake._warm_idle_since is None
    assert fake._warm_round_done is False


def test_busy_warm_replay_keeps_idle_window(wl_path):
    # A warm replay makes the engine busy but is not traffic: resetting the
    # idle window here requeued the whole head list every idle period while
    # the server sat idle (endless 8k prefill bursts).
    sched = FakeSched()
    sched.running = [_req(warm=True)]
    fake = make_fake(sched)
    EngineCore._warm_tick(fake, 100.0, False)
    assert fake._warm_idle_since == 100.0
    EngineCore._warm_tick(fake, 110.0, True)
    assert fake._warm_idle_since == 100.0


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
    # Warm work must lose admission ordering to any real request: with the
    # default priority a probe arriving mid-round queued behind the whole
    # warm prefill burst (+4.9s median, Gate W3).
    assert req.priority > 0
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


def test_drained_queue_latches_round_until_real_traffic(wl_path):
    # After the head queue drains, idle time alone must never requeue —
    # only real traffic re-arms the round latch.
    sched = FakeSched()
    fake = make_fake(sched)
    t0 = 100.0
    thr = warm_list.warm_idle_threshold_s()
    EngineCore._warm_tick(fake, t0, False)
    EngineCore._warm_tick(fake, t0 + thr + 1, False)
    EngineCore._warm_tick(fake, t0 + thr + 2, False)
    EngineCore._warm_tick(fake, t0 + thr + 3, False)
    assert len(sched.added) == 2
    assert fake._warm_round_done is True
    assert not fake._warm_pending
    far = t0 + thr + 3 + thr * 10
    EngineCore._warm_tick(fake, far, False)
    EngineCore._warm_tick(fake, far + 1, False)
    assert len(sched.added) == 2
    # Real traffic clears the latch; the next idle window queues again.
    sched.running = [_req(warm=False)]
    EngineCore._warm_tick(fake, far + 2, True)
    sched.running = []
    fake._warm_idle_since = None
    EngineCore._warm_tick(fake, far + 3, False)
    EngineCore._warm_tick(fake, far + 3 + thr + 1, False)
    while fake._warm_pending:
        EngineCore._warm_tick(fake, far + 3 + thr + 2, False)
    assert len(sched.added) == 4
    assert fake._warm_round_done is True


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


def test_short_heads_never_queued(tmp_path, monkeypatch):
    # Default threshold filters tiny recorded prompts out of the replay
    # path even when they hold the top hit counts.
    monkeypatch.setenv("FUSION_WARMLIST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FUSION_WARMLIST_MIN_TOKENS", "256")
    wl = warm_list.WarmList(str(tmp_path / "short.json"))
    wl.record("tiny", "warmup:", 4)
    wl.flush()
    warm_list._default = wl
    try:
        sched = FakeSched()
        fake = make_fake(sched)
        t0 = 100.0
        EngineCore._warm_tick(fake, t0, False)
        EngineCore._warm_tick(fake, t0 + warm_list.warm_idle_threshold_s() + 1, False)
        assert not fake._warm_pending
        assert not sched.added
    finally:
        warm_list._default = None
