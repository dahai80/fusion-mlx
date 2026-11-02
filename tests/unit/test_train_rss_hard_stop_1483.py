# SPDX-License-Identifier: Apache-2.0
# #1483 item 2b: training-side RSS hard stop. A job whose process RSS stays
# over FUSION_MLX_TRAIN_HARD_STOP_RSS_GB for _RSS_STRIKES consecutive
# between-iteration checks aborts with a FAILED job carrying the reason —
# before the enforcer's process-level fatal-exit and long before the host
# watchdog (#1473 panic class).

import threading

import pytest

from fusion_mlx.training import abort as ta


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("FUSION_MLX_TRAIN_HARD_STOP_RSS_GB", raising=False)
    with ta._lock:
        ta._active.clear()
        ta._rss_strikes.clear()
    yield
    with ta._lock:
        ta._active.clear()
        ta._rss_strikes.clear()


def _set_limit(monkeypatch, gb):
    monkeypatch.setenv("FUSION_MLX_TRAIN_HARD_STOP_RSS_GB", str(gb))


def _fake_rss(monkeypatch, gb):
    monkeypatch.setattr(ta, "_current_rss_gb", lambda: gb)


class TestKnobParsing:
    def test_unset_disables(self):
        assert ta.hard_stop_limit_gb() == 0.0

    def test_zero_disables(self, monkeypatch):
        _set_limit(monkeypatch, 0)
        assert ta.hard_stop_limit_gb() == 0.0

    def test_negative_disables(self, monkeypatch):
        _set_limit(monkeypatch, -5)
        assert ta.hard_stop_limit_gb() == 0.0

    def test_valid_value(self, monkeypatch):
        _set_limit(monkeypatch, 42.5)
        assert ta.hard_stop_limit_gb() == 42.5

    def test_invalid_disables_with_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("FUSION_MLX_TRAIN_HARD_STOP_RSS_GB", "abc")
        with caplog.at_level("WARNING"):
            assert ta.hard_stop_limit_gb() == 0.0
        assert "FUSION_MLX_TRAIN_HARD_STOP_RSS_GB" in caplog.text


class TestCheckRssHardStop:
    def test_disabled_never_raises(self, monkeypatch):
        _fake_rss(monkeypatch, 9999.0)
        for _ in range(10):
            ta.check_rss_hard_stop("job-1")

    def test_under_limit_resets_strikes(self, monkeypatch):
        _set_limit(monkeypatch, 10)
        _fake_rss(monkeypatch, 20.0)
        ta.check_rss_hard_stop("job-1")
        _fake_rss(monkeypatch, 5.0)
        ta.check_rss_hard_stop("job-1")
        assert ta._rss_strikes["job-1"] == 0

    def test_raises_after_sustained_breach(self, monkeypatch, caplog):
        _set_limit(monkeypatch, 10)
        _fake_rss(monkeypatch, 20.0)
        with caplog.at_level("WARNING"):
            with pytest.raises(ta.TrainingAbortedError) as ei:
                for _ in range(ta._RSS_STRIKES):
                    ta.check_rss_hard_stop("job-1")
        assert "RSS hard stop (#1483)" in str(ei.value)
        assert "20.0GB" in str(ei.value)
        # a WARNING per over-limit observation before the raise
        assert caplog.text.count("over hard-stop limit") >= ta._RSS_STRIKES - 1

    def test_single_spike_does_not_raise(self, monkeypatch):
        _set_limit(monkeypatch, 10)
        seq = [20.0, 5.0, 20.0, 5.0, 20.0, 5.0]
        for gb in seq:
            _fake_rss(monkeypatch, gb)
            ta.check_rss_hard_stop("job-1")

    def test_rss_read_failure_skips(self, monkeypatch):
        _set_limit(monkeypatch, 10)
        monkeypatch.setattr(ta, "_current_rss_gb", lambda: None)
        for _ in range(10):
            ta.check_rss_hard_stop("job-1")

    def test_deregister_clears_strikes(self, monkeypatch):
        _set_limit(monkeypatch, 10)
        _fake_rss(monkeypatch, 20.0)
        ta.register_job("job-1", "dpo")
        ta.check_rss_hard_stop("job-1")
        assert ta._rss_strikes.get("job-1") == 1
        ta.deregister_job("job-1")
        assert "job-1" not in ta._rss_strikes


class TestCheckAbortIntegration:
    def test_check_abort_runs_rss_gate(self, monkeypatch):
        _set_limit(monkeypatch, 10)
        _fake_rss(monkeypatch, 20.0)
        ta.register_job("job-1", "dpo")
        with pytest.raises(ta.TrainingAbortedError) as ei:
            for _ in range(ta._RSS_STRIKES):
                ta.check_abort("job-1")
        assert "RSS hard stop (#1483)" in str(ei.value)

    def test_check_abort_still_honors_requested_abort(self, monkeypatch):
        _set_limit(monkeypatch, 10)
        _fake_rss(monkeypatch, 5.0)
        ta.register_job("job-1", "dpo")
        ta.request_abort("job-1", "enforcer said stop")
        with pytest.raises(ta.TrainingAbortedError, match="enforcer said stop"):
            ta.check_abort("job-1")

    def test_requested_abort_wins_when_both_pending(self, monkeypatch):
        # explicit abort requests take effect on the very next check; the
        # RSS gate needs _RSS_STRIKES sustained readings — so with both
        # pending the requested reason is the one that fires.
        _set_limit(monkeypatch, 10)
        _fake_rss(monkeypatch, 20.0)
        ta.register_job("job-1", "dpo")
        ta.request_abort("job-1", "enforcer said stop")
        with pytest.raises(ta.TrainingAbortedError) as ei:
            ta.check_abort("job-1")
        assert "enforcer said stop" in str(ei.value)


class _FakeProc:
    def __init__(self, pid=4242):
        self.pid = pid
        self._alive = True

    def is_alive(self):
        return self._alive


class TestChildWatcher:
    def _watch(self, monkeypatch, rss_seq, limit):
        # limit=None runs the watcher with the knob unset (disabled).
        import psutil

        import fusion_mlx.training.service as svc

        if limit is None:
            monkeypatch.delenv("FUSION_MLX_TRAIN_HARD_STOP_RSS_GB", raising=False)
        else:
            monkeypatch.setenv("FUSION_MLX_TRAIN_HARD_STOP_RSS_GB", str(limit))
        reads = list(rss_seq)

        class _FakeMem:
            rss = 0

        class _FakePsutilProc:
            def __init__(self, pid=None):
                pass

            def memory_info(self):
                if not reads:
                    raise psutil.NoSuchProcess(pid=1)
                gb = reads.pop(0)
                m = _FakeMem()
                m.rss = int(gb * (2**30))
                return m

        monkeypatch.setattr(psutil, "Process", _FakePsutilProc)

        # Each loop iteration consumes one rss reading; stop when the
        # scripted sequence is exhausted (or after a hard cap of 10 loops —
        # the disabled-knob path consumes no readings and must not spin).
        calls = {"n": 0}

        def _wait(self, t=None):
            calls["n"] += 1
            if not reads or calls["n"] > 10:
                self.set()
            return self.is_set()

        monkeypatch.setattr(threading.Event, "wait", _wait)
        stop = threading.Event()
        notes = []
        proc = _FakeProc()
        killed = []

        def _kill(p, grace_seconds=3.0):
            killed.append(p.pid)
            proc._alive = False
            return True

        import fusion_mlx.training.proc as proc_mod

        monkeypatch.setattr(proc_mod, "kill_training_process", _kill)
        svc._child_rss_hard_stop_watcher(
            lambda: proc if proc.is_alive() else None,
            stop,
            notes.append,
        )
        return notes, killed

    def test_watcher_kills_on_sustained_breach(self, monkeypatch, caplog):
        with caplog.at_level("WARNING"):
            notes, killed = self._watch(monkeypatch, [20.0, 20.0, 20.0], 10)
        assert killed == [4242]
        assert len(notes) == 1
        assert "RSS hard stop (#1483)" in notes[0]
        assert "terminated by RSS hard stop" in caplog.text

    def test_watcher_idle_when_under_limit(self, monkeypatch):
        notes, killed = self._watch(monkeypatch, [5.0, 5.0, 5.0], 10)
        assert killed == []
        assert notes == []

    def test_watcher_intermittent_spike_never_kills(self, monkeypatch):
        notes, killed = self._watch(monkeypatch, [20.0, 5.0, 20.0, 5.0, 20.0, 5.0], 10)
        assert killed == []
        assert notes == []

    def test_watcher_disabled_knob_never_kills(self, monkeypatch):
        notes, killed = self._watch(monkeypatch, [200.0, 200.0, 200.0], None)
        assert killed == []
        assert notes == []
