# SPDX-License-Identifier: Apache-2.0
# #1716: RSS hard stop for process-mode training children must be ON by
# default. With the env knob unset the #1483 watcher used to be disabled
# entirely, so a ballooning child died to an external killer (jetsam/OOM,
# exitcode=-9) with no enforcer and no record. All RSS readings here are
# mocked psutil — no model load.

import threading

import pytest

import fusion_mlx.training.proc as proc_mod
from fusion_mlx.training.proc import default_hard_stop_limit_gb


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("FUSION_MLX_TRAIN_HARD_STOP_RSS_GB", raising=False)
    yield


class TestDefaultDerivation:
    def test_derived_from_child_memory_limit(self):
        # child MLX budget 20GB -> +25% non-MLX headroom
        assert default_hard_stop_limit_gb(20.0) == pytest.approx(25.0)

    def test_unknown_basis_uses_auto_fraction(self, monkeypatch):
        # memory_limit_gb=0 -> child auto budget = 60% of physical; +25%.
        monkeypatch.setattr(proc_mod, "physical_memory_gb", lambda: 128.0)
        assert default_hard_stop_limit_gb(0.0) == pytest.approx(128.0 * 0.60 * 1.25)

    def test_capped_at_85pct_of_physical(self, monkeypatch):
        # a misconfigured huge child budget must not disable the guard
        monkeypatch.setattr(proc_mod, "physical_memory_gb", lambda: 128.0)
        assert default_hard_stop_limit_gb(400.0) == pytest.approx(128.0 * 0.85)

    def test_default_always_positive(self):
        assert default_hard_stop_limit_gb(0.0) > 0


class _FakeProc:
    def __init__(self, pid=91716):
        self.pid = pid
        self._alive = True

    def is_alive(self):
        return self._alive


def _run_watcher(monkeypatch, rss_seq, env, memory_limit_gb):
    # Scripted-RSS watcher run: each loop iteration pops one reading; the
    # sequence ends the run. env=None means knob unset (default path).
    import psutil

    if env is None:
        monkeypatch.delenv("FUSION_MLX_TRAIN_HARD_STOP_RSS_GB", raising=False)
    else:
        monkeypatch.setenv("FUSION_MLX_TRAIN_HARD_STOP_RSS_GB", str(env))
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

    monkeypatch.setattr(proc_mod, "kill_training_process", _kill)
    proc_mod.child_rss_hard_stop_watcher(
        lambda: proc if proc.is_alive() else None,
        stop,
        notes.append,
        memory_limit_gb,
    )
    return notes, killed


class TestWatcherDefaultEnabled:
    def test_unset_env_derives_default_and_kills(self, monkeypatch):
        # knob unset + child budget 1GB -> default limit 1.25GB; sustained
        # 2GB RSS must be terminated gracefully with a recorded note.
        notes, killed = _run_watcher(
            monkeypatch, [2.0, 2.0, 2.0], None, memory_limit_gb=1.0
        )
        assert killed == [91716]
        assert len(notes) == 1
        assert "RSS hard stop (#1483/#1716)" in notes[0]
        assert "source: default" in notes[0]
        assert "1.2GB" in notes[0]

    def test_env_override_wins_over_default(self, monkeypatch):
        # env 5GB beats the 1.25GB default; 2GB RSS is under it -> no kill.
        notes, killed = _run_watcher(
            monkeypatch, [2.0, 2.0, 2.0], 5, memory_limit_gb=1.0
        )
        assert killed == []
        assert notes == []

    def test_env_override_limit_kills_with_env_source(self, monkeypatch):
        notes, killed = _run_watcher(
            monkeypatch, [6.0, 6.0, 6.0], 5, memory_limit_gb=1.0
        )
        assert killed == [91716]
        assert "source: FUSION_MLX_TRAIN_HARD_STOP_RSS_GB" in notes[0]

    def test_explicit_zero_opts_out(self, monkeypatch):
        notes, killed = _run_watcher(
            monkeypatch, [200.0, 200.0, 200.0], 0, memory_limit_gb=1.0
        )
        assert killed == []
        assert notes == []

    def test_invalid_env_opts_out(self, monkeypatch):
        notes, killed = _run_watcher(
            monkeypatch, [200.0, 200.0, 200.0], "abc", memory_limit_gb=1.0
        )
        assert killed == []
        assert notes == []

    def test_default_capped_by_host_fraction_kills(self, monkeypatch):
        # unknown child budget on a 128GB box -> default 96GB; 120GB RSS
        # sustained must still die before host jetsam.
        monkeypatch.setattr(proc_mod, "physical_memory_gb", lambda: 128.0)
        notes, killed = _run_watcher(
            monkeypatch, [120.0, 120.0, 120.0], None, memory_limit_gb=0.0
        )
        assert killed == [91716]
        assert "source: default" in notes[0]

    def test_intermittent_spike_never_kills_under_default(self, monkeypatch):
        notes, killed = _run_watcher(
            monkeypatch, [2.0, 0.5, 2.0, 0.5, 2.0, 0.5], None, memory_limit_gb=1.0
        )
        assert killed == []
        assert notes == []


class TestAbortRegistryComment:
    def test_hard_stop_disabled_explicitly_vs_unset(self, monkeypatch):
        from fusion_mlx.training.abort import hard_stop_disabled_explicitly

        monkeypatch.delenv("FUSION_MLX_TRAIN_HARD_STOP_RSS_GB", raising=False)
        assert hard_stop_disabled_explicitly() is False
        monkeypatch.setenv("FUSION_MLX_TRAIN_HARD_STOP_RSS_GB", "0")
        assert hard_stop_disabled_explicitly() is True
        monkeypatch.setenv("FUSION_MLX_TRAIN_HARD_STOP_RSS_GB", "-3")
        assert hard_stop_disabled_explicitly() is True
        monkeypatch.setenv("FUSION_MLX_TRAIN_HARD_STOP_RSS_GB", "abc")
        assert hard_stop_disabled_explicitly() is True
        monkeypatch.setenv("FUSION_MLX_TRAIN_HARD_STOP_RSS_GB", "12")
        assert hard_stop_disabled_explicitly() is False
