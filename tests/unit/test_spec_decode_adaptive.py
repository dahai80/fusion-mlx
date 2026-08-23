# SPDX-License-Identifier: Apache-2.0
# Phase-2 item 3: adaptive pause/resume for SpecDecodeState.
#
# The original resume branch in record_accepted was dead code — once
# _spec_paused=True, should_speculate() always returned False, so no
# drafts ran, so record_accepted() never fired to un-pause. Phase-2
# item 3 adds a periodic re-probe: while paused, should_speculate()
# returns True every SPEC_RESUME_CHECK_INTERVAL steps, record_accepted()
# accumulates probe samples, and resumes when the probe rate recovers.
# These tests pin that contract without a real model (the scheduler
# fast path is mocked).

from __future__ import annotations

from fusion_mlx.scheduler import spec_decode


class _FakeDraft:
    def reset(self):
        pass

    def on_new_request(self, request_id, prompt_tokens):
        pass

    def record_accepted(self, n_accepted):
        pass


def _state(monkeypatch, *, window=5, min_rate=0.05, interval=2):
    monkeypatch.setattr(spec_decode, "SPEC_ADAPTIVE_WINDOW", window)
    monkeypatch.setattr(spec_decode, "SPEC_MIN_ACCEPT_RATE", min_rate)
    monkeypatch.setattr(spec_decode, "SPEC_RESUME_CHECK_INTERVAL", interval)
    monkeypatch.setattr(spec_decode, "SPEC_WARMUP_STEPS", 0)
    st = spec_decode.SpecDecodeState(draft_model_decoder=_FakeDraft())
    return st


def test_pauses_when_acceptance_below_threshold(monkeypatch):
    st = _state(monkeypatch, window=5, min_rate=0.5, interval=2)
    # 5 windows, all 0 accepted -> rate 0.0 < 0.5 -> pause
    for _ in range(5):
        st.record_accepted(0, 3)
    assert st._spec_paused is True


def test_should_speculate_blocks_when_paused(monkeypatch):
    st = _state(monkeypatch, window=2, min_rate=0.5, interval=10)
    st.record_accepted(0, 3)
    st.record_accepted(0, 3)
    assert st._spec_paused is True
    # interval=10, first few steps stay blocked
    for _ in range(9):
        st.add_token(1)
        assert st.should_speculate() is False


def test_paused_reprobe_lets_one_through(monkeypatch):
    st = _state(monkeypatch, window=2, min_rate=0.5, interval=3)
    st.record_accepted(0, 3)
    st.record_accepted(0, 3)
    assert st._spec_paused is True
    # 2 blocked steps, 3rd lets a probe through
    st.add_token(1)
    assert st.should_speculate() is False
    st.add_token(1)
    assert st.should_speculate() is False
    st.add_token(1)
    assert st.should_speculate() is True  # probe step


def test_resumes_after_probe_recovery(monkeypatch):
    st = _state(monkeypatch, window=2, min_rate=0.5, interval=1)
    st.record_accepted(0, 3)
    st.record_accepted(0, 3)
    assert st._spec_paused is True
    # 3 probe steps, all accepted -> rate 1.0 >= 0.5 -> resume
    for _ in range(3):
        st.add_token(1)
        assert st.should_speculate() is True
        st.record_accepted(3, 3)
    assert st._spec_paused is False
    assert st._paused_steps == 0
    assert st._probe_records == []


def test_stays_paused_when_probe_still_low(monkeypatch):
    st = _state(monkeypatch, window=2, min_rate=0.5, interval=1)
    st.record_accepted(0, 3)
    st.record_accepted(0, 3)
    assert st._spec_paused is True
    # 3 probe steps, all 0 accepted -> stays paused, probe records cleared
    for _ in range(3):
        st.add_token(1)
        assert st.should_speculate() is True
        st.record_accepted(0, 3)
    assert st._spec_paused is True
    assert st._probe_records == []


def test_new_request_resets_paused_state(monkeypatch):
    st = _state(monkeypatch, window=2, min_rate=0.5, interval=10)
    st.record_accepted(0, 3)
    st.record_accepted(0, 3)
    assert st._spec_paused is True
    assert st._paused_steps >= 0
    st.on_new_request("r2", [1, 2, 3])
    assert st._spec_paused is False
    assert st._paused_steps == 0
    assert st._probe_records == []


def test_two_probes_do_not_resume(monkeypatch):
    # Guard: need >=3 probe samples to resume (avoid lucky single draft).
    st = _state(monkeypatch, window=2, min_rate=0.5, interval=1)
    st.record_accepted(0, 3)
    st.record_accepted(0, 3)
    assert st._spec_paused is True
    for _ in range(2):
        st.add_token(1)
        assert st.should_speculate() is True
        st.record_accepted(3, 3)
    # only 2 probes — still paused
    assert st._spec_paused is True
