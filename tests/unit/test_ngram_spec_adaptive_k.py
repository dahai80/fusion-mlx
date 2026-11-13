# SPDX-License-Identifier: Apache-2.0
"""Adaptive draft length (K) for n-gram speculative decode.

The suffix drafter is built with the K_MAX ceiling; the controller trims
proposals to a dynamic _current_k that follows the rolling acceptance.
Verify is exact greedy rejection, so K never changes output identity —
these tests lock the K dynamics only:
  - base K + proposal trimming
  - upshift after a strong rolling window (capped at the drafter ceiling)
  - downshift in the mediocre band (no pause)
  - pause resets K to base so the resume probe re-measures conservatively
  - break-even reads the EMA of actual proposals, not the static config
  - env kill-switch keeps K constant and proposals untrimmed
"""

import fusion_mlx.scheduler.ngram_spec as ns


class _StubPredictor:
    def __init__(self, num_draft):
        self.num_draft = num_draft
        self._next = list(range(1, num_draft + 1))

    def predict(self):
        return list(self._next)

    def predict_top1(self):
        return self._next[0] if self._next else None

    def record_accepted(self, n_accepted):
        pass

    def get_stats(self):
        return {}


def _state(num_draft=16):
    return ns.NGramSpecState(predictor=_StubPredictor(num_draft))


def test_base_k_and_proposal_trimming(monkeypatch):
    monkeypatch.setattr(ns, "NGRAM_SPEC_ADAPTIVE_K", True)
    st = _state(num_draft=16)
    assert st._base_k == 3
    assert st._current_k == 3
    assert st.get_drafts() == [1, 2, 3]


def test_upshift_after_strong_window(monkeypatch):
    monkeypatch.setattr(ns, "NGRAM_SPEC_ADAPTIVE_K", True)
    st = _state(num_draft=16)
    for _ in range(12):
        st.record_result(3, 3, verify_dt=0.02, decode_dt=0.01)
    assert st._current_k == 9
    assert len(st.get_drafts()) == 9
    for _ in range(12):
        st.record_result(9, 9, verify_dt=0.02, decode_dt=0.01)
    assert st._current_k == 16
    assert len(st.get_drafts()) == 16


def test_downshift_in_mediocre_band_without_pause(monkeypatch):
    monkeypatch.setattr(ns, "NGRAM_SPEC_ADAPTIVE_K", True)
    st = _state(num_draft=16)
    st._current_k = 9
    for _ in range(12):
        st.record_result(4, 9, verify_dt=0.012, decode_dt=0.01)
    assert not st._paused
    assert st._current_k == 4


def test_pause_resets_k_to_base(monkeypatch):
    monkeypatch.setattr(ns, "NGRAM_SPEC_ADAPTIVE_K", True)
    st = _state(num_draft=16)
    st._current_k = 16
    for _ in range(12):
        st.record_result(1, 16, verify_dt=0.05, decode_dt=0.01)
    assert st._paused
    assert st._current_k == st._base_k == 3


def test_break_even_uses_actual_k_ema(monkeypatch):
    monkeypatch.setattr(ns, "NGRAM_SPEC_ADAPTIVE_K", True)
    st = _state(num_draft=16)
    for _ in range(20):
        st.record_result(6, 12, verify_dt=0.03, decode_dt=0.01)
    assert abs(st._k_ema - 12.0) < 0.5
    expected = (3.0 - 1.0) / 12
    assert abs(st._break_even() - expected) < 0.02


def test_adaptive_disabled_keeps_k_constant(monkeypatch):
    monkeypatch.setattr(ns, "NGRAM_SPEC_ADAPTIVE_K", False)
    st = _state(num_draft=16)
    assert st.get_drafts() == list(range(1, 17))
    for _ in range(12):
        st.record_result(3, 3, verify_dt=0.02, decode_dt=0.01)
    assert st._current_k == st._base_k
