# SPDX-License-Identifier: Apache-2.0
# Eagle3 adaptive skip: when recent acceptance is below break-even,
# generate_draft_tokens returns [] (no GPU forward wasted). Re-probes
# periodically to detect recovery. These tests pin that contract
# without a real model (the speculator's draft model is stubbed).

from __future__ import annotations

from fusion_mlx.speculative.eagle3 import speculator as e3


class _StubModel:
    # Minimal stand-in: forward_standalone returns logits whose argmax
    # is always 0 — drafts will be [0, 0, 0, ...] (content doesn't
    # matter; the test drives acceptance via record_accepted directly).
    def forward_standalone(self, input_ids, cache=None, hidden_state=None):
        import mlx.core as mx

        return mx.zeros((1, 1, 32000))


def _spec(monkeypatch, *, window=4, break_even=0.15, probe=4, num_draft=3):
    monkeypatch.setattr(e3, "EAGLE3_ADAPTIVE_WINDOW", window)
    monkeypatch.setattr(e3, "EAGLE3_ADAPTIVE_BREAK_EVEN", break_even)
    monkeypatch.setattr(e3, "EAGLE3_ADAPTIVE_PROBE_INTERVAL", probe)
    cfg = e3.Eagle3DraftConfig(num_draft=num_draft, temperature=0.0)
    s = e3.Eagle3Speculator(config=cfg)
    s.model = _StubModel()
    s._loaded = True
    # Stub cache: a bare object with offset so forward_standalone runs.
    s._draft_cache = type("C", (), {"offset": 0})()
    # Stub hidden capture so _get_decode_hidden returns a dummy.
    s._get_decode_hidden = lambda: __import__("mlx.core", fromlist=["core"]).zeros(
        (1, 1, 4096)
    )
    return s


def test_skips_after_low_acceptance_window(monkeypatch):
    s = _spec(monkeypatch, window=4, break_even=0.15)
    # Fill window with 0% acceptance → should pause.
    for _ in range(4):
        drafts = s.generate_draft_tokens(1)
        assert len(drafts) == 3  # still drafting until window fills
        s.record_accepted(0)
    # Window now full, rate=0% < 15% → next call skips.
    assert s._adaptive_paused is True
    drafts = s.generate_draft_tokens(1)
    assert drafts == []


def test_does_not_pause_below_window(monkeypatch):
    s = _spec(monkeypatch, window=4, break_even=0.15)
    # Only 3 records (< window=4) → not enough data to pause.
    for _ in range(3):
        s.generate_draft_tokens(1)
        s.record_accepted(0)
    assert s._adaptive_paused is False


def test_reprobe_lets_draft_through(monkeypatch):
    s = _spec(monkeypatch, window=4, break_even=0.15, probe=2)
    for _ in range(4):
        s.generate_draft_tokens(1)
        s.record_accepted(0)
    assert s._adaptive_paused is True
    # skip 1: return []
    assert s.generate_draft_tokens(1) == []
    # skip 2 (probe interval): should generate drafts
    drafts = s.generate_draft_tokens(1)
    assert len(drafts) == 3


def test_resumes_after_probe_recovery(monkeypatch):
    s = _spec(monkeypatch, window=4, break_even=0.15, probe=2)
    for _ in range(4):
        s.generate_draft_tokens(1)
        s.record_accepted(0)
    assert s._adaptive_paused is True
    # Probe: generate + record high acceptance.
    s.generate_draft_tokens(1)
    s.record_accepted(3)  # 100% acceptance
    # Fill window with good acceptance to flush stale records.
    for _ in range(4):
        s.generate_draft_tokens(1)
        s.record_accepted(3)
    assert s._adaptive_paused is False


def test_new_request_resets_adaptive(monkeypatch):
    s = _spec(monkeypatch, window=4, break_even=0.15)
    for _ in range(4):
        s.generate_draft_tokens(1)
        s.record_accepted(0)
    assert s._adaptive_paused is True
    s.on_new_request("req2", [1, 2, 3])
    assert s._adaptive_paused is False
    assert len(s._recent_accept) == 0


def test_stats_include_adaptive_state(monkeypatch):
    s = _spec(monkeypatch)
    stats = s.get_stats()
    assert "adaptive_paused" in stats
    assert "adaptive_skip_count" in stats
