# N11 hybrid drafting unit tests (design:
# fusion-research/docs/n11-hybrid-drafting-1113.zh-CN.md).
# Covers: n-gram head planning + caps, no-hit regression guard, telemetry
# counters, accepted-from-ngram degradation signal, the prefix-aware
# refinement tail slicing, and lossless verification of a hybrid chain
# (one-hot q rows) through the sampled rejection verifier. CPU-only —
# drafter/tail/model are duck-typed fakes; no model weights are loaded.
import mlx.core as mx
import pytest

from fusion_mlx.scheduler.spec_decode import DSpineSpecState, _run_spec_verify_sampled
from fusion_mlx.speculative.dspine.drafter import DSpineDrafter
from fusion_mlx.speculative.dspine.hybrid import (
    HybridDraftConfig,
    HybridDraftState,
    hybrid_suffix_len,
    plan_hybrid_step,
)
from fusion_mlx.speculative.dspine.propose import RefinementTail
from fusion_mlx.speculative.dspine.t0_verify import sampled_capped_reject

V = 64


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TestHybridConfig:
    def test_defaults_are_off(self, monkeypatch):
        for k in (
            "FUSION_DSPINE_HYBRID",
            "FUSION_DSPINE_HYBRID_NGRAM_TOKENS",
            "FUSION_DSPINE_HYBRID_MIN_PREFIX",
            "FUSION_DSPINE_HYBRID_MAX_TOTAL",
            "FUSION_DSPINE_HYBRID_DEGRADE_WINDOW",
            "FUSION_DSPINE_HYBRID_DEGRADE_MIN_ACCEPT",
        ):
            monkeypatch.delenv(k, raising=False)
        cfg = HybridDraftConfig.from_env()
        assert cfg.enabled is False
        assert cfg.ngram_draft_tokens == 4
        assert cfg.min_prefix_tokens == 2
        assert cfg.max_draft_total == 8
        assert cfg.degrade_window == 32
        assert cfg.degrade_min_accept == 1.2

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("FUSION_DSPINE_HYBRID", "1")
        monkeypatch.setenv("FUSION_DSPINE_HYBRID_NGRAM_TOKENS", "6")
        monkeypatch.setenv("FUSION_DSPINE_HYBRID_MAX_TOTAL", "10")
        cfg = HybridDraftConfig.from_env()
        assert cfg.enabled is True
        assert cfg.ngram_draft_tokens == 6
        assert cfg.max_draft_total == 10

    def test_head_cap_clamped_below_total(self, monkeypatch):
        monkeypatch.setenv("FUSION_DSPINE_HYBRID_NGRAM_TOKENS", "12")
        monkeypatch.setenv("FUSION_DSPINE_HYBRID_MAX_TOTAL", "8")
        cfg = HybridDraftConfig.from_env()
        assert cfg.ngram_draft_tokens < cfg.max_draft_total

    def test_invalid_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("FUSION_DSPINE_HYBRID_NGRAM_TOKENS", "abc")
        cfg = HybridDraftConfig.from_env()
        assert cfg.ngram_draft_tokens == 4


# ---------------------------------------------------------------------------
# Hybrid state: head planning, telemetry, degradation
# ---------------------------------------------------------------------------
def _feed_repeat(state: HybridDraftState, pattern: list[int], rounds: int):
    for _ in range(rounds):
        for t in pattern:
            state.add_token(t)


class TestHybridState:
    def test_hit_builds_head_capped(self, monkeypatch):
        monkeypatch.setenv("FUSION_DSPINE_HYBRID", "1")
        monkeypatch.setenv("FUSION_DSPINE_HYBRID_NGRAM_TOKENS", "4")
        state = HybridDraftState(HybridDraftConfig.from_env())
        _feed_repeat(state, [1, 2, 3, 4, 5, 6], rounds=12)
        prefix = state.prefix_for_step()
        assert len(prefix) >= 2
        assert len(prefix) <= 4
        assert all(isinstance(t, int) for t in prefix)

    def test_no_hit_returns_empty(self, monkeypatch):
        monkeypatch.setenv("FUSION_DSPINE_HYBRID", "1")
        state = HybridDraftState(HybridDraftConfig.from_env())
        # Diverse tokens, no repetition -> no n-gram hit.
        for i in range(40):
            state.add_token(1000 + i * 7)
        assert state.prefix_for_step() == []

    def test_disabled_state_is_noop(self):
        state = HybridDraftState(HybridDraftConfig(enabled=False))
        _feed_repeat(state, [1, 2, 3, 1, 2, 3], rounds=12)
        assert state.prefix_for_step() == []
        # Predictor history untouched when disabled (zero overhead).
        assert state.predictor.get_stats()["history_len"] == 0

    def test_min_prefix_gate(self, monkeypatch):
        monkeypatch.setenv("FUSION_DSPINE_HYBRID", "1")
        state = HybridDraftState(HybridDraftConfig.from_env())
        # Stub the predictor: the gate itself is length-based.
        state.predictor.predict = lambda: [5]
        assert state.prefix_for_step() == []
        state.predictor.predict = lambda: [5, 6]
        assert state.prefix_for_step() == [5, 6]

    def test_request_reset_clears_predictor(self, monkeypatch):
        monkeypatch.setenv("FUSION_DSPINE_HYBRID", "1")
        state = HybridDraftState(HybridDraftConfig.from_env())
        _feed_repeat(state, [1, 2, 3, 4, 5, 6], rounds=12)
        state.reset_request()
        assert state.predictor.get_stats()["history_len"] == 0
        # Counters are lifecycle-cumulative and survive resets.
        assert state.total_hybrid_steps >= 0

    def test_record_round_telemetry_counters(self):
        state = HybridDraftState(HybridDraftConfig(enabled=True))
        state.record_round(4, 4, 6)
        state.record_round(4, 4, 2)
        stats = state.get_stats()
        assert stats["ngram_prefix_hit"] == 2
        assert stats["ngram_len"] == 8
        assert stats["drafter_len"] == 8
        # accepted_from_ngram = min(accepted, prefix): 4 then 2.
        assert stats["accepted_from_ngram"] == 6
        assert stats["hybrid_enabled"] is True
        assert stats["hybrid_degraded"] is False

    def test_degradation_signal(self):
        cfg = HybridDraftConfig(enabled=True, degrade_window=4, degrade_min_accept=1.2)
        state = HybridDraftState(cfg)
        # Prefix accepted length below 1.2 across the window.
        for _ in range(4):
            state.record_round(4, 4, 1)
        assert state.degraded is True
        assert state.get_stats()["hybrid_degraded"] is True
        # Degraded state plans pure DSpine only.
        prefix, n_draft = plan_hybrid_step(state, 16)
        assert prefix == []
        assert n_draft == 15

    def test_no_degradation_on_healthy_window(self):
        cfg = HybridDraftConfig(enabled=True, degrade_window=4, degrade_min_accept=1.2)
        state = HybridDraftState(cfg)
        for _ in range(4):
            state.record_round(4, 4, 4)
        assert state.degraded is False

    def test_degrade_window_rolls(self):
        cfg = HybridDraftConfig(enabled=True, degrade_window=4, degrade_min_accept=1.2)
        state = HybridDraftState(cfg)
        for _ in range(4):
            state.record_round(4, 4, 0)
        assert state.degraded is True
        assert state.get_stats()["hybrid_window_filled"] == 4


# ---------------------------------------------------------------------------
# Step planning: caps + no-hit regression guard
# ---------------------------------------------------------------------------
class TestPlanHybridStep:
    def test_no_hit_matches_pure_dspine_shape(self, monkeypatch):
        monkeypatch.setenv("FUSION_DSPINE_HYBRID", "1")
        state = HybridDraftState(HybridDraftConfig.from_env())
        for i in range(40):
            state.add_token(1000 + i * 7)
        prefix, n_draft = plan_hybrid_step(state, 16)
        assert prefix == []
        assert n_draft == 15  # max(1, block_size - 1), the pre-N11 shape

    def test_hit_respects_total_cap(self, monkeypatch):
        monkeypatch.setenv("FUSION_DSPINE_HYBRID", "1")
        monkeypatch.setenv("FUSION_DSPINE_HYBRID_NGRAM_TOKENS", "4")
        monkeypatch.setenv("FUSION_DSPINE_HYBRID_MAX_TOTAL", "8")
        state = HybridDraftState(HybridDraftConfig.from_env())
        _feed_repeat(state, [1, 2, 3, 4, 5, 6], rounds=12)
        prefix, n_draft = plan_hybrid_step(state, 16)
        assert len(prefix) >= 2
        assert len(prefix) + n_draft <= 8
        assert n_draft >= 1

    def test_suffix_len_caps(self):
        cfg = HybridDraftConfig(enabled=True, ngram_draft_tokens=4, max_draft_total=8)
        assert hybrid_suffix_len(cfg, 4, 16) == 4
        assert hybrid_suffix_len(cfg, 2, 16) == 6
        # Block capacity bound: min(max_total, block_size - 1); a b4 block
        # fits only 3 draft positions, so prefix 2 leaves 1 suffix row.
        assert hybrid_suffix_len(cfg, 2, 4) == 1
        assert hybrid_suffix_len(cfg, 4, 2) == 1

    def test_prefix_does_not_fit_block_falls_back(self, monkeypatch):
        monkeypatch.setenv("FUSION_DSPINE_HYBRID", "1")
        state = HybridDraftState(HybridDraftConfig.from_env())
        _feed_repeat(state, [1, 2, 3, 4, 5, 6], rounds=12)
        # block_size 4 -> at most 3 draft positions; a 2+ head cannot fit
        # with a suffix row, so the step must fall back to pure DSpine.
        prefix, n_draft = plan_hybrid_step(state, 4)
        if len(prefix) + 1 > 3:
            assert prefix == []
            assert n_draft == 3


# ---------------------------------------------------------------------------
# Prefix-aware refinement tail (duck-typed DSpine internals, tiny shapes)
# ---------------------------------------------------------------------------
H = 8
RANK = 2
CODE = 4


class _FakeHead:
    def __init__(self, v: int, h: int):
        self.w = mx.random.normal((v, h), key=mx.random.key(7)) * 0.5

    def matmul(self, rows):
        return rows @ self.w.T

    def rows(self, cand):
        flat = cand.reshape(-1).astype(mx.int32)
        return self.w[flat].reshape(cand.shape[0], cand.shape[1], H)


def _make_tail(v: int = V, k: int = 2) -> RefinementTail:
    class _Draft:
        pass

    class _Bridge:
        pass

    class _Norm:
        pass

    class _Gate:
        pass

    class _PGate:
        pass

    class _Prev:
        pass

    draft = _Draft()
    draft.head_topk = k
    draft.norm = _Norm()
    draft.norm.weight = mx.random.normal((H,), key=mx.random.key(1))
    draft.norm.eps = 1e-6
    bridge = _Bridge()
    bridge.rank = RANK
    bridge.code_size = CODE
    bridge.gate = _Gate()
    bridge.gate.bias = mx.zeros((RANK,))
    bridge.codebook = mx.random.normal((v, CODE), key=mx.random.key(2)) * 0.7
    bridge.predecessor_gate = _PGate()
    bridge.predecessor_gate.weight = (
        mx.random.normal((RANK, CODE), key=mx.random.key(3)) * 0.1
    )
    bridge.prev = _Prev()
    bridge.prev.weight = mx.random.normal((RANK, CODE), key=mx.random.key(4)) * 0.1
    bridge.feature_write_weights = [
        mx.random.normal((H, RANK), key=mx.random.key(5)) * 0.1
    ]
    draft.selected_prefix_bridge = bridge
    head = _FakeHead(v, H)
    tg = mx.random.normal((v, RANK), key=mx.random.key(6)) * 0.1
    tsv = mx.random.normal((v, 2 * RANK), key=mx.random.key(8)) * 0.1
    return RefinementTail(draft, head, tg, tsv)


def _make_stash(n_total: int):
    # stash rows: [anchor] + n_total draft positions.
    return {
        "raw": mx.random.normal((n_total + 1, H), key=mx.random.key(11)),
        "rms": mx.random.uniform(0.5, 1.5, (n_total + 1,), key=mx.random.key(12)),
        "packed": mx.random.normal((n_total + 1, 3 * RANK), key=mx.random.key(13)),
    }


class TestSampledHybridTail:
    def test_shapes_and_sender_anchor(self):
        tail = _make_tail()
        p = 2
        n_suffix = 3
        n_total = p + n_suffix
        rows = mx.random.normal((n_total, H), key=mx.random.key(14))
        stash = _make_stash(n_total)
        prefix = [5, 9]

        captured = {}
        orig = tail._serial_table

        def spy(r, raw, rms, q, anchor):
            captured["rows"] = r
            captured["raw"] = raw
            captured["rms"] = rms
            captured["q"] = q
            captured["anchor"] = anchor
            return orig(r, raw, rms, q, anchor)

        tail._serial_table = spy
        uniforms = mx.random.uniform(0.0, 1.0, (n_suffix,), key=mx.random.key(15))
        tokens, q_rows, cand = tail.sampled_hybrid(
            rows,
            stash,
            123,
            prefix,
            inv_temp=1.0,
            top_k=2,
            top_p=1.0,
            min_p=0.0,
            uniforms=uniforms,
        )
        # Serial chain starts from the LAST prefix token (exact_sender).
        assert captured["anchor"] == 9
        # Only the suffix rows are refined (the sender row at stash index p
        # is dropped; the sender enters via the anchor token lookup).
        assert int(captured["rows"].shape[0]) == n_suffix
        assert int(captured["raw"].shape[0]) == n_suffix
        assert int(captured["q"].shape[0]) == n_suffix
        assert int(tokens.shape[0]) == n_suffix
        assert cand.shape == (n_suffix, 2)
        assert q_rows.shape == (n_suffix, 2)

    def test_prefix_too_long_raises(self):
        tail = _make_tail()
        rows = mx.random.normal((3, H), key=mx.random.key(16))
        stash = _make_stash(3)
        with pytest.raises(ValueError):
            tail.sampled_hybrid(
                rows,
                stash,
                1,
                [2, 3, 4],
                inv_temp=1.0,
                top_k=2,
                top_p=1.0,
                min_p=0.0,
                uniforms=mx.random.uniform(0.0, 1.0, (1,), key=mx.random.key(17)),
            )


# ---------------------------------------------------------------------------
# Drafter hybrid assembly (block layout + point-mass prefix rows)
# ---------------------------------------------------------------------------
class _FakeDraftModel:
    def __init__(self, mask_token_id: int):
        self.mask_token_id = mask_token_id
        self.last_block = None

    def __call__(self, target_hidden, block, caches):
        self.last_block = [int(x) for x in block.reshape(-1).tolist()]
        n_total = block.shape[1] - 1
        out = mx.random.normal((1, n_total + 1, H), key=mx.random.key(21))
        stash = _make_stash(n_total)
        return out, stash


class _FakeTail:
    def __init__(self, k: int = 2):
        self.k = k
        self.calls = []

    def sampled_hybrid(self, rows, stash, anchor, prefix_tokens, **kw):
        # Mirror the real tail's contract: rows cover ALL draft positions;
        # the suffix is what remains after the prefix.
        n = int(rows.shape[0]) - len(prefix_tokens)
        self.calls.append({"anchor": anchor, "prefix": list(prefix_tokens), "n": n})
        cand = mx.stack([mx.array([3, 4]) for _ in range(n)]).astype(mx.int32)
        q = mx.full((n, 2), 0.5)
        tokens = mx.array([3] * n, dtype=mx.int32)
        return tokens, q, cand


def _make_drafter(mask_token_id: int = 99):
    d = DSpineDrafter.__new__(DSpineDrafter)
    d._draft = _FakeDraftModel(mask_token_id)
    d._tail = _FakeTail()
    d._caches = None
    return d


class TestDrafterHybridAssembly:
    def test_block_layout_and_aligned_outputs(self):
        drafter = _make_drafter()
        prefix = [7, 9]
        uniforms = mx.random.uniform(0.0, 1.0, (3,), key=mx.random.key(22))
        tokens, q_rows, cand_rows = drafter.draft_block_sampled_hybrid(
            mx.zeros((1, 1, 4)),
            123,
            prefix,
            None,
            inv_temp=1.0,
            top_k=2,
            top_p=1.0,
            min_p=0.0,
            uniforms=uniforms,
        )
        # Block: [anchor, *prefix, mask * n_suffix].
        assert drafter._draft.last_block == [123, 7, 9, 99, 99, 99]
        # Chain = prefix head + drafter suffix.
        assert tokens == [7, 9, 3, 3, 3]
        assert len(q_rows) == 5
        assert len(cand_rows) == 5
        # Prefix rows are point-mass proposals: one candidate, q = 1.
        for j in range(2):
            assert cand_rows[j].tolist() == [prefix[j]]
            assert float(q_rows[j][0]) == 1.0
        # Tail received the anchor + full prefix; the last prefix token
        # becomes the first serial sender inside sampled_hybrid (covered
        # by TestSampledHybridTail).
        assert drafter._tail.calls[0]["anchor"] == 123
        assert drafter._tail.calls[0]["prefix"] == [7, 9]
        assert drafter._tail.calls[0]["n"] == 3

    def test_empty_prefix_raises(self):
        drafter = _make_drafter()
        with pytest.raises(ValueError):
            drafter.draft_block_sampled_hybrid(
                mx.zeros((1, 1, 4)),
                1,
                [],
                None,
                inv_temp=1.0,
                top_k=2,
                top_p=1.0,
                min_p=0.0,
                uniforms=mx.random.uniform(0.0, 1.0, (2,), key=mx.random.key(23)),
            )


# ---------------------------------------------------------------------------
# Lossless verification of a hybrid chain through the sampled verifier
# ---------------------------------------------------------------------------
class _FakeModel:
    def __init__(self, logits_rows):
        self.logits_rows = logits_rows

    def __call__(self, tokens, cache=None):
        k = int(tokens.shape[1])
        return mx.stack(self.logits_rows[:k])[None]


def _target_row(seed: int, peak: int) -> mx.array:
    logits = mx.random.normal((V,), key=mx.random.key(seed))
    # +12 keeps the peak mass above 0.999 so a 0.999 acceptance coin
    # clears min(1, p/q) deterministically on the agree-path tests.
    logits = mx.where(mx.arange(V) == peak, logits + 12.0, logits)
    return mx.softmax(logits.astype(mx.float32))


class TestHybridChainVerification:
    def test_full_chain_accepted_when_target_agrees(self):
        # Target peaks exactly on the n-gram head tokens and the suffix.
        p0 = _target_row(31, 7)
        p1 = _target_row(32, 9)
        p2 = _target_row(33, 11)
        p3 = _target_row(34, 13)
        draft_tokens = [7, 9, 11, 13]
        model = _FakeModel([p1 * 1e6, p2 * 1e6, p3 * 1e6, p1 * 1e6])
        q_rows = [
            mx.array([1.0]),
            mx.array([1.0]),
            p2,
            p3,
        ]
        cand_rows = [
            mx.array([7], dtype=mx.int32),
            mx.array([9], dtype=mx.int32),
            mx.arange(V, dtype=mx.int32),
            mx.arange(V, dtype=mx.int32),
        ]
        verified, n_accepted, processed = _run_spec_verify_sampled(
            model,
            draft_tokens,
            q_rows,
            cand_rows,
            mx.log(mx.maximum(p0, 1e-30)),
            prompt_cache=[],
            inv_temp=1.0,
            top_k=V,
            top_p=1.0,
            min_p=0.0,
            accept_uniforms=mx.array([0.9, 0.9, 0.9, 0.9]),
            final_uniform=0.5,
        )
        assert n_accepted == 4
        assert processed == 4
        assert verified[:4] == draft_tokens
        assert len(verified) == 5

    def test_ngram_head_rejection_resamples_from_target(self):
        # The n-gram head token has ~zero target probability -> the point
        # mass q row forces rejection and the emitted token is resampled
        # from the target row (residual fallback).
        p0 = _target_row(41, 5)
        bad = (5 + 1) % V
        p1 = _target_row(42, 6)
        model = _FakeModel([p1 * 1e6, p1 * 1e6])
        verified, n_accepted, processed = _run_spec_verify_sampled(
            model,
            [bad, 6],
            [mx.array([1.0]), mx.array([1.0])],
            [
                mx.array([bad], dtype=mx.int32),
                mx.array([6], dtype=mx.int32),
            ],
            mx.log(mx.maximum(p0, 1e-30)),
            prompt_cache=[],
            inv_temp=1.0,
            top_k=V,
            top_p=1.0,
            min_p=0.0,
            accept_uniforms=mx.array([1.0, 1.0]),
            final_uniform=0.5,
        )
        assert n_accepted == 0
        assert processed == 2
        assert len(verified) == 1
        assert verified[0] != bad

    def test_point_mass_position_marginal_is_lossless(self):
        # Monte Carlo: a one-hot q row must reproduce the target row
        # exactly at that position — accept w.p. p[g], rejection resamples
        # from (p - q)+, so P(emitted = x) == p[x] for every x.
        p = _target_row(51, 9)
        g = 9
        trials = 4000
        counts = [0] * V
        onehot_g = (mx.arange(V) == g).astype(mx.float32)
        zero_row = mx.zeros((V,))
        anchor_row = zero_row
        for i in range(trials):
            key = mx.random.key(1000 + i)
            u1, u2 = mx.random.uniform(0.0, 1.0, (2,), key=key).tolist()
            cand = mx.array([0, g], dtype=mx.int32)
            target = mx.stack([p, anchor_row])
            draft = mx.stack([onehot_g, zero_row])
            _acc, _bonus, emitted = sampled_capped_reject(
                cand,
                target,
                draft,
                verify_len=2,
                uniforms=mx.array([u1]),
                uniform_final=u2,
            )
            counts[int(emitted[0])] += 1
        for x in range(V):
            expected = float(p[x])
            observed = counts[x] / trials
            assert (
                abs(observed - expected) < 0.02
            ), f"token {x}: observed {observed:.4f} vs target {expected:.4f}"

    def test_partial_accept_counts_ngram_prefix(self):
        # accepted_from_ngram semantics: rejection sampling accepts a
        # prefix, so accepted positions map onto the head first.
        p0 = _target_row(61, 3)
        p1 = _target_row(62, 4)
        p2 = _target_row(63, 5)
        bad = (4 + 1) % V
        model = _FakeModel([p1 * 1e6, p2 * 1e6, p2 * 1e6])
        verified, n_accepted, _ = _run_spec_verify_sampled(
            model,
            [3, bad, 5],
            [mx.array([1.0]), mx.array([1.0]), mx.array([1.0])],
            [
                mx.array([3], dtype=mx.int32),
                mx.array([bad], dtype=mx.int32),
                mx.array([5], dtype=mx.int32),
            ],
            mx.log(mx.maximum(p0, 1e-30)),
            prompt_cache=[],
            inv_temp=1.0,
            top_k=V,
            top_p=1.0,
            min_p=0.0,
            accept_uniforms=mx.array([0.0, 1.0, 0.0]),
            final_uniform=0.5,
        )
        assert n_accepted == 1
        assert verified[0] == 3
        state = HybridDraftState(HybridDraftConfig(enabled=True))
        state.record_round(2, 1, n_accepted)
        assert state.get_stats()["accepted_from_ngram"] == 1


# ---------------------------------------------------------------------------
# Scheduler state wiring
# ---------------------------------------------------------------------------
class _FakeRuntime:
    def __init__(self):
        self.block_size = 16
        self.drafter = None

    def record_accept(self, avg_accept):
        pass


class TestDSpineSpecStateHybridWiring:
    def test_stats_export_hybrid_keys(self):
        state = DSpineSpecState(_FakeRuntime())
        stats = state.get_stats()
        for key in (
            "hybrid_enabled",
            "hybrid_degraded",
            "ngram_prefix_hit",
            "ngram_len",
            "drafter_len",
            "accepted_from_ngram",
        ):
            assert key in stats

    def test_add_token_feeds_predictor_when_enabled(self, monkeypatch):
        monkeypatch.setenv("FUSION_DSPINE_HYBRID", "1")
        state = DSpineSpecState(_FakeRuntime())
        for _ in range(12):
            for t in (1, 2, 3, 4, 5, 6):
                state.add_token(t)
        assert state._hybrid.prefix_for_step()

    def test_reset_request_resets_predictor(self, monkeypatch):
        monkeypatch.setenv("FUSION_DSPINE_HYBRID", "1")
        state = DSpineSpecState(_FakeRuntime())
        for t in (1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6):
            state.add_token(t)
        state.on_new_request("req-b")
        assert state._hybrid.predictor.get_stats()["history_len"] == 0

    def test_disabled_by_default_no_predictor_growth(self):
        state = DSpineSpecState(_FakeRuntime())
        for t in range(100):
            state.add_token(t)
        assert state._hybrid.predictor.get_stats()["history_len"] == 0
        assert state.get_stats()["hybrid_enabled"] is False
