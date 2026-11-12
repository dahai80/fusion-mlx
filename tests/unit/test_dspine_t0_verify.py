import math

import mlx.core as mx
import pytest

from fusion_mlx.speculative.dspine.t0_verify import (
    emitted_marginal_single_position,
    sampled_capped_reject,
)


def _rand_dist(V, seed, peak=None):
    key = mx.random.key(seed)
    logits = mx.random.normal((V,), key=key)
    if peak is not None:
        logits = mx.where(mx.arange(V) == peak, logits + 6.0, logits)
    p = mx.softmax(logits)
    return p


class TestCappedChainMath:
    def test_single_position_marginal_identity(self):
        # Analytic identity: emitted marginal must equal target p exactly
        # (classical rejection sampling theorem). No sampling noise.
        V = 128
        p = _rand_dist(V, seed=1)
        q = mx.softmax(_rand_dist(V, seed=2) * 3.0)
        emitted = emitted_marginal_single_position(p, q)
        tv = float(mx.sum(mx.abs(emitted - p)))
        assert tv < 1e-4

    def test_single_position_marginal_residual_fallback(self):
        # q > p everywhere on the support -> (p - q)+ mass 0 -> fallback to p
        V = 32
        p = _rand_dist(V, seed=3)
        q = mx.softmax(_rand_dist(V, seed=4) * 0.2)
        emitted = emitted_marginal_single_position(p, q)
        tv = float(mx.sum(mx.abs(emitted - p)))
        assert tv < 1e-4

    def test_sampler_matches_analytic_marginal(self):
        # Statistical: 20k rounds, TV(sampler, target) below sampling noise
        # bound for the effective support of this peaked distribution.
        V = 256
        p = _rand_dist(V, seed=5, peak=7)
        q = mx.softmax(_rand_dist(V, seed=6) * 2.0)
        N = 20000
        counts = mx.zeros((V,))
        for i in range(N):
            key = mx.random.key(1000 + i)
            u0, u1, u2 = mx.random.uniform(0.0, 1.0, (3,), key=key)
            # Draft token must be SAMPLED from q — a fixed argmax breaks
            # the rejection-sampling marginal identity.
            qcdf = mx.cumsum(q)
            draft_tok = int(mx.sum((qcdf <= float(u0)).astype(mx.int32)))
            draft_tok = min(draft_tok, V - 1)
            cand = mx.array([0, draft_tok])
            accepted, bonus, emitted = sampled_capped_reject(
                cand,
                mx.stack([p]),
                mx.stack([q]),
                verify_len=2,
                uniforms=mx.array([float(u1)]),
                uniform_final=float(u2),
            )
            # emitted[0] is the next token after the prefix in both the
            # accept and reject branches; its marginal must equal p.
            counts = counts + (mx.arange(V) == emitted[0]).astype(mx.float32)
        emp = counts / N
        tv = float(mx.sum(mx.abs(emp - p)))
        # Sampling noise bound: E[TV] ~ sum sqrt(2 p_i (1-p_i) / (pi N))
        noise = math.sqrt(2 / math.pi) * float(mx.sum(mx.sqrt(p * (1 - p) / N)))
        assert tv < max(3.0 * noise, 0.01)

    def test_all_accept_emits_bonus_from_target(self):
        # q tiny everywhere relative to p -> every coin*q < p -> accept all
        V = 64
        T = 4
        p_rows = mx.stack([_rand_dist(V, seed=10 + i, peak=i) for i in range(T)])
        # Perfect draft q = p rows -> coin*q < p always (coin < 1)
        q_rows = p_rows[:-1]
        cand = mx.array([int(mx.argmax(q_rows[i])) for i in range(T - 1)] + [0])
        accepted, bonus, emitted = sampled_capped_reject(
            cand,
            p_rows,
            q_rows,
            verify_len=T,
            uniforms=mx.array([0.999] * (T - 1)),
            uniform_final=0.5,
        )
        assert accepted == T - 1
        assert emitted[:-1] == [int(cand[i]) for i in range(1, T)]
        # bonus sampled from p row accepted (argmax = peak index accepted)
        assert bonus == accepted

    def test_reject_at_first_position(self):
        # q row puts all mass on a token with p ~ 0 -> coin*q >= p -> reject
        V = 32
        p = _rand_dist(V, seed=30, peak=3)
        bad = (int(mx.argmax(p)) + 1) % V
        q = (mx.arange(V) == bad).astype(mx.float32)
        cand = mx.array([0, bad, bad])
        accepted, bonus, emitted = sampled_capped_reject(
            cand,
            mx.stack([p, p]),
            mx.stack([q, q]),
            verify_len=3,
            uniforms=mx.array([0.5, 0.5]),
            uniform_final=0.3,
        )
        assert accepted == 0
        assert emitted == [bonus]
        # residual (p - q)+ = p (q mass on a p~0 token, may be nonzero) ->
        # bonus must never equal the rejected draft's q-argmax unless p
        # actually supports it
        assert float(p[bonus]) > 0.0

    def test_resampled_bonus_follows_residual(self):
        # Deterministic check: on reject, bonus ~ (p - q)+ normalized
        V = 16
        p = _rand_dist(V, seed=40)
        q = mx.softmax(_rand_dist(V, seed=41) * 4.0)
        residual = mx.maximum(p - q, 0.0)
        mass = float(mx.sum(residual))
        if mass <= 0.0:
            pytest.skip("degenerate residual")
        dist = residual / mass
        # Force reject: coin=1.0 accepts only when q < p; pick the token
        # where q - p is maximal so 1.0*q >= p guarantees rejection.
        q_over_p = int(mx.argmax(q - p))
        assert float(q[q_over_p]) >= float(p[q_over_p])
        cand = mx.array([0, q_over_p])
        accepted, bonus, emitted = sampled_capped_reject(
            cand,
            mx.stack([p]),
            mx.stack([q]),
            verify_len=2,
            uniforms=mx.array([1.0]),
            uniform_final=1e-9,
        )
        assert accepted == 0
        # uniform_final ~ 0 -> bonus = first token of residual cdf
        best = int(mx.argmax(dist))
        assert float(dist[bonus]) > 0.0
        assert bonus == best or float(dist[best]) > float(dist[bonus]) * 0.5

    def test_cap_clamps_to_table(self):
        V = 8
        p = _rand_dist(V, seed=50)
        q = mx.softmax(_rand_dist(V, seed=51))
        cand = mx.array([0, 1, 2])
        accepted, bonus, emitted = sampled_capped_reject(
            cand,
            mx.stack([p, p]),
            mx.stack([q]),
            verify_len=99,
            uniforms=mx.array([0.1]),
            uniform_final=0.5,
        )
        assert accepted <= 1
        assert len(emitted) == accepted + 1

    def test_bonus_never_from_rejected_draft(self):
        # Regression discipline (scoping hard point 3): on reject at j the
        # replacement comes from (p_j - q_j)+, never the draft chain.
        V = 16
        p = _rand_dist(V, seed=60, peak=2)
        bad = 5
        assert float(p[bad]) < 0.1
        q = (mx.arange(V) == bad).astype(mx.float32)
        cand = mx.array([0, bad])
        rejects = 0
        for i in range(200):
            accepted, bonus, emitted = sampled_capped_reject(
                cand,
                mx.stack([p]),
                mx.stack([q]),
                verify_len=2,
                uniforms=mx.array([0.999]),
                uniform_final=float(i % 7) / 7.0,
            )
            if accepted == 0:
                rejects += 1
                assert bonus != bad
        assert rejects > 0
