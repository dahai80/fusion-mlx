import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from fusion_mlx.video.skyreels_v3.speculative_denoise import (
    DraftDiTMixin,
    LayerPrunedDraft,
    SpecStats,
    SpeculativeConfig,
    baseline_euler,
    speculative_denoise,
    speculative_enabled,
)


def _t_batch(t_batch, ndim):
    return mx.array(t_batch).reshape((-1,) + (1,) * (ndim - 1))


def full_velocity(x_batch, t_batch):
    t = _t_batch(t_batch, x_batch.ndim).astype(x_batch.dtype)
    return -x_batch + 0.3 * t


def make_draft_diverge(diverge_below):
    def draft(x_batch, t_batch):
        v = full_velocity(x_batch, t_batch)
        t = _t_batch(t_batch, x_batch.ndim).astype(v.dtype)
        mask = (t <= diverge_below).astype(v.dtype)
        return v + mask * 5.0

    return draft


def make_schedule(N, t0=1.0, t1=0.0):
    return mx.array(np.linspace(t0, t1, N).astype(np.float32))


def make_latents(C=2, F=3, H=4, W=4, seed=7):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((C, F, H, W)).astype(np.float32))


def _allclose(a, b, atol=1e-4):
    return float(mx.max(mx.abs(mx.array(a) - mx.array(b)))) < atol


def test_speculative_accepts_all_when_draft_equals_full():
    N, K = 8, 4
    timesteps = make_schedule(N)
    latents = make_latents()
    cfg = SpeculativeConfig(K=K, epsilon=0.1)
    out, stats = speculative_denoise(
        full_velocity, full_velocity, latents, timesteps, cfg
    )
    base = baseline_euler(full_velocity, latents, timesteps)
    assert _allclose(out, base)
    assert stats.macro_steps == 2
    assert stats.accepted == [4, 3]
    assert stats.full_forwards == 2
    assert stats.speedup == pytest.approx((N - 1) / 2.0)


def test_speculative_matches_baseline_under_divergence():
    N, K = 10, 4
    timesteps = make_schedule(N)
    latents = make_latents()
    cfg = SpeculativeConfig(K=K, epsilon=0.1)
    draft = make_draft_diverge(diverge_below=0.6)
    out, stats = speculative_denoise(full_velocity, draft, latents, timesteps, cfg)
    base = baseline_euler(full_velocity, latents, timesteps)
    assert _allclose(out, base)
    assert stats.macro_steps >= 1
    assert any(a < K for a in stats.accepted), "expected at least one divergence"
    assert all(a <= K for a in stats.accepted)


def test_speculative_zero_accept_still_advances_one_step():
    N, K = 6, 4
    timesteps = make_schedule(N)
    latents = make_latents()
    cfg = SpeculativeConfig(K=K, epsilon=1e-6)
    draft = make_draft_diverge(diverge_below=2.0)
    out, stats = speculative_denoise(full_velocity, draft, latents, timesteps, cfg)
    base = baseline_euler(full_velocity, latents, timesteps)
    assert _allclose(out, base)
    assert all(a == 0 for a in stats.accepted), "every macro-step should diverge at j=0"
    assert stats.macro_steps == N - 1
    assert stats.full_forwards == N - 1


def test_batched_verify_matches_sequential():
    timesteps = make_schedule(5)
    latents = make_latents()
    xs = [latents + 0.1 * float(t) for t in timesteps[:-1]]
    x_batch = mx.stack(xs, axis=0)
    t_batch = mx.array([float(t) for t in timesteps[:-1]])
    batched = full_velocity(x_batch, t_batch)
    for j, xj in enumerate(xs):
        single = full_velocity(mx.expand_dims(xj, 0), mx.array([float(timesteps[j])]))[
            0
        ]
        assert _allclose(batched[j], single)


def test_speculative_clamps_K_at_end():
    N, K = 3, 8
    timesteps = make_schedule(N)
    latents = make_latents()
    cfg = SpeculativeConfig(K=K, epsilon=0.1)
    out, stats = speculative_denoise(
        full_velocity, full_velocity, latents, timesteps, cfg
    )
    base = baseline_euler(full_velocity, latents, timesteps)
    assert _allclose(out, base)
    assert stats.accepted == [2]
    assert stats.full_forwards == 1


def test_speculative_too_few_timesteps_returns_input():
    latents = make_latents()
    timesteps = mx.array([1.0])
    out, stats = speculative_denoise(full_velocity, full_velocity, latents, timesteps)
    assert _allclose(out, latents)
    assert stats.macro_steps == 0


def test_speculative_disabled_flag(monkeypatch):
    monkeypatch.delenv("FUSION_SPECULATIVE_DENOISE", raising=False)
    assert speculative_enabled() is False
    monkeypatch.setenv("FUSION_SPECULATIVE_DENOISE", "1")
    assert speculative_enabled() is True
    monkeypatch.setenv("FUSION_SPECULATIVE_DENOISE", "false")
    assert speculative_enabled() is False


def test_specstats_speedup():
    s = SpecStats(baseline_steps=10, full_forwards=4)
    assert s.speedup == pytest.approx(2.5)
    s2 = SpecStats()
    assert s2.speedup == 0.0
    assert s2.avg_accept == 0.0


class TinyDiT(nn.Module, DraftDiTMixin):
    def __init__(self, dim=4, n_blocks=4, seed=3):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.n_blocks = n_blocks
        self.blocks = [nn.Linear(dim, dim) for _ in range(n_blocks)]
        self.head = nn.Linear(dim, dim)
        for m in self.blocks + [self.head]:
            m.weight = mx.array(
                rng.standard_normal(m.weight.shape).astype(np.float32) * 0.1
            )
            m.bias = mx.zeros(m.bias.shape)

    def _run(self, x, n_blocks):
        for b in self.blocks[:n_blocks]:
            x = x + b(x)
        return x

    def forward_full(self, x_batch, t_batch):
        t = _t_batch(t_batch, x_batch.ndim).astype(x_batch.dtype)
        return self.head(self._run(x_batch, self.n_blocks)) * t

    def forward_partial(self, x_batch, t_batch, n_blocks, **kwargs):
        t = _t_batch(t_batch, x_batch.ndim).astype(x_batch.dtype)
        return self.head(self._run(x_batch, n_blocks)) * t


def test_layer_pruned_draft_runs_fewer_blocks():
    dit = TinyDiT(dim=4, n_blocks=4)
    x = mx.array(np.random.default_rng(5).standard_normal((1, 4)).astype(np.float32))
    t = mx.array([0.5])
    draft2 = LayerPrunedDraft(dit, n_blocks=2)
    v2 = draft2(x, t)
    v2_manual = dit.forward_partial(x, t, n_blocks=2)
    v4 = dit.forward_partial(x, t, n_blocks=4)
    assert _allclose(v2, v2_manual)
    assert not _allclose(v2, v4)


def test_speculative_with_layer_pruned_draft_equals_full_matches_baseline():
    dit = TinyDiT(dim=4, n_blocks=4)
    N, K = 6, 3
    timesteps = make_schedule(N)
    latents = mx.array(
        np.random.default_rng(1).standard_normal((1, 4)).astype(np.float32)
    )
    full = dit.forward_full
    draft_full = LayerPrunedDraft(dit, n_blocks=4)
    cfg = SpeculativeConfig(K=K, epsilon=0.1)
    out, stats = speculative_denoise(full, draft_full, latents, timesteps, cfg)
    base = baseline_euler(full, latents, timesteps)
    assert _allclose(out, base, atol=1e-3)
    assert stats.accepted == [3, 2]
