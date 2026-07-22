import logging
import os

os.environ.setdefault("FUSION_DISABLE_COMPILE", "1")

import mlx.core as mx

from fusion_mlx.video.skyreels_v3.scheduler.fm_solvers_unipc import perform_guidance
from fusion_mlx.video.skyreels_v3.speculative_denoise import (
    SpeculativeConfig,
    baseline_euler,
    speculative_denoise,
)
from fusion_mlx.video.skyreels_v3.transformer_r2v import SkyReelsR2VDiT

logger = logging.getLogger(__name__)

TINY_CFG = {
    "dim": 64,
    "ffn_dim": 128,
    "num_heads": 4,
    "num_kv_heads": 4,
    "num_layers": 4,
    "patch_size": (1, 2, 2),
    "in_dim": 16,
    "out_dim": 16,
    "text_dim": 64,
    "text_len": 32,
    "freq_dim": 32,
    "window_size": (-1, -1),
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,
}

H = W = 8
T = 1
C = 16
L_PATCH = T * (H // 2) * (W // 2)
L_CTX = 8


def _build_dit():
    mx.random.seed(7)
    return SkyReelsR2VDiT(dict(TINY_CFG))


def _inputs(b):
    x = mx.random.normal((b, C, T, H, W))
    t = mx.array([0.5 + 0.1 * i for i in range(b)])
    context = mx.random.normal((b, L_CTX, 64))
    seq_lens = [L_PATCH] * b
    grid_sizes = [(T, H // 2, W // 2)] * b
    return x, t, context, seq_lens, grid_sizes


def _allclose(a, b, atol=1e-4):
    return float(mx.max(mx.abs(a - b)).item()) < atol


def test_r2v_forward_partial_bit_identical_b1():
    dit = _build_dit()
    x, t, context, seq_lens, grid_sizes = _inputs(1)
    full = dit(x, t, context, seq_lens, grid_sizes)
    part = dit.forward_partial(
        x, t, context, seq_lens, grid_sizes, n_blocks=dit.num_layers
    )
    mx.eval(full)
    mx.eval(part)
    assert mx.array_equal(full, part).item()


def test_r2v_forward_partial_bit_identical_b2():
    dit = _build_dit()
    x, t, context, seq_lens, grid_sizes = _inputs(2)
    full = dit(x, t, context, seq_lens, grid_sizes)
    part = dit.forward_partial(
        x, t, context, seq_lens, grid_sizes, n_blocks=dit.num_layers
    )
    mx.eval(full)
    mx.eval(part)
    assert mx.array_equal(full, part).item()


def test_r2v_forward_runs_multibatch():
    for b in (1, 2, 4):
        dit = _build_dit()
        x, t, context, seq_lens, grid_sizes = _inputs(b)
        out = dit(x, t, context, seq_lens, grid_sizes)
        mx.eval(out)
        assert tuple(out.shape) == (b, C, T, H, W)


def test_r2v_draft_runs_fewer_blocks():
    dit = _build_dit()
    x, t, context, seq_lens, grid_sizes = _inputs(2)
    full = dit(x, t, context, seq_lens, grid_sizes)
    draft = dit.forward_partial(x, t, context, seq_lens, grid_sizes, n_blocks=2)
    mx.eval(full)
    mx.eval(draft)
    assert tuple(draft.shape) == (2, C, T, H, W)
    assert not _allclose(full, draft, atol=1e-6)


def test_r2v_spec_correctness_draft_equals_full():
    dit = _build_dit()
    num_layers = dit.num_layers
    guidance = 5.0
    x, t, context, seq_lens, grid_sizes = _inputs(1)
    latents = x[0]
    timesteps = mx.array([1.0, 0.75, 0.5, 0.25, 0.0])

    def _cfg_expand(x_batch, t_batch):
        k = x_batch.shape[0]
        x_2k = mx.concatenate([x_batch, x_batch], axis=0)
        t_2k = mx.concatenate([t_batch, t_batch], axis=0)
        ctx_2k = mx.concatenate([context] * (2 * k), axis=0)
        seq_2k = list(seq_lens) * (2 * k)
        grid_2k = list(grid_sizes) * (2 * k)
        return x_2k, t_2k, ctx_2k, seq_2k, grid_2k

    def full_velocity(x_batch, t_batch):
        x_2k, t_2k, ctx_2k, seq_2k, grid_2k = _cfg_expand(x_batch, t_batch)
        noise = dit(x_2k, t_2k, ctx_2k, seq_2k, grid_2k)
        return perform_guidance(noise, guidance)

    def draft_velocity(x_batch, t_batch):
        x_2k, t_2k, ctx_2k, seq_2k, grid_2k = _cfg_expand(x_batch, t_batch)
        noise = dit.forward_partial(
            x_2k, t_2k, ctx_2k, seq_2k, grid_2k, n_blocks=num_layers
        )
        return perform_guidance(noise, guidance)

    config = SpeculativeConfig(K=3, epsilon=0.1, eval_steps=True)
    spec_out, stats = speculative_denoise(
        full_velocity, draft_velocity, latents, timesteps, config
    )
    base_out = baseline_euler(full_velocity, latents, timesteps)
    mx.eval(spec_out)
    mx.eval(base_out)
    logger.info(
        "spec invariant: macro=%d accepted=%s full_fwds=%d draft_fwds=%d",
        stats.macro_steps,
        stats.accepted,
        stats.full_forwards,
        stats.draft_forwards,
    )
    assert _allclose(spec_out, base_out, atol=1e-4)
    assert sum(stats.accepted) == len(timesteps) - 1
