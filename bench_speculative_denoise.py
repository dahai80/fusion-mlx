#!/usr/bin/env python3
import os
import sys
import time

os.environ.setdefault("FUSION_DISABLE_COMPILE", "1")

import mlx.core as mx

sys.path.insert(0, "/Users/dahai/claude-home/fusion-mlx")
from fusion_mlx.video.skyreels_v3.scheduler.fm_solvers_unipc import perform_guidance
from fusion_mlx.video.skyreels_v3.speculative_denoise import (
    SpeculativeConfig,
    baseline_euler,
    speculative_denoise,
)
from fusion_mlx.video.skyreels_v3.transformer_r2v import SkyReelsR2VDiT
from fusion_mlx.video.skyreels_v3.weights import load_dit_weights

MODEL_PATH = "/Users/dahai/.fusion-mlx/models/Skywork/SkyReels-V3-R2V-14B-MLX"


def log(msg):
    print(f"[spec-sweep] {msg}", flush=True)


def main():
    log("=== loading real 14B R2V DiT ===")
    t0 = time.time()
    dit = SkyReelsR2VDiT()
    load_dit_weights(dit, MODEL_PATH, quantization=None)
    mx.eval(dit.parameters())
    num_layers = dit.num_layers
    log(f"loaded in {time.time()-t0:.1f}s num_layers={num_layers}")

    guidance = 5.0
    C = 16
    T = 2
    H = 16
    W = 16
    L_patch = T * (H // 2) * (W // 2)
    L_ctx = 256
    text_dim = 4096
    mx.random.seed(7)
    latents = mx.random.normal((C, T, H, W), dtype=mx.bfloat16)
    context = mx.random.normal((1, L_ctx, text_dim), dtype=mx.bfloat16) * 0.01
    seq_lens = [L_patch]
    grid_sizes = [(T, H // 2, W // 2)]
    timesteps = mx.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])

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

    def draft_velocity_factory(n_blocks):
        def _dv(x_batch, t_batch):
            x_2k, t_2k, ctx_2k, seq_2k, grid_2k = _cfg_expand(x_batch, t_batch)
            noise = dit.forward_partial(
                x_2k, t_2k, ctx_2k, seq_2k, grid_2k, n_blocks=n_blocks
            )
            return perform_guidance(noise, guidance)

        return _dv

    log("=== warmup ===")
    w = full_velocity(mx.expand_dims(latents, 0), mx.array([1.0]))
    mx.eval(w)
    w = draft_velocity_factory(num_layers)(mx.expand_dims(latents, 0), mx.array([1.0]))
    mx.eval(w)
    log("warmup done")

    log("=== baseline euler ===")
    t0 = time.time()
    base_out = baseline_euler(full_velocity, latents, timesteps)
    mx.eval(base_out)
    base_t = time.time() - t0
    log(f"baseline: {base_t:.3f}s")

    log("=" * 80)
    log(
        f"{'n_blocks':>9} {'%':>5} {'eps':>6} {'avg_acc':>8} "
        f"{'full_fw':>8} {'drft_fw':>8} {'wall_s':>8} {'speedup':>8} {'maxdiff':>10}"
    )

    configs = [
        (10, 0.1),
        (20, 0.1),
        (30, 0.1),
        (38, 0.1),
        (20, 0.3),
        (20, 0.5),
        (30, 0.3),
    ]
    for n_blocks, eps in configs:
        dv = draft_velocity_factory(n_blocks)
        cfg = SpeculativeConfig(K=3, epsilon=eps, eval_steps=True)
        t0 = time.time()
        spec_out, stats = speculative_denoise(
            full_velocity, dv, latents, timesteps, cfg
        )
        mx.eval(spec_out)
        st = time.time() - t0
        diff = float(mx.max(mx.abs(spec_out - base_out)).item())
        pct = 100.0 * n_blocks / num_layers
        log(
            f"{n_blocks:>9} {pct:>4.0f}% {eps:>6.2f} {stats.avg_accept:>8.2f} "
            f"{stats.full_forwards:>8} {stats.draft_forwards:>8} {st:>8.2f} "
            f"{base_t / st:>7.3f}x {diff:>10.5f}"
        )
        mx.eval(spec_out)
    log("=" * 80)
    log(f"baseline wall: {base_t:.3f}s (reference for speedup column)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
