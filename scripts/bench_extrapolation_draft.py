#!/usr/bin/env python3
# #177 Phase-4 E2E: ExtrapolationDraft acceptance + speedup on real 14B DiT.
#
# Tests zero-cost velocity extrapolation (linear + quadratic) against baseline
# Euler on real SkyReels R2V-14B weights. Measures acceptance rate and wall-clock
# speedup across multiple epsilon/K/configurations.

import argparse
import json
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

MODEL_PATH = os.environ.get(
    "BENCH_MODEL",
    "/Users/dahai/.fusion-mlx/models/Skywork/SkyReels-V3-R2V-14B-MLX",
)


def log(msg):
    print(f"[extrap-bench] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="#177 Phase-4 ExtrapolationDraft E2E bench")
    parser.add_argument("--model", default=MODEL_PATH, help="model path")
    parser.add_argument("--steps", type=int, default=6, help="denoise timesteps")
    parser.add_argument("--K", type=int, default=4, help="speculative lookahead K")
    parser.add_argument("--quant", default=None, choices=["w8a16", "w4"], help="DiT quantization")
    args = parser.parse_args()

    log("=== loading real 14B R2V DiT ===")
    t0 = time.time()
    dit = SkyReelsR2VDiT()
    quant_cfg = None
    if args.quant == "w8a16":
        quant_cfg = {"bits": 8, "group_size": 64}
    elif args.quant == "w4":
        quant_cfg = {"bits": 4, "group_size": 64}
    load_dit_weights(dit, args.model, quantization=quant_cfg)
    mx.eval(dit.parameters())
    num_layers = dit.num_layers
    log(f"loaded in {time.time()-t0:.1f}s num_layers={num_layers} quant={args.quant}")

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

    ts_vals = [1.0 - i / args.steps for i in range(args.steps + 1)]
    timesteps = mx.array(ts_vals)
    log(f"timesteps: {ts_vals}")

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

    log("=== warmup (2x full_velocity) ===")
    for _ in range(2):
        w = full_velocity(mx.expand_dims(latents, 0), mx.array([1.0]))
        mx.eval(w)
    log("warmup done")

    # baseline euler
    log("=== baseline euler ===")
    mx.random.seed(7)
    latents_b = mx.random.normal((C, T, H, W), dtype=mx.bfloat16)
    t0 = time.time()
    base_out = baseline_euler(full_velocity, latents_b, timesteps)
    mx.eval(base_out)
    base_t = time.time() - t0
    log(f"baseline: {base_t:.3f}s ({args.steps} steps)")

    results = []

    # extrapolation sweep: linear + quadratic x (K, epsilon) grid
    log("=" * 90)
    log(f"{'strategy':>14} {'mode':>8} {'K':>3} {'eps':>6} {'avg_acc':>8} "
        f"{'full_fw':>8} {'drft_fw':>8} {'macro':>6} {'wall_s':>8} {'speedup':>8} {'maxdiff':>10}")
    log("-" * 90)

    sweep = [
        ("extrapolation", "linear", 2, 0.1),
        ("extrapolation", "linear", 3, 0.1),
        ("extrapolation", "linear", 4, 0.1),
        ("extrapolation", "linear", 2, 0.3),
        ("extrapolation", "linear", 3, 0.3),
        ("extrapolation", "linear", 4, 0.3),
        ("extrapolation", "linear", 2, 0.5),
        ("extrapolation", "linear", 3, 0.5),
        ("extrapolation", "linear", 4, 0.5),
        ("extrapolation", "quadratic", 2, 0.1),
        ("extrapolation", "quadratic", 3, 0.1),
        ("extrapolation", "quadratic", 4, 0.1),
        ("extrapolation", "quadratic", 2, 0.3),
        ("extrapolation", "quadratic", 3, 0.3),
        ("extrapolation", "quadratic", 4, 0.3),
        ("extrapolation", "quadratic", 2, 0.5),
        ("extrapolation", "quadratic", 3, 0.5),
        ("extrapolation", "quadratic", 4, 0.5),
    ]

    for strategy, mode, K, eps in sweep:
        mx.random.seed(7)
        latents_r = mx.random.normal((C, T, H, W), dtype=mx.bfloat16)
        cfg = SpeculativeConfig(
            K=K,
            epsilon=eps,
            eval_steps=True,
            draft_strategy=strategy,
        )
        os.environ["FUSION_SPEC_DRAFT_STRATEGY"] = strategy
        os.environ["FUSION_SPEC_EXTRAP_MODE"] = mode
        t0 = time.time()
        spec_out, stats = speculative_denoise(
            full_velocity, None, latents_r, timesteps, cfg
        )
        mx.eval(spec_out)
        st = time.time() - t0
        diff = float(mx.max(mx.abs(spec_out - base_out)).item())
        speedup = base_t / st if st > 0 else 0.0
        row = {
            "strategy": strategy,
            "mode": mode,
            "K": K,
            "epsilon": eps,
            "avg_accept": stats.avg_accept,
            "full_forwards": stats.full_forwards,
            "draft_forwards": stats.draft_forwards,
            "macro_steps": stats.macro_steps,
            "accepted": stats.accepted,
            "wall_s": round(st, 3),
            "speedup": round(speedup, 3),
            "maxdiff": round(diff, 5),
        }
        results.append(row)
        log(
            f"{strategy:>14} {mode:>8} {K:>3} {eps:>6.2f} {stats.avg_accept:>8.2f} "
            f"{stats.full_forwards:>8} {stats.draft_forwards:>8} {stats.macro_steps:>6} "
            f"{st:>8.2f} {speedup:>7.3f}x {diff:>10.5f}"
        )

    log("=" * 90)
    log(f"baseline wall: {base_t:.3f}s ({args.steps} steps)")

    # summary
    best = max(results, key=lambda r: r["speedup"])
    log(f"\nBEST: strategy={best['strategy']} mode={best['mode']} K={best['K']} "
        f"eps={best['epsilon']} speedup={best['speedup']:.3f}x avg_accept={best['avg_accept']:.2f}")

    # save JSON
    out_path = os.path.join(os.path.dirname(__file__), "bench_extrapolation_draft_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "baseline_wall_s": round(base_t, 3),
            "steps": args.steps,
            "K": args.K,
            "model": os.path.basename(args.model),
            "quant": args.quant,
            "results": results,
        }, f, indent=2)
    log(f"results saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
