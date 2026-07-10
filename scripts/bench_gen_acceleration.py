#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark diffusion-acceleration knobs (step count / compile / quantize).

文生图 (Flux) and 文生视频 (LTX-2 / Wan2) are latent-diffusion samplers: their
wall-time is ~linear in num_inference_steps (each denoise step does equal
work), plus a fixed overhead (model load, VAE encode/decode, IO). So reducing
steps is the dominant speed lever and the speedup is *computable* up front:

    expected_speedup_on_loop = baseline_steps / reduced_steps
    expected_wall_speedup    = (k * baseline_steps + c) / (k * reduced_steps + c)

where k = per-step cost, c = fixed overhead. For aggressive reduction
(baseline 40 -> reduced 10) the loop-speedup is 4x, which exceeds the +100%
(2x) target on the denoise portion. Real measured wall-clock depends on c.

Two modes:
  --explain   Print the deterministic expected-speedup table (no weights, no
              compute). Always runnable. This is the honest, computable answer
              to "how much faster" - NOT a fabricated benchmark.
  --run       Actually generate at each config and time it. Needs model weights
              + compute (multi-GB). Use --model {ltx2,wan2,flux} --model-path.
"""

import argparse
import json
import sys
import time


def expected_loop_speedup(baseline_steps: int, reduced_steps: int) -> float:
    if reduced_steps <= 0:
        return float("inf")
    return baseline_steps / reduced_steps


def expected_wall_speedup(
    baseline_steps: int, reduced_steps: int, fixed_overhead_ratio: float
) -> float:
    # fixed_overhead_ratio = c / (k * baseline_steps): 0.0 = no overhead
    # (pure loop), 1.0 = overhead equals the whole baseline loop.
    k = 1.0
    c = fixed_overhead_ratio * k * baseline_steps
    base = k * baseline_steps + c
    reduced = k * reduced_steps + c
    if reduced <= 0:
        return float("inf")
    return base / reduced


def explain(args):
    print("=== Diffusion step-reduction speedup (deterministic) ===")
    print(
        "Wall-time ≈ k*steps + c (per-step cost k, fixed overhead c). "
        "Loop speedup = baseline/reduced steps.\n"
    )
    baseline = args.baseline_steps
    ratios = [0.0, 0.25, 0.5, 1.0]
    print(
        f"{'reduced':>8} {'loop_x':>8} {'wall_x(oh=0)':>14} "
        f"{'wall_x(oh=.25)':>15} {'wall_x(oh=.5)':>13} {'wall_x(oh=1.0)':>14}"
    )
    for reduced in args.reduced_steps:
        loop = expected_loop_speedup(baseline, reduced)
        cells = [f"{expected_wall_speedup(baseline, reduced, r):.2f}x" for r in ratios]
        print(f"{reduced:>8} {loop:>7.2f}x " + " ".join(f"{c:>13}" for c in cells))
    msg = (
        "\nInterpretation: baseline={} steps. Smallest config ({} steps) gives "
        "~{}x on the denoise loop. +100% (2x wall-clock) is reached at "
        "reduced<={} steps when overhead is low. Real wall-clock is "
        "lower-bounded by fixed overhead (load/VAE/IO). Measure with --run."
    ).format(
        baseline,
        args.reduced_steps[-1] if args.reduced_steps else baseline,
        (
            expected_loop_speedup(baseline, args.reduced_steps[-1])
            if args.reduced_steps
            else 1.0
        ),
        baseline // 2,
    )
    print(msg)


def run_ltx2(model_path, prompt, configs):
    from fusion_mlx.video.ltx2.generate import generate_video

    results = []
    for cfg in configs:
        out = f"/tmp/fusion_bench_ltx2_{cfg['steps']}.mp4"
        t0 = time.monotonic()
        generate_video(
            model_path,
            None,
            prompt,
            num_inference_steps=cfg["steps"],
            cfg_scale=cfg.get("cfg", 4.0),
            height=cfg.get("height", 512),
            width=cfg.get("width", 512),
            num_frames=cfg.get("num_frames", 33),
            seed=42,
            output_path=out,
            verbose=False,
        )
        elapsed = time.monotonic() - t0
        results.append({"config": cfg, "elapsed_s": round(elapsed, 3)})
        print(f"LTX-2 steps={cfg['steps']}: {elapsed:.2f}s -> {out}")
    return results


def run_wan2(model_path, prompt, configs):
    from fusion_mlx.video.wan2.generate import generate_video

    results = []
    for cfg in configs:
        out = f"/tmp/fusion_bench_wan2_{cfg['steps']}.mp4"
        t0 = time.monotonic()
        generate_video(
            model_path,
            prompt,
            width=cfg.get("width", 832),
            height=cfg.get("height", 480),
            num_frames=cfg.get("num_frames", 41),
            steps=cfg["steps"],
            seed=42,
            output_path=out,
            scheduler="unipc",
            no_compile=cfg.get("no_compile", True),
        )
        elapsed = time.monotonic() - t0
        results.append({"config": cfg, "elapsed_s": round(elapsed, 3)})
        print(
            f"Wan2 steps={cfg['steps']} compile={'off' if cfg.get('no_compile', True) else 'on'}: "
            f"{elapsed:.2f}s -> {out}"
        )
    return results


def run_flux(model_path, prompt, configs):
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux.variants.txt2img.flux import Flux1

    flux = Flux1(model_config=ModelConfig.schnell(), model_path=model_path)
    results = []
    for cfg in configs:
        t0 = time.monotonic()
        img = flux.generate_image(
            seed=42,
            prompt=prompt,
            num_inference_steps=cfg["steps"],
            height=cfg.get("height", 1024),
            width=cfg.get("width", 1024),
            guidance=cfg.get("guidance", 4.0),
        )
        elapsed = time.monotonic() - t0
        out = f"/tmp/fusion_bench_flux_{cfg['steps']}.png"
        img.image.save(out)
        results.append({"config": cfg, "elapsed_s": round(elapsed, 3)})
        print(f"Flux steps={cfg['steps']}: {elapsed:.2f}s -> {out}")
    return results


def run(args):
    configs = [{"steps": s} for s in args.reduced_steps]
    if args.model == "ltx2":
        results = run_ltx2(args.model_path, args.prompt, configs)
    elif args.model == "wan2":
        results = run_wan2(args.model_path, args.prompt, configs)
    elif args.model == "flux":
        results = run_flux(args.model_path, args.prompt, configs)
    else:
        print(f"unknown model: {args.model}", file=sys.stderr)
        sys.exit(2)

    if len(results) >= 2:
        base = results[0]["elapsed_s"]
        print("\n=== Measured speedup vs first config ===")
        for r in results[1:]:
            speedup = base / r["elapsed_s"] if r["elapsed_s"] > 0 else float("inf")
            print(
                f"  steps {results[0]['config']['steps']} -> {r['config']['steps']}: "
                f"{speedup:.2f}x  (+{(speedup - 1) * 100:.0f}%)"
            )
    print("\n" + json.dumps(results, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--explain",
        action="store_true",
        help="print deterministic speedup table, no compute",
    )
    p.add_argument(
        "--run", action="store_true", help="actually generate + time (needs weights)"
    )
    p.add_argument("--model", choices=["ltx2", "wan2", "flux"], default="ltx2")
    p.add_argument("--model-path", default=None, help="local model dir/repo for --run")
    p.add_argument("--prompt", default="a cat walking on a sunny beach, cinematic")
    p.add_argument(
        "--baseline-steps", type=int, default=40, help="reference step count"
    )
    p.add_argument(
        "--reduced-steps",
        type=int,
        nargs="+",
        default=[40, 20, 10, 5],
        help="step counts to evaluate (first = baseline for measured speedup)",
    )
    args = p.parse_args()

    if args.explain or not args.run:
        explain(args)
        if not args.run:
            print("\n(--run not set; only the deterministic table was printed.)")
            return
    if not args.model_path:
        print("error: --run requires --model-path", file=sys.stderr)
        sys.exit(2)
    run(args)


if __name__ == "__main__":
    main()
