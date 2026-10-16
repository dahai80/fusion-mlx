#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#1202 bench: image quality-tier routing end-to-end.

Two measurements against a running fusion-mlx instance:

1. Tier latency (DoD gate): default fast route (quality=fast, distilled
   sibling, 4 steps) vs high route (quality=high, dev model, 28 Euler
   steps), same prompt/resolution. Gate: >= 4x faster e2e.
2. Dev-tier sampler tuning: dev 28-step Euler reference vs dev at reduced
   steps with the DPM-Solver++(2M) scheduler (dpmpp_2m). Quality gates =
   no-reference stats (Laplacian-variance sharpness, Hasik-Susstrunk
   colorfulness) retained >= 85% vs the reference. PSNR/SSIM vs the
   reference is reported against a cross-seed noise floor (same config,
   different seed) for context — same-seed stochastic divergence between
   configs dominates it. Timing uses adjacent same-step-count
   Euler/dpmpp pairs (order alternated) because sustained GPU load
   thermally throttles mid-run and would otherwise contaminate
   cross-phase comparisons. All computed programmatically — no human
   image viewing.

Usage (debug instance on 11433):
  python benchmarks/image_quality_tier_1202_bench.py \
      --base-url http://127.0.0.1:11433 --api-key <key> \
      --dev-model black-forest-labs/FLUX.1-dev \
      --out benchmarks/reports/image_quality_tier_1202.json
"""

import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from scipy.ndimage import uniform_filter

PROMPT = (
    "a serene mountain lake at golden hour, snow-capped peaks reflected in "
    "still water, soft warm light, photographic detail"
)
WIDTH = 1024
HEIGHT = 1024
SEED = 20261002
DPMPP_STEPS = (12, 16, 20)
SEED_ALTS = (SEED + 1000, SEED + 2000)
# Retention floors for the no-reference quality proxy: reduced-step dpmpp
# output must keep >= 85% of the dev-28 reference sharpness (blur
# detector) and colorfulness (washed-out detector) to count as no
# user-perceptible regression. PSNR/SSIM vs the reference is dominated by
# same-seed stochastic divergence between configs, so it is reported
# against the cross-seed noise floor for context, not gated on.
SHARPNESS_FLOOR = 0.85
COLORFULNESS_FLOOR = 0.85


def _generate(base_url, headers, model, quality, steps=None, scheduler=None, seed=None):
    body = {
        "model": model,
        "prompt": PROMPT,
        "width": WIDTH,
        "height": HEIGHT,
        "seed": SEED if seed is None else seed,
        "quality": quality,
        "response_format": "b64_json",
        "n": 1,
    }
    if steps is not None:
        body["steps"] = steps
    if scheduler is not None:
        body["scheduler"] = scheduler
    t0 = time.perf_counter()
    r = requests.post(
        f"{base_url}/v1/images/generations", headers=headers, json=body, timeout=900
    )
    dt = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()["data"][0]
    if "b64_json" not in data:
        raise RuntimeError(f"no b64_json in response: {list(data)}")
    img = Image.open(io.BytesIO(base64.b64decode(data["b64_json"])))
    # The route substitutes the model for the fast tier; the response does
    # not echo which model served — the server log line
    # "image quality tier: fast routed ... -> distilled ..." is the record.
    return img, dt


def _to_luma(img):
    return np.asarray(img.convert("L"), dtype=np.float64)


def sharpness(luma):
    # Variance of the discrete Laplacian — standard blur detector. A
    # under-denoised (too few steps) diffusion output goes soft; this
    # drops sharply, so it is the programmatic proxy for the #823-style
    # "blurry" regression without any human image viewing.
    lap = (
        np.roll(luma, 1, 0)
        + np.roll(luma, -1, 0)
        + np.roll(luma, 1, 1)
        + np.roll(luma, -1, 1)
        - 4.0 * luma
    )
    return float(np.var(lap))


def colorfulness(img):
    # Hasik-Susstrunk metric: rg/yb opponent-channel spread. Desaturated
    # or washed-out output scores low vs the reference.
    a = np.asarray(img.convert("RGB"), dtype=np.float64)
    rg = a[..., 0] - a[..., 1]
    yb = 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    root = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
    mean = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    return float(root + 0.3 * mean)


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))


def ssim(a, b, win=8):
    # Standard SSIM with a uniform (box) window — deterministic, adequate
    # as a relative-quality proxy between the same-model step variants.
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = uniform_filter(a, win)
    mu_b = uniform_filter(b, win)
    mu_aa, mu_bb, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = uniform_filter(a * a, win) - mu_aa
    var_b = uniform_filter(b * b, win) - mu_bb
    cov = uniform_filter(a * b, win) - mu_ab
    num = (2 * mu_ab + c1) * (2 * cov + c2)
    den = (mu_aa + mu_bb + c1) * (var_a + var_b + c2)
    return float(np.mean(num / den))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:11433")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--dev-model", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--out", default="benchmarks/reports/image_quality_tier_1202.json")
    args = ap.parse_args()
    headers = {"Authorization": f"Bearer {args.api_key}"}
    dev = args.dev_model

    print(f"[1/5] baseline: dev 28-step Euler (quality=high), {WIDTH}x{HEIGHT}")
    ref_img, ref_dt = _generate(
        base_url=args.base_url, headers=headers, model=dev, quality="high"
    )
    print(f"      dev-28 e2e: {ref_dt:.1f}s")
    ref_l = _to_luma(ref_img)

    print(f"[2/5] noise floor: dev 28-step Euler, seeds={SEED_ALTS} (cross-seed)")
    nf_runs = []
    for sd in SEED_ALTS:
        nf_img, _ = _generate(
            base_url=args.base_url,
            headers=headers,
            model=dev,
            quality="high",
            seed=sd,
        )
        nf_l = _to_luma(nf_img)
        nf_runs.append(
            {
                "seed": sd,
                "psnr_vs_dev28": round(psnr(ref_l, nf_l), 2),
                "ssim_vs_dev28": round(ssim(ref_l, nf_l), 4),
                "sharpness": round(sharpness(nf_l), 1),
                "colorfulness": round(colorfulness(nf_img), 1),
            }
        )

    print("[3/5] fast tier: default route (quality=fast -> distilled sibling)")
    fast_img, fast_dt = _generate(
        base_url=args.base_url, headers=headers, model=dev, quality="fast"
    )
    speedup = ref_dt / fast_dt
    print(f"      fast e2e: {fast_dt:.1f}s  speedup vs dev-28: {speedup:.2f}x")

    nf_psnr = min(r["psnr_vs_dev28"] for r in nf_runs)
    nf_sharps = [r["sharpness"] for r in nf_runs]
    nf_colors = [r["colorfulness"] for r in nf_runs]
    noise_floor = {
        "seeds": SEED_ALTS,
        "runs": nf_runs,
        "sharpness_band": [min(nf_sharps), max(nf_sharps)],
        "colorfulness_band": [min(nf_colors), max(nf_colors)],
        "psnr_floor_db": nf_psnr,
    }
    print(
        f"      noise floor: PSNR {nf_psnr:.1f}dB (min of {len(nf_runs)} seeds)  "
        f"sharpness band {min(nf_sharps):.0f}-{max(nf_sharps):.0f}  "
        f"colorfulness band {min(nf_colors):.1f}-{max(nf_colors):.1f}"
    )
    ref_sharp, ref_color = sharpness(ref_l), colorfulness(ref_img)
    print(f"      dev-28 sharpness {ref_sharp:.1f}  colorfulness {ref_color:.1f}")

    print("[4/5] dev-tier reduced steps with dpmpp_2m (paired Euler timing)")
    # Sustained GPU load thermally throttles the SoC mid-run (observed
    # per-step cost ramping ~3.2s -> ~5.8s over a sequential bench), which
    # would make late dpmpp runs look slower than the early dev-28 baseline
    # at MORE steps. Each step count is therefore timed as an adjacent
    # same-step-count Euler/dpmpp pair (order alternated to cancel order
    # bias) — the pair ratio isolates the scheduler's per-step overhead
    # from thermal drift, and the step-count ratio vs dev-28 carries the
    # speed claim.
    tuning = []
    for idx, n in enumerate(DPMPP_STEPS):
        order = ("euler", "dpmpp") if idx % 2 == 0 else ("dpmpp", "euler")
        runs = {}
        for which in order:
            runs[which] = _generate(
                base_url=args.base_url,
                headers=headers,
                model=dev,
                quality="high",
                steps=n,
                scheduler="dpmpp_2m" if which == "dpmpp" else None,
            )
        img, dt = runs["dpmpp"]
        _, dt_e = runs["euler"]
        lum = _to_luma(img)
        sh, col = sharpness(lum), colorfulness(img)
        p, s = psnr(ref_l, lum), ssim(ref_l, lum)
        entry = {
            "steps": n,
            "scheduler": "dpmpp_2m",
            "e2e_s": round(dt, 2),
            "euler_same_steps_e2e_s": round(dt_e, 2),
            "dpmpp_vs_euler_ratio": round(dt / dt_e, 3),
            "steps_vs_dev28": f"{n}/28",
            "psnr_vs_dev28": round(p, 2),
            "ssim_vs_dev28": round(s, 4),
            "psnr_gte_noise_floor": p >= nf_psnr,
            "sharpness": round(sh, 1),
            "colorfulness": round(col, 1),
            "sharpness_retention": round(sh / ref_sharp, 3),
            "colorfulness_retention": round(col / ref_color, 3),
            "quality_ok": (
                sh / ref_sharp >= SHARPNESS_FLOOR
                and col / ref_color >= COLORFULNESS_FLOOR
            ),
        }
        tuning.append(entry)
        print(
            f"      dev-{n} dpmpp_2m: {dt:.1f}s (Euler-{n} {dt_e:.1f}s, "
            f"ratio {dt / dt_e:.2f}x)  PSNR {p:.1f}dB  "
            f"sharp {sh / ref_sharp:.2f}x  color {col / ref_color:.2f}x  "
            f"{'OK' if entry['quality_ok'] else 'REGRESSION'}"
        )

    fast_l = _to_luma(fast_img)
    result = {
        "prompt": PROMPT,
        "resolution": f"{WIDTH}x{HEIGHT}",
        "seed": SEED,
        "dev_model": dev,
        "dev28_e2e_s": round(ref_dt, 2),
        "fast_e2e_s": round(fast_dt, 2),
        "fast_speedup_x": round(speedup, 2),
        "dod_gate_4x": speedup >= 4.0,
        "noise_floor": noise_floor,
        "image_stats": {
            "dev28": {
                "sharpness": round(ref_sharp, 1),
                "colorfulness": round(ref_color, 1),
            },
            "fast_distilled": {
                "sharpness": round(sharpness(fast_l), 1),
                "colorfulness": round(colorfulness(fast_img), 1),
            },
        },
        "quality_floors": {
            "sharpness": SHARPNESS_FLOOR,
            "colorfulness": COLORFULNESS_FLOOR,
        },
        "dod_quality_gate": all(e["quality_ok"] for e in tuning),
        "dev_tuning": tuning,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")
    print(
        f"DoD gate (>=4x default-route speedup): "
        f"{'PASS' if result['dod_gate_4x'] else 'FAIL'}"
    )
    print(
        f"DoD gate (dev-tier quality, sharpness/colorfulness >= "
        f"{SHARPNESS_FLOOR:.0%}): "
        f"{'PASS' if result['dod_quality_gate'] else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
