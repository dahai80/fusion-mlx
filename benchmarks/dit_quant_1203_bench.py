#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#1203 bench: DiT linear-layer quantization quality regression.

Loads a DiT (image or video) backend in-process at full precision, then at
INT4/INT8, generates with an identical prompt/seed/steps, and reports
quality (PSNR/SSIM against the full-precision reference plus no-reference
sharpness/colorfulness), throughput (s/step) and peak memory.

Gate: a config ships quantized only if SSIM >= 0.90 and both no-reference
proxies retain >= 90% of the reference. Anything below stays fp16/bf16 and
is recorded in the report with its numbers.

Usage:
  python benchmarks/dit_quant_1203_bench.py \
      --modality image --model black-forest-labs/FLUX.1-dev \
      --bits 4 8 --out benchmarks/reports/dit_quant_1203.json
"""

import argparse
import json
import resource
import time
from pathlib import Path

SSIM_FLOOR = 0.90
PROXY_FLOOR = 0.90


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=("image", "video"), default="image")
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--frames", type=int, default=9)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument(
        "--prompt",
        default=(
            "a red fox sitting in fresh snow at dusk, sharp detail, "
            "cinematic lighting"
        ),
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-run", action="store_true")
    return ap.parse_args()


def resolve_model_path(model: str) -> str:
    """HF repo id -> local snapshot dir (the image worker needs a real dir)."""
    if Path(model).is_dir():
        return model
    root = Path.home() / ".fusion-mlx" / "models"
    repo = root / ("models--" + model.replace("/", "--"))
    snaps = repo / "snapshots"
    if snaps.is_dir():
        candidates = sorted(p for p in snaps.iterdir() if p.is_dir())
        if candidates:
            chosen = candidates[-1]
            print(f"resolved {model} -> {chosen}")
            return str(chosen)
    raise SystemExit(f"model not found locally: {model} (pass a directory)")


def peak_rss_gib():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return round(ru.ru_maxrss / 1024**3, 2)


def ssim(a, b):
    import numpy as np

    x = a.astype(np.float64)
    y = b.astype(np.float64)
    c1, c2 = (0.01 * 1.0) ** 2, (0.03 * 1.0) ** 2
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cxy = ((x - mx) * (y - my)).mean()
    return float(
        ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
    )


def psnr(a, b, peak=1.0):
    import numpy as np

    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 0:
        return 99.0
    return float(10.0 * np.log10(peak**2 / mse))


def sharpness(img):
    import numpy as np
    from scipy.ndimage import uniform_filter

    gray = img.astype(np.float64)
    if gray.ndim == 3:
        gray = gray.mean(axis=-1)
    lap = gray - uniform_filter(gray, size=3)
    return float(lap.var())


def colorfulness(img):
    import numpy as np

    arr = img.astype(np.float64)
    if arr.ndim < 3 or arr.shape[-1] < 3:
        return 0.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(
        (rg.std() ** 2 + yb.std() ** 2) ** 0.5
        + (0.3 * (rg.mean() ** 2 + yb.mean() ** 2) ** 0.5)
    )


def make_engine(args, bits):
    from fusion_mlx.engines.image_gen import ImageGenEngine

    return ImageGenEngine(
        model_name=args.model_path,
        variant=None,
        quantize=bits,
    )


def run_image(args, bits):
    import asyncio

    import numpy as np

    async def _one():
        eng = make_engine(args, bits)
        t0 = time.perf_counter()
        await eng.start()
        load_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        out = await eng.generate(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            steps=args.steps,
            seed=args.seed,
            n_images=1,
        )
        gen_s = time.perf_counter() - t1
        await eng.stop()
        return out[0], load_s, gen_s

    png, load_s, gen_s = asyncio.run(_one())
    from io import BytesIO

    from PIL import Image

    img = np.asarray(Image.open(BytesIO(png)).convert("RGB"), dtype=np.float32) / 255.0
    return img, load_s, gen_s


def main():
    args = parse_args()
    report = {
        "issue": "#1203",
        "modality": args.modality,
        "model": args.model,
        "steps": args.steps,
        "seed": args.seed,
        "resolution": [args.width, args.height],
        "gate": {"ssim_floor": SSIM_FLOOR, "proxy_floor": PROXY_FLOOR},
        "runs": [],
    }
    if args.skip_run:
        print("skip-run: harness only")
        return
    args.model_path = resolve_model_path(args.model)
    if args.modality != "image":
        raise SystemExit(
            "video DiT quantization is not a runtime knob: ltx2/ltx2_5 load "
            "quantized checkpoints (checkpoint keys carry .scales) and "
            "quantize_bits comes from split_model.json. Produce an INT4 "
            "checkpoint offline, point --model at it, then re-run this "
            "harness. See benchmarks/reports/dit_quant_1203.md."
        )

    print("stage: fp16 reference run", flush=True)
    ref, ref_load, ref_gen = run_image(args, None)
    ref_sharp, ref_color = sharpness(ref), colorfulness(ref)
    report["runs"].append(
        {
            "config": "fp16",
            "bits": None,
            "load_s": round(ref_load, 2),
            "gen_s": round(ref_gen, 2),
            "s_per_step": round(ref_gen / args.steps, 3),
            "peak_rss_gib": peak_rss_gib(),
            "sharpness": round(ref_sharp, 5),
            "colorfulness": round(ref_color, 5),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for bits in args.bits:
        print(f"stage: int{bits} run", flush=True)
        img, load_s, gen_s = run_image(args, bits)
        s = ssim(ref, img)
        p = psnr(ref, img)
        sharp_ret = sharpness(img) / ref_sharp if ref_sharp else 0.0
        color_ret = colorfulness(img) / ref_color if ref_color else 0.0
        passed = s >= SSIM_FLOOR and min(sharp_ret, color_ret) >= PROXY_FLOOR
        report["runs"].append(
            {
                "config": f"int{bits}",
                "bits": bits,
                "load_s": round(load_s, 2),
                "gen_s": round(gen_s, 2),
                "s_per_step": round(gen_s / args.steps, 3),
                "speedup_vs_fp16": round(ref_gen / gen_s, 2) if gen_s else None,
                "peak_rss_gib": peak_rss_gib(),
                "psnr": round(p, 3),
                "ssim": round(s, 5),
                "sharpness_retained": round(sharp_ret, 4),
                "colorfulness_retained": round(color_ret, 4),
                "passed": passed,
            }
        )
        print(json.dumps(report["runs"][-1], indent=2), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
