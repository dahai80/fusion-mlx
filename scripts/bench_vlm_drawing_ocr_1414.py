# SPDX-License-Identifier: Apache-2.0
"""VLM architectural-drawing OCR benchmark (#1414).

Generates synthetic ground-truth drawing samples (printed annotations,
dimension numbers, tables, handwriting-style notes), renders an A1 sheet
at 200 DPI, and measures per-model OCR accuracy, structured-output
success rate, long-image slice stability, and latency against a live
fusion-mlx / fusion-gateway OpenAI-compatible endpoint.

Usage:
    python scripts/bench_vlm_drawing_ocr_1414.py \
        --base-url http://127.0.0.1:11432 \
        --models mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
        --out-dir /tmp/vlm_ocr_bench
"""

from __future__ import annotations

import argparse
import base64
import difflib
import io
import json
import logging
import statistics
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("bench_vlm_ocr_1414")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

A1_MM = (841, 594)  # landscape width x height
FONT_CANDIDATES = {
    "print": [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ],
    "dimension": [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ],
    "handwriting": [
        "/System/Library/Fonts/Supplemental/Snell Roundhand.ttf",
        "/System/Library/Fonts/Supplemental/Zapfino.ttf",
        "/System/Library/Fonts/Supplemental/Brush Script.ttf",
        "/System/Library/Fonts/Supplemental/Chalkduster.ttf",
    ],
    "table": [
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ],
}

GT_PRINTED = [
    "FLOOR PLAN - LEVEL 02",
    "EXTERNAL WALL TYPE A - 200mm CAVITY BRICK",
    "THERMAL INSULATION TO ALL EXTERNAL WALLS",
    "CEILING HEIGHT 2800mm ABOVE FFL",
    "WATERPROOF MEMBRANE AT BALCONY THRESHOLD",
    "STEEL BEAM 254x146x31 UB OVER OPENING",
]

GT_HANDWRITING = [
    "check with structural engineer",
    "verify levels on site",
    "RCP to match above",
    "confirm with client",
    "not to scale drawing",
]

# ---------------------------------------------------------------------------
# Sample generation (deterministic ground truth)
# ---------------------------------------------------------------------------


def _load_font(kind: str, size: int):
    from PIL import ImageFont

    for path in FONT_CANDIDATES[kind]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise RuntimeError(
        f"no usable font for kind={kind!r} — checked {FONT_CANDIDATES[kind]}"
    )


def _draw_grid(draw, w, h, step=64, color=(210, 210, 210)):
    for x in range(0, w, step):
        draw.line([(x, 0), (x, h)], fill=color, width=1)
    for y in range(0, h, step):
        draw.line([(0, y), (w, y)], fill=color, width=1)


def _text_image(text: str, font_kind: str, px_size: int, fg=(16, 16, 16)) -> tuple:
    from PIL import Image, ImageDraw

    font = _load_font(font_kind, int(px_size * 0.05) + 18)
    tmp = Image.new("RGB", (10, 10), "white")
    d = ImageDraw.Draw(tmp)
    box = d.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    img = Image.new("RGB", (px_size, px_size), "white")
    draw = ImageDraw.Draw(img)
    _draw_grid(draw, px_size, px_size)
    draw.text(((px_size - tw) // 2, (px_size - th) // 2), text, fill=fg, font=font)
    return img, text


def gen_table_image() -> tuple:
    from PIL import Image, ImageDraw

    rows = [
        ("MARK", "DESCRIPTION", "QTY"),
        ("A-01", "200mm CAVITY BRICK WALL", "12"),
        ("A-02", "REINFORCED CONCRETE SLAB", "8"),
        ("A-03", "ALUMINIUM CURTAIN WALL", "5"),
        ("A-04", "TIMBER DECKING 32mm", "3"),
        ("A-05", "STEEL CHANNEL 152x76", "16"),
    ]
    w, h = 1024, 640
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    _draw_grid(draw, w, h)
    font = _load_font("table", 30)
    header_font = _load_font("table", 32)
    row_h = 80
    cols = [(40, 200), (210, 700), (720, 980)]
    for ri, row in enumerate(rows):
        y = 60 + ri * row_h
        f = header_font if ri == 0 else font
        for ci, cell in enumerate(row):
            draw.text((cols[ci][0] + 8, y + 24), cell, fill=(16, 16, 16), font=f)
        draw.rectangle([30, y, w - 30, y + row_h - 8], outline=(40, 40, 40), width=2)
    draw.line(
        [(200, 60), (200, 60 + len(rows) * row_h - 8)], fill=(40, 40, 40), width=2
    )
    draw.line(
        [(710, 60), (710, 60 + len(rows) * row_h - 8)], fill=(40, 40, 40), width=2
    )
    gt = [list(r) for r in rows]
    return img, gt


def gen_dimension_image() -> tuple:
    from PIL import Image, ImageDraw

    dims = ["3600", "2400", "1800", "1200", "900", "450"]
    w, h = 1024, 512
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    _draw_grid(draw, w, h)
    font = _load_font("dimension", 34)
    seg = (w - 160) // len(dims)
    y = 220
    draw.line([(80, y), (w - 80, y)], fill=(30, 30, 30), width=3)
    for i, d in enumerate(dims):
        x = 80 + i * seg
        draw.line([(x, y - 28), (x, y + 28)], fill=(30, 30, 30), width=3)
        bbox = font.getbbox(d)
        tw = bbox[2] - bbox[0]
        draw.text((x + seg // 2 - tw // 2, y - 76), d, fill=(16, 16, 16), font=font)
    for i in range(len(dims) + 1):
        x = min(80 + i * seg, w - 80)
        draw.line([(x, y - 28), (x, y + 28)], fill=(30, 30, 30), width=3)
    return img, dims


def gen_category_samples() -> dict:
    samples = {"printed": [], "dimensions": [], "table": [], "handwriting": []}
    for text in GT_PRINTED:
        img, gt = _text_image(text, "print", 768)
        samples["printed"].append((img, gt))
    dim_img, dim_gt = gen_dimension_image()
    samples["dimensions"].append((dim_img, dim_gt))
    tab_img, tab_gt = gen_table_image()
    samples["table"].append((tab_img, tab_gt))
    for text in GT_HANDWRITING:
        img, gt = _text_image(text, "handwriting", 768, fg=(10, 30, 90))
        samples["handwriting"].append((img, gt))
    return samples


def gen_a1_sheet(dpi: int) -> tuple:
    from PIL import Image, ImageDraw

    px_w = int(A1_MM[0] / 25.4 * dpi)
    px_h = int(A1_MM[1] / 25.4 * dpi)
    img = Image.new("RGB", (px_w, px_h), "white")
    draw = ImageDraw.Draw(img)
    _draw_grid(draw, px_w, px_h, step=128, color=(225, 225, 225))
    font = _load_font("print", 44)
    hw_font = _load_font("handwriting", 40)
    region_truth = []
    regions = [
        (100, 100, "FLOOR PLAN - LEVEL 02", font, (16, 16, 16)),
        (px_w // 2, 100, "SCALE 1:100 ON A1 SHEET", font, (16, 16, 16)),
        (
            100,
            px_h // 3,
            "EXTERNAL WALL TYPE A - 200mm CAVITY BRICK",
            font,
            (16, 16, 16),
        ),
        (
            px_w // 2,
            px_h // 2,
            "DIMENSION 3600mm BETWEEN GRID B AND C",
            font,
            (16, 16, 16),
        ),
        (100, px_h - 300, "check with structural engineer", hw_font, (10, 30, 90)),
        (px_w // 2, px_h - 300, "verify levels on site", hw_font, (10, 30, 90)),
        (px_w - 1400, px_h - 300, "not to scale drawing", hw_font, (10, 30, 90)),
    ]
    for i, (x, y, text, f, fg) in enumerate(regions):
        draw.text((x, y), text, fill=fg, font=f)
        region_truth.append({"idx": i, "x": x, "y": y, "text": text})
    border = 40
    draw.rectangle(
        [border, border, px_w - border, px_h - border], outline=(20, 20, 20), width=4
    )
    logger.info(
        "A1 sheet %dx%d px (%d DPI), %d text regions", px_w, px_h, dpi, len(regions)
    )
    return img, region_truth


def slice_sheet(image, slice_px: int, overlap_pct: int) -> list:
    w, h = image.size
    stride = int(slice_px * (100 - overlap_pct) / 100)
    crops = []
    ys = []
    y = 0
    while y + slice_px < h:
        ys.append(y)
        y += stride
    ys.append(h - slice_px)
    xs = []
    x = 0
    while x + slice_px < w:
        xs.append(x)
        x += stride
    xs.append(w - slice_px)
    for yi, sy in enumerate(ys):
        for xi, sx in enumerate(xs):
            crops.append(
                (image.crop((sx, sy, sx + slice_px, sy + slice_px)), (xi, yi, sx, sy))
            )
    return crops


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def _api_key() -> str:
    settings_path = Path.home() / ".fusion-mlx" / "settings.json"
    try:
        cfg = json.loads(settings_path.read_text())
        key = (cfg.get("auth") or {}).get("api_key") or ""
        if key:
            return key
        logger.warning(
            "settings.json has empty auth.api_key — proceeding unauthenticated"
        )
        return ""
    except FileNotFoundError:
        logger.warning("no ~/.fusion-mlx/settings.json — proceeding unauthenticated")
        return ""
    except Exception as exc:
        raise RuntimeError(
            f"failed reading settings.json: {type(exc).__name__}"
        ) from exc


def _encode_image(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ocr_request(
    client, model: str, image, prompt: str, schema: dict | None, timeout: float
) -> dict:
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _encode_image(image)}},
    ]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2048,
        "temperature": 0.1,
    }
    if schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "ocr_result", "strict": True, "schema": schema},
        }
    t0 = time.monotonic()
    try:
        r = client.post("/v1/chat/completions", json=body, timeout=timeout)
    except Exception as exc:
        logger.warning("request failed: %s(%s)", type(exc).__name__, exc)
        return {
            "text": "",
            "finish_reason": None,
            "latency_s": time.monotonic() - t0,
            "http_status": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    latency = time.monotonic() - t0
    if r.status_code != 200:
        logger.warning("HTTP %d: %s", r.status_code, r.text[:200])
        return {
            "text": "",
            "finish_reason": None,
            "latency_s": latency,
            "http_status": r.status_code,
            "error": r.text[:300],
        }
    data = r.json()
    choices = data.get("choices") or [{}]
    msg = choices[0].get("message") or {}
    return {
        "text": msg.get("content") or "",
        "finish_reason": choices[0].get("finish_reason"),
        "latency_s": latency,
        "http_status": 200,
        "error": None,
        "usage": data.get("usage") or {},
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _normalize(s: str) -> str:
    return " ".join(s.upper().split())


def score_category(results: list, gt: list) -> dict:
    exact = 0
    sims = []
    valid_json = 0
    for res in results:
        text = res.get("text") or ""
        if res.get("error") is None:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    valid_json += 1
            except (ValueError, TypeError):
                pass
    found_texts = []
    for res in results:
        found_texts.append(res.get("text") or "")
    corpus = _normalize(" ".join(found_texts))
    for gt_item in gt:
        needle = _normalize(gt_item if isinstance(gt_item, str) else " ".join(gt_item))
        if needle in corpus:
            exact += 1
            sims.append(1.0)
        else:
            words = needle.split()
            probe = " ".join(words[:6])
            best = 0.0
            for chunk in corpus.split("  "):
                best = max(
                    best,
                    difflib.SequenceMatcher(
                        None, probe, chunk[: max(len(probe) * 2, 40)]
                    ).ratio(),
                )
            sims.append(best)
    latencies = [r["latency_s"] for r in results if r.get("latency_s") is not None]
    return {
        "n_samples": len(results),
        "exact_match_rate": round(exact / len(gt), 4) if gt else None,
        "median_similarity": round(statistics.median(sims), 4) if sims else None,
        "valid_json_rate": round(valid_json / len(results), 4) if results else None,
        "median_latency_s": (
            round(statistics.median(latencies), 3) if latencies else None
        ),
        "p90_latency_s": (
            round(sorted(latencies)[int(0.9 * (len(latencies) - 1))], 3)
            if latencies
            else None
        ),
        "errors": sum(1 for r in results if r.get("error")),
        "truncated": sum(1 for r in results if r.get("finish_reason") == "length"),
    }


# ---------------------------------------------------------------------------
# Main benchmark phases
# ---------------------------------------------------------------------------


OCR_SCHEMA = {
    "type": "object",
    "properties": {
        "texts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Every text string legible in the image, verbatim.",
        }
    },
    "required": ["texts"],
    "additionalProperties": False,
}

PROMPT_PLAIN = (
    "This is a crop of an architectural drawing. Transcribe every text "
    "string you can read in the image, verbatim, as a JSON object with a "
    'single key "texts" holding an array of strings. Do not translate, '
    "do not paraphrase, do not add commentary."
)


def bench_categories(client, model: str, samples: dict, out: Path) -> dict:
    results = {}
    for cat, items in samples.items():
        cat_results = []
        gt = []
        for img, gt_item in items:
            res = ocr_request(client, model, img, PROMPT_PLAIN, OCR_SCHEMA, timeout=300)
            logger.info(
                "cat=%s model=%s latency=%.1fs finish=%s err=%s",
                cat,
                model.split("/")[-1],
                res["latency_s"],
                res["finish_reason"],
                (res.get("error") or "none")[:80],
            )
            cat_results.append(res)
            if isinstance(gt_item, list):
                gt.extend(gt_item)
            else:
                gt.append(gt_item)
        results[cat] = {"score": score_category(cat_results, gt), "gt": gt}
    return results


def bench_long_image(client, model: str, sheet, region_truth, args, out: Path) -> dict:
    crops = slice_sheet(sheet, args.slice_px, args.overlap_pct)
    logger.info(
        "long-image: %d slices of %dpx (overlap %d%%)",
        len(crops),
        args.slice_px,
        args.overlap_pct,
    )
    per_slice = []
    recovered = set()
    for crop, (xi, yi, sx, sy) in crops:
        res = ocr_request(client, model, crop, PROMPT_PLAIN, OCR_SCHEMA, timeout=600)
        text = res.get("text") or ""
        upper = _normalize(text)
        hit = []
        for reg in region_truth:
            if _normalize(reg["text"]) in upper and reg["idx"] not in recovered:
                recovered.add(reg["idx"])
                hit.append(reg["idx"])
        logger.info(
            "slice (%d,%d) offset=(%d,%d) latency=%.1fs finish=%s status=%s new_hits=%d",
            xi,
            yi,
            sx,
            sy,
            res["latency_s"],
            res["finish_reason"],
            res["http_status"],
            len(hit),
        )
        per_slice.append(
            {
                "offset": [sx, sy],
                "latency_s": round(res["latency_s"], 3),
                "finish_reason": res["finish_reason"],
                "http_status": res["http_status"],
                "error": res.get("error"),
                "new_region_hits": hit,
            }
        )
    lat = [s["latency_s"] for s in per_slice]
    return {
        "slice_px": args.slice_px,
        "overlap_pct": args.overlap_pct,
        "n_slices": len(crops),
        "regions_recovered": len(recovered),
        "regions_total": len(region_truth),
        "oom_or_5xx": sum(1 for s in per_slice if s["http_status"] != 200),
        "truncated_slices": sum(1 for s in per_slice if s["finish_reason"] == "length"),
        "median_latency_s": round(statistics.median(lat), 3) if lat else None,
        "max_latency_s": round(max(lat), 3) if lat else None,
        "per_slice": per_slice,
    }


def bench_slice_sizes(client, model: str, sheet, args, out: Path) -> dict:
    from argparse import Namespace

    results = {}
    for size in args.slice_sizes:
        sub_args = Namespace(slice_px=size, overlap_pct=args.overlap_pct)
        logger.info("slice-size probe: %dpx", size)
        results[str(size)] = bench_long_image(
            client, model, sheet, region_truth=[], args=sub_args, out=out
        )
    return results


def write_report(all_results: dict, out: Path) -> Path:
    lines = [
        "# VLM Architectural-Drawing OCR Benchmark (#1414)",
        "",
        "Endpoint, models, per-category accuracy, structured-output success,",
        "long-image slicing stability, and slice-size safety probes.",
        "",
    ]
    for model, mres in all_results["models"].items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            "| category | exact-match | median sim | valid JSON | med latency (s) | p90 (s) | errors | truncated |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for cat, cat_res in mres["categories"].items():
            s = cat_res["score"]
            lines.append(
                f"| {cat} | {s['exact_match_rate']} | {s['median_similarity']} "
                f"| {s['valid_json_rate']} | {s['median_latency_s']} | {s['p90_latency_s']} "
                f"| {s['errors']} | {s['truncated']} |"
            )
        li = mres.get("long_image") or {}
        if li:
            lines.append("")
            lines.append(
                f"Long image A1@{all_results['dpi']}DPI: {li['n_slices']} slices @ {li['slice_px']}px "
                f"(overlap {li['overlap_pct']}%) — regions recovered {li['regions_recovered']}/{li['regions_total']}, "
                f"OOM/5xx {li['oom_or_5xx']}, truncated {li['truncated_slices']}, "
                f"median {li['median_latency_s']}s, max {li['max_latency_s']}s."
            )
        ss = mres.get("slice_sizes") or {}
        if ss:
            lines.append("")
            lines.append(
                "| slice px | slices | OOM/5xx | truncated | median latency (s) |"
            )
            lines.append("| --- | --- | --- | --- | --- |")
            for size, r in ss.items():
                lines.append(
                    f"| {size} | {r['n_slices']} | {r['oom_or_5xx']} | {r['truncated_slices']} "
                    f"| {r['median_latency_s']} |"
                )
        lines.append("")
    rec = all_results.get("recommendation") or {}
    if rec:
        lines.append("## Recommendation")
        lines.append("")
        lines.append(f"- Quant tier: **{rec.get('quant_tier')}**")
        lines.append(f"- Max safe slice size: **{rec.get('max_safe_slice_px')} px**")
        lines.append(f"- Rationale: {rec.get('rationale')}")
        lines.append("")
    md_path = out / "report.md"
    md_path.write_text("\n".join(lines))
    logger.info("report written: %s", md_path)
    return md_path


def _recommend(all_results: dict, args) -> dict:
    best_tier = None
    for model, mres in all_results["models"].items():
        cats = mres["categories"]
        accs = [c["score"]["exact_match_rate"] or 0 for c in cats.values()]
        jsons = [c["score"]["valid_json_rate"] or 0 for c in cats.values()]
        li = mres.get("long_image") or {}
        mean_acc = sum(accs) / len(accs) if accs else 0
        mean_json = sum(jsons) / len(jsons) if jsons else 0
        mres["summary"] = {
            "mean_exact_match": round(mean_acc, 4),
            "mean_valid_json": round(mean_json, 4),
            "long_image_stable": bool(li)
            and li["oom_or_5xx"] == 0
            and li["truncated_slices"] == 0,
        }
        tier = "8bit" if "8bit" in model else ("4bit" if "4bit" in model else model)
        if mean_acc >= 0.8 and mean_json >= 0.9:
            if best_tier is None:
                best_tier = model
    max_safe = None
    for model, mres in all_results["models"].items():
        for size, r in (mres.get("slice_sizes") or {}).items():
            if r["oom_or_5xx"] == 0 and r["truncated_slices"] == 0:
                max_safe = max(max_safe or 0, int(size))
    quant_tier = best_tier or "none-met-threshold"
    rationale = (
        f"mean exact-match >= 0.8 and valid-JSON >= 0.9 on category suite; "
        f"slice probes up to {max_safe or args.slice_px}px had zero OOM/5xx and zero truncation. "
        "If no model met the accuracy threshold, drawing-parser M2 should use the cloud fallback path."
    )
    return {
        "quant_tier": quant_tier,
        "max_safe_slice_px": max_safe or args.slice_px,
        "rationale": rationale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11432")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
            "mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
        ],
    )
    parser.add_argument("--out-dir", default="/tmp/vlm_ocr_bench_1414")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--slice-px", type=int, default=2048)
    parser.add_argument("--overlap-pct", type=int, default=15)
    parser.add_argument(
        "--slice-sizes", type=int, nargs="+", default=[2048, 2560, 3072]
    )
    parser.add_argument("--skip-slice-probe", action="store_true")
    parser.add_argument(
        "--api-key", default=None, help="bearer token override (default: ~/.fusion-mlx/settings.json auth.api_key)"
    )
    args = parser.parse_args()

    import httpx

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    headers = {}
    key = args.api_key or _api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    client = httpx.Client(base_url=args.base_url, headers=headers, timeout=600)

    logger.info("endpoint %s, models %s", args.base_url, args.models)
    all_results = {"dpi": args.dpi, "base_url": args.base_url, "models": {}}
    samples = gen_category_samples()
    sheet, region_truth = gen_a1_sheet(args.dpi)
    sheet_gt_path = out / "a1_ground_truth.json"
    sheet_gt_path.write_text(json.dumps(region_truth, indent=2))
    for model in args.models:
        logger.info("=== model %s ===", model)
        mres = {"categories": bench_categories(client, model, samples, out)}
        mres["long_image"] = bench_long_image(
            client, model, sheet, region_truth, args, out
        )
        if not args.skip_slice_probe:
            mres["slice_sizes"] = bench_slice_sizes(client, model, sheet, args, out)
        all_results["models"][model] = mres
        (out / f"raw_{model.replace('/', '_')}.json").write_text(
            json.dumps(mres, indent=2, default=str)
        )
    all_results["recommendation"] = _recommend(all_results, args)
    (out / "results.json").write_text(json.dumps(all_results, indent=2, default=str))
    md = write_report(all_results, out)
    logger.info("DONE — results %s, report %s", out / "results.json", md)


if __name__ == "__main__":
    main()
