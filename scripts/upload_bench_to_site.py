#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Upload local fusion-mlx benchmark SUMMARY reports to bench.dpdns.org.

Reads benchmarks/reports/SUMMARY_*.json, maps each result to the bench-site
speed-entry schema, POSTs to https://bench.dpdns.org/api/benchmarks.
Idempotent: server returns 409 for duplicates.

Usage:
    python scripts/upload_bench_to_site.py [--dry-run] [SUMMARY_*.json ...]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger("upload_bench")

API_URL = "https://bench.dpdns.org/api/benchmarks"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "reports"

# Hardware — matches prior on-site entries for this machine (M5 Max).
CHIP_NAME = "M5 Max"
MEMORY_GB = 128
GPU_CORES = 40
OMLX_VERSION = "fusion-mlx"
SUBMISSION_GROUP = "fusion-mlx"


def parse_quant(model: str) -> str:
    m = model.lower()
    if "mxfp8" in m:
        return "mxfp8"
    if "mxfp4" in m:
        return "mxfp4"
    if "nvfp4" in m:
        return "nvfp4"
    if "bf16" in m:
        return "bf16"
    if "a4b" in m:
        return "a4b"
    if "8bit" in m or "-8bit" in m or "_8bit" in m:
        return "8bit"
    if "4bit" in m or "-4bit" in m or "_4bit" in m:
        return "4bit"
    if "q4_k_m" in m or "q4_k_m" in m:
        return "Q4_K_M"
    return "4bit"


def normalize_model_name(model: str) -> str:
    # Strip mlx-community-- prefix double-dash to match on-site naming.
    return model


def result_to_payload(r: dict, timestamp: str) -> dict:
    model = normalize_model_name(r.get("model", ""))
    tg_tps = float(r.get("tokens_per_second") or 0.0)
    ttft_s = float(r.get("ttft_seconds") or 0.0)
    ttft_ms = round(ttft_s * 1000.0, 1) if ttft_s > 0 else 0.0
    prompt_tokens = float(r.get("prompt_tokens") or 0.0)
    pp_tps = round(prompt_tokens / ttft_s, 1) if ttft_s > 0 else 0.0
    quant = parse_quant(model)
    payload = {
        "chip_name": CHIP_NAME,
        "chip_variant": "",
        "memory_gb": MEMORY_GB,
        "gpu_cores": GPU_CORES,
        "os_version": "",
        "omlx_version": OMLX_VERSION,
        "model_name": model,
        "quantization": quant,
        "context_length": 4096,
        "pp_tps": pp_tps,
        "tg_tps": round(tg_tps, 1),
        "ttft_ms": ttft_ms,
        "submission_group": SUBMISSION_GROUP,
        "benchmark_type": "speed",
        "task_name": f"decode-{model}",
        "metric_name": "decode_speed",
        "metric_value": round(tg_tps, 1),
        "detail": json.dumps(
            {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(r.get("completion_tokens") or 0),
                "wall_seconds": round(float(r.get("wall_seconds") or 0.0), 3),
                "source_report": timestamp,
            },
            ensure_ascii=False,
        ),
    }
    return payload


def collect_reports(explicit: list[str]) -> list[Path]:
    if explicit:
        return [Path(p) for p in explicit]
    return sorted(REPORTS_DIR.glob("SUMMARY_*.json"))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Print payloads, do not POST")
    ap.add_argument("--api-url", default=API_URL)
    ap.add_argument("reports", nargs="*", help="SUMMARY_*.json paths (default: all in reports/)")
    args = ap.parse_args()

    reports = collect_reports(args.reports)
    if not reports:
        logger.error("no SUMMARY reports found in %s", REPORTS_DIR)
        return 1

    try:
        import httpx
    except ImportError:
        logger.error("httpx not installed; pip install httpx")
        return 2

    # Pre-submit dedup: the bench-site API has NO server-side dedup (POST 201
    # always, no 409). Fetch existing entries once and skip any whose
    # (model, quant, tg, pp, ttft) key already exists. Without this, re-running
    # the uploader pollutes the site with duplicate rows that cannot be
    # deleted (no DELETE endpoint — only GET/POST).
    existing_keys: set[tuple] = set()
    try:
        resp = httpx.get(args.api_url, params={"limit": 500}, timeout=20.0)
        if resp.status_code == 200:
            for e in resp.json().get("data", []):
                existing_keys.add(
                    (
                        e.get("modelName"),
                        e.get("quantization"),
                        round(float(e.get("tgTps") or 0), 1),
                        round(float(e.get("ppTps") or 0), 1),
                        round(float(e.get("ttftMs") or 0), 1),
                    )
                )
            logger.info("dedup: %d existing entries on site", len(existing_keys))
    except Exception as e:
        logger.warning("dedup fetch failed (%s); proceeding without guard", e)

    total = 0
    created = 0
    dup = 0
    errs = 0
    for rep in reports:
        data = json.loads(rep.read_text())
        ts = data.get("timestamp", rep.stem)
        results = data.get("results", [])
        logger.info("report %s: %d results", rep.name, len(results))
        for r in results:
            payload = result_to_payload(r, ts)
            if payload["tg_tps"] <= 0.0:
                logger.info("  skip errored/empty: %s", payload["model_name"])
                continue
            key = (
                payload["model_name"],
                payload["quantization"],
                payload["tg_tps"],
                payload["pp_tps"],
                payload["ttft_ms"],
            )
            if key in existing_keys:
                logger.info(
                    "  ⏭️  already on site, skip: %s %s tg=%.1f",
                    payload["model_name"],
                    payload["quantization"],
                    payload["tg_tps"],
                )
                dup += 1
                continue
            total += 1
            if args.dry_run:
                logger.info(
                    "  [dry-run] %s %s tg=%.1f pp=%.1f ttft=%.1fms",
                    payload["model_name"],
                    payload["quantization"],
                    payload["tg_tps"],
                    payload["pp_tps"],
                    payload["ttft_ms"],
                )
                continue
            try:
                resp = httpx.post(args.api_url, json=payload, timeout=20.0)
                if resp.status_code == 201:
                    body = resp.json()
                    logger.info(
                        "  ✅ created id=%s  %s %s tg=%.1f",
                        body.get("id"),
                        payload["model_name"],
                        payload["quantization"],
                        payload["tg_tps"],
                    )
                    created += 1
                elif resp.status_code == 409:
                    body = resp.json()
                    logger.info(
                        "  ⏭️  duplicate id=%s  %s",
                        body.get("existing_id") or body.get("id"),
                        payload["model_name"],
                    )
                    dup += 1
                else:
                    logger.warning(
                        "  ⚠️ %s %s -> HTTP %s: %s",
                        payload["model_name"],
                        payload["quantization"],
                        resp.status_code,
                        resp.text[:200],
                    )
                    errs += 1
            except Exception as e:
                logger.error("  ✗ %s: %s", payload["model_name"], e)
                errs += 1

    logger.info(
        "done: total=%d created=%d duplicate=%d error=%d%s",
        total,
        created,
        dup,
        errs,
        " (dry-run)" if args.dry_run else "",
    )
    return 0 if errs == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
