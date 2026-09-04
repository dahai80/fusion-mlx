#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproducible fusion-mlx throughput benchmark.

Drives a RUNNING fusion-mlx server's public /v1/chat/completions endpoint
with a fixed prompt + sampling, measures tokens/sec, TTFT, and wall time
per model, and writes one JSON report per model under benchmarks/reports/.

Usage (server must be up: ~/claude-home/fusion-mlx/start.sh start):
    .venv/bin/python benchmarks/run_bench.py --model qwen3-4b-4bit
    .venv/bin/python benchmarks/run_bench.py --all
    .venv/bin/python benchmarks/run_bench.py --models a,b,c --prompt-tokens 512 --gen 256

Reproducibility: fixed prompt, temperature=0, top_p=1, no streaming for
the timed body (one non-stream request measures total tok/s; an optional
stream pass measures TTFT). Seeds are fixed. No model weights are touched
by this script — it only sends HTTP.

Reports: benchmarks/reports/<model>_<timestamp>.json
Summary:  benchmarks/reports/SUMMARY_<timestamp>.json
Matrix:   benchmarks/MATRIX.md + benchmarks/matrix.json, built from reports by
          `generate_matrix.py` (run that, not this, to update the public table).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [bench] %(message)s",
)
logger = logging.getLogger("fusion_bench")

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_PROMPT_TOKENS = 512
DEFAULT_GEN_TOKENS = 256
WARMUP_PROMPT = "Say hello in one word."

PROMPT_TEMPLATE = (
    "Write a clear, factual explanation of how a transformer neural network "
    "handles long-range dependencies, covering self-attention, positional "
    "encoding, and layer normalization. Be precise and technical. "
    "Continue in detail: {padding}"
)


def _pad_prompt(target_tokens: int) -> str:
    pad = "The quick brown fox jumps over the lazy dog. " * 64
    return PROMPT_TEMPLATE.format(padding=pad)


def _hdr(api_key: str | None) -> dict:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _health(base_url: str, api_key: str | None) -> bool:
    try:
        r = requests.get(f"{base_url}/health", headers=_hdr(api_key), timeout=5)
        ok = r.status_code == 200
        logger.info("health %s -> %s", base_url, ok)
        return ok
    except Exception as exc:
        logger.error("health check failed: %s", exc)
        return False


def _resolve_model_alias(base_url: str, model: str, api_key: str | None) -> str:
    try:
        r = requests.get(
            f"{base_url}/v1/models", headers=_hdr(api_key), timeout=10
        )
        if r.status_code != 200:
            return model
        ids = {m.get("id", "") for m in r.json().get("data", [])}
        if model in ids:
            return model
        for cand in ids:
            if cand and model.lower() in cand.lower():
                logger.info("alias %s -> %s", model, cand)
                return cand
    except Exception as exc:
        logger.debug("model resolve failed: %s", exc)
    return model


def _bench_one(
    base_url: str,
    model: str,
    prompt_tokens: int,
    gen_tokens: int,
    api_key: str | None,
    warmup: bool = True,
) -> dict:
    model = _resolve_model_alias(base_url, model, api_key)
    prompt = _pad_prompt(prompt_tokens)
    result: dict = {
        "model": model,
        "prompt_tokens_requested": prompt_tokens,
        "gen_tokens_requested": gen_tokens,
    }

    if warmup:
        try:
            requests.post(
                f"{base_url}/v1/chat/completions",
                headers=_hdr(api_key),
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": WARMUP_PROMPT}],
                    "max_tokens": 8,
                    "temperature": 0,
                    "stream": False,
                },
                timeout=120,
            )
            logger.info("warmup done for %s", model)
        except Exception as exc:
            logger.warning("warmup failed for %s: %s", model, exc)

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen_tokens,
        "temperature": 0,
        "top_p": 1.0,
        "stream": False,
    }
    logger.info("timed request: %s gen=%d", model, gen_tokens)
    t0 = time.perf_counter()
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        headers=_hdr(api_key),
        json=body,
        timeout=600,
    )
    wall = time.perf_counter() - t0
    if resp.status_code != 200:
        result["error"] = f"HTTP {resp.status_code}: {resp.text[:300]}"
        logger.error("%s failed: %s", model, result["error"])
        return result

    data = resp.json()
    usage = data.get("usage", {}) or {}
    comp_tokens = usage.get("completion_tokens", 0)
    prompt_tok = usage.get("prompt_tokens", 0)
    result.update(
        {
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp_tokens,
            "wall_seconds": round(wall, 3),
            "tokens_per_second": round(comp_tokens / wall, 2) if wall > 0 else 0,
            "ttft_seconds": None,
        }
    )
    logger.info(
        "%s: %d tok / %.2fs = %.1f tok/s",
        model,
        comp_tokens,
        wall,
        result["tokens_per_second"],
    )

    try:
        s0 = time.perf_counter()
        ttft = None
        with requests.post(
            f"{base_url}/v1/chat/completions",
            headers=_hdr(api_key),
            json={**body, "stream": True},
            stream=True,
            timeout=600,
        ) as sr:
            if sr.status_code == 200:
                for line in sr.iter_lines():
                    if line and line.startswith(b"data: ") and b"content" in line:
                        ttft = time.perf_counter() - s0
                        break
        if ttft is not None:
            result["ttft_seconds"] = round(ttft, 3)
            logger.info("%s TTFT: %.3fs", model, ttft)
    except Exception as exc:
        logger.debug("ttft stream failed for %s: %s", model, exc)

    return result


def _stamp() -> str:
    import datetime

    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def main() -> int:
    ap = argparse.ArgumentParser(description="fusion-mlx reproducible benchmark")
    ap.add_argument("--model", action="append", default=[], help="model id (repeatable)")
    ap.add_argument("--models", help="comma-separated model ids")
    ap.add_argument("--all", action="store_true", help="bench all /v1/models ids")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    ap.add_argument("--gen", type=int, default=DEFAULT_GEN_TOKENS, help="max_tokens")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "reports"))
    args = ap.parse_args()

    models: list[str] = list(args.model)
    if args.models:
        models.extend(m.strip() for m in args.models.split(",") if m.strip())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not _health(args.base_url, args.api_key):
        logger.error("server not healthy at %s — start it first", args.base_url)
        return 2

    if args.all or not models:
        try:
            r = requests.get(
                f"{args.base_url}/v1/models", headers=_hdr(args.api_key), timeout=10
            )
            ids = [m.get("id", "") for m in r.json().get("data", [])]
            models = [i for i in ids if i]
            logger.info("--all resolved %d models", len(models))
        except Exception as exc:
            logger.error("failed to list models: %s", exc)
            return 3

    if not models:
        logger.error("no models to bench")
        return 4

    stamp = _stamp()
    reports: list[dict] = []
    for m in models:
        logger.info("==== bench %s ====", m)
        rep = _bench_one(
            args.base_url, m, args.prompt_tokens, args.gen, args.api_key
        )
        rep["timestamp"] = stamp
        rep["base_url"] = args.base_url
        safe = m.replace("/", "_")
        Path(out_dir, f"{safe}_{stamp}.json").write_text(
            json.dumps(rep, indent=2, ensure_ascii=False)
        )
        reports.append(rep)

    summary = {
        "timestamp": stamp,
        "prompt_tokens": args.prompt_tokens,
        "gen_tokens": args.gen,
        "base_url": args.base_url,
        "results": reports,
    }
    Path(out_dir, f"SUMMARY_{stamp}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    logger.info("wrote %d reports + summary to %s", len(reports), out_dir)

    print("\n=== SUMMARY ===")
    print(f"{'model':40} {'tok/s':>8} {'ttft(s)':>8} {'tokens':>7} {'wall(s)':>8}")
    for r in reports:
        if r.get("error"):
            print(f"{r['model']:40} {'ERR':>8}")
            continue
        print(
            f"{r.get('model','?'):40} {r.get('tokens_per_second',0):>8.1f} "
            f"{str(r.get('ttft_seconds','-')):>8} {r.get('completion_tokens',0):>7} "
            f"{r.get('wall_seconds',0):>8.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
