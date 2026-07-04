#!/usr/bin/env python3
"""fusion-mlx decode throughput benchmark (HTTP).

Matches the v0.4.0 baseline methodology (bench_fusion_only.json):
  - HTTP /v1/chat/completions, non-stream
  - Qwen3.6-27B-mxfp8, spec-decode K=3 (server default)
  - 1 warmup + 3 measured runs, single request
  - tok/s = completion_tokens / wall_time

Success criterion: mean_tok_s >= 29.24 (v0.4.0 baseline).
"""
import argparse
import json
import logging
import statistics
import sys
import time
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bench_fusion")

BASE = "http://127.0.0.1:11434"
MODEL = "Qwen3.6-27B-mxfp8"
API_KEY = "dahai168"
BASELINE_TOK_S = 29.24


def chat(max_tokens: int, timeout: float = 180.0) -> dict:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Write a short essay about the ocean."}],
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    dt = time.perf_counter() - t0
    ct = body.get("usage", {}).get("completion_tokens", 0)
    return {"dt_s": round(dt, 3), "ct": ct, "tok_s": round(ct / dt, 2) if dt > 0 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    logger.info("benchmark: model=%s max_tokens=%d runs=%d warmup=%d", MODEL, args.max_tokens, args.runs, args.warmup)

    logger.info("warmup (%d run(s))...", args.warmup)
    for i in range(args.warmup):
        r = chat(args.max_tokens)
        logger.info("warmup[%d]: %s", i, r)

    runs = []
    for i in range(args.runs):
        r = chat(args.max_tokens)
        runs.append(r)
        logger.info("run[%d]: %s", i, r)

    toks = [r["tok_s"] for r in runs]
    mean = statistics.mean(toks)
    median = statistics.median(toks)
    result = {
        "model": MODEL,
        "max_tokens": args.max_tokens,
        "runs": runs,
        "mean_tok_s": round(mean, 2),
        "median_tok_s": round(median, 2),
        "baseline_tok_s": BASELINE_TOK_S,
        "ratio_vs_baseline": round(mean / BASELINE_TOK_S, 3),
        "pass": mean >= BASELINE_TOK_S,
    }
    logger.info("RESULT: mean=%.2f tok/s median=%.2f tok/s baseline=%.2f ratio=%.3fx pass=%s",
                mean, median, BASELINE_TOK_S, mean / BASELINE_TOK_S, result["pass"])
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
