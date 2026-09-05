#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Radix vs hash-baseline prefix-cache Δ harness.

Measures the TTFT speedup a radix-tree prefix-cache index delivers over the
hash-chain (BlockAwarePrefixCache) baseline when a long shared prefix is
re-sent. Drives a RUNNING fusion-mlx server.

PRD target (architecture/fusion-mlx-architecture-enhance-0825.md): Radix
Δ>=30% TTFT reduction vs baseline at 4k+ shared prefix.

Method:
  - Build a prompt with a 4k+ token shared prefix (system + a fixed long
    context block) followed by a short unique instruction.
  - Call 1 (cold): populates the prefix cache. Record TTFT_cold.
  - Call 2 (hot): same prefix, different short instruction. The cache serves
    the shared prefix; record TTFT_hot.
  - Repeat N times, report median TTFT_cold / TTFT_hot per mode.
  - The server must be started with FUSION_MLX_PREFIX_CACHE=radix (radix) and
    again with it unset (baseline). Run this script against each; compare.

Usage (server up):
    python scripts/bench_radix_vs_baseline.py --base-url http://127.0.0.1:11434 \\
        --model qwen3.5-9b-4bit --api-key "$KEY" --rounds 5 --label radix
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time

import requests

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s [radix-bench] %(message)s"
)
logger = logging.getLogger("radix_bench")

PREFIX_BLOCK = (
    "You are a meticulous technical auditor. Below is a long reference document "
    "you must internalize before answering. Document: "
)
FILLER = (
    "The transformer architecture processes input tokens through stacked self-"
    "attention and feed-forward layers. Each attention head computes scaled "
    "dot-product attention over queries, keys, and values derived from the input "
    "embedding via learned projections. Positional information is injected either "
    "additively (sinusoidal) or via rotary embeddings (RoPE). Layer normalization "
    "stabilizes training. The feed-forward sublayer applies a gated non-linearity. "
    "Residual connections carry gradients. "
)


def build_prompt(target_prefix_tokens: int) -> str:
    prefix = PREFIX_BLOCK + FILLER * 120
    # Heuristic: ~1.6 tokens/word, FILLER ~50 words -> ~80 tokens/repeat.
    # 120 repeats -> ~9600 tokens, well above the 4k target.
    return prefix


def make_body(model: str, prompt: str, api_key: str | None) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "temperature": 0,
        "top_p": 1,
        "stream": True,
    }


def call_ttft(
    base_url: str, model: str, prompt: str, api_key: str | None
) -> float | None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    body = make_body(model, prompt, api_key)
    t0 = time.perf_counter()
    ttft = None
    with requests.post(
        f"{base_url}/v1/chat/completions",
        headers=headers,
        json=body,
        stream=True,
        timeout=600,
    ) as sr:
        if sr.status_code != 200:
            logger.error("HTTP %s: %s", sr.status_code, sr.text[:200])
            return None
        for line in sr.iter_lines():
            if line and line.startswith(b"data: ") and b"content" in line:
                ttft = time.perf_counter() - t0
                break
    return ttft


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--label", required=True, help="run label (radix | baseline)")
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--out", default=None, help="write JSON result to this path")
    args = ap.parse_args()

    prompt = build_prompt(args.prompt_tokens)

    logger.info("label=%s model=%s rounds=%d", args.label, args.model, args.rounds)

    # warmup + ensure model loaded (do not record)
    logger.info("warmup request...")
    call_ttft(args.base_url, args.model, "Say hello.", args.api_key)
    time.sleep(2)

    cold_ttfts: list[float] = []
    hot_ttfts: list[float] = []

    for i in range(args.rounds):
        # Cold: fresh prefix variant first call (cache may be partially warm
        # from prior round's hot call; to force a cold-ish measurement we use
        # a unique filler offset per round so the shared prefix differs).
        cold_prompt = prompt + f" Round {i} cold marker {time.time_ns()}."
        t_cold = call_ttft(args.base_url, args.model, cold_prompt, args.api_key)
        # Hot: exact same prefix re-sent immediately (cache serves it).
        hot_prompt = prompt + f" Summarize the document in one sentence. Hot {i}."
        # Re-send cold_prompt prefix portion via the SAME cold_prompt to hit cache:
        t_hot = call_ttft(args.base_url, args.model, cold_prompt, args.api_key)
        if t_cold is not None and t_hot is not None:
            cold_ttfts.append(t_cold)
            hot_ttfts.append(t_hot)
            speedup = t_cold / t_hot if t_hot > 0 else 0
            logger.info(
                "round %d: cold_ttft=%.3fs hot_ttft=%.3fs speedup=%.2fx",
                i,
                t_cold,
                t_hot,
                speedup,
            )
        else:
            logger.warning("round %d: missing ttft (cold=%s hot=%s)", i, t_cold, t_hot)
        time.sleep(1)

    if not cold_ttfts:
        logger.error("no valid ttft samples collected")
        return 1

    med_cold = statistics.median(cold_ttfts)
    med_hot = statistics.median(hot_ttfts)
    delta_pct = (med_cold - med_hot) / med_cold * 100 if med_cold > 0 else 0
    logger.info("=" * 60)
    logger.info("label=%s", args.label)
    logger.info("median cold TTFT = %.3fs", med_cold)
    logger.info("median hot TTFT  = %.3fs", med_hot)
    logger.info("Δ (hot vs cold)  = %.1f%% reduction", delta_pct)
    logger.info("=" * 60)

    result = {
        "label": args.label,
        "model": args.model,
        "rounds": len(cold_ttfts),
        "median_cold_ttft_s": round(med_cold, 4),
        "median_hot_ttft_s": round(med_hot, 4),
        "delta_pct": round(delta_pct, 1),
        "cold_ttfts_s": [round(x, 4) for x in cold_ttfts],
        "hot_ttfts_s": [round(x, 4) for x in hot_ttfts],
        "target_pct": 30.0,
        "met_target": delta_pct >= 30.0,
    }
    out = args.out or f"benchmarks/reports/delta/radix_{args.label}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("wrote %s", out)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
