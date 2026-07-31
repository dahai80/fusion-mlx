#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TTFT prefix-cache benchmark for fusion-mlx.

Measures Time-To-First-Token with prefix cache ON vs OFF, proving that
fusion-mlx's BlockAwarePrefixCache delivers competitive TTFT.

Usage:
    # Start server first:
    fusion-mlx serve --model Qwen3.6-27B-mxfp8

    # Run TTFT benchmark:
    python scripts/bench_ttft.py

    # Custom endpoint and model:
    python scripts/bench_ttft.py --base-url http://localhost:8897 --model Qwen3.5-9B-6bit

    # More iterations for stable results:
    python scripts/bench_ttft.py --iterations 20
"""

import argparse
import json
import logging
import statistics
import sys
import time

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROMPT_CONFIGS = [
    {
        "label": "short",
        "prefix": "The capital of France is",
        "suffix": " Paris. What is the capital of Germany?",
    },
    {
        "label": "medium",
        "prefix": (
            "Artificial intelligence has transformed many industries. "
            "From healthcare to finance, AI systems are being deployed to solve "
            "complex problems. Machine learning models can now process vast amounts "
            "of data and extract meaningful patterns. Deep learning has enabled "
            "breakthroughs in computer vision, natural language processing, and "
            "speech recognition. The field continues to evolve rapidly with new "
            "architectures and training techniques emerging regularly. "
            "Reinforcement learning has shown remarkable results in game playing "
            "and robotic control. Transfer learning allows models trained on one "
            "task to be adapted for another with minimal additional training. "
        ),
        "suffix": "What are the key challenges in deploying AI systems in production?",
    },
    {
        "label": "long",
        "prefix": (
            "The history of computing spans several centuries, from early mechanical "
            "calculators to modern quantum computers. The abacus, invented thousands "
            "of years ago, represents one of the earliest computing devices. In the "
            "17th century, Blaise Pascal built the mechanical calculator. Charles "
            "Babbage designed the Analytical Engine in the 1830s, which contained "
            "many concepts found in modern computers. Ada Lovelace wrote what is "
            "considered the first computer program for Babbage's machine. The 20th "
            "century saw the development of electronic computers. ENIAC, completed "
            "in 1945, was one of the first general-purpose electronic digital "
            "computers. The invention of the transistor in 1947 revolutionized "
            "computing, leading to smaller, faster, and more reliable machines. "
            "Integrated circuits further miniaturized computing components. The "
            "microprocessor, introduced in the early 1970s, put an entire CPU on "
            "a single chip. Personal computers emerged in the late 1970s and early "
            "1980s, bringing computing to homes and offices. The internet, "
            "originally a military research project called ARPANET, transformed "
            "into a global communication network. The World Wide Web, invented by "
            "Tim Berners-Lee in 1989, made the internet accessible to everyone. "
            "Mobile computing took off with smartphones in the late 2000s. Cloud "
            "computing emerged as a paradigm for on-demand computing resources. "
            "Machine learning and artificial intelligence have become central to "
            "modern computing, with neural networks achieving human-level "
            "performance on many tasks. Quantum computing promises to solve "
            "problems that are intractable for classical computers. The field "
            "continues to advance at an unprecedented pace, with new breakthroughs "
            "announced regularly. "
        ),
        "suffix": "What was the significance of the transistor invention for computing?",
    },
]


def measure_ttft(
    base_url: str,
    model: str,
    prompt: str,
    stream: bool = True,
) -> float | None:
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": stream,
    }

    start = time.perf_counter()

    if stream:
        try:
            resp = requests.post(url, json=payload, stream=True, timeout=60)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    line_str = line.decode("utf-8", errors="replace")
                    if line_str.startswith("data: ") and line_str != "data: [DONE]":
                        first_token_time = time.perf_counter()
                        resp.close()
                        return first_token_time - start
        except requests.RequestException as e:
            logger.warning("Stream request failed: %s", e)
            return None
    else:
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            first_token_time = time.perf_counter()
            return first_token_time - start
        except requests.RequestException as e:
            logger.warning("Non-stream request failed: %s", e)
            return None

    return None


def run_benchmark(
    base_url: str,
    model: str,
    iterations: int,
) -> dict:
    logger.info("Starting TTFT prefix-cache benchmark")
    logger.info("Model: %s  Iterations: %d", model, iterations)

    results = {}

    for config in PROMPT_CONFIGS:
        label = config["label"]
        prefix = config["prefix"]
        suffix = config["suffix"]
        full_prompt = prefix + suffix

        logger.info("  Config: %s (prefix=%d chars, total=%d chars)", label, len(prefix), len(full_prompt))

        cold_ttfts = []
        for i in range(iterations):
            unique_prompt = f"[Run {i}] " + full_prompt
            ttft = measure_ttft(base_url, model, unique_prompt)
            if ttft is not None:
                cold_ttfts.append(ttft)
            if (i + 1) % 5 == 0:
                logger.info("    cold %d/%d", i + 1, iterations)

        measure_ttft(base_url, model, full_prompt)
        time.sleep(0.1)

        warm_ttfts = []
        for i in range(iterations):
            ttft = measure_ttft(base_url, model, full_prompt)
            if ttft is not None:
                warm_ttfts.append(ttft)
            if (i + 1) % 5 == 0:
                logger.info("    warm %d/%d", i + 1, iterations)

        def stats(vals):
            if not vals:
                return {"mean": None, "median": None, "p95": None, "min": None, "max": None}
            sorted_vals = sorted(vals)
            p95_idx = int(len(sorted_vals) * 0.95)
            return {
                "mean": round(statistics.mean(vals) * 1000, 1),
                "median": round(statistics.median(vals) * 1000, 1),
                "p95": round(sorted_vals[min(p95_idx, len(sorted_vals) - 1)] * 1000, 1),
                "min": round(min(vals) * 1000, 1),
                "max": round(max(vals) * 1000, 1),
                "n": len(vals),
            }

        cold_stats = stats(cold_ttfts)
        warm_stats = stats(warm_ttfts)

        speedup = None
        if cold_stats["mean"] and warm_stats["mean"] and warm_stats["mean"] > 0:
            speedup = round(cold_stats["mean"] / warm_stats["mean"], 2)

        results[label] = {
            "prefix_chars": len(prefix),
            "total_chars": len(full_prompt),
            "cold_ttft_ms": cold_stats,
            "warm_ttft_ms": warm_stats,
            "speedup": speedup,
        }

        logger.info(
            "    cold_mean=%.1fms  warm_mean=%.1fms  speedup=%.2fx",
            cold_stats.get("mean", 0),
            warm_stats.get("mean", 0),
            speedup or 0,
        )

    return {"model": model, "iterations": iterations, "configs": results}


def main():
    parser = argparse.ArgumentParser(description="TTFT prefix-cache benchmark")
    parser.add_argument("--base-url", default="http://localhost:8897", help="Server URL")
    parser.add_argument("--model", default=None, help="Model name (auto-detect if omitted)")
    parser.add_argument("--iterations", type=int, default=10, help="Iterations per config")
    args = parser.parse_args()

    model = args.model
    if not model:
        try:
            resp = requests.get(f"{args.base_url}/v1/models", timeout=5)
            models = resp.json().get("data", [])
            if models:
                model = models[0]["id"]
                logger.info("Auto-detected model: %s", model)
            else:
                logger.error("No models loaded on server")
                sys.exit(1)
        except Exception as e:
            logger.error("Cannot reach server at %s: %s", args.base_url, e)
            sys.exit(1)

    report = run_benchmark(args.base_url, model, args.iterations)

    print("\n" + "=" * 70)
    print("TTFT Prefix-Cache Benchmark Report")
    print("=" * 70)
    print(f"Model:      {report['model']}")
    print(f"Iterations: {report['iterations']}")
    print()

    for label, data in report["configs"].items():
        cold = data["cold_ttft_ms"]
        warm = data["warm_ttft_ms"]
        print(f"--- {label} (prefix={data['prefix_chars']} chars, total={data['total_chars']} chars) ---")
        if cold["mean"] is not None:
            print(f"  Cold TTFT:  mean={cold['mean']}ms  median={cold['median']}ms  p95={cold['p95']}ms  (n={cold['n']})")
        if warm["mean"] is not None:
            print(f"  Warm TTFT:  mean={warm['mean']}ms  median={warm['median']}ms  p95={warm['p95']}ms  (n={warm['n']})")
        if data["speedup"]:
            print(f"  Prefix cache speedup: {data['speedup']}×")
        print()

    out_path = f"bench_ttft_{model.replace('/', '_')}_{args.iterations}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Full report saved to: {out_path}")


if __name__ == "__main__":
    main()
