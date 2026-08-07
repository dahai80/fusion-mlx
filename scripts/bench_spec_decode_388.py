#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Speculative-decoding speedup benchmark for issue #388.
# Importers/callers: run manually against a live fusion-mlx server; results
#   feed the #388 benchmark report (BENCHMARK_SPEC_388.md).
# Affected API: reads existing POST /v1/completions (streaming) endpoint.
# Data schemas: per-run JSON {ttft_s, total_s, n_tokens, decode_tps,
#   overall_tps}; emits a summary table + JSON dump.
# User verbatim instruction: "启动3个功能issue的修复落地" (#388 acceptance:
#   >=1.5x speedup + benchmark report).
# Usage:
#   python scripts/bench_spec_decode_388.py --model llama8b --runs 3 \
#       --tag spec_off --out scripts/bench_spec_388_off.json
import argparse
import json
import time
import urllib.request

API = "http://127.0.0.1:11434/v1/completions"
TOKEN = "dahai168"

# Per-family chat-template prompts so the target doesn't hit a stop token
# on the first generated token (the bare bench prompt does for Llama).
PROMPTS = {
    "qwen": (
        "<|im_start|>user\n"
        "Write a detailed 400-word essay about the ocean, its ecosystems, "
        "and its importance to life on Earth.<|im_end|>\n"
        "<|im_start|>assistant\n\n"
    ),
    "llama": (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "Write a detailed 400-word essay about the ocean, its ecosystems, "
        "and its importance to life on Earth.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
}


def stream_bench(model, prompt, max_tokens, temperature):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        API,
        data=body,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )
    t_start = time.perf_counter()
    ttft = None
    token_times = []
    n_tokens = 0
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                text = delta.get("content") or ""
                if text:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t_start
                    token_times.append(now)
            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens"):
                n_tokens = max(n_tokens, usage["completion_tokens"])
    t_end = time.perf_counter()
    total = t_end - t_start
    decode_span = t_end - (t_start + (ttft or 0.0))
    decode_tps = (max(n_tokens - 1, 1) / decode_span) if decode_span > 0 else 0.0
    overall_tps = n_tokens / total if total > 0 else 0.0
    return {
        "model": model,
        "max_tokens": max_tokens,
        "n_tokens": n_tokens,
        "ttft_s": round(ttft or 0.0, 3),
        "total_s": round(total, 3),
        "decode_tps": round(decode_tps, 2),
        "overall_tps": round(overall_tps, 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama8b")
    p.add_argument("--family", default="llama", choices=["qwen", "llama"])
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--tag", default="run")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    prompt = PROMPTS[args.family]
    results = []
    for i in range(args.runs):
        r = stream_bench(args.model, prompt, args.max_tokens, args.temperature)
        r["tag"] = args.tag
        results.append(r)
        print(f"[{args.tag}] run {i + 1}: {r}")
    if results:
        avg_ttft = sum(r["ttft_s"] for r in results) / len(results)
        avg_decode = sum(r["decode_tps"] for r in results) / len(results)
        avg_overall = sum(r["overall_tps"] for r in results) / len(results)
        print(
            f"\n[{args.tag}] AVG ttft={avg_ttft:.3f}s "
            f"decode_tps={avg_decode:.2f} overall_tps={avg_overall:.2f}"
        )
        summary = {
            "tag": args.tag,
            "model": args.model,
            "family": args.family,
            "runs": results,
            "avg_ttft_s": round(avg_ttft, 3),
            "avg_decode_tps": round(avg_decode, 2),
            "avg_overall_tps": round(avg_overall, 2),
        }
        if args.out:
            with open(args.out, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
