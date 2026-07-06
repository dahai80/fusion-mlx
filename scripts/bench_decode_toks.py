#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure decode tok/s for a fusion-mlx server via streaming.

Streams a completion, records first-token time (TTFT) and inter-token
timestamps, then reports prefill vs decode throughput.
"""
import argparse
import json
import time
import urllib.request

API = "http://127.0.0.1:11434/v1/completions"
TOKEN = "dahai168"
PROMPT = (
    "<|im_start|>user\n"
    "Write a detailed 500-word essay about the ocean, its ecosystems, "
    "and its importance to life on Earth.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
)


def stream_bench(model: str, max_tokens: int, temperature: float) -> dict:
    body = json.dumps({
        "model": model,
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
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
            payload = line[len("data:"):].strip()
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
    # Decode phase = everything after TTFT (prefill + first token excluded).
    decode_span = (t_end - (t_start + (ttft or 0.0)))
    decode_tps = (max(n_tokens - 1, 1) / decode_span) if decode_span > 0 else 0.0
    overall_tps = n_tokens / total if total > 0 else 0.0
    chunk_decode = 0.0
    if len(token_times) > 1:
        cs = token_times[-1] - token_times[0]
        chunk_decode = (len(token_times) - 1) / cs if cs > 0 else 0.0
    return {
        "model": model,
        "max_tokens": max_tokens,
        "n_tokens": n_tokens,
        "n_chunks": len(token_times),
        "ttft_s": round(ttft or 0.0, 3),
        "total_s": round(total, 3),
        "decode_tps": round(decode_tps, 2),
        "chunk_decode_tps": round(chunk_decode, 2),
        "overall_tps": round(overall_tps, 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen3.6-27B-mxfp8")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--runs", type=int, default=3)
    args = p.parse_args()
    results = []
    for i in range(args.runs):
        r = stream_bench(args.model, args.max_tokens, args.temperature)
        results.append(r)
        print(f"run {i + 1}: {r}")
    if results:
        avg_decode = sum(r["decode_tps"] for r in results) / len(results)
        avg_overall = sum(r["overall_tps"] for r in results) / len(results)
        avg_chunk = sum(r["chunk_decode_tps"] for r in results) / len(results)
        print(f"\nAVG decode_tps={avg_decode:.2f} chunk_decode_tps={avg_chunk:.2f} overall_tps={avg_overall:.2f}")


if __name__ == "__main__":
    main()
