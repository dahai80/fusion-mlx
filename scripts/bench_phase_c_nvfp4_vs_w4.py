#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase C: nvfp4 vs affine-W4 weight-quant comparison.

Loads each converted model, generates a fixed prompt, reports tok/s and the
raw output text (for a coherence eyeball). Run from the fusion-mlx venv.
"""
import sys
import time

import mlx.core as mx
from mlx_lm import load, generate

PROMPT = "The capital of France is"
MODELS = [
    ("affine-w4", "/tmp/qwen-w4-test"),
    ("nvfp4", "/tmp/qwen-nvfp4-test"),
]


def bench(name, path):
    model, tokenizer = load(path)
    messages = [{"role": "user", "content": PROMPT}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    t0 = time.perf_counter()
    out = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=64,
        verbose=False,
    )
    dt = time.perf_counter() - t0
    tps = 64.0 / dt if dt > 0 else 0.0
    text = out.strip().replace("\n", " ")[:120]
    print(f"[{name}] {tps:6.2f} tok/s | {dt:5.2f}s | out: {text!r}")
    return tps


def main():
    mx.set_default_device(mx.gpu)
    results = {}
    for name, path in MODELS:
        try:
            results[name] = bench(name, path)
        except Exception as e:
            print(f"[{name}] FAILED: {e!r}", file=sys.stderr)
            results[name] = None
    print("---")
    if all(results.values()):
        a, n = results["affine-w4"], results["nvfp4"]
        delta = (n - a) / a * 100.0 if a else 0.0
        print(f"nvfp4 vs affine-w4: {delta:+.1f}% ({a:.2f} -> {n:.2f} tok/s)")


if __name__ == "__main__":
    main()
