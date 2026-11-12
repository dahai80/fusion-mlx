# SPDX-License-Identifier: Apache-2.0
"""Scheduler overhead profile: engine floor vs live scheduler step wall.

Catch-up plan P1-3 (audit X3): the 2.2x-vs-llama.cpp single-stream gap is
attributed to the Python scheduler's per-step dispatch, but there was no
in-repo artifact measuring it. This script produces the numbers:

  Phase A  engine floor   - in-process greedy decode of a raw mlx-lm model,
                            per-step wall (model forward + eval only).
  Phase A' cProfile split - the same loop under cProfile: top cost centers
                            (Metal sync vs Python bookkeeping).
  Phase B  scheduler wall - from a RUNNING fusion-mlx server's /metrics
                            (fusion_mlx_sched_decode_*), per-step wall of
                            the full scheduler pure-decode path.

  overhead % = B / A - 1   (how much the scheduler fabric adds per token)

Usage:
  # Phase A + A' only (offline, no server):
  python benchmarks/scheduler_overhead_profile.py --model <path> --steps 256

  # Full comparison against a serving engine on the same machine/model:
  python benchmarks/scheduler_overhead_profile.py --model <path> \
      --metrics-url http://127.0.0.1:11434/metrics

Methodology: run on an idle machine (check `fusion-sv status` first), one
model, same dtype. Phase B averages over the server's whole lifetime
counter, so restart the server or read right after a decode burst for a
clean window.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
import urllib.request

import mlx.core as mx


def engine_floor(model, tokenizer, prompt_ids, steps):
    """Greedy decode loop; returns (per_step_ms, last_logits)."""
    cache = None
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt_ids], dtype=mx.uint32), cache=cache)
    mx.eval(logits)
    token = int(mx.argmax(logits[0, -1]))
    t = mx.array([[token]], dtype=mx.uint32)
    t0 = time.perf_counter()
    for _ in range(steps):
        logits = model(t, cache=cache)
        token = int(mx.argmax(logits[0, -1]))
        t = mx.array([[token]], dtype=mx.uint32)
    mx.eval(logits)
    wall = time.perf_counter() - t0
    return wall / steps * 1000.0


def profile_split(model, tokenizer, prompt_ids, steps, top=20):
    """cProfile the decode loop; print top cumtime rows."""
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt_ids], dtype=mx.uint32), cache=cache)
    mx.eval(logits)
    token = int(mx.argmax(logits[0, -1]))
    t = mx.array([[token]], dtype=mx.uint32)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(steps):
        logits = model(t, cache=cache)
        token = int(mx.argmax(logits[0, -1]))
        t = mx.array([[token]], dtype=mx.uint32)
    mx.eval(logits)
    pr.disable()
    buf = io.StringIO()
    stats = pstats.Stats(pr, stream=buf)
    stats.sort_stats("cumulative").print_stats(top)
    return buf.getvalue()


def scheduler_wall_from_metrics(url):
    """Read fusion_mlx_sched_decode_* from a running server's /metrics."""
    import json
    import os

    headers = {}
    key_path = os.path.expanduser("~/.fusion-mlx/settings.json")
    try:
        with open(key_path) as f:
            api_key = json.load(f).get("auth", {}).get("api_key")
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
    except OSError:
        pass
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as r:
        text = r.read().decode()
    steps = wall = None
    for line in text.splitlines():
        if line.startswith("fusion_mlx_sched_decode_steps_total"):
            steps = float(line.rsplit(" ", 1)[1])
        elif line.startswith("fusion_mlx_sched_decode_wall_seconds"):
            wall = float(line.rsplit(" ", 1)[1])
    if not steps or wall is None:
        return None
    return wall / steps * 1000.0, int(steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. ")
    ap.add_argument("--profile-steps", type=int, default=128)
    ap.add_argument("--metrics-url", default=None)
    args = ap.parse_args()

    from mlx_lm import load

    print("loading", args.model, flush=True)
    model, tokenizer = load(args.model)
    ids = tokenizer.encode(args.prompt, add_special_tokens=True)

    floor_ms = engine_floor(model, tokenizer, ids, args.steps)
    print(f"engine_floor_ms_per_step {floor_ms:.3f} (steps={args.steps})")

    print(profile_split(model, tokenizer, ids, args.profile_steps))

    if args.metrics_url:
        got = scheduler_wall_from_metrics(args.metrics_url)
        if got is None:
            print("scheduler_wall: metrics not available (need fusion-mlx "
                  "server with decode traffic; counters land in "
                  "fusion_mlx_sched_decode_*)")
        else:
            sched_ms, steps = got
            print(f"scheduler_wall_ms_per_step {sched_ms:.3f} (steps={steps})")
            print(f"scheduler_overhead_vs_floor {(sched_ms / floor_ms - 1) * 100:.1f}%")


if __name__ == "__main__":
    main()
