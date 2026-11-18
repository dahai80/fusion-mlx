#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M9 long-trajectory bench (arXiv 2610.28577): KV segment drops under
cache capacity pressure.

Two arms, identical prompts, T=0, dedicated server per arm:
  nodrop - tool-output blocks stay indexed; capacity evicts oldest
           blocks, destroying the retained prefix -> re-prefill.
  drop   - each turn drops the previous tool-output span mid-decode
           (POST /v1/cache/segments/drop); footprint stays under
           capacity -> retained prefix keeps hitting.

Prompt shape keeps the shared prefix intact across turns: tool output
is a SUFFIX of each turn's prompt (retained text -> question -> tool
output), so turn k+1 shares the retained prefix chain with turn k and
diverges only at new content.

Usage:
    python scripts/bench_m9_trajectory.py --model mlx-community--Qwen3-0.6B-4bit \
        --port 8898 --turns 12 --tool-tokens 1200 --max-cache-blocks 96 \
        --json /tmp/m9_bench.json
"""

import argparse
import json
import logging
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

logger = logging.getLogger("bench_m9")

SYS = logging.StreamHandler()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[SYS])

MARKER = "\n[assistant note {}]\n"
QUESTION = "\nuser: Summarize finding {} in one short line.\nassistant:\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", default=None, help="HF snapshot dir with tokenizer.json (offsets)")
    parser.add_argument("--port", type=int, default=8898)
    parser.add_argument("--api-key", default="bench-key")
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--tool-tokens", type=int, default=1200)
    parser.add_argument("--max-cache-blocks", type=int, default=96)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args()

    tokenizer = _load_tokenizer(args.model_dir, args.model)
    turns = _build_trajectory(tokenizer, args.turns, args.tool_tokens)

    results = {}
    for arm in ("nodrop", "drop"):
        logger.info("=== arm %s starting ===", arm)
        proc = _start_server(args, arm)
        try:
            _wait_ready(args.port)
            results[arm] = _run_trajectory(args, tokenizer, turns, arm)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()

    _report(results, args)


def _load_tokenizer(model_dir, model):
    from transformers import AutoTokenizer

    path = model_dir
    if not path:
        import glob
        import os

        repo_dir = model.replace("/", "--")
        pat = os.path.expanduser(
            f"~/.fusion-mlx/models/models--{repo_dir}/snapshots/*/"
        )
        hits = sorted(glob.glob(pat))
        if not hits:
            raise SystemExit(f"no snapshot dir for {model}")
        path = hits[-1]
    logger.info("tokenizer: %s", path)
    return AutoTokenizer.from_pretrained(path, use_fast=True)


TOOL_MARK = "tool_output: "


def _tool_span(tokenizer, prompt):
    """Token offsets [start, end) of the trailing tool-output span."""
    idx = prompt.rindex(TOOL_MARK)
    start = len(tokenizer.encode(prompt[:idx]))
    end = len(tokenizer.encode(prompt))
    return start, end


def _start_server(args, arm):
    import os

    env = dict(os.environ)
    env["HF_MIRROR"] = "https://hf-mirror.com"
    if arm == "drop":
        env["FUSION_MLX_KV_DROP_API"] = "1"
    model_arg = args.model
    cmd = [
        sys.executable,
        "-m",
        "fusion_mlx",
        "serve",
        model_arg,
        "--port",
        str(args.port),
        "--api-key",
        args.api_key,
        "--max-cache-blocks",
        str(args.max_cache_blocks),
        "--paged-cache-block-size",
        str(args.block_size),
    ]
    logger.info("server cmd: %s (arm=%s)", " ".join(cmd[:6]), arm)
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=open(f"/tmp/m9_bench_{arm}_server.log", "w"),
        stderr=subprocess.STDOUT,
    )


def _wait_ready(port, timeout_s=600):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as r:
                if r.status == 200:
                    logger.info("server ready on %d", port)
                    return
        except urllib.error.URLError:
            pass
        time.sleep(2)
    raise SystemExit(f"server not ready after {timeout_s}s")


def _post_json(url, body, api_key, timeout=30):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _stream_turn(args, prompt, drop_span=None):
    """Stream one completion; returns (ttft, request_id, usage, text).

    drop_span: token offsets [start, end) posted to the drop route once
    the request is active (after the first content chunk) — the M9 V1
    semantic: the request drops its own trailing stale segment
    mid-decode. Posting inline blocks SSE reads for ~1 RTT; the server
    keeps decoding.
    """
    body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/completions",
        data=data,
        headers={"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"},
    )
    ttft = None
    request_id = None
    usage = None
    drop_queued = None
    pieces = []
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if request_id is None:
                request_id = chunk.get("id")
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("text") or ""
                if delta and ttft is None:
                    ttft = time.perf_counter() - t0
                if delta and drop_span and request_id:
                    try:
                        resp = _post_json(
                            f"http://127.0.0.1:{args.port}/v1/cache/segments/drop",
                            {"request_id": request_id, "segments": [list(drop_span)]},
                            args.api_key,
                        )
                        drop_queued = resp.get("queued_segments")
                    except urllib.error.HTTPError as exc:
                        logger.warning(
                            "drop failed: %s %s", exc.code, exc.read()[:200]
                        )
                        drop_queued = None
                    drop_span = None
                pieces.append(delta)
    return ttft, request_id, usage, "".join(pieces), drop_queued


def _run_trajectory(args, tokenizer, prompts, arm):
    rows = []
    for k, prompt in enumerate(prompts, start=1):
        drop_span = _tool_span(tokenizer, prompt) if arm == "drop" and k > 1 else None
        ttft, request_id, usage, text, drop_queued = _stream_turn(args, prompt, drop_span)
        row = {
            "turn": k,
            "ttft_s": round(ttft, 4) if ttft else None,
            "prompt_tokens": usage.get("prompt_tokens") if usage else None,
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") if usage else None,
            "completion_tokens": usage.get("completion_tokens") if usage else None,
            "text": text,
            "drop_queued": drop_queued,
        }
        rows.append(row)
        logger.info(
            "arm=%s turn=%d ttft=%s prompt=%s cached=%s drop=%s",
            arm,
            k,
            row["ttft_s"],
            row["prompt_tokens"],
            row["cached_tokens"],
            row.get("drop_queued"),
        )
        prev_span = _tool_span(tokenizer, prompt)
    return rows


def _report(results, args):
    summary = {}
    for arm, rows in results.items():
        ttfts = [r["ttft_s"] for r in rows if r["ttft_s"]]
        cached = [r["cached_tokens"] or 0 for r in rows]
        summary[arm] = {
            "turns": len(rows),
            "ttft_median_s": round(statistics.median(ttfts), 4),
            "ttft_last5_median_s": round(statistics.median(ttfts[-5:]), 4),
            "cached_total": sum(cached),
            "cached_last5": sum(cached[-5:]),
        }
    if "nodrop" in summary and "drop" in summary:
        a, b = summary["nodrop"], summary["drop"]
        if a["ttft_last5_median_s"]:
            summary["ttft_improvement_last5"] = round(
                1 - b["ttft_last5_median_s"] / a["ttft_last5_median_s"], 4
            )
    out = {"config": vars(args), "summary": summary, "rows": results}
    logger.info("SUMMARY %s", json.dumps(summary, indent=2))
    if args.json_path:
        with open(args.json_path, "w") as fh:
            json.dump(out, fh, indent=2)
        logger.info("wrote %s", args.json_path)


def _filler_text(tokenizer, n_tokens, seed):
    import random

    rng = random.Random(seed)
    vocab = [f"tok{i}" for i in range(5000)]
    words = []
    while True:
        words.append(rng.choice(vocab))
        if len(tokenizer.encode(" ".join(words))) >= n_tokens:
            break
    return " ".join(words)


def _build_trajectory(tokenizer, turns, tool_tokens):
    """Turn k prompt: system + notes 1..k + questions 1..k + tool_k.

    Tool output is the trailing span so turn k+1 shares the full
    retained prefix chain (system+notes+questions) with turn k.
    Per-turn filler text avoids cross-turn block-hash aliasing.
    """
    tool_texts = [
        _filler_text(tokenizer, tool_tokens, seed=100 + k) for k in range(1, turns + 1)
    ]
    prompts = []
    for k in range(1, turns + 1):
        parts = ["system: You are a research agent.\n"]
        for j in range(1, k + 1):
            parts.append(MARKER.format(j))
            parts.append(QUESTION.format(j))
        parts.append("tool_output: ")
        parts.append(tool_texts[k - 1])
        prompts.append("".join(parts))
    return prompts


if __name__ == "__main__":
    main()
