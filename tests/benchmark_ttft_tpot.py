# SPDX-License-Identifier: Apache-2.0
"""TTFT/TPOT benchmark for fusion-mlx OpenAI-compatible API.

Measures Time To First Token and Tokens Per Output Token
against a running fusion-mlx serve instance.

Usage:
    python -m tests.benchmark_ttft_tpot
    python -m tests.benchmark_ttft_tpot --url http://localhost:11435
    python -m tests.benchmark_ttft_tpot --prompt-tokens 1000 5000 100000
    python -m tests.benchmark_ttft_tpot --output-tokens 256 512 4096
"""

import argparse
import asyncio
import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_URL = "http://localhost:11435"
DEFAULT_MODEL = "qwen3.6:27B-mxfp8"

# Prompt length variants (token counts)
DEFAULT_PROMPT_TOKENS = [1_000, 5_000, 10_000, 50_000, 100_000]

# Output length variants (token counts)
DEFAULT_OUTPUT_TOKENS = [256, 512, 1_024, 2_048, 4_096]

# Number of runs per combination
DEFAULT_RUNS = 3


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    prompt_tokens: int
    output_tokens: int
    run_index: int
    ttft_sec: float = 0.0
    tpot_sec: float = 0.0
    total_sec: float = 0.0
    actual_output_tokens: int = 0
    error: str | None = None


@dataclass
class BenchmarkPlan:
    results: list[BenchmarkResult] = field(default_factory=list)

    def summary(self) -> list[dict[str, Any]]:
        groups = defaultdict(list)
        for r in self.results:
            groups[(r.prompt_tokens, r.output_tokens)].append(r)

        summary = []
        for (pt, ot), runs in sorted(groups.items()):
            ttfts = [r.ttft_sec for r in runs if not r.error]
            tpos = [r.tpot_sec for r in runs if not r.error]
            totals = [r.total_sec for r in runs if not r.error]
            actuals = [r.actual_output_tokens for r in runs if not r.error]
            if not ttfts:
                continue
            summary.append(
                {
                    "prompt_tokens": pt,
                    "output_tokens": ot,
                    "ttft_mean_s": f"{sum(ttfts) / len(ttfts):.3f}",
                    "ttft_min_s": f"{min(ttfts):.3f}",
                    "ttft_max_s": f"{max(ttfts):.3f}",
                    "tpot_mean_s": f"{sum(tpos) / len(tpos):.6f}",
                    "tpot_min_s": f"{min(tpos):.6f}",
                    "tpot_max_s": f"{max(tpos):.6f}",
                    "total_mean_s": f"{sum(totals) / len(totals):.3f}",
                    "actual_tokens_mean": f"{sum(actuals) / len(actuals):.0f}",
                }
            )
        return summary


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_prompt(target_tokens: int) -> str:
    """Generate a prompt that tokenizes to roughly target_tokens tokens.

    Uses repetitive English sentences with predictable tokenization.
    Each sentence is ~10 tokens for Qwen-like tokenizers.
    """
    sentence = "The quick brown fox jumps over the lazy dog. "
    reps = max(1, target_tokens // 10)
    return sentence * reps


# ---------------------------------------------------------------------------
# SSE stream parser
# ---------------------------------------------------------------------------


async def run_single_benchmark(
    url: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
    run_index: int,
) -> BenchmarkResult:
    """Run a single streaming request and measure TTFT + TPOT."""

    result = BenchmarkResult(
        prompt_tokens=len(prompt.split()),
        output_tokens=max_output_tokens,
        run_index=run_index,
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_output_tokens,
        "stream": True,
        "temperature": 0.1,
        "top_p": 0.9,
    }

    headers = {"Content-Type": "application/json"}

    try:
        start_time = time.monotonic()
        first_token_time = None
        chunk_times: list[float] = []
        total_content = ""

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(600.0, connect=30.0)
        ) as client:
            async with client.stream(
                "POST", f"{url}/v1/chat/completions", json=payload, headers=headers
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    result.error = f"HTTP {resp.status_code}: {body[:200]}"
                    return result

                line_buffer = ""
                async for chunk in resp.aiter_bytes():
                    now = time.monotonic()
                    line_buffer += chunk.decode("utf-8", errors="replace")

                    while "\n\n" in line_buffer:
                        line, line_buffer = line_buffer.split("\n\n", 1)
                        line = line.strip()

                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                continue

                            try:
                                data = json.loads(data_str)
                                content = (
                                    data.get("choices", [{}])[0]
                                    .get("delta", {})
                                    .get("content", "")
                                )
                            except json.JSONDecodeError:
                                continue

                            if content:
                                total_content += content
                                if first_token_time is None:
                                    first_token_time = now
                                chunk_times.append(now)

    except httpx.ConnectError as e:
        result.error = f"Cannot connect to {url}: {e}"
        return result
    except Exception as e:
        result.error = f"Request error: {e}"
        return result

    total_sec = time.monotonic() - start_time

    # Calculate metrics
    if first_token_time:
        result.ttft_sec = round(first_token_time - start_time, 4)

    result.actual_output_tokens = len(total_content.split())
    result.total_sec = round(total_sec, 4)

    # TPOT: average time between chunks
    if len(chunk_times) > 1:
        intervals = [
            chunk_times[i + 1] - chunk_times[i] for i in range(len(chunk_times) - 1)
        ]
        result.tpot_sec = round(sum(intervals) / len(intervals), 6)
    elif chunk_times:
        result.tpot_sec = 0.0

    return result


# ---------------------------------------------------------------------------
# Terminal table
# ---------------------------------------------------------------------------


def print_table(summary: list[dict[str, Any]]) -> None:
    """Print benchmark results as a formatted terminal table."""
    if not summary:
        print("No results to display.")
        return

    header = f"{'Prompt Tokens':>14} {'Output Tokens':>14} {'TTFT (s)':>12} {'TPOT (s)':>14} {'Total (s)':>12} {'Actual Tokens':>14}"
    sep = "-" * len(header)

    print()
    print(sep)
    print(header)
    print(sep)

    for row in summary:
        line = (
            f"{row['prompt_tokens']:>14} "
            f"{row['output_tokens']:>14} "
            f"{row['ttft_mean_s']:>12} "
            f"{row['tpot_mean_s']:>14} "
            f"{row['total_mean_s']:>12} "
            f"{row['actual_tokens_mean']:>14}"
        )
        print(line)

    print(sep)
    print()


def write_csv(results: list[BenchmarkResult], csv_path: str) -> None:
    """Write raw results to CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "prompt_tokens",
                "output_tokens",
                "run_index",
                "ttft_sec",
                "tpot_sec",
                "total_sec",
                "actual_output_tokens",
                "error",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.prompt_tokens,
                    r.output_tokens,
                    r.run_index,
                    f"{r.ttft_sec:.4f}",
                    f"{r.tpot_sec:.6f}",
                    f"{r.total_sec:.4f}",
                    r.actual_output_tokens,
                    r.error or "",
                ]
            )
    print(f"Raw results written to: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_benchmark(
    url: str,
    model: str,
    prompt_token_sizes: list[int],
    output_token_sizes: list[int],
    runs: int,
) -> BenchmarkPlan:
    """Run the full benchmark matrix."""

    plan = BenchmarkPlan()
    total_tests = len(prompt_token_sizes) * len(output_token_sizes) * runs
    completed = 0

    print("Benchmark config:")
    print(f"  URL:         {url}")
    print(f"  Model:       {model}")
    print(f"  Prompt sizes: {prompt_token_sizes}")
    print(f"  Output sizes: {output_token_sizes}")
    print(f"  Runs:         {runs}")
    print(f"  Total tests:  {total_tests}")
    print()

    # Verify server is reachable
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{url}/v1/models")
            if resp.status_code == 200:
                models_data = resp.json()
                model_names = [
                    m.get("id", m.get("name", "?")) for m in models_data.get("data", [])
                ]
                print(f"  Server models: {model_names}")
            else:
                print(f"  Warning: /v1/models returned {resp.status_code}")
    except Exception as e:
        print(f"  Warning: Cannot reach server at {url}: {e}")
        print("  Continuing anyway — benchmarks will likely fail.")

    for pt in prompt_token_sizes:
        prompt = _build_prompt(pt)
        actual_pt = len(prompt.split())
        print(f"\n{'=' * 60}")
        print(f"Prompt ~{pt} tokens (words: {actual_pt})")
        print(f"{'=' * 60}")

        for ot in output_token_sizes:
            for run_i in range(runs):
                completed += 1
                label = f"[{completed}/{total_tests}] prompt={pt}, output={ot}, run={run_i + 1}/{runs}"
                print(f"  {label}... ", end="", flush=True)

                res = await run_single_benchmark(url, model, prompt, ot, run_i + 1)
                plan.results.append(res)

                if res.error:
                    print(f"ERROR: {res.error}")
                else:
                    print(
                        f"TTFT={res.ttft_sec:.3f}s, "
                        f"TPOT={res.tpot_sec:.6f}s, "
                        f"Total={res.total_sec:.3f}s, "
                        f"Tokens={res.actual_output_tokens}"
                    )

    return plan


def main():
    parser = argparse.ArgumentParser(description="TTFT/TPOT Benchmark for fusion-mlx")
    parser.add_argument("--url", default=DEFAULT_URL, help="fusion-mlx API URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        nargs="+",
        default=DEFAULT_PROMPT_TOKENS,
        help="Prompt token sizes to test",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        nargs="+",
        default=DEFAULT_OUTPUT_TOKENS,
        help="Output token sizes to test",
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS, help="Number of runs per combination"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark-results",
        help="Output directory for CSV",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_file = output_dir / f"benchmark-{timestamp}.csv"

    plan = asyncio.run(
        run_benchmark(
            url=args.url,
            model=args.model,
            prompt_token_sizes=args.prompt_tokens,
            output_token_sizes=args.output_tokens,
            runs=args.runs,
        )
    )

    summary = plan.summary()
    print_table(summary)
    write_csv(plan.results, str(csv_file))

    # Write summary JSON
    json_file = output_dir / f"benchmark-{timestamp}-summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to: {json_file}")


if __name__ == "__main__":
    main()
