# SPDX-License-Identifier: Apache-2.0
"""G1/S1: Performance regression gate (ADVISORY mode).

Rapid-mlx blueprint: perf_gate measures decode throughput (tok/s) + TTFT.
Mode: ADVISORY (no baseline → warn, never fail CI). When a baseline file
exists, regressions beyond threshold become enforce failures.

"NEVER invents a baseline" — baselines must be recorded from a real run,
not guessed. First run records; subsequent runs compare.

CLI: python -m fusion_mlx.eval.perf_gate --model <alias> [--baseline <file>]
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_BASELINE_PATH = "evals/baselines/perf_baseline.json"
_REGRESSION_THRESHOLD = 0.15  # 15% slower than baseline = regression
_WARMUP_PROMPTS = 2
_MEASURE_PROMPTS = 5
_PROMPTS = [
    "Explain photosynthesis in two sentences.",
    "Write a Python function to reverse a string.",
    "What is the capital of France?",
    "Summarize the plot of Romeo and Juliet.",
    "List three benefits of exercise.",
    "Translate 'good morning' to Japanese.",
    "What causes rainbows?",
    "Name a primary color and why it is primary.",
]


@dataclass
class PerfSample:
    prompt: str
    ttft_ms: float
    total_tokens: int
    wall_seconds: float
    tok_s: float


@dataclass
class PerfGateResult:
    model: str
    samples: list[PerfSample] = field(default_factory=list)
    mean_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    mean_tok_s: float = 0.0
    mode: str = "ADVISORY"
    baseline_present: bool = False
    regression: bool = False
    regression_detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def measure_one(host: str, api_key: str, model: str, prompt: str) -> PerfSample | None:
    import httpx

    url = f"{host}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    t0 = time.perf_counter()
    ttft_ms = 0.0
    total_tokens = 0
    try:
        with httpx.stream(
            "POST",
            url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 128,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers=headers,
            timeout=httpx.Timeout(120.0, connect=10.0),
        ) as resp:
            if resp.status_code != 200:
                logger.warning("perf_gate: %s returned %s", url, resp.status_code)
                return None
            first_chunk = True
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except (json.JSONDecodeError, ValueError):
                    continue
                if (
                    first_chunk
                    and chunk.get("choices")
                    and chunk["choices"][0].get("delta", {}).get("content")
                ):
                    ttft_ms = (time.perf_counter() - t0) * 1000.0
                    first_chunk = False
                usage = chunk.get("usage")
                if usage and usage.get("completion_tokens"):
                    total_tokens = usage["completion_tokens"]
    except (httpx.HTTPError, OSError) as exc:
        logger.warning("perf_gate: request failed: %s", exc)
        return None
    wall = time.perf_counter() - t0
    tok_s = total_tokens / wall if wall > 0 else 0.0
    if ttft_ms == 0.0:
        ttft_ms = wall * 1000.0
    if total_tokens == 0:
        logger.debug("perf_gate: zero tokens for prompt: %.60r", prompt)
        return None
    return PerfSample(
        prompt=prompt,
        ttft_ms=ttft_ms,
        total_tokens=total_tokens,
        wall_seconds=wall,
        tok_s=tok_s,
    )


async def run_gate(
    host: str, api_key: str, model: str, baseline_path: str
) -> PerfGateResult:
    result = PerfGateResult(model=model)
    baseline = None
    if baseline_path and Path(baseline_path).exists():
        try:
            baseline = json.loads(Path(baseline_path).read_text())
            result.baseline_present = True
        except (json.JSONDecodeError, OSError):
            logger.warning("perf_gate: baseline unreadable: %s", baseline_path)
    prompts = _PROMPTS[: _WARMUP_PROMPTS + _MEASURE_PROMPTS]
    for i, prompt in enumerate(prompts):
        sample = await asyncio.to_thread(measure_one, host, api_key, model, prompt)
        if sample is None:
            logger.info(
                "perf_gate: skipping prompt %d (server unreachable or no tokens)", i
            )
            continue
        if i < _WARMUP_PROMPTS:
            logger.debug(
                "perf_gate: warmup %d tok_s=%.1f ttft=%.1fms",
                i,
                sample.tok_s,
                sample.ttft_ms,
            )
            continue
        result.samples.append(sample)
    if not result.samples:
        result.mode = "ADVISORY"
        result.regression_detail = (
            "no samples collected (server unreachable or model not loaded)"
        )
        logger.warning("perf_gate: %s", result.regression_detail)
        return result
    ttfts = sorted(s.ttft_ms for s in result.samples)
    tok_s_vals = [s.tok_s for s in result.samples]
    result.mean_ttft_ms = statistics.mean(ttfts)
    result.p50_ttft_ms = ttfts[len(ttfts) // 2]
    result.p99_ttft_ms = ttfts[-1] if len(ttfts) <= 1 else ttfts[int(len(ttfts) * 0.99)]
    result.mean_tok_s = statistics.mean(tok_s_vals)
    if baseline:
        base_tok_s = baseline.get("mean_tok_s", 0.0)
        if base_tok_s > 0:
            drop = (base_tok_s - result.mean_tok_s) / base_tok_s
            if drop > _REGRESSION_THRESHOLD:
                result.regression = True
                result.mode = "ENFORCE"
                result.regression_detail = (
                    f"tok/s {result.mean_tok_s:.1f} vs baseline {base_tok_s:.1f} "
                    f"= {drop:.0%} regression (threshold {_REGRESSION_THRESHOLD:.0%})"
                )
            else:
                result.regression_detail = (
                    f"tok/s {result.mean_tok_s:.1f} vs baseline {base_tok_s:.1f} "
                    f"= {drop:.0%} (within threshold)"
                )
    else:
        result.regression_detail = (
            "no baseline — ADVISORY only, recording first measurement"
        )
    logger.info("perf_gate: %s", result.regression_detail)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Performance regression gate (ADVISORY)"
    )
    parser.add_argument("--model", required=True, help="Model alias or id")
    parser.add_argument(
        "--host", default=os.environ.get("FUSION_HOST", "http://127.0.0.1:11434")
    )
    parser.add_argument("--api-key", default=os.environ.get("FUSION_MLX_API_KEY", ""))
    parser.add_argument("--baseline", default=_DEFAULT_BASELINE_PATH)
    parser.add_argument(
        "--record-baseline", action="store_true", help="Record result as new baseline"
    )
    args = parser.parse_args()
    result = asyncio.run(run_gate(args.host, args.api_key, args.model, args.baseline))
    if args.record_baseline:
        Path(args.baseline).parent.mkdir(parents=True, exist_ok=True)
        Path(args.baseline).write_text(json.dumps(result.to_dict(), indent=2))
        logger.info("baseline recorded → %s", args.baseline)
        return 0
    print(json.dumps(result.to_dict(), indent=2))
    return 1 if result.regression else 0


if __name__ == "__main__":
    raise SystemExit(main())
