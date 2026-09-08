# SPDX-License-Identifier: Apache-2.0
"""Community benchmark runner — standardized B=1 bench.

Runs a minimal benchmark (2 buckets × 5 rounds + 1 warmup) on a loaded
engine, returning structured results compatible with the community DB.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Standardized prompts — short (~20 tokens) and long (~200 tokens)
_SHORT_PROMPT = "Write a short poem about nature."
_LONG_PROMPT = (
    "You are a helpful coding assistant. Explain the difference between "
    "a stack and a queue data structure. Include time complexity for push, "
    "pop, and peek operations. Give examples of when you would use each one."
)

_ROUNDS = 5
_TOKENS_PER_BUCKET = 64


@dataclass
class StatResult:
    values: list[float] = field(default_factory=list)

    @property
    def median(self) -> float:
        return statistics.median(self.values) if self.values else 0.0

    @property
    def mean(self) -> float:
        return statistics.mean(self.values) if self.values else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.values) if len(self.values) >= 2 else 0.0

    def __getitem__(self, key: str) -> float:
        if key == "median":
            return self.median
        if key == "mean":
            return self.mean
        if key == "stdev":
            return self.stdev
        raise KeyError(key)


@dataclass
class BucketResult:
    decode_stat: StatResult = field(default_factory=StatResult)
    prefill_stat: StatResult = field(default_factory=StatResult)
    ttft_stat: StatResult = field(default_factory=StatResult)


@dataclass
class BenchResult:
    short: BucketResult = field(default_factory=BucketResult)
    long: BucketResult = field(default_factory=BucketResult)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * pct / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_v):
        return sorted_v[-1]
    return sorted_v[f] + (k - f) * (sorted_v[c] - sorted_v[f])


async def run_standardized_bench(
    engine: Any,
    tokenizer: Any,
    sampling: str = "greedy",
) -> BenchResult:
    from ..request import SamplingParams

    result = BenchResult()

    for bucket_name, prompt in [("short", _SHORT_PROMPT), ("long", _LONG_PROMPT)]:
        bucket = result.short if bucket_name == "short" else result.long

        # Warmup round (not counted)
        params = SamplingParams(
            max_tokens=_TOKENS_PER_BUCKET,
            temperature=0.0 if sampling == "greedy" else 0.7,
            top_p=0.9 if sampling == "sampled" else 1.0,
        )
        try:
            await engine.generate(prompt, sampling_params=params)
        except Exception as e:
            logger.warning("bench warmup failed: %s", e)

        for _ in range(_ROUNDS):
            prompt_ids = tokenizer.encode(prompt) if hasattr(tokenizer, "encode") else []
            prefill_tokens = len(prompt_ids)

            t0 = time.perf_counter()
            output = await engine.generate(prompt, sampling_params=params)
            wall_time = time.perf_counter() - t0

            output_tokens = getattr(output, "completion_tokens", 0) or 0
            completion_text = getattr(output, "output_text", "") or ""

            # Decode speed (tok/s)
            if wall_time > 0 and output_tokens > 0:
                decode_tps = output_tokens / wall_time
            else:
                decode_tps = 0.0
            bucket.decode_stat.values.append(decode_tps)

            # Prefill speed (tok/s)
            if prefill_tokens > 0:
                # Estimate prefill time as fraction of total (rough)
                prefill_tps = prefill_tokens / max(wall_time * 0.3, 0.001)
            else:
                prefill_tps = 0.0
            bucket.prefill_stat.values.append(prefill_tps)

            # TTFT (ms) — approximate as time to first output token
            # For a simple bench, approximate as wall_time * (prefill_ratio)
            ttft_ms = wall_time * 1000 * min(prefill_tokens / max(prefill_tokens + output_tokens, 1), 0.5)
            bucket.ttft_stat.values.append(ttft_ms)

    logger.info(
        "bench: short decode=%.1f tok/s, long decode=%.1f tok/s",
        result.short.decode_stat.median,
        result.long.decode_stat.median,
    )
    return result
