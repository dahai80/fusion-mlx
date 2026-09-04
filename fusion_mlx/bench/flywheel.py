# SPDX-License-Identifier: Apache-2.0
"""D1 flywheel: bench -> store -> recommend -> apply -> re-bench closed loop.

Consumes bench results, recommends a runtime config (batch_size,
max_kv_tokens, quant_level) with best throughput under a memory budget,
applies it to the live ServerConfig, and re-benchs to confirm improvement.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import SchedulerConfig, ServerConfig, get_config

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path.home() / ".fusion-mlx" / "flywheel"


@dataclass
class BenchResult:
    config_id: str
    batch_size: int
    max_kv_tokens: int
    quant_level: str
    tok_per_sec: float
    vram_used_gb: float = 0.0
    ttft_ms: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Recommendation:
    batch_size: int
    max_kv_tokens: int
    quant_level: str
    expected_tok_per_sec: float
    memory_budget_gb: float
    rationale: str = ""


@dataclass
class FlywheelReport:
    before: BenchResult
    after: BenchResult
    recommendation: Recommendation
    tok_per_sec_delta: float
    improved: bool


def _results_dir() -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return _RESULTS_DIR


def store_result(result: BenchResult) -> Path:
    path = _results_dir() / f"{result.config_id}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    logger.info(
        "flywheel: stored result config_id=%s tps=%.1f vram=%.2f",
        result.config_id,
        result.tok_per_sec,
        result.vram_used_gb,
    )
    return path


def load_results() -> list[BenchResult]:
    out: list[BenchResult] = []
    if not _RESULTS_DIR.exists():
        return out
    for f in sorted(_RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            out.append(BenchResult(**data))
        except (json.JSONDecodeError, OSError, TypeError) as e:
            logger.warning("flywheel: skipping bad result %s: %s", f, e)
    return out


def recommend(
    results: list[BenchResult] | None = None,
    memory_budget_gb: float = 0.0,
) -> Recommendation:
    if results is None:
        results = load_results()
    if not results:
        raise ValueError("no bench results to recommend from")

    eligible = results
    if memory_budget_gb > 0:
        eligible = [r for r in results if r.vram_used_gb <= memory_budget_gb]
        if not eligible:
            logger.warning(
                "flywheel: no result fits budget %.2f, falling back to all",
                memory_budget_gb,
            )
            eligible = results

    best = max(eligible, key=lambda r: r.tok_per_sec)
    logger.info(
        "flywheel: recommend config_id=%s tps=%.1f batch=%d kv=%d quant=%s",
        best.config_id,
        best.tok_per_sec,
        best.batch_size,
        best.max_kv_tokens,
        best.quant_level,
    )
    return Recommendation(
        batch_size=best.batch_size,
        max_kv_tokens=best.max_kv_tokens,
        quant_level=best.quant_level,
        expected_tok_per_sec=best.tok_per_sec,
        memory_budget_gb=memory_budget_gb,
        rationale=f"best throughput among {len(eligible)} eligible configs",
    )


_QUANT_BITS: dict[str, int] = {
    "fp16": 16,
    "bf16": 16,
    "q8": 8,
    "q8_0": 8,
    "q6_k": 6,
    "q5": 5,
    "q4": 4,
    "q4_k_m": 4,
    "q3": 3,
    "q2": 2,
}


def _quant_rank(level: str) -> int:
    return _QUANT_BITS.get(level.lower(), 4)


def apply(reco: Recommendation, config: ServerConfig | None = None) -> ServerConfig:
    cfg = config if config is not None else get_config()
    sched: SchedulerConfig = cfg.scheduler
    sched.completion_batch_size = reco.batch_size
    sched.max_num_batched_tokens = max(
        reco.batch_size * 512, sched.max_num_batched_tokens
    )
    if reco.max_kv_tokens > 0:
        sched.cache_memory_mb = int(reco.max_kv_tokens * 2)
    sched.kv_cache_quantization = _quant_rank(reco.quant_level) <= 8
    if sched.kv_cache_quantization:
        sched.kv_cache_quantization_bits = _quant_rank(reco.quant_level) or 8
    logger.info(
        "flywheel: applied batch=%d kv=%d quant=%s kv_quant=%s",
        sched.completion_batch_size,
        reco.max_kv_tokens,
        reco.quant_level,
        sched.kv_cache_quantization,
    )
    return cfg


def _run_bench_once(
    config_id: str,
    batch_size: int,
    max_kv_tokens: int,
    quant_level: str,
    runner: Any = None,
) -> BenchResult:
    if runner is not None:
        raw = runner(
            config_id=config_id,
            batch_size=batch_size,
            max_kv_tokens=max_kv_tokens,
            quant_level=quant_level,
        )
        return BenchResult(
            config_id=raw.get("config_id", config_id),
            batch_size=int(raw.get("batch_size", batch_size)),
            max_kv_tokens=int(raw.get("max_kv_tokens", max_kv_tokens)),
            quant_level=raw.get("quant_level", quant_level),
            tok_per_sec=float(raw.get("tok_per_sec", 0.0)),
            vram_used_gb=float(raw.get("vram_used_gb", 0.0)),
            ttft_ms=float(raw.get("ttft_ms", 0.0)),
            notes=raw.get("notes", ""),
        )
    from . import run_benchmark

    raw = run_benchmark(config_id) or {}
    return BenchResult(
        config_id=config_id,
        batch_size=batch_size,
        max_kv_tokens=max_kv_tokens,
        quant_level=quant_level,
        tok_per_sec=float(raw.get("tokens_per_second", 0.0)),
        vram_used_gb=float(raw.get("vram_used_gb", 0.0)),
        ttft_ms=float(raw.get("ttft_ms", 0.0)),
        notes="local-stub-runner",
    )


def flywheel(
    before: BenchResult,
    candidate: Recommendation | None = None,
    memory_budget_gb: float = 0.0,
    runner: Any = None,
    config: ServerConfig | None = None,
) -> FlywheelReport:
    store_result(before)
    if candidate is None:
        candidate = recommend([before], memory_budget_gb=memory_budget_gb)
    apply(candidate, config=config)
    after = _run_bench_once(
        config_id=f"{candidate.batch_size}_{candidate.max_kv_tokens}_{candidate.quant_level}",
        batch_size=candidate.batch_size,
        max_kv_tokens=candidate.max_kv_tokens,
        quant_level=candidate.quant_level,
        runner=runner,
    )
    store_result(after)
    delta = after.tok_per_sec - before.tok_per_sec
    logger.info(
        "flywheel: before=%.1f after=%.1f delta=%+.1f improved=%s",
        before.tok_per_sec,
        after.tok_per_sec,
        delta,
        delta > 0,
    )
    return FlywheelReport(
        before=before,
        after=after,
        recommendation=candidate,
        tok_per_sec_delta=delta,
        improved=delta > 0,
    )
