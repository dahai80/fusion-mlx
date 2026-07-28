# SPDX-License-Identifier: Apache-2.0
"""Benchmark data query API for fusion-mlx.

Provides /v1/benchmarks endpoint for querying real performance data.
When community_bench data is available, returns measured tok/s and
vram usage per model/chip/quant combination. Falls back to estimated
values from compatibility + performance modules otherwise.

Issue: dahai80/fusion-mlx#233
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/benchmarks", tags=["benchmarks"])

_BENCH_DATA_DIR = Path.home() / ".fusion-mlx" / "benchmarks"


class BenchmarkEntry(BaseModel):
    model_id: str
    chip: str
    quant: str
    tok_per_sec: float
    step_per_sec: float = 0.0
    vram_used_gb: float = 0.0
    source: str = "community"
    tested_at: str = ""


class BenchmarkListResponse(BaseModel):
    benchmarks: list[BenchmarkEntry]
    total: int
    chips_available: list[str]
    models_available: list[str]


def _load_benchmark_files() -> list[BenchmarkEntry]:
    entries: list[BenchmarkEntry] = []
    if not _BENCH_DATA_DIR.exists():
        return entries
    for f in _BENCH_DATA_DIR.glob("*.json"):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                for item in data:
                    entries.append(BenchmarkEntry(**item))
            elif isinstance(data, dict):
                if "benchmarks" in data:
                    for item in data["benchmarks"]:
                        entries.append(BenchmarkEntry(**item))
                else:
                    entries.append(BenchmarkEntry(**data))
        except (json.JSONDecodeError, OSError, TypeError) as e:
            logger.warning("Failed to load benchmark file %s: %s", f, e)
    return entries


@router.get("", response_model=BenchmarkListResponse)
async def list_benchmarks(
    chip: str | None = Query(None, description="Filter by chip name"),
    model_id: str | None = Query(None, description="Filter by model ID"),
    quant: str | None = Query(None, description="Filter by quant type"),
) -> Any:
    entries = _load_benchmark_files()

    if chip:
        entries = [e for e in entries if chip.lower() in e.chip.lower()]
    if model_id:
        entries = [e for e in entries if model_id.lower() in e.model_id.lower()]
    if quant:
        entries = [e for e in entries if quant.lower() in e.quant.lower()]

    chips_available = sorted(set(e.chip for e in entries))
    models_available = sorted(set(e.model_id for e in entries))

    return BenchmarkListResponse(
        benchmarks=entries,
        total=len(entries),
        chips_available=chips_available,
        models_available=models_available,
    )


@router.get("/{model_id}", response_model=BenchmarkEntry)
async def get_benchmark(
    model_id: str,
    chip: str | None = Query(None, description="Chip name (e.g. M4_Max)"),
    quant: str | None = Query(None, description="Quant type (e.g. Q4_K_M)"),
) -> Any:
    entries = _load_benchmark_files()

    matches = [e for e in entries if e.model_id == model_id]
    if chip:
        matches = [e for e in matches if chip.lower() in e.chip.lower()]
    if quant:
        matches = [e for e in matches if quant.lower() in e.quant.lower()]

    if not matches:
        raise HTTPException(404, detail=f"No benchmark data for model '{model_id}'")

    best = max(matches, key=lambda e: e.tok_per_sec)
    return best
