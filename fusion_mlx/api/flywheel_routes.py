# SPDX-License-Identifier: Apache-2.0
"""D1 flywheel HTTP routes: bench -> store -> recommend -> apply -> re-bench.

Endpoints:
  GET  /v1/bench/flywheel/results          list stored bench results
  POST /v1/bench/flywheel/run              run a single bench, store it
  POST /v1/bench/flywheel/recommend        recommend a config from stored results
  POST /v1/bench/flywheel/apply            apply a recommendation to live config
  POST /v1/bench/flywheel                  full loop: bench->recommend->apply->rebench
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..bench import flywheel as fw
from ..bench.flywheel import (
    BenchResult,
    Recommendation,
    load_results,
    recommend,
    store_result,
)
from ..middleware.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/bench/flywheel", tags=["flywheel"])


class BenchResultModel(BaseModel):
    config_id: str
    batch_size: int
    max_kv_tokens: int
    quant_level: str
    tok_per_sec: float = 0.0
    vram_used_gb: float = 0.0
    ttft_ms: float = 0.0
    notes: str = ""


class RunRequest(BaseModel):
    config_id: str
    batch_size: int = 32
    max_kv_tokens: int = 4096
    quant_level: str = "q4"
    tok_per_sec: float = Field(0.0, description="Measured throughput to store")
    vram_used_gb: float = 0.0
    ttft_ms: float = 0.0
    notes: str = ""


class RecommendRequest(BaseModel):
    memory_budget_gb: float = 0.0
    results: list[BenchResultModel] | None = None


class RecommendationModel(BaseModel):
    batch_size: int
    max_kv_tokens: int
    quant_level: str
    expected_tok_per_sec: float
    memory_budget_gb: float
    rationale: str = ""


class ApplyRequest(BaseModel):
    recommendation: RecommendationModel


class FlywheelRequest(BaseModel):
    before: BenchResultModel
    memory_budget_gb: float = 0.0
    recommendation: RecommendationModel | None = None


def _to_result(m: BenchResultModel) -> BenchResult:
    return BenchResult(
        config_id=m.config_id,
        batch_size=m.batch_size,
        max_kv_tokens=m.max_kv_tokens,
        quant_level=m.quant_level,
        tok_per_sec=m.tok_per_sec,
        vram_used_gb=m.vram_used_gb,
        ttft_ms=m.ttft_ms,
        notes=m.notes,
    )


def _to_reco(m: RecommendationModel) -> Recommendation:
    return Recommendation(
        batch_size=m.batch_size,
        max_kv_tokens=m.max_kv_tokens,
        quant_level=m.quant_level,
        expected_tok_per_sec=m.expected_tok_per_sec,
        memory_budget_gb=m.memory_budget_gb,
        rationale=m.rationale,
    )


@router.get("/results")
async def get_results(_auth: bool = Depends(verify_api_key)) -> Any:
    results = load_results()
    return {
        "results": [r.to_dict() for r in results],
        "total": len(results),
    }


@router.post("/run")
async def run_bench(req: RunRequest, _auth: bool = Depends(verify_api_key)) -> Any:
    result = BenchResult(
        config_id=req.config_id,
        batch_size=req.batch_size,
        max_kv_tokens=req.max_kv_tokens,
        quant_level=req.quant_level,
        tok_per_sec=req.tok_per_sec,
        vram_used_gb=req.vram_used_gb,
        ttft_ms=req.ttft_ms,
        notes=req.notes,
    )
    path = store_result(result)
    logger.info("flywheel route: stored bench at %s", path)
    return {"stored": True, "result": result.to_dict()}


@router.post("/recommend")
async def recommend_endpoint(
    req: RecommendRequest, _auth: bool = Depends(verify_api_key)
) -> Any:
    results = [_to_result(m) for m in req.results] if req.results is not None else None
    try:
        reco = recommend(results, memory_budget_gb=req.memory_budget_gb)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "batch_size": reco.batch_size,
        "max_kv_tokens": reco.max_kv_tokens,
        "quant_level": reco.quant_level,
        "expected_tok_per_sec": reco.expected_tok_per_sec,
        "memory_budget_gb": reco.memory_budget_gb,
        "rationale": reco.rationale,
    }


@router.post("/apply")
async def apply_endpoint(
    req: ApplyRequest, _auth: bool = Depends(verify_api_key)
) -> Any:
    cfg = fw.apply(_to_reco(req.recommendation))
    sched = cfg.scheduler
    logger.info("flywheel route: applied config to scheduler")
    return {
        "applied": True,
        "completion_batch_size": sched.completion_batch_size,
        "max_num_batched_tokens": sched.max_num_batched_tokens,
        "cache_memory_mb": sched.cache_memory_mb,
        "kv_cache_quantization": sched.kv_cache_quantization,
        "kv_cache_quantization_bits": sched.kv_cache_quantization_bits,
    }


@router.post("")
async def flywheel_endpoint(
    req: FlywheelRequest, _auth: bool = Depends(verify_api_key)
) -> Any:
    candidate = _to_reco(req.recommendation) if req.recommendation else None
    report = fw.flywheel(
        before=_to_result(req.before),
        candidate=candidate,
        memory_budget_gb=req.memory_budget_gb,
    )
    return {
        "before": report.before.to_dict(),
        "after": report.after.to_dict(),
        "recommendation": {
            "batch_size": report.recommendation.batch_size,
            "max_kv_tokens": report.recommendation.max_kv_tokens,
            "quant_level": report.recommendation.quant_level,
            "expected_tok_per_sec": report.recommendation.expected_tok_per_sec,
            "memory_budget_gb": report.recommendation.memory_budget_gb,
            "rationale": report.recommendation.rationale,
        },
        "tok_per_sec_delta": report.tok_per_sec_delta,
        "improved": report.improved,
    }
