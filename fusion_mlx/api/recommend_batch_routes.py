# SPDX-License-Identifier: Apache-2.0
"""Batch model recommendation API for fusion-mlx.

Provides /v1/recommend/batch endpoint that evaluates multiple models
simultaneously on the same hardware and returns ranked results.

Issue: dahai80/fusion-mlx#231
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..compatibility import check_compatibility
from ..hardware.detector import detect_hardware
from ..performance import estimate_tok_per_sec

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/recommend", tags=["recommend"])


class ModelSpec(BaseModel):
    model_id: str = Field(..., description="Model identifier")
    params: int = Field(..., gt=0, description="Parameter count in billions")
    quant_type: str = Field("Q4_K_M", description="Quantization type")


class BatchRecommendRequest(BaseModel):
    models: list[ModelSpec] = Field(
        ..., min_length=1, max_length=50, description="Models to evaluate"
    )
    task: str | None = Field(
        None, description="Task type: text2image|text2video|llm|embedding"
    )
    preference: str = Field(
        "balanced",
        description="Preference: quality|balanced|speed",
    )


class ModelRankResult(BaseModel):
    model_id: str
    can_run: bool
    fit_type: str
    vram_required_gb: float
    vram_available_gb: float
    estimated_tok_per_sec: float
    rank_score: float


class HardwareSummary(BaseModel):
    chip: str
    gpu_vram_gb: float
    ram_gb: float


class BatchRecommendResponse(BaseModel):
    results: list[ModelRankResult]
    hardware: HardwareSummary


_WEIGHT_PROFILES = {
    "quality": {"vram_fit": 0.3, "speed": 0.2, "quant_level": 0.5},
    "balanced": {"vram_fit": 0.4, "speed": 0.3, "quant_level": 0.3},
    "speed": {"vram_fit": 0.2, "speed": 0.5, "quant_level": 0.3},
}

_QUANT_SCORES = {
    "FP16": 100,
    "BF16": 100,
    "Q8_0": 85,
    "Q6_K": 75,
    "Q5_K_M": 65,
    "Q5_0": 60,
    "Q4_K_M": 55,
    "Q4_0": 50,
    "Q3_K_M": 40,
    "IQ3_M": 35,
    "Q2_K": 25,
    "IQ1_M": 15,
}


def _score_vram_fit(vram_required_gb: float, vram_available_gb: float) -> float:
    if vram_available_gb <= 0:
        return 0.0
    ratio = vram_required_gb / vram_available_gb
    if ratio <= 0.5:
        return 100.0
    if ratio <= 0.8:
        return 80.0 + 20.0 * (0.8 - ratio) / 0.3
    if ratio <= 1.0:
        return 40.0 + 40.0 * (1.0 - ratio) / 0.2
    return max(0.0, 40.0 * (1.0 - min(ratio - 1.0, 1.0)))


def _score_speed(tok_per_sec: float) -> float:
    if tok_per_sec <= 0:
        return 0.0
    return min(100.0, tok_per_sec / 0.5)


def _score_quant(quant_type: str) -> float:
    return _QUANT_SCORES.get(quant_type, 50.0)


@router.post("/batch", response_model=BatchRecommendResponse)
async def recommend_batch(req: BatchRecommendRequest) -> Any:
    if req.preference not in _WEIGHT_PROFILES:
        raise HTTPException(
            400,
            detail=f"preference must be one of {list(_WEIGHT_PROFILES.keys())}",
        )

    try:
        hw = detect_hardware()
    except Exception as e:
        logger.error("Hardware detection failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

    gpu = hw.gpus[0] if hw.gpus else None
    weights = _WEIGHT_PROFILES[req.preference]

    chip_name = gpu.name if gpu else hw.cpu_name
    gpu_vram = gpu.vram_bytes / 1e9 if gpu else 0.0
    ram_gb = hw.ram_bytes / 1e9

    results: list[ModelRankResult] = []
    for spec in req.models:
        compat = check_compatibility(
            model_id=spec.model_id,
            params=spec.params,
            quant_type=spec.quant_type,
            hardware=hw,
        )

        tok_per_sec = 0.0
        if compat.can_run and gpu:
            tok_per_sec = estimate_tok_per_sec(
                params=spec.params,
                quant_type=spec.quant_type,
                gpu=gpu,
                fit_type=compat.fit_type,
            )

        s_vram = _score_vram_fit(
            compat.vram_required_bytes / 1e9,
            compat.vram_available_bytes / 1e9,
        )
        s_speed = _score_speed(tok_per_sec)
        s_quant = _score_quant(spec.quant_type)

        rank_score = (
            weights["vram_fit"] * s_vram
            + weights["speed"] * s_speed
            + weights["quant_level"] * s_quant
        )

        if not compat.can_run:
            rank_score = 0.0

        results.append(
            ModelRankResult(
                model_id=spec.model_id,
                can_run=compat.can_run,
                fit_type=compat.fit_type,
                vram_required_gb=round(compat.vram_required_bytes / 1e9, 2),
                vram_available_gb=round(compat.vram_available_bytes / 1e9, 2),
                estimated_tok_per_sec=round(tok_per_sec, 1),
                rank_score=round(rank_score, 1),
            )
        )

    results.sort(key=lambda r: r.rank_score, reverse=True)

    return BatchRecommendResponse(
        results=results,
        hardware=HardwareSummary(
            chip=chip_name,
            gpu_vram_gb=round(gpu_vram, 2),
            ram_gb=round(ram_gb, 2),
        ),
    )
