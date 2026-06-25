"""Hardware detection and model recommendation API for fusion-mlx."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..compatibility import check_compatibility
from ..hardware.detector import detect_hardware
from ..performance import estimate_tok_per_sec

logger = logging.getLogger(__name__)

router = APIRouter()


class HardwareRequest(BaseModel):
    model_id: str
    params: int
    quant_type: str = "Q4_K_M"
    context_length: int = 4096
    max_context: Optional[int] = None


@router.get("/v1/hardware")
async def get_hardware():
    """Detect and return local hardware info."""
    try:
        hw = detect_hardware()
    except Exception as e:
        logger.error("Hardware detection failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Hardware detection failed: {e}")

    gpu_info = None
    if hw.gpus:
        gpu = hw.gpus[0]
        gpu_info = {
            "name": gpu.name,
            "vendor": gpu.vendor,
            "vram_bytes": gpu.vram_bytes,
            "memory_bandwidth_gbps": gpu.memory_bandwidth_gbps,
            "shared_memory": gpu.shared_memory,
        }

    return {
        "gpu": gpu_info,
        "cpu": {
            "name": hw.cpu_name,
            "cores": hw.cpu_cores,
        },
        "ram": {
            "total_bytes": hw.ram_bytes,
            "total_gb": round(hw.ram_bytes / 1e9, 2),
        },
        "disk": {
            "free_bytes": hw.disk_free_bytes,
            "free_gb": round(hw.disk_free_bytes / 1e9, 2),
        },
        "os": hw.os,
    }


@router.post("/v1/recommend")
async def recommend_model(req: HardwareRequest):
    """Check if a model can run on this hardware and estimate speed."""
    if req.params <= 0:
        raise HTTPException(status_code=400, detail="params must be a positive integer")

    try:
        hw = detect_hardware()
    except Exception as e:
        logger.error("Hardware detection failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Hardware detection failed: {e}")

    compat = check_compatibility(
        model_id=req.model_id,
        params=req.params,
        quant_type=req.quant_type,
        hardware=hw,
        context_length=req.context_length,
        max_context=req.max_context,
    )

    tok_per_sec = 0.0
    if compat.can_run and hw.gpus:
        tok_per_sec = estimate_tok_per_sec(
            params=req.params,
            quant_type=req.quant_type,
            gpu=hw.gpus[0],
            fit_type=compat.fit_type,
        )

    return {
        "model_id": req.model_id,
        "can_run": compat.can_run,
        "fit_type": compat.fit_type,
        "vram_required_gb": round(compat.vram_required_bytes / 1e9, 2),
        "vram_available_gb": round(compat.vram_available_bytes / 1e9, 2),
        "offload_ratio": round(compat.offload_ratio, 3),
        "context_fits": compat.context_fits,
        "estimated_tok_per_sec": round(tok_per_sec, 1),
        "warnings": compat.warnings,
    }
