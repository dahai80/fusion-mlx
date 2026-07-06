# SPDX-License-Identifier: Apache-2.0
"""Admin panel routes for Fusion-MLX server configuration.

This module provides HTTP routes for the admin panel including:
- Login/logout with API key authentication
- Dashboard for server monitoring
- Model settings management (per-model sampling parameters, pinning, default)
- Global settings management
"""

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from .auth import (
    require_admin,
)

logger = logging.getLogger(__name__)

PRESET_REMOTE_URL = "http://bench.dpdns.org/assets/fusionmlx_preset.json"


from .helpers import (
    _oq_manager,
    _paroquant_compat_for_model,
)
from .models import (
    OQStartRequest,
)

_router = APIRouter()


def _validate_model_path(model_path: str) -> str:
    resolved = Path(model_path).resolve()
    model_root = Path.home() / ".cache" / "fusion-mlx" / "models"
    model_root = model_root.resolve()
    if not str(resolved).startswith(str(model_root)):
        raise HTTPException(
            status_code=403,
            detail="model_path must reside under the configured model directory",
        )
    return str(resolved)


# =============================================================================
# oQ Quantization API Routes
# =============================================================================


@_router.get("/api/oq/models")
async def list_oq_models(is_admin: bool = Depends(require_admin)):
    """List non-quantized models available for oQ quantization."""
    if _oq_manager is None:
        return {"models": [], "all_models": []}
    source_models, all_models = await _oq_manager.list_quantizable_models()
    return {"models": source_models, "all_models": all_models}


@_router.get("/api/oq/recipes")
async def list_mlx_recipes(is_admin: bool = Depends(require_admin)):
    """List available MLX quantization recipes with benchmark data."""
    recipes = [
        {
            "name": "mixed_3_4",
            "label": "Mixed 3/4-bit",
            "description": "Good quality/speed tradeoff. 3-bit for sensitive layers, 4-bit for rest.",
            "category": "recommended",
            "bpw": 3.68,
            "relative_speed": "+96%",
        },
        {
            "name": "mixed_2_6",
            "label": "Mixed 2/6-bit",
            "description": "Faster decode, decent quality. 2-bit for robust layers, 6-bit for sensitive.",
            "category": "recommended",
            "bpw": 3.25,
            "relative_speed": "+112%",
        },
        {
            "name": "mixed_2_4",
            "label": "Mixed 2/4-bit",
            "description": "Fast decode, lower quality. 2-bit for robust layers, 4-bit for sensitive.",
            "category": "aggressive",
            "bpw": 2.95,
            "relative_speed": "+131%",
        },
        {
            "name": "mixed_3_6",
            "label": "Mixed 3/6-bit",
            "description": "Balanced quality and speed. 3-bit for robust, 6-bit for sensitive layers.",
            "category": "balanced",
            "bpw": 4.0,
            "relative_speed": "+75%",
        },
        {
            "name": "mixed_4_6",
            "label": "Mixed 4/6-bit",
            "description": "High quality, moderate speed gain. 4-bit for robust, 6-bit for sensitive.",
            "category": "conservative",
            "bpw": 4.85,
            "relative_speed": "+57%",
        },
        {
            "name": "quant2_all",
            "label": "quant2-all",
            "description": "Best speed/quality balance. 2-bit middle, 3-bit first/last, 4-bit embed/lm_head.",
            "category": "recommended",
            "bpw": 2.37,
            "relative_speed": "+162%",
        },
        {
            "name": "quant2",
            "label": "quant2",
            "description": "Ultra-aggressive 2-bit. 2-bit for most layers, 4-bit for embed/lm_head.",
            "category": "aggressive",
            "bpw": 2.72,
            "relative_speed": "+144%",
        },
        {
            "name": "quant2_128",
            "label": "quant2-g128",
            "description": "2-bit with group_size=128. Slightly faster than quant2, similar quality.",
            "category": "aggressive",
            "bpw": 2.46,
            "relative_speed": "+161%",
        },
        {
            "name": "quant2_flat",
            "label": "quant2-flat",
            "description": "Max speed but 2-bit embeddings degrade quality. Use quant2_all instead.",
            "category": "experimental",
            "bpw": 2.25,
            "relative_speed": "+167%",
        },
        {
            "name": "mxfp4",
            "label": "MLX FP4",
            "description": "Uniform 4-bit. Simple, good quality, moderate speed gain.",
            "category": "conservative",
            "bpw": 4.0,
            "relative_speed": "+75%",
        },
        {
            "name": "mxfp8",
            "label": "MLX FP8",
            "description": "Uniform 8-bit. Best quality, smallest speed gain. Baseline.",
            "category": "conservative",
            "bpw": 8.0,
            "relative_speed": "baseline",
        },
    ]
    return {"recipes": recipes}


@_router.post("/api/oq/estimate")
async def estimate_oq(
    model_path: str,
    oq_level: float,
    preserve_mtp: bool = False,
    is_admin: bool = Depends(require_admin),
):
    """Estimate effective bpw and output size for a model at given oQ level."""
    model_path = _validate_model_path(model_path)
    from ..oq import estimate_bpw_and_size

    try:
        result = await asyncio.to_thread(
            estimate_bpw_and_size,
            model_path,
            oq_level,
            64,  # group_size (default)
            preserve_mtp,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@_router.post("/api/oq/start")
async def start_oq_quantization(
    request: OQStartRequest,
    is_admin: bool = Depends(require_admin),
):
    """Start an oQ quantization task (or MLX recipe-based conversion)."""
    if _oq_manager is None:
        raise HTTPException(status_code=503, detail="oQ quantizer not initialized")
    is_recipe = bool(request.recipe)
    if not is_recipe and request.oq_level not in (2, 3, 3.5, 4, 5, 6, 8):
        raise HTTPException(
            status_code=400,
            detail="Invalid oQ level. Must be 2, 3, 4, 5, 6, or 8",
        )
    if request.dtype not in ("bfloat16", "float16"):
        raise HTTPException(
            status_code=400,
            detail="Invalid dtype. Must be 'bfloat16' or 'float16'",
        )
    _validate_model_path(request.model_path)
    is_paro, _ = _paroquant_compat_for_model({"model_path": request.model_path})
    if is_paro and not is_recipe:
        raise HTTPException(
            status_code=400,
            detail=(
                "Model is already quantized with paroquant; "
                "oQ re-quantization is not supported"
            ),
        )
    try:
        task = await _oq_manager.start_quantization(
            model_path=request.model_path,
            oq_level=request.oq_level,
            group_size=request.group_size,
            sensitivity_model_path=request.sensitivity_model_path,
            text_only=request.text_only,
            dtype=request.dtype,
            preserve_mtp=request.preserve_mtp,
            auto_proxy_sensitivity=request.auto_proxy_sensitivity,
            recipe=request.recipe,
        )
        return {"success": True, "task": task.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@_router.get("/api/oq/tasks")
async def list_oq_tasks(is_admin: bool = Depends(require_admin)):
    """List all quantization tasks."""
    if _oq_manager is None:
        return {"tasks": []}
    return {"tasks": _oq_manager.get_tasks()}


@_router.post("/api/oq/cancel/{task_id}")
async def cancel_oq_task(task_id: str, is_admin: bool = Depends(require_admin)):
    """Cancel an active quantization task."""
    if _oq_manager is None:
        raise HTTPException(status_code=503, detail="oQ quantizer not initialized")
    success = await _oq_manager.cancel_quantization(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or not cancellable")
    return {"success": True}


@_router.delete("/api/oq/task/{task_id}")
async def remove_oq_task(task_id: str, is_admin: bool = Depends(require_admin)):
    """Remove a completed/failed/cancelled task."""
    if _oq_manager is None:
        raise HTTPException(status_code=503, detail="oQ quantizer not initialized")
    success = _oq_manager.remove_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or still active")
    return {"success": True}


router = _router
