# SPDX-License-Identifier: Apache-2.0
"""Admin panel routes for Fusion-MLX server configuration.

This module provides HTTP routes for the admin panel including:
- Login/logout with API key authentication
- Dashboard for server monitoring
- Model settings management (per-model sampling parameters, pinning, default)
- Global settings management
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .auth import (
    require_admin,
)

logger = logging.getLogger(__name__)

PRESET_REMOTE_URL = "http://bench.dpdns.org/assets/fusionmlx_preset.json"


from .helpers import (
    _ms_downloader,
    get_system_memory_info,
)
from .models import (
    MSDownloadRequest,
    MSRetryRequest,
)

_router = APIRouter()

# =============================================================================
# ModelScope Downloader API Routes
# =============================================================================


@_router.get("/api/ms/status")
async def ms_status(is_admin: bool = Depends(require_admin)):
    """Check if ModelScope downloader is available."""
    return {"available": _ms_downloader is not None}


@_router.post("/api/ms/download")
async def start_ms_download(
    request: MSDownloadRequest,
    is_admin: bool = Depends(require_admin),
):
    """Start downloading a model from ModelScope."""
    if _ms_downloader is None:
        raise HTTPException(
            status_code=503, detail="ModelScope downloader not initialized"
        )

    try:
        task = await _ms_downloader.start_download(request.model_id, request.ms_token)
        return {"success": True, "task": task.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@_router.get("/api/ms/tasks")
async def list_ms_tasks(is_admin: bool = Depends(require_admin)):
    """List all ModelScope download tasks."""
    if _ms_downloader is None:
        return {"tasks": []}

    return {"tasks": _ms_downloader.get_tasks()}


@_router.post("/api/ms/cancel/{task_id}")
async def cancel_ms_download(
    task_id: str,
    is_admin: bool = Depends(require_admin),
):
    """Cancel an active ModelScope download."""
    if _ms_downloader is None:
        raise HTTPException(
            status_code=503, detail="ModelScope downloader not initialized"
        )

    success = await _ms_downloader.cancel_download(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or not cancellable")
    return {"success": True}


class MSRetryRequest(BaseModel):
    ms_token: str | None = None


@_router.post("/api/ms/retry/{task_id}")
async def retry_ms_download(
    task_id: str,
    request: MSRetryRequest = MSRetryRequest(),
    is_admin: bool = Depends(require_admin),
):
    """Retry a failed or cancelled ModelScope download."""
    if _ms_downloader is None:
        raise HTTPException(
            status_code=503, detail="ModelScope downloader not initialized"
        )

    try:
        task = await _ms_downloader.retry_download(task_id, request.ms_token)
        return {"success": True, "task": task.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@_router.delete("/api/ms/task/{task_id}")
async def remove_ms_task(
    task_id: str,
    is_admin: bool = Depends(require_admin),
):
    """Remove a completed, failed, or cancelled ModelScope task."""
    if _ms_downloader is None:
        raise HTTPException(
            status_code=503, detail="ModelScope downloader not initialized"
        )

    success = _ms_downloader.remove_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or still active")
    return {"success": True}


@_router.get("/api/ms/recommended")
async def get_ms_recommended_models(
    mlx_only: bool = True,
    is_admin: bool = Depends(require_admin),
):
    """Get recommended models from ModelScope filtered by system memory."""
    if _ms_downloader is None:
        return {"models": []}

    memory_info = get_system_memory_info()
    max_memory = memory_info["total_bytes"] or 16 * 1024**3

    from .ms_downloader import MSDownloader

    try:
        result = await MSDownloader.get_recommended_models(
            max_memory_bytes=max_memory, result_limit=50, mlx_only=mlx_only
        )
        return result
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="ModelScope API request timed out. The service may be temporarily unavailable.",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@_router.get("/api/ms/search")
async def search_ms_models(
    q: str = "",
    sort: str = "trending",
    limit: int = 100,
    mlx_only: bool = True,
    is_admin: bool = Depends(require_admin),
):
    """Search ModelScope models by query."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")

    from .ms_downloader import MSDownloader

    try:
        result = await MSDownloader.search_models(
            query=q.strip(),
            sort=sort,
            limit=min(limit, 100),
            mlx_only=mlx_only,
        )
        return result
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="ModelScope API request timed out. The service may be temporarily unavailable.",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@_router.get("/api/ms/model-info")
async def get_ms_model_info(
    model_id: str = "",
    is_admin: bool = Depends(require_admin),
):
    """Get detailed model information from ModelScope."""
    if not model_id.strip():
        raise HTTPException(
            status_code=400, detail="Query parameter 'model_id' is required"
        )

    from .ms_downloader import MSDownloader

    try:
        result = await MSDownloader.get_model_info(model_id=model_id.strip())
        return result
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="ModelScope API request timed out. The service may be temporarily unavailable.",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        if "NotExistError" in type(e).__name__ or "404" in str(e):
            raise HTTPException(
                status_code=404, detail=f"Model '{model_id.strip()}' not found"
            )
        raise HTTPException(status_code=502, detail=str(e))


router = _router
