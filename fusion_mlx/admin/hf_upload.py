# SPDX-License-Identifier: Apache-2.0
"""Admin panel routes for oMLX server configuration.

This module provides HTTP routes for the admin panel including:
- Login/logout with API key authentication
- Dashboard for server monitoring
- Model settings management (per-model sampling parameters, pinning, default)
- Global settings management
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from .auth import (
    require_admin,
)

logger = logging.getLogger(__name__)

PRESET_REMOTE_URL = "http://bench.dpdns.org/assets/omlx_preset.json"


from .helpers import (
    _hf_uploader,
)
from .models import (
    HFUploadRequest,
    HFValidateTokenRequest,
)

_router = APIRouter()

# =============================================================================
# HuggingFace Upload Endpoints
# =============================================================================


@_router.post("/api/upload/validate-token")
async def validate_upload_token(
    request: HFValidateTokenRequest,
    is_admin: bool = Depends(require_admin),
):
    """Validate a HuggingFace token and return user info."""
    if _hf_uploader is None:
        raise HTTPException(status_code=503, detail="HF Uploader not initialized")
    try:
        result = await _hf_uploader.validate_token(request.hf_token)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@_router.get("/api/upload/oq-models")
async def list_upload_oq_models(is_admin: bool = Depends(require_admin)):
    """List local oQ models available for upload."""
    if _hf_uploader is None:
        return {"oq_models": [], "all_models": []}
    oq_models = await _hf_uploader.list_oq_models()
    all_models = await _hf_uploader.list_all_models()
    return {"oq_models": oq_models, "all_models": all_models}


@_router.post("/api/upload/start")
async def start_upload(
    request: HFUploadRequest,
    is_admin: bool = Depends(require_admin),
):
    """Start an upload task to HuggingFace Hub."""
    if _hf_uploader is None:
        raise HTTPException(status_code=503, detail="HF Uploader not initialized")
    try:
        task = await _hf_uploader.start_upload(
            model_path=request.model_path,
            repo_id=request.repo_id,
            token=request.hf_token,
            readme_source_path=request.readme_source_path,
            auto_readme=request.auto_readme,
            redownload_notice=request.redownload_notice,
            private=request.private,
        )
        return {"success": True, "task": task.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@_router.get("/api/upload/tasks")
async def list_upload_tasks(is_admin: bool = Depends(require_admin)):
    """List all upload tasks."""
    if _hf_uploader is None:
        return {"tasks": []}
    return {"tasks": _hf_uploader.get_tasks()}


@_router.post("/api/upload/cancel/{task_id}")
async def cancel_upload_task(task_id: str, is_admin: bool = Depends(require_admin)):
    """Cancel an active or pending upload task."""
    if _hf_uploader is None:
        raise HTTPException(status_code=503, detail="HF Uploader not initialized")
    success = await _hf_uploader.cancel_upload(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or not cancellable")
    return {"success": True}


@_router.delete("/api/upload/task/{task_id}")
async def remove_upload_task(task_id: str, is_admin: bool = Depends(require_admin)):
    """Remove a completed/failed/cancelled upload task."""
    if _hf_uploader is None:
        raise HTTPException(status_code=503, detail="HF Uploader not initialized")
    success = _hf_uploader.remove_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or still active")
    return {"success": True}


router = _router
