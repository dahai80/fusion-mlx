# SPDX-License-Identifier: Apache-2.0
"""Admin panel routes for Fusion-MLX server configuration.

This module provides HTTP routes for the admin panel including:
- Login/logout with API key authentication
- Dashboard for server monitoring
- Model settings management (per-model sampling parameters, pinning, default)
- Global settings management
"""

import logging
import shutil
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .auth import (
    require_admin,
)

logger = logging.getLogger(__name__)

PRESET_REMOTE_URL = "http://bench.dpdns.org/assets/fusionmlx_preset.json"


from .helpers import (
    _get_engine_pool,
    _get_hf_downloader,
    _get_model_dirs,
    _get_settings_manager,
    format_size,
    get_system_memory_info,
)
from .models import (
    HFDownloadRequest,
    HFRetryRequest,
)

_router = APIRouter()

# =============================================================================
# HuggingFace Downloader API Routes
# =============================================================================


@_router.post("/api/hf/download")
async def start_hf_download(
    request: HFDownloadRequest,
    is_admin: bool = Depends(require_admin),
):
    """Start downloading a model from HuggingFace."""
    dl = _get_hf_downloader()
    if dl is None:
        raise HTTPException(status_code=503, detail="Downloader not initialized")

    try:
        task = await dl.start_download(request.repo_id, request.hf_token)
        return {"success": True, "task": task.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@_router.get("/api/hf/tasks")
async def list_hf_tasks(is_admin: bool = Depends(require_admin)):
    """List all download tasks."""
    dl = _get_hf_downloader()
    if dl is None:
        raise HTTPException(status_code=503, detail="Downloader not initialized")

    return {"tasks": dl.get_tasks()}


@_router.post("/api/hf/cancel/{task_id}")
async def cancel_hf_download(
    task_id: str,
    is_admin: bool = Depends(require_admin),
):
    """Cancel an active download."""
    dl = _get_hf_downloader()
    if dl is None:
        raise HTTPException(status_code=503, detail="Downloader not initialized")

    success = await dl.cancel_download(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or not cancellable")
    return {"success": True}


class HFRetryRequest(BaseModel):
    hf_token: str | None = None


@_router.post("/api/hf/retry/{task_id}")
async def retry_hf_download(
    task_id: str,
    request: HFRetryRequest = HFRetryRequest(),
    is_admin: bool = Depends(require_admin),
):
    """Retry a failed or cancelled download, resuming from existing files."""
    dl = _get_hf_downloader()
    if dl is None:
        raise HTTPException(status_code=503, detail="Downloader not initialized")

    try:
        task = await dl.retry_download(task_id, request.hf_token)
        return {"success": True, "task": task.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@_router.delete("/api/hf/task/{task_id}")
async def remove_hf_task(
    task_id: str,
    is_admin: bool = Depends(require_admin),
):
    """Remove a completed, failed, or cancelled task."""
    dl = _get_hf_downloader()
    if dl is None:
        raise HTTPException(status_code=503, detail="Downloader not initialized")

    success = dl.remove_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or still active")
    return {"success": True}


@_router.get("/api/hf/recommended")
async def get_recommended_models(
    mlx_only: bool = True,
    is_admin: bool = Depends(require_admin),
):
    """Get recommended models filtered by system memory."""
    dl = _get_hf_downloader()
    if dl is None:
        raise HTTPException(status_code=503, detail="Downloader not initialized")

    memory_info = get_system_memory_info()
    max_memory = memory_info["total_bytes"] or 16 * 1024**3

    from .hf_downloader import HFDownloader

    try:
        result = await HFDownloader.get_recommended_models(
            max_memory_bytes=max_memory, result_limit=50, mlx_only=mlx_only
        )
        return result
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="HuggingFace API request timed out. The service may be temporarily unavailable.",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@_router.get("/api/hf/search")
async def search_hf_models(
    q: str = "",
    sort: str = "trending",
    limit: int = 100,
    mlx_only: bool = True,
    # Filtering
    min_params: int | None = None,
    max_params: int | None = None,
    min_size: int | None = None,  # bytes
    max_size: int | None = None,  # bytes
    # Sorting
    sort_by_size: bool = False,
    sort_ascending: bool = False,
    is_admin: bool = Depends(require_admin),
):
    """Search HuggingFace models by query with filtering and sorting.

    Query Parameters:
        q: Search query string (required)
        sort: Sort order - trending/downloads/created/updated/most_params/least_params/largest/smallest
        limit: Maximum results (max 100)
        mlx_only: Restrict to MLX library models
        min_params: Minimum parameter count
        max_params: Maximum parameter count
        min_size: Minimum model size in bytes
        max_size: Maximum model size in bytes
        sort_by_size: Sort results by size instead of default sort
        sort_ascending: Sort in ascending order
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")

    from .hf_downloader import HFDownloader

    try:
        result = await HFDownloader.search_models(
            query=q.strip(),
            sort=sort,
            limit=min(limit, 100),
            mlx_only=mlx_only,
            min_params=min_params,
            max_params=max_params,
            min_size=min_size,
            max_size=max_size,
            sort_by_size=sort_by_size,
            sort_ascending=sort_ascending,
        )
        return result
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="HuggingFace API request timed out. The service may be temporarily unavailable.",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@_router.get("/api/hf/model-info")
async def get_hf_model_info(
    repo_id: str = "",
    is_admin: bool = Depends(require_admin),
):
    """Get detailed model information from HuggingFace."""
    if not repo_id.strip():
        raise HTTPException(
            status_code=400, detail="Query parameter 'repo_id' is required"
        )

    from huggingface_hub.utils import RepositoryNotFoundError

    from .hf_downloader import HFDownloader

    try:
        result = await HFDownloader.get_model_info(repo_id=repo_id.strip())
        return result
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="HuggingFace API request timed out. The service may be temporarily unavailable.",
        )
    except RepositoryNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Model '{repo_id.strip()}' not found"
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@_router.get("/api/hf/models")
async def list_hf_models(is_admin: bool = Depends(require_admin)):
    """List models in all model directories with disk size info."""
    model_dirs = _get_model_dirs()
    if not model_dirs:
        raise HTTPException(status_code=503, detail="No model directories configured")

    from ..pool.model_discovery import _resolve_hf_cache_entry

    def _add_model(model_path: Path, model_name: str) -> None:
        if model_name in seen_names:
            return
        seen_names.add(model_name)
        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        models.append(
            {
                "name": model_name,
                "path": str(model_path),
                "size": total_size,
                "size_formatted": format_size(total_size),
            }
        )

    models = []
    seen_names: set[str] = set()
    for model_dir in model_dirs:
        if not model_dir.exists():
            continue
        for subdir in sorted(model_dir.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue

            if (subdir / "config.json").exists():
                # Level 1: direct model folder
                _add_model(subdir, subdir.name)
            else:
                # HF Hub cache entry: models--Org--Name/snapshots/<hash>/
                hf_resolved = _resolve_hf_cache_entry(subdir)
                if hf_resolved is not None:
                    snapshot_path, model_name = hf_resolved
                    if (snapshot_path / "config.json").exists():
                        _add_model(snapshot_path, model_name)
                    continue

                # Level 2: organization folder — scan children
                for child in sorted(subdir.iterdir()):
                    if not child.is_dir() or child.name.startswith("."):
                        continue
                    if (child / "config.json").exists():
                        _add_model(child, child.name)

    # Sort case-insensitively by name for a stable, user-friendly order.
    models.sort(key=lambda m: m["name"].lower())
    return {"models": models}


@_router.delete("/api/hf/models/{model_name}")
async def delete_hf_model(
    model_name: str,
    is_admin: bool = Depends(require_admin),
):
    """Delete a downloaded model from disk and refresh the model pool."""
    model_dirs = _get_model_dirs()
    engine_pool = _get_engine_pool()

    if not model_dirs:
        raise HTTPException(status_code=503, detail="No model directories configured")

    # Search for model across all directories in both flat and org-folder layouts
    model_path = None
    parent_model_dir = None
    for model_dir in model_dirs:
        if not model_dir.exists():
            continue
        candidate = model_dir / model_name
        if candidate.is_dir() and (candidate / "config.json").exists():
            model_path = candidate
            parent_model_dir = model_dir
            break
        # Try two-level: search inside organization folders
        for subdir in model_dir.iterdir():
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue
            candidate = subdir / model_name
            if candidate.is_dir() and (candidate / "config.json").exists():
                model_path = candidate
                parent_model_dir = model_dir
                break
        if model_path is not None:
            break

    if model_path is None:
        raise HTTPException(status_code=404, detail="Model not found")

    # Validate path traversal against parent model directory
    try:
        if not model_path.resolve().is_relative_to(parent_model_dir.resolve()):
            raise HTTPException(status_code=400, detail="Invalid model name")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model name")

    if not model_path.is_dir():
        raise HTTPException(status_code=400, detail="Not a model directory")

    # Unload model if loaded
    if engine_pool is not None:
        loaded_ids = engine_pool.get_loaded_model_ids()
        if model_name in loaded_ids:
            try:
                await engine_pool.unload_engine_async(model_name)
                logger.info(f"Unloaded model '{model_name}' before deletion")
            except Exception as e:
                logger.warning(f"Failed to unload model '{model_name}': {e}")

    # Delete from disk
    # Handle macOS resource fork files (._*) that may disappear on non-native
    # filesystems (exFAT, NTFS). Use onexc (Python 3.12+) to avoid
    # DeprecationWarning, with onerror fallback for older versions.
    def _handle_onexc(func, path, exc):
        if isinstance(exc, FileNotFoundError) and Path(path).name.startswith("._"):
            logger.debug(f"Ignoring missing resource fork file: {path}")
            return
        raise exc

    def _handle_onerror(func, path, exc_info):
        if exc_info[0] is FileNotFoundError and Path(path).name.startswith("._"):
            logger.debug(f"Ignoring missing resource fork file: {path}")
            return
        raise exc_info[1].with_traceback(exc_info[2])

    try:
        if sys.version_info >= (3, 12):
            shutil.rmtree(model_path, onexc=_handle_onexc)
        else:
            shutil.rmtree(model_path, onerror=_handle_onerror)
        logger.info(f"Deleted model directory: {model_path}")
    except Exception as e:
        logger.error(f"Failed to delete model directory {model_path}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete model")

    # If the model was inside an org folder (organized layout) and that
    # folder is now empty, drop it so the listing stays tidy.
    parent = model_path.parent
    if parent != parent_model_dir and parent.exists() and not any(parent.iterdir()):
        try:
            parent.rmdir()
            logger.info(f"Removed empty org folder: {parent}")
        except OSError as e:
            logger.debug(f"Could not remove empty org folder {parent}: {e}")

    # Re-discover models
    if engine_pool is not None:
        settings_manager = _get_settings_manager()
        pinned_models = []
        if settings_manager:
            pinned_models = settings_manager.get_pinned_model_ids()

        engine_pool._entries.pop(model_name, None)
        # Release the deleted model's persisted settings (including its alias)
        # so they can be reused by another model.
        if settings_manager:
            settings_manager.delete_settings(model_name)
        await engine_pool.discover_models_async(
            [str(d) for d in model_dirs], pinned_models
        )
        if settings_manager:
            engine_pool.apply_settings_overrides(settings_manager)
        logger.info("Model pool refreshed after deletion")

    return {"success": True, "message": f"Model '{model_name}' deleted"}


router = _router
