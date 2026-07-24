# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..admin.auth import require_admin
from ..middleware.auth import verify_api_key
from ..cache.protocol import (
    PROTOCOL_VERSION,
    InvalidExportPathError,
    MalformedManifestError,
    ManifestMismatchError,
    ManifestNotFoundError,
    read_manifest,
    resolve_cache_dir,
)

logger = logging.getLogger(__name__)

_NOT_IMPLEMENTED_MSG = "engine integration pending"

_NOT_IMPLEMENTED_DETAIL = {
    "error": {
        "message": _NOT_IMPLEMENTED_MSG,
        "type": "not_implemented_error",
        "code": None,
    }
}

_SANDBOX_ESCAPE_MSG = "destination must resolve under the cache-export sandbox"
_SANDBOX_ESCAPE_DETAIL = {
    "error": {
        "message": _SANDBOX_ESCAPE_MSG,
        "type": "invalid_request_error",
        "code": "sandbox_escape",
    }
}

router = APIRouter(
    prefix="/v1/cache",
    tags=["cache"],
    dependencies=[Depends(verify_api_key), Depends(require_admin)],
)


class ExportRequest(BaseModel):
    destination: str | None = Field(
        default=None,
        description=(
            "Path under FUSION_MLX_CACHE_EXPORT_DIR (default "
            "~/.cache/fusion-mlx/cache_exports/). May be relative (resolved "
            "against the sandbox root) or absolute (must resolve inside "
            "the sandbox). Omit to use the sandbox root itself."
        ),
    )
    max_bytes: int | None = Field(
        default=None,
        ge=1,
        description="Optional cap on the exported blob size.",
    )


class ImportRequest(BaseModel):
    source: str = Field(
        ...,
        description="Path to an export root containing manifest.json + index.json.",
    )
    expected_protocol_version: str = Field(
        default=PROTOCOL_VERSION,
        description=f"Manifest protocol version the caller expects. Current: {PROTOCOL_VERSION!r}.",
    )
    expected_model_id: str | None = Field(
        default=None,
        description="If set, manifest.model_id must match exactly. Mismatch -> 409.",
    )
    merge_strategy: Literal["replace", "merge"] = Field(
        default="merge",
        description="'merge' keeps existing entries, 'replace' clears first.",
    )


def _resolve_or_400(caller_path: str | None) -> Path:
    try:
        return resolve_cache_dir(caller_path)
    except InvalidExportPathError as exc:
        logger.warning(
            "cache: sandbox-escape rejected (caller_path=%r): %s",
            caller_path,
            exc,
        )
        raise HTTPException(status_code=403, detail=_SANDBOX_ESCAPE_DETAIL) from exc


def _read_manifest_or_http(root: Path):
    try:
        return read_manifest(root)
    except ManifestNotFoundError as exc:
        logger.info("cache: manifest not found at %s", root)
        raise HTTPException(
            status_code=404,
            detail="no manifest.json at the requested cache path",
        ) from exc
    except MalformedManifestError as exc:
        logger.warning("cache: malformed manifest at %s: %s", root, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/export", status_code=501)
async def export_cache(req: ExportRequest):
    destination = _resolve_or_400(req.destination)
    logger.info(
        "cache/export: validated destination=%s max_bytes=%s — %s",
        destination,
        req.max_bytes,
        _NOT_IMPLEMENTED_MSG,
    )
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED_DETAIL)


@router.post("/import", status_code=501)
async def import_cache(req: ImportRequest):
    source = _resolve_or_400(req.source)
    manifest = _read_manifest_or_http(source)

    if manifest.protocol_version != req.expected_protocol_version:
        raise HTTPException(
            status_code=409,
            detail=str(
                ManifestMismatchError(
                    "protocol_version",
                    req.expected_protocol_version,
                    manifest.protocol_version,
                )
            ),
        )

    if req.expected_model_id is not None and manifest.model_id != req.expected_model_id:
        raise HTTPException(
            status_code=409,
            detail=str(
                ManifestMismatchError(
                    "model_id", req.expected_model_id, manifest.model_id
                )
            ),
        )

    logger.info(
        "cache/import: validated source=%s manifest=%s merge=%s — %s",
        source,
        manifest.model_id,
        req.merge_strategy,
        _NOT_IMPLEMENTED_MSG,
    )
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED_DETAIL)


@router.get("/info")
async def cache_info(path: str | None = None):
    root = _resolve_or_400(path)
    manifest = _read_manifest_or_http(root)
    logger.debug(
        "cache/info: resolved root=%s model_id=%s entries=%s",
        root,
        manifest.model_id,
        manifest.entries,
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "manifest": manifest.to_dict(),
    }


@router.post("/clear")
async def clear_cache(is_admin: bool = Depends(require_admin)):
    import gc

    from ..service.helpers import _server_state

    pool = _server_state.get("engine_pool")
    if pool is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    gc.collect()
    return {"status": "ok", "message": "Cache cleared"}


@router.get("/stats")
async def cache_stats(is_admin: bool = Depends(require_admin)):
    # #178 Phase-2: aggregate diffusion text-encoding radix cache stats.
    # Reports per-encoder caches (UMT5/CLIP) via the module-level weakref
    # registry. This is NOT the LLM KV/prefix cache - labeled explicitly.
    from ..cache.radix_diffusion_cache import all_cache_stats

    caches = all_cache_stats()
    if not caches:
        return {
            "cache_type": "diffusion_text_encoding",
            "caches": [],
            "message": "No diffusion text caches active",
        }

    total_hits = sum(c.get("hits", 0) for c in caches)
    total_misses = sum(c.get("misses", 0) for c in caches)
    totals = {
        "cache_count": len(caches),
        "hits": total_hits,
        "misses": total_misses,
        "evictions": sum(c.get("evictions", 0) for c in caches),
        "insertions": sum(c.get("insertions", 0) for c in caches),
        "total_bytes": sum(c.get("total_bytes", 0) for c in caches),
        "hit_rate": total_hits / max(total_hits + total_misses, 1),
    }
    logger.debug(
        "cache stats: %d caches, %d hits/%d misses",
        len(caches),
        total_hits,
        total_misses,
    )
    return {
        "cache_type": "diffusion_text_encoding",
        "caches": caches,
        "totals": totals,
    }
