# SPDX-License-Identifier: Apache-2.0
import hashlib
import logging
import secrets
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException

from ..settings import SubKeyEntry
from .auth import (
    require_admin,
    validate_api_key,
)

logger = logging.getLogger(__name__)

PRESET_REMOTE_URL = "http://bench.dpdns.org/assets/fusionmlx_preset.json"


from .helpers import (
    _get_global_settings,
)
from .models import (
    CreateSubKeyRequest,
    DeleteSubKeyRequest,
)

_router = APIRouter()


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@_router.post("/api/sub-keys")
async def create_sub_key(
    request: CreateSubKeyRequest, is_admin: bool = Depends(require_admin)
):
    global_settings = _get_global_settings()
    if global_settings is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    is_valid, error_msg = validate_api_key(request.key)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    if global_settings.auth.api_key and secrets.compare_digest(
        request.key, global_settings.auth.api_key
    ):
        raise HTTPException(
            status_code=400, detail="Sub key cannot be the same as the main key"
        )

    new_hash = _hash_key(request.key)
    for sk in global_settings.auth.sub_keys:
        if sk.key_hash == new_hash:
            raise HTTPException(status_code=400, detail="This key already exists")

    entry = SubKeyEntry(
        name=request.name or "",
        key_hash=new_hash,
        created_at=datetime.now(UTC).isoformat(),
    )
    global_settings.auth.sub_keys.append(entry)

    try:
        global_settings.save()
    except Exception as e:
        global_settings.auth.sub_keys.pop()
        logger.error("Failed to save settings after subkey creation: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save settings")
    logger.info(f"Sub key created: {request.name or '(unnamed)'}")
    return {
        "success": True,
        "sub_key": {"name": entry.name, "created_at": entry.created_at},
    }


@_router.delete("/api/sub-keys")
async def delete_sub_key(
    request: DeleteSubKeyRequest, is_admin: bool = Depends(require_admin)
):
    global_settings = _get_global_settings()
    if global_settings is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    request_hash = _hash_key(request.key)
    for i, sk in enumerate(global_settings.auth.sub_keys):
        if sk.key_hash == request_hash:
            removed = global_settings.auth.sub_keys.pop(i)
            try:
                global_settings.save()
            except Exception as e:
                global_settings.auth.sub_keys.insert(i, removed)
                raise HTTPException(
                    status_code=500, detail="Failed to save settings"
                )
            logger.info(f"Sub key deleted: {sk.name or '(unnamed)'}")
            return {"success": True}

    raise HTTPException(status_code=404, detail="Sub key not found")


router = _router
