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

PRESET_REMOTE_URL = "https://fusion_mlx.ai/assets/omlx_preset.json"



from .helpers import (
    _require_model,
    _require_settings_manager,
)
from .models import (
    CreateProfileRequest,
    CreateTemplateRequest,
    UpdateProfileRequest,
    UpdateTemplateRequest,
)

_router = APIRouter()

# =============================================================================
# Profile & Template endpoints
# =============================================================================




@_router.get("/api/models/{model_id}/profiles")
async def list_model_profiles(
    model_id: str,
    is_admin: bool = Depends(require_admin),
):
    mgr = _require_settings_manager()
    _require_model(model_id)
    return {"profiles": mgr.list_profiles(model_id)}


@_router.post("/api/models/{model_id}/profiles")
async def create_model_profile(
    model_id: str,
    request: CreateProfileRequest,
    is_admin: bool = Depends(require_admin),
):
    from ..model_profiles import InvalidProfileNameError, filter_universal_fields

    mgr = _require_settings_manager()
    _require_model(model_id)
    try:
        profile = mgr.save_profile(
            model_id=model_id,
            name=request.name,
            display_name=request.display_name,
            description=request.description,
            settings=request.settings or {},
            source_template=request.source_template,
        )
    except InvalidProfileNameError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    if request.also_save_as_template:
        try:
            mgr.upsert_template(
                name=request.name,
                display_name=request.display_name,
                description=request.description,
                settings=filter_universal_fields(request.settings or {}),
            )
        except InvalidProfileNameError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return {"profile": profile}


@_router.put("/api/models/{model_id}/profiles/{name}")
async def update_model_profile(
    model_id: str,
    name: str,
    request: UpdateProfileRequest,
    is_admin: bool = Depends(require_admin),
):
    from ..model_profiles import InvalidProfileNameError, filter_universal_fields

    mgr = _require_settings_manager()
    _require_model(model_id)
    try:
        updated = mgr.update_profile(
            model_id=model_id,
            name=name,
            new_name=request.new_name,
            display_name=request.display_name,
            description=request.description,
            settings=request.settings,
            source_template=request.source_template,
        )
    except InvalidProfileNameError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Profile not found: {name}")

    if request.also_save_as_template and request.settings is not None:
        try:
            mgr.upsert_template(
                name=updated["name"],
                display_name=updated["display_name"],
                description=updated.get("description"),
                settings=filter_universal_fields(request.settings),
            )
        except InvalidProfileNameError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return {"profile": updated}


@_router.delete("/api/models/{model_id}/profiles/{name}")
async def delete_model_profile(
    model_id: str,
    name: str,
    is_admin: bool = Depends(require_admin),
):
    mgr = _require_settings_manager()
    _require_model(model_id)
    if not mgr.delete_profile(model_id, name):
        raise HTTPException(status_code=404, detail=f"Profile not found: {name}")
    return {"deleted": True, "name": name}


@_router.post("/api/models/{model_id}/profiles/{name}/apply")
async def apply_model_profile(
    model_id: str,
    name: str,
    is_admin: bool = Depends(require_admin),
):
    mgr = _require_settings_manager()
    _require_model(model_id)
    applied = mgr.apply_profile(model_id, name)
    if applied is None:
        raise HTTPException(status_code=404, detail=f"Profile not found: {name}")
    return {"model_id": model_id, "settings": applied.to_dict()}


@_router.get("/api/profile-fields")
async def get_profile_fields(is_admin: bool = Depends(require_admin)):
    from ..model_profiles import (
        MODEL_SPECIFIC_PROFILE_FIELDS,
        UNIVERSAL_PROFILE_FIELDS,
    )

    return {
        "universal": list(UNIVERSAL_PROFILE_FIELDS),
        "model_specific": list(MODEL_SPECIFIC_PROFILE_FIELDS),
    }


@_router.get("/api/profile-templates")
async def list_templates(is_admin: bool = Depends(require_admin)):
    mgr = _require_settings_manager()
    return {"templates": mgr.list_templates()}


@_router.post("/api/profile-templates")
async def create_template(
    request: CreateTemplateRequest,
    is_admin: bool = Depends(require_admin),
):
    from ..model_profiles import InvalidProfileNameError

    mgr = _require_settings_manager()
    try:
        tmpl = mgr.save_template(
            name=request.name,
            display_name=request.display_name,
            description=request.description,
            settings=request.settings or {},
        )
    except InvalidProfileNameError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"template": tmpl}


@_router.put("/api/profile-templates/{name}")
async def update_template(
    name: str,
    request: UpdateTemplateRequest,
    is_admin: bool = Depends(require_admin),
):
    from ..model_profiles import InvalidProfileNameError

    mgr = _require_settings_manager()
    try:
        updated = mgr.update_template(
            name=name,
            new_name=request.new_name,
            display_name=request.display_name,
            description=request.description,
            settings=request.settings,
        )
    except InvalidProfileNameError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Template not found: {name}")
    return {"template": updated}


@_router.delete("/api/profile-templates/{name}")
async def delete_template(
    name: str,
    is_admin: bool = Depends(require_admin),
):
    mgr = _require_settings_manager()
    if not mgr.delete_template(name):
        raise HTTPException(status_code=404, detail=f"Template not found: {name}")
    return {"deleted": True, "name": name}



router = _router
