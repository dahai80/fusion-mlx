# SPDX-License-Identifier: Apache-2.0
"""Admin panel routes for oMLX server configuration.

This module provides HTTP routes for the admin panel including:
- Login/logout with API key authentication
- Dashboard for server monitoring
- Model settings management (per-model sampling parameters, pinning, default)
- Global settings management
"""

import asyncio
import logging
from pathlib import Path

import requests
from fastapi import APIRouter, Depends, HTTPException

from .auth import (
    require_admin,
)

logger = logging.getLogger(__name__)

PRESET_REMOTE_URL = "http://bench.dpdns.org/assets/omlx_preset.json"



from .helpers import (
    _get_engine_pool,
)

_router = APIRouter()

# =============================================================================
# Preset refresh (proxy to bench.dpdns.org to avoid CORS)
# =============================================================================


@_router.post("/api/presets/refresh")
async def refresh_presets(is_admin: bool = Depends(require_admin)):
    """Fetch the latest preset bundle from bench.dpdns.org and return it.

    The client uses this instead of fetching bench.dpdns.org directly so we do not
    depend on CORS headers on the remote host. Any failure is surfaced as 502
    so the client can silently fall back to the bundled presets.
    """
    try:
        resp = await asyncio.to_thread(
            requests.get,
            PRESET_REMOTE_URL,
            timeout=10,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Fetch failed: {e}")
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Remote returned {resp.status_code}",
        )
    try:
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Invalid JSON: {e}")


@_router.get("/api/models/{model_id}/generation_config")
async def get_generation_config(
    model_id: str,
    is_admin: bool = Depends(require_admin),
):
    """
    Read model config files and return recommended defaults.

    Reads generation_config.json for sampling parameters and config.json
    for max_context_window (max_position_embeddings).

    Args:
        model_id: The model identifier.

    Returns:
        JSON with recommended parameters from the model's config files.

    Raises:
        HTTPException: 404 if model not found or no config files exist.
    """
    import json as json_module

    engine_pool = _get_engine_pool()
    if engine_pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")

    entry = engine_pool.get_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    model_path = Path(entry.model_path)
    result = {}

    # Read generation_config.json for sampling parameters
    gen_config_path = model_path / "generation_config.json"
    if gen_config_path.exists():
        try:
            with open(gen_config_path, encoding="utf-8") as f:
                gen_config = json_module.load(f)

            # Temperature: if do_sample is false, effective temperature is 0
            do_sample = gen_config.get("do_sample", True)
            if "temperature" in gen_config:
                result["temperature"] = 0.0 if not do_sample else gen_config["temperature"]

            if "top_p" in gen_config:
                result["top_p"] = gen_config["top_p"]

            if "top_k" in gen_config:
                result["top_k"] = gen_config["top_k"]

            if "repetition_penalty" in gen_config:
                result["repetition_penalty"] = gen_config["repetition_penalty"]

        except (json_module.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to parse generation_config.json for {model_id}: {e}")

    # Read config.json for max_position_embeddings → max_context_window
    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                model_config = json_module.load(f)

            max_pos = (
                model_config.get("max_position_embeddings")
                or model_config.get("max_seq_len")
                or model_config.get("seq_length")
                or model_config.get("n_positions")
            )

            # Nested config fallback (VLM, MoE models like Qwen3.5, GLM-4V)
            if not max_pos:
                text_config = model_config.get("text_config", {})
                if isinstance(text_config, dict):
                    max_pos = (
                        text_config.get("max_position_embeddings")
                        or text_config.get("max_seq_len")
                        or text_config.get("seq_length")
                        or text_config.get("n_positions")
                    )

            if max_pos and isinstance(max_pos, int):
                result["max_context_window"] = max_pos

        except (json_module.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to parse config.json for {model_id}: {e}")

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No config files with defaults found for {model_id}",
        )

    return result



router = _router
