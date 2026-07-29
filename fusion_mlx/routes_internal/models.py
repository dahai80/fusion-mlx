# SPDX-License-Identifier: Apache-2.0
"""/v1/models route surfacing EFFECTIVE parsers (live CLI override /
auto-detect / alias-profile default) - not just static alias defaults.

Lookup order (effective_parsers_for):
  Tier 1 - per-entry live state: registry entry whose matches(id) is True.
           Strict ``is True`` guard rejects truthy-non-bool returns.
  Tier 2 - per-server live state: id is the served model_name / model_alias.
           Reads ServerConfig.tool_call_parser / .reasoning_parser_name. Each side
           independent - no backfill from profile.
  Tier 3 - alias-profile default (profile.tool_call_parser / .reasoning_parser).
  Tier 4 - None.
"""

import logging

from fastapi import APIRouter, Depends

from fusion_mlx.config import get_config
from fusion_mlx.middleware.auth import verify_api_key
from fusion_mlx.model_aliases import resolve_profile

logger = logging.getLogger(__name__)

router = APIRouter()

_pool: object | None = None

_MODEL_TYPE_TO_MODALITY: dict[str, str] = {
    "llm": "text",
    "vlm": "text",
    "embedding": "text",
    "reranker": "text",
    "ner": "text",
    "audio_stt": "audio",
    "audio_tts": "audio",
    "audio_sts": "audio",
    "image": "image",
    "video": "video",
}


def set_models_context(pool: object | None) -> None:
    global _pool
    _pool = pool


def _is_served_model(model_id: str) -> bool:
    cfg = get_config()
    if cfg.model_name and model_id == cfg.model_name:
        return True
    if cfg.model_alias and model_id == cfg.model_alias:
        return True
    return False


def effective_parsers_for(model_id, profile_tool, profile_reasoning):
    # Tier 1: per-entry live state (registry match). Strict ``is True`` guard.
    cfg = get_config()
    registry = cfg.model_registry
    if registry is not None:
        entry = registry.get_entry(model_id)
        if entry is not None and entry.matches(model_id) is True:
            return (entry.tool_call_parser, entry.reasoning_parser)
    # Tier 2: per-server live state from ServerConfig (#50 globals consolidation).
    # Each side independent - no profile backfill.
    if _is_served_model(model_id):
        return (cfg.tool_call_parser, cfg.reasoning_parser_name)
    # Tier 3/4: alias-profile default, or None.
    return (profile_tool, profile_reasoning)


def _resolve_modality(model_id: str) -> str:
    profile = resolve_profile(model_id)
    if profile is not None and profile.modality:
        return profile.modality
    if _pool is not None:
        entry = _pool.get_entry(model_id)
        if entry is not None:
            mt = getattr(entry, "model_type", None)
            if mt and mt in _MODEL_TYPE_TO_MODALITY:
                return _MODEL_TYPE_TO_MODALITY[mt]
    return "text"


def _entry_payload(model_id, tool, reasoning, modality="text"):
    return {
        "id": model_id,
        "object": "model",
        "tool_call_parser": tool,
        "reasoning_parser": reasoning,
        "modality": modality,
    }


@router.get("/v1/models")
async def list_models(_auth: bool = Depends(verify_api_key)):
    cfg = get_config()
    data = []
    if cfg.model_registry is not None:
        for entry in cfg.model_registry:
            tool, reasoning = effective_parsers_for(entry.model_name, None, None)
            modality = _resolve_modality(entry.model_name)
            data.append(_entry_payload(entry.model_name, tool, reasoning, modality))
    elif cfg.model_name:
        profile = resolve_profile(cfg.model_alias) if cfg.model_alias else None
        profile_tool = profile.tool_call_parser if profile else None
        profile_reasoning = profile.reasoning_parser if profile else None
        tool, reasoning = effective_parsers_for(
            cfg.model_name, profile_tool, profile_reasoning
        )
        modality = _resolve_modality(cfg.model_name)
        data.append(_entry_payload(cfg.model_name, tool, reasoning, modality))
        if cfg.model_alias:
            tool, reasoning = effective_parsers_for(
                cfg.model_alias, profile_tool, profile_reasoning
            )
            alias_modality = _resolve_modality(cfg.model_alias)
            data.append(
                _entry_payload(cfg.model_alias, tool, reasoning, alias_modality)
            )
    logger.info("routes_internal.models: /v1/models listed %d entries", len(data))
    return {"object": "list", "data": data}
