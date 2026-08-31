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
from types import SimpleNamespace

from fastapi import APIRouter, Depends

from fusion_mlx.audio.registry import resolve_audio_alias
from fusion_mlx.config import get_config
from fusion_mlx.middleware.auth import verify_api_key
from fusion_mlx.model_aliases import resolve_profile

logger = logging.getLogger(__name__)

router = APIRouter()

_pool: object | None = None

# R11-B-F4 (#505): audio type -> advertised capability tag. TTS aliases
# report ``audio.speech``, STT aliases report ``audio.transcription``.
_AUDIO_TYPE_TO_CAPABILITY: dict[str, str] = {
    "tts": "audio.speech",
    "stt": "audio.transcription",
}

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
    # R11-B-F4 (#505): audio registry wins — short alias + HF id both
    # resolve via resolve_audio_alias (reverse HF-id index).
    audio_entry = resolve_audio_alias(model_id)
    if audio_entry is not None:
        return "audio"
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


def _capabilities_for(model_id: str, profile) -> list[str]:
    # R11-B-F4 (#726): unified capability resolver wired into the live
    # /v1/models listing + retrieve route. Audio aliases (tts/stt) have
    # NO profile — resolve_audio_alias owns their capability tag
    # (``audio.speech`` / ``audio.transcription``), so they no longer
    # ship with an empty ``capabilities=[]`` that drop-in OpenAI clients
    # cannot route. Text models keep ``profile.capabilities``; models
    # with neither profile nor audio entry fall back to ``[]`` (NOT
    # ``["text"]``) — pinned by test_capabilities_field's
    # unregistered-text-path contract.
    audio_entry = resolve_audio_alias(model_id)
    if audio_entry is not None:
        return [_AUDIO_TYPE_TO_CAPABILITY.get(audio_entry.type, "audio")]
    if profile is not None and profile.capabilities:
        return sorted(profile.capabilities)
    return []


def _build_model_info(model_id: str) -> SimpleNamespace:
    # R11-B-F4 (#505/#726): single-id model card builder. Audio aliases
    # (tts/stt) short-circuit to an audio-only capability set — no
    # ``text`` leak. Text models keep ``capabilities=["text"]`` and
    # resolve modality via _resolve_modality. Wired into the live
    # /v1/models/{model_id} retrieve route (separate contract from the
    # listing's _capabilities_for — the single-id card always carries a
    # non-empty capability set, so unregistered text ids report
    # ``["text"]`` here, NOT the listing's ``[]``).
    audio_entry = resolve_audio_alias(model_id)
    if audio_entry is not None:
        cap = _AUDIO_TYPE_TO_CAPABILITY.get(audio_entry.type, "audio")
        return SimpleNamespace(
            id=model_id,
            modality="audio",
            capabilities=[cap],
        )
    return SimpleNamespace(
        id=model_id,
        modality=_resolve_modality(model_id),
        capabilities=["text"],
    )


def _entry_payload(
    model_id,
    tool,
    reasoning,
    modality="text",
    capabilities=None,
    loaded=True,
    state="loaded",
):
    # #577: every entry carries a ``loaded`` bool + ``state`` ("loaded" |
    # "registered") so consumers can distinguish resident-in-memory models
    # from on-disk-registered-but-not-yet-loaded ones. Without this the
    # gateway / studio / design checkers saw a non-empty list and assumed
    # every id was immediately servable — a 502-on-generate "fake green".
    payload = {
        "id": model_id,
        "object": "model",
        "tool_call_parser": tool,
        "reasoning_parser": reasoning,
        "modality": modality,
        "loaded": loaded,
        "state": state,
    }
    if capabilities is not None:
        payload["capabilities"] = capabilities
    return payload


@router.get("/v1/models")
async def list_models(_auth: bool = Depends(verify_api_key)):
    cfg = get_config()
    data = []
    if cfg.model_registry is not None:
        for entry in cfg.model_registry:
            tool, reasoning = effective_parsers_for(entry.model_name, None, None)
            modality = _resolve_modality(entry.model_name)
            profile = resolve_profile(entry.model_name)
            caps = _capabilities_for(entry.model_name, profile)
            data.append(
                _entry_payload(entry.model_name, tool, reasoning, modality, caps)
            )
    elif cfg.model_name:
        profile = resolve_profile(cfg.model_alias) if cfg.model_alias else None
        profile_tool = profile.tool_call_parser if profile else None
        profile_reasoning = profile.reasoning_parser if profile else None
        tool, reasoning = effective_parsers_for(
            cfg.model_name, profile_tool, profile_reasoning
        )
        modality = _resolve_modality(cfg.model_name)
        caps = _capabilities_for(cfg.model_name, profile)
        data.append(_entry_payload(cfg.model_name, tool, reasoning, modality, caps))
        if cfg.model_alias:
            tool, reasoning = effective_parsers_for(
                cfg.model_alias, profile_tool, profile_reasoning
            )
            alias_modality = _resolve_modality(cfg.model_alias)
            data.append(
                _entry_payload(cfg.model_alias, tool, reasoning, alias_modality, caps)
            )
    # H-13: surface the boot-locked embedding model so discovery clients
    # (langchain / llamaindex / openai-python) find an
    # ``capabilities=["embedding"]`` card via client.models.list(). In
    # multi-model pool mode the pool branch below already lists it (the
    # embed model is preloaded into the pool); this single-model branch
    # covers single-route mounts + pool-less boots where the embed model
    # would otherwise be invisible. No-op when no embed model is locked.
    if cfg.embedding_model_locked:
        _listed = {entry["id"] for entry in data}
        if cfg.embedding_model_locked not in _listed:
            _tool, _reasoning = effective_parsers_for(
                cfg.embedding_model_locked, None, None
            )
            data.append(
                _entry_payload(
                    cfg.embedding_model_locked,
                    _tool,
                    _reasoning,
                    _resolve_modality(cfg.embedding_model_locked),
                    ["embedding"],
                )
            )
    # #577: surface on-disk-registered models that are NOT currently loaded
    # (resident in memory) so consumers can tell "registered" from "loaded".
    # The engine pool discovers every model under the configured model dirs;
    # an entry whose ``engine`` is None is registered-but-not-loaded. We only
    # append ids absent from ``data`` (loaded entries already listed above)
    # to avoid duplicates. Skipped when no pool is wired (single-route test
    # mounts) — those callers see only the served model, unchanged.
    if _pool is not None:
        listed_ids = {entry["id"] for entry in data}
        try:
            for model_id in _pool.list_models():
                if model_id in listed_ids:
                    continue
                pool_entry = _pool.get_entry(model_id)
                is_loaded = pool_entry is not None and pool_entry.engine is not None
                tool, reasoning = effective_parsers_for(model_id, None, None)
                modality = _resolve_modality(model_id)
                profile = resolve_profile(model_id)
                caps = _capabilities_for(model_id, profile)
                data.append(
                    _entry_payload(
                        model_id,
                        tool,
                        reasoning,
                        modality,
                        caps,
                        loaded=is_loaded,
                        state="loaded" if is_loaded else "registered",
                    )
                )
                listed_ids.add(model_id)
        except Exception:
            logger.warning(
                "routes_internal.models: pool enumeration failed", exc_info=True
            )
    logger.info("routes_internal.models: /v1/models listed %d entries", len(data))
    return {"object": "list", "data": data}


@router.get("/v1/models/{model_id}")
async def retrieve_model(model_id: str, _auth: bool = Depends(verify_api_key)):
    # R11-B-F4 (#726): single-id retrieve route. Audio aliases advertise
    # their audio capability + modality; text models report ``["text"]``.
    # Mirrors the listing's per-entry shape so clients bootstrapping state
    # via the single-id endpoint see the same envelope (not a stale text
    # card for an audio alias).
    info = _build_model_info(model_id)
    profile = resolve_profile(model_id)
    tool, reasoning = effective_parsers_for(
        model_id,
        getattr(profile, "tool_call_parser", None),
        getattr(profile, "reasoning_parser", None),
    )
    payload = _entry_payload(
        model_id,
        tool,
        reasoning,
        info.modality,
        info.capabilities,
    )
    logger.info(
        "routes_internal.models: /v1/models/%s modality=%s caps=%s",
        model_id,
        info.modality,
        info.capabilities,
    )
    return payload
