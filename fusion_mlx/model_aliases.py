# SPDX-License-Identifier: Apache-2.0
"""Model alias definitions for fusion-mlx."""

import logging
import os
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

logger = logging.getLogger(__name__)

_ALIASES_FILE = Path(__file__).parent / "aliases.json"

_POPULAR_ALIAS_NAMES = [
    "qwen3.5-4b-4bit",
    "qwen3.5-9b-4bit",
    "qwen3.5-9b-8bit",
    "qwen3.5-27b-4bit",
    "qwen3.5-27b-8bit",
    "gemma-4-4b-4bit",
    "gemma-4-12b-4bit",
    "llama4-8b-4bit",
]

POPULAR_ALIASES = _POPULAR_ALIAS_NAMES


@dataclass
class AliasProfile:
    name: str
    hf_path: str
    supports_dflash: bool = False
    is_moe: bool = False
    drafter_hf_path: str | None = None
    description: str = ""
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    is_hybrid: bool = False
    supports_spec_decode: bool = True
    supports_mllm: bool = False
    is_audio: bool = False
    supports_dspark: bool = False
    modality: str = ""


def _load_aliases() -> dict[str, str]:
    if not _ALIASES_FILE.exists():
        return {}
    import json

    try:
        with open(_ALIASES_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load aliases.json: %s", e)
    return {}


def list_aliases() -> list[str]:
    return sorted(_load_aliases().keys())


def list_profiles() -> list[AliasProfile]:
    aliases = _load_aliases()
    profiles = []
    for name, hf_path in aliases.items():
        if isinstance(hf_path, str):
            profiles.append(AliasProfile(name=name, hf_path=hf_path))
        elif isinstance(hf_path, dict):
            profiles.append(
                AliasProfile(
                    name=name,
                    hf_path=hf_path.get("hf_path", hf_path.get("path", "")),
                    supports_dflash=hf_path.get("supports_dflash", False),
                    is_moe=hf_path.get("is_moe", False),
                    drafter_hf_path=hf_path.get("drafter_hf_path"),
                    description=hf_path.get("description", ""),
                    tool_call_parser=hf_path.get("tool_call_parser"),
                    reasoning_parser=hf_path.get("reasoning_parser"),
                    is_hybrid=hf_path.get("is_hybrid", False),
                    supports_spec_decode=hf_path.get("supports_spec_decode", True),
                    supports_mllm=hf_path.get("supports_mllm", False),
                    supports_dspark=hf_path.get("supports_dspark", False),
                    modality=hf_path.get("modality", ""),
                )
            )
    return profiles


def resolve_model(name: str) -> str:
    aliases = _load_aliases()
    if name in aliases:
        entry = aliases[name]
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            return entry.get("hf_path", entry.get("path", name))
    if "/" in name:
        return name
    if os.path.exists(name):
        return name
    return name


def resolve_profile(name: str) -> AliasProfile | None:
    for profile in list_profiles():
        if profile.name == name:
            return profile
    return None


def suggest_similar(name: str, n: int = 3, cutoff: float = 0.6) -> list[str]:
    aliases = _load_aliases()
    return get_close_matches(name, aliases.keys(), n=n, cutoff=cutoff)
