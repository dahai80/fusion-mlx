# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

AudioType = Literal["tts", "stt"]


@dataclass(frozen=True)
class AudioAliasEntry:
    alias: str
    type: AudioType
    hf_id: str
    family: str
    default_voice: str | None = None
    languages: str | None = None
    notes: str = ""


_REGISTRY: dict[str, AudioAliasEntry] | None = None
_HF_ID_INDEX: dict[str, str] = {}


def _reset_registry_cache() -> None:
    global _REGISTRY
    _REGISTRY = None
    _HF_ID_INDEX.clear()


def _registry_path() -> str:
    return os.path.join(os.path.dirname(__file__), "aliases.json")


def _load_registry() -> dict[str, AudioAliasEntry]:
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY

    path = _registry_path()
    with open(path) as f:
        raw = json.load(f)

    entries: dict[str, AudioAliasEntry] = {}
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        if not isinstance(value, dict):
            raise ValueError(
                f"audio aliases.json: entry {key!r} must be an object, "
                f"got {type(value).__name__}"
            )
        try:
            kind = value["type"]
            hf_id = value["hf_id"]
            family = value["family"]
        except KeyError as e:
            raise ValueError(
                f"audio aliases.json: entry {key!r} missing required "
                f"field {e.args[0]!r}"
            ) from e
        if kind not in ("tts", "stt"):
            raise ValueError(
                f"audio aliases.json: entry {key!r} has invalid type "
                f"{kind!r}; must be 'tts' or 'stt'"
            )
        if "/" not in hf_id:
            raise ValueError(
                f"audio aliases.json: entry {key!r}.hf_id={hf_id!r} "
                "must be a HuggingFace org/name repo id"
            )
        entries[key] = AudioAliasEntry(
            alias=key,
            type=kind,
            hf_id=hf_id,
            family=family,
            default_voice=value.get("default_voice"),
            languages=value.get("languages"),
            notes=value.get("notes", ""),
        )

    _REGISTRY = entries
    _HF_ID_INDEX.clear()
    for alias, entry in entries.items():
        _HF_ID_INDEX.setdefault(entry.hf_id.lower(), alias)
    return entries


def resolve_audio_alias(name: str | None) -> AudioAliasEntry | None:
    if not isinstance(name, str) or not name:
        return None
    registry = _load_registry()
    lc = name.lower()
    entry = registry.get(lc)
    if entry is not None:
        return entry
    alias = _HF_ID_INDEX.get(lc)
    if alias is not None:
        return registry[alias]
    return None


def is_audio_name(name: str | None) -> bool:
    return resolve_audio_alias(name) is not None


def list_audio_aliases() -> list[AudioAliasEntry]:
    return sorted(_load_registry().values(), key=lambda e: e.alias)


def stt_aliases() -> dict[str, str]:
    return {e.alias: e.hf_id for e in _load_registry().values() if e.type == "stt"}


def tts_aliases() -> dict[str, str]:
    return {e.alias: e.hf_id for e in _load_registry().values() if e.type == "tts"}
