# SPDX-License-Identifier: Apache-2.0
"""Model alias definitions for fusion-mlx."""

import logging
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

logger = logging.getLogger(__name__)

_ALIASES_FILE = Path(__file__).parent / "aliases.json"


def _allowed_model_dirs() -> list[str]:
    home = os.path.realpath(os.path.expanduser("~"))
    dirs = [
        os.path.join(home, ".fusion-mlx", "models"),
        os.path.join(home, ".cache", "huggingface"),
    ]
    cwd = os.path.realpath(os.getcwd())
    if cwd != "/" and len(Path(cwd).parts) >= 2:
        dirs.append(cwd)
    return dirs

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


def _is_path_like(name: str) -> bool:
    if os.path.isabs(name):
        return True
    if "/" in name or "\\" in name:
        return True
    if name.startswith("."):
        return True
    return False


def resolve_model(name: str) -> str:
    if ".." in name.split(os.sep) or ".." in name.split("/"):
        logger.warning("resolve_model: path traversal component rejected: %s", name)
        raise ValueError(f"Path not allowed: {name}. Path traversal (..) is forbidden.")
    if os.path.isabs(name):
        resolved = os.path.realpath(name)
        allowed = _allowed_model_dirs()
        if not any(resolved.startswith(p) for p in allowed):
            logger.warning("resolve_model: absolute path outside allowed dirs: %s", name)
            raise ValueError(f"Path not allowed: {name}. Must be within allowed model directories.")
        return name
    if _is_path_like(name) and os.path.exists(name):
        resolved = os.path.realpath(name)
        allowed = _allowed_model_dirs()
        if not any(resolved.startswith(p) for p in allowed):
            logger.warning("resolve_model: path outside allowed dirs: %s", name)
            raise ValueError(f"Path not allowed: {name}. Must be within allowed model directories.")
        return name
    if _is_path_like(name):
        resolved = os.path.realpath(name)
        allowed = _allowed_model_dirs()
        if not any(resolved.startswith(p) for p in allowed):
            logger.warning("resolve_model: path-like name outside allowed dirs: %s", name)
            raise ValueError(f"Path not allowed: {name}. Must be within allowed model directories.")
        return name
    if os.path.exists(name):
        resolved = os.path.realpath(name)
        allowed = _allowed_model_dirs()
        if any(resolved.startswith(p) for p in allowed):
            return name
    aliases = _load_aliases()
    if name in aliases:
        entry = aliases[name]
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            return entry.get("hf_path", entry.get("path", name))
    return name


def resolve_profile(name: str) -> AliasProfile | None:
    for profile in list_profiles():
        if profile.name == name:
            return profile
    return None


_SIZE_PATTERN = re.compile(r"\d+(?:\.\d+)?[bm](?:it)?\b", re.IGNORECASE)
_SIZE_TOKEN_START = re.compile(r"^\d")
_SEPARATOR_SPLIT = re.compile(r"[-_./]")


def _letters_only_prefix(raw: str) -> str:
    out = []
    for ch in raw:
        if ch.isalpha():
            out.append(ch.lower())
        else:
            break
    return "".join(out)


def _has_size_token(name: str) -> bool:
    return bool(_SIZE_PATTERN.search(name))


def _family_prefix(name: str) -> str:
    tokens = [t for t in _SEPARATOR_SPLIT.split(name) if t]
    family_tokens = []
    for tok in tokens:
        if _SIZE_TOKEN_START.match(tok):
            break
        family_tokens.append(tok)
    if not family_tokens:
        return _letters_only_prefix(name)
    return "-".join(family_tokens).lower()


def suggest_similar(name: str, n: int = 3, cutoff: float = 0.6) -> list[str]:
    # Family-aware suggest: gate on family-prefix match (or letter-only fallback
    # for separator-mismatch / collapsed-hyphen inputs). Family filter IS the
    # matcher - SequenceMatcher only ranks, cutoff is not applied as a gate
    # (prefix matches like "hermes"->"hermes3-8b-4bit" sit below 0.6 but are
    # correct). cutoff retained for API signature stability.
    aliases = _load_aliases()
    alias_names = list(aliases.keys())
    if not alias_names:
        return []
    if len(name) < 2:
        return []
    has_size = _has_size_token(name)
    tokens = [t for t in _SEPARATOR_SPLIT.split(name) if t]
    is_multi_segment = len(tokens) > 1
    # Legit-looking HF id: multi-segment without a size token
    # (bert-base-uncased, qwen-coder), or single-segment with a digit but no
    # size token (gpt2, xyzabc12345). Must NOT bait-and-switch into an alias.
    if is_multi_segment and not has_size:
        return []
    if not is_multi_segment and not name.isalpha() and not has_size:
        return []
    family = _family_prefix(name)
    candidates = [a for a in alias_names if a.lower().startswith(family)]
    if not candidates:
        # Letter-only fallback (collapsed separator / version-digit mash).
        letters = _letters_only_prefix(name)
        if len(letters) < 3:
            return []
        candidates = [a for a in alias_names if a.lower().startswith(letters)]
    if not candidates:
        return []
    ranked = sorted(
        candidates,
        key=lambda a: SequenceMatcher(None, name, a).ratio(),
        reverse=True,
    )
    logger.debug("suggest_similar: name=%s family=%s -> %s", name, family, ranked[:n])
    return ranked[:n]
