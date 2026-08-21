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

_hf_to_alias: dict[str, str] | None = None


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


@dataclass(frozen=True)
class AliasProfile:
    name: str = ""
    hf_path: str = ""
    supports_dflash: bool = False
    is_moe: bool = False
    drafter_hf_path: str | None = None
    dflash_draft_model: str | None = None
    description: str = ""
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    is_hybrid: bool = False
    is_hybrid_explicit: bool = False
    supports_spec_decode: bool = True
    supports_mllm: bool = False
    is_audio: bool = False
    supports_dspark: bool = False
    supports_dflash2: bool = False
    modality: str = "text"
    recommended_sampling: tuple[tuple[str, float], ...] | None = None
    suffix_decoding_tier: str = "unknown"
    pflash_tier: str = "unknown"
    turboquant_tier: str = "unknown"
    default_max_tokens: int | None = None
    suffix_bench_speedup: tuple[tuple[str, float], ...] | None = None
    # Phase 2: unified spec-decode routing fields
    model_family: str | None = None
    spec_methods: tuple[str, ...] = ()
    spec_drafter_map: dict[str, str] | None = None
    spec_constraints: tuple[str, ...] = ()

    @property
    def capabilities(self) -> frozenset[str]:
        caps = []
        if self.supports_dflash:
            caps.append("dflash")
        if self.supports_dflash2:
            caps.append("dflash2")
        if self.supports_dspark:
            caps.append("dspark")
        if self.supports_spec_decode:
            caps.append("spec_decode")
        if self.tool_call_parser:
            caps.append("tool_call")
        if self.reasoning_parser:
            caps.append("reasoning")
        if self.supports_mllm:
            caps.append("vision")
        if self.is_audio:
            caps.append("audio")
        if self.is_moe:
            caps.append("moe")
        if self.is_hybrid:
            caps.append("hybrid")
        for m in self.spec_methods:
            if m not in caps:
                caps.append(m)
        return frozenset(caps)


_aliases: dict[str, AliasProfile] | None = None


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


def list_aliases() -> dict[str, str]:
    raw = _load_aliases()
    out = {}
    for name, value in raw.items():
        if isinstance(value, str):
            out[name] = value
        elif isinstance(value, dict):
            out[name] = value.get("hf_path", value.get("path", ""))
    return out


_VALID_MODALITIES = frozenset({"text", "text-diffusion", "embedding"})
_RESERVED_MODALITIES = frozenset({"vision", "image-gen"})

_SUPPORTED_SAMPLING_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
    }
)


def _coerce_recommended_sampling(
    raw,
) -> tuple[tuple[str, float], ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("recommended_sampling must be an object")
    if not raw:
        return None
    for key in raw:
        if key not in _SUPPORTED_SAMPLING_KEYS:
            raise ValueError(f"unsupported key {key!r} in recommended_sampling")
    coerced = []
    for key in sorted(raw):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"recommended_sampling {key!r} must be a number")
        coerced.append((key, float(value)))
    logger.debug("_coerce_recommended_sampling: %d keys", len(coerced))
    return tuple(coerced)


def _coerce(name: str, raw: str | dict) -> AliasProfile:
    if isinstance(raw, str):
        if not raw:
            raise ValueError(f"alias {name!r} has empty hf_path (string form)")
        return AliasProfile(name=name, hf_path=raw, modality="text")
    if not isinstance(raw, dict):
        raise ValueError(
            f"alias {name!r} must be a string or dict, got {type(raw).__name__}"
        )
    hf_path = raw.get("hf_path", raw.get("path", ""))
    if not hf_path or not isinstance(hf_path, str) or not hf_path.strip():
        raise ValueError(f"alias {name!r} must have a non-empty string hf_path")
    modality = raw.get("modality", "text")
    _allowed = sorted(_VALID_MODALITIES | _RESERVED_MODALITIES)
    if not isinstance(modality, str):
        raise ValueError(
            f"modality must be one of {_allowed} (alias {name!r}, "
            f"got {type(modality).__name__})"
        )
    if modality in _RESERVED_MODALITIES:
        raise ValueError(
            f"modality {modality!r} is reserved but not yet implemented "
            f"(alias {name!r})"
        )
    if modality not in _VALID_MODALITIES:
        raise ValueError(
            f"modality must be one of {_allowed} (alias {name!r}, got {modality!r})"
        )
    supports_spec_decode = raw.get("supports_spec_decode", True)
    supports_dflash = raw.get("supports_dflash", False)
    if modality != "text":
        if supports_spec_decode:
            raise ValueError(
                f"supports_spec_decode must be false for non-text modality "
                f"{modality!r} (alias {name!r})"
            )
        if supports_dflash:
            raise ValueError(
                f"supports_dflash must be false for non-text modality "
                f"{modality!r} (alias {name!r})"
            )
    recommended_sampling = _coerce_recommended_sampling(raw.get("recommended_sampling"))
    is_hybrid = raw.get("is_hybrid", False)
    is_hybrid_explicit = raw.get("is_hybrid_explicit", is_hybrid)
    suffix_bench_speedup = None
    raw_speedup = raw.get("suffix_bench_speedup")
    if raw_speedup and isinstance(raw_speedup, dict):
        suffix_bench_speedup = tuple(sorted(raw_speedup.items(), key=lambda kv: kv[0]))

    # Phase 2: unified spec-decode routing fields
    model_family = raw.get("model_family")
    raw_spec_methods = raw.get("spec_methods")
    spec_methods = tuple(raw_spec_methods) if raw_spec_methods else ()
    raw_drafter_map = raw.get("spec_drafter_map")
    spec_drafter_map = dict(raw_drafter_map) if raw_drafter_map else None
    raw_constraints = raw.get("spec_constraints")
    spec_constraints = tuple(raw_constraints) if raw_constraints else ()

    profile = AliasProfile(
        name=name,
        hf_path=hf_path,
        supports_dflash=supports_dflash,
        is_moe=raw.get("is_moe", False),
        drafter_hf_path=raw.get("drafter_hf_path"),
        dflash_draft_model=raw.get("dflash_draft_model") or raw.get("drafter_hf_path"),
        description=raw.get("description", ""),
        tool_call_parser=raw.get("tool_call_parser"),
        reasoning_parser=raw.get("reasoning_parser"),
        is_hybrid=is_hybrid,
        is_hybrid_explicit=is_hybrid_explicit,
        supports_spec_decode=supports_spec_decode,
        supports_mllm=raw.get("supports_mllm", False),
        supports_dspark=raw.get("supports_dspark", False),
        supports_dflash2=raw.get("supports_dflash2", False),
        modality=modality,
        recommended_sampling=recommended_sampling,
        suffix_decoding_tier=raw.get("suffix_decoding_tier", "unknown"),
        pflash_tier=raw.get("pflash_tier", "unknown"),
        turboquant_tier=raw.get("turboquant_tier", "unknown"),
        default_max_tokens=raw.get("default_max_tokens"),
        suffix_bench_speedup=suffix_bench_speedup,
        model_family=model_family,
        spec_methods=spec_methods,
        spec_drafter_map=spec_drafter_map,
        spec_constraints=spec_constraints,
    )
    return _migrate_legacy_spec_fields(profile)


def _migrate_legacy_spec_fields(profile: AliasProfile) -> AliasProfile:
    """Map old supports_dflash/supports_dspark to new spec_methods when
    spec_methods is empty. Backward compat: existing aliases.json entries
    that use the legacy flags still work without spec_methods.
    """
    if profile.spec_methods:
        return profile
    methods = []
    if profile.supports_dflash:
        methods.append("ddtree")
    if profile.supports_dspark:
        methods.append("dspark")
    if not methods:
        return profile
    logger.debug(
        "migrating legacy spec fields for %s: dflash=%s dspark=%s -> %s",
        profile.name,
        profile.supports_dflash,
        profile.supports_dspark,
        methods,
    )
    return AliasProfile(
        **{k: v for k, v in profile.__dict__.items() if k != "spec_methods"},
        spec_methods=tuple(methods),
    )


def list_profiles() -> dict[str, AliasProfile]:
    global _aliases, _hf_to_alias
    if _aliases is not None:
        return _aliases
    aliases = _load_aliases()
    profiles = {}
    skipped = 0
    skipped_errors: list[str] = []
    for name, raw in aliases.items():
        try:
            profiles[name] = _coerce(name, raw)
        except ValueError as e:
            logger.warning("list_profiles: skipping alias %r: %s", name, e)
            skipped += 1
            skipped_errors.append(str(e))
    if skipped:
        raise ValueError(
            f"list_profiles: {skipped}/{len(aliases)} aliases invalid: "
            f"{'; '.join(skipped_errors)}"
        )
    else:
        logger.debug("list_profiles: %d aliases loaded", len(aliases))
    _aliases = profiles
    # Build reverse index on first load
    if _hf_to_alias is None:
        _hf_to_alias = {}
        for alias_name, profile in profiles.items():
            if profile.hf_path:
                _hf_to_alias[profile.hf_path] = alias_name
    return profiles


def _is_path_like(name: str) -> bool:
    if os.path.isabs(name):
        return True
    if "/" in name or "\\" in name:
        return True
    if name.startswith("."):
        return True
    return False


def _check_path_allowed(name: str) -> None:
    resolved = os.path.realpath(name)
    allowed = _allowed_model_dirs()
    if not any(resolved.startswith(p) for p in allowed):
        logger.warning("resolve_model: path outside allowed dirs: %s", name)
        raise ValueError(
            f"Path not allowed: {name}. Must be within allowed model directories."
        )


def resolve_model(name: str) -> str:
    if ".." in name.split(os.sep) or ".." in name.split("/"):
        logger.warning("resolve_model: path traversal component rejected: %s", name)
        raise ValueError(f"Path not allowed: {name}. Path traversal (..) is forbidden.")
    if os.path.isabs(name):
        _check_path_allowed(name)
        return name
    if _is_path_like(name):
        _check_path_allowed(name)
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
    profiles = list_profiles()
    if name in profiles:
        return profiles[name]
    # Issue #256: reverse-lookup by hf_path so that raw HF paths like
    # "mlx-community/diffusiongemma-26B-A4B-it-4bit" resolve to the
    # same profile as the alias name "diffusion-gemma-26b-4bit".
    # This is critical for engine dispatch: the server receives an HF
    # path and needs to know the modality to route to DiffusionEngine.
    for profile in profiles.values():
        if profile.hf_path and profile.hf_path == name:
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
