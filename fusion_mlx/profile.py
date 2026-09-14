# SPDX-License-Identifier: Apache-2.0
"""Server profile gate (R-7, simple-final §4).

Three presets control which modalities, routes, and engines are
available at boot — converging the commercial failure surface.

    lite      — LLM text only (default for minimal deployments)
    standard  — LLM + embeddings + audio + ner + rerank + ocr + spec + mcp
    full      — everything (image, video, agent, multitenant)

Profile is parsed once at startup and read-only thereafter. Explicit
CLI flags (--enable-dflash2, --profile full) always override profile
defaults.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

ALL_MODALITIES = frozenset(
    {
        "llm",
        "vlm",
        "embedding",
        "reranker",
        "ner",
        "audio",
        "image",
        "video",
        "agent",
        "multitenant",
        "mcp",
        "bench",
        "tools",
        "ocr",
    }
)

_LITE_DISABLED = frozenset(
    {
        "vlm",
        "embedding",
        "reranker",
        "ner",
        "audio",
        "image",
        "video",
        "agent",
        "multitenant",
        "mcp",
        "bench",
        "tools",
        "ocr",
    }
)

_STANDARD_DISABLED = frozenset({"image", "video", "agent", "multitenant"})

_FULL_DISABLED: frozenset[str] = frozenset()

_PRESET_DISABLED: dict[str, frozenset[str]] = {
    "lite": _LITE_DISABLED,
    "standard": _STANDARD_DISABLED,
    "full": _FULL_DISABLED,
    # D2.5: turbo = full modalities + aggressive cache/spec/quant params.
    # is_turbo flag signals the serve path to default-enable DFlash2 +
    # radix prefix cache + KV turboquant. No yaml double-track — the
    # profile name drives param composition via ServerProfile.is_turbo.
    "turbo": _FULL_DISABLED,
}

_PRESET_SPEC_DEFAULT: dict[str, bool] = {
    "lite": False,
    "standard": True,
    "full": True,
    "turbo": True,
}


@dataclass
class ServerProfile:
    name: str = "standard"
    disabled: set[str] = field(default_factory=set)
    api_routes: set[str] | None = None

    @property
    def is_turbo(self) -> bool:
        # D2.5: turbo profile = aggressive cache (radix) + spec (DFlash2)
        # + KV quant (turboquant) defaults. Serve path reads this to
        # compose the param layer above the base profile.
        return self.name == "turbo"

    def engine_allowed(self, modality: str) -> bool:
        if modality == "llm":
            return True
        return modality not in self.disabled

    def routes_enabled(self) -> set[str] | None:
        return self.api_routes

    def spec_decode_default(self) -> bool:
        return _PRESET_SPEC_DEFAULT.get(self.name, True)

    def summary(self) -> str:
        enabled = sorted(ALL_MODALITIES - self.disabled)
        return (
            f"profile={self.name} enabled_modalities=[{','.join(enabled)}] "
            f"disabled=[{','.join(sorted(self.disabled))}] "
            f"spec_decode_default={self.spec_decode_default()} "
            f"is_turbo={self.is_turbo}"
        )


def _from_preset(name: str) -> ServerProfile:
    disabled = set(_PRESET_DISABLED.get(name, _STANDARD_DISABLED))
    return ServerProfile(name=name, disabled=disabled)


def suggest_profile_from_hardware() -> str | None:
    """Auto-suggest a profile based on detected RAM and chip tier.

    Combines RAM thresholds (from doctor _RAM_PROFILE_MAP) with chip tier
    classification. Returns None on non-macOS or detection failure.

    Thresholds:
      < 16 GB → lite  (too constrained for standard modalities)
      16-23 GB → lite (27B-4bit=15.7G weights, tight headroom)
      >= 24 GB → standard
    Chip tier overrides: base M-chip (no Pro/Max/Ultra) with < 32 GB → lite.
    """
    try:
        from .hardware.memory import detect_ram_bytes

        ram_bytes = detect_ram_bytes()
        ram_gb = ram_bytes / (1024**3)
    except Exception:
        logger.debug("RAM detection failed, skipping hardware profile suggestion")
        return None

    from .hardware.apple import detect_chip_tier

    chip_tier = detect_chip_tier()
    logger.info("hardware auto-detect: ram_gb=%.1f chip_tier=%s", ram_gb, chip_tier)

    if ram_gb < 24:
        return "lite"
    if chip_tier == "lite" and ram_gb < 32:
        return "lite"
    return "standard"


def resolve_profile(
    explicit: str | None = None,
    settings_profile: str | None = None,
    disabled_modules: list[str] | None = None,
) -> ServerProfile:
    if explicit:
        name = explicit
        logger.info("profile from explicit flag: %s", name)
    elif settings_profile:
        name = settings_profile
        logger.info("profile from settings.json: %s", name)
    else:
        hw = suggest_profile_from_hardware()
        name = hw or "standard"
        if hw:
            logger.info("profile auto-selected from hardware: %s", name)
        else:
            logger.info("profile default: standard (hardware detection unavailable)")
    if name not in _PRESET_DISABLED:
        logger.warning("unknown profile '%s', falling back to 'standard'", name)
        name = "standard"
    profile = _from_preset(name)
    if disabled_modules:
        for mod in disabled_modules:
            if mod not in ALL_MODALITIES:
                logger.warning(
                    "disabled_modules entry '%s' not a known modality, ignoring", mod
                )
                continue
            profile.disabled.add(mod)
            logger.info("profile: modality '%s' disabled via disabled_modules", mod)
    logger.info("profile resolved: %s", profile.summary())
    return profile


def profile_from_config(config) -> ServerProfile:
    explicit = getattr(config, "profile", None)
    disabled_modules = getattr(config, "disabled_modules", None)
    return resolve_profile(explicit=explicit, disabled_modules=disabled_modules)
