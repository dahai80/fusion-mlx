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
}

_PRESET_SPEC_DEFAULT: dict[str, bool] = {
    "lite": False,
    "standard": True,
    "full": True,
}


@dataclass
class ServerProfile:
    name: str = "standard"
    disabled: set[str] = field(default_factory=set)
    api_routes: set[str] | None = None

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
            f"spec_decode_default={self.spec_decode_default()}"
        )


def _from_preset(name: str) -> ServerProfile:
    disabled = set(_PRESET_DISABLED.get(name, _STANDARD_DISABLED))
    return ServerProfile(name=name, disabled=disabled)


def resolve_profile(
    explicit: str | None = None,
    settings_profile: str | None = None,
    disabled_modules: list[str] | None = None,
) -> ServerProfile:
    name = explicit or settings_profile or "standard"
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
