# SPDX-License-Identifier: Apache-2.0
# Video backend registry. VideoGenEngine resolves a concrete backend here.
# Phase 0 ships only LTX2Backend (LTX-2 + LTX-2.3 via mlx-video). Phases 1/3/4
# register wan2, ltx_video_legacy, cogvideo without touching the engine.

from __future__ import annotations

from typing import Any

from .base import (
    VideoBackend,
    VideoConstraints,
    VideoGenParams,
    validate_params,
)
from .ltx2 import LTX2Backend
from .ltx_video_legacy import LegacyLTXBackend
from .unimplemented import CogVideoBackend
from .wan2 import Wan2Backend

BACKENDS: dict[str, type[VideoBackend]] = {
    "ltx2": LTX2Backend,
    "wan2": Wan2Backend,
    "ltx_video_legacy": LegacyLTXBackend,
    "cogvideo": CogVideoBackend,
}

# Stable name aliases -> canonical registry key.
_ALIASES: dict[str, str] = {
    "ltx-2": "ltx2",
    "ltx_2": "ltx2",
    "ltx-2.3": "ltx2",
    "ltx2.3": "ltx2",
    "wan": "wan2",
    "wan2.1": "wan2",
    "wan2.2": "wan2",
    "wan-2.1": "wan2",
    "wan-2.2": "wan2",
    "ltx-video": "ltx_video_legacy",
    "ltx_video": "ltx_video_legacy",
    "cogvideox": "cogvideo",
    "cog_video": "cogvideo",
    "cogvideo-x": "cogvideo",
}


def resolve_backend(
    model_name: str,
    *,
    explicit: str | None = None,
    **kwargs: Any,
) -> VideoBackend:
    # Explicit hint wins; else auto-detect via per-backend detect(); else
    # fall back to LTX2Backend so Phase 0 preserves the prior single-backend
    # behavior for any text-to-video model.
    if explicit:
        key = _ALIASES.get(explicit.lower(), explicit.lower())
        cls = BACKENDS.get(key)
        if cls is None:
            raise ValueError(f"unknown video backend: {explicit}")
        return cls(model_name, **kwargs)

    for cls in BACKENDS.values():
        if cls.detect(model_name):
            return cls(model_name, **kwargs)

    return LTX2Backend(model_name, **kwargs)


def constraints_for(
    model_name: str, *, explicit: str | None = None
) -> VideoConstraints:
    # Lightweight backend-aware constraint lookup for the API layer. Builds a
    # throwaway backend (no model loading) to read its static constraints.
    return resolve_backend(model_name, explicit=explicit).constraints()


__all__ = [
    "BACKENDS",
    "VideoBackend",
    "VideoConstraints",
    "VideoGenParams",
    "validate_params",
    "resolve_backend",
    "constraints_for",
    "LTX2Backend",
    "Wan2Backend",
    "LegacyLTXBackend",
    "CogVideoBackend",
]
