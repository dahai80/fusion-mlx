# SPDX-License-Identifier: Apache-2.0
# Video backend registry. VideoGenEngine resolves a concrete backend here.
# LTX2Backend (LTX-2 + LTX-2.3) and Wan2Backend run on vendored pure-MLX ports
# (fusion_mlx.video.ltx2 / wan2, Phases 4/5); ltx_video_legacy is a direct
# pure-MLX impl. No mlx-video runtime dependency remains.

from __future__ import annotations

from typing import Any

from .base import (
    VideoBackend,
    VideoConstraints,
    VideoGenParams,
    validate_params,
)
from .cogvideox import CogVideoBackend
from .cosmos import CosmosBackend
from .hunyuanvideo import HunyuanVideoBackend
from .ltx2 import LTX2Backend
from .ltx2_5 import LTX2_5Backend
from .ltx_video_legacy import LegacyLTXBackend
from .minimax_h3 import MiniMaxH3Backend
from .opensora import OpenSoraBackend
from .skyreels import SkyReelsBackend
from .svd import SVDBackend
from .uniworld import UniWorldBackend
from .wan2 import Wan2Backend

BACKENDS: dict[str, type[VideoBackend]] = {
    "ltx2_5": LTX2_5Backend,
    "ltx2": LTX2Backend,
    "cosmos": CosmosBackend,
    "svd": SVDBackend,
    "wan2": Wan2Backend,
    "skyreels": SkyReelsBackend,
    "ltx_video_legacy": LegacyLTXBackend,
    "cogvideo": CogVideoBackend,
    "hunyuanvideo": HunyuanVideoBackend,
    "opensora": OpenSoraBackend,
    "uniworld": UniWorldBackend,
    "minimax_h3": MiniMaxH3Backend,
}

# Stable name aliases -> canonical registry key.
_ALIASES: dict[str, str] = {
    "ltx-2": "ltx2",
    "ltx_2": "ltx2",
    "ltx-2.3": "ltx2",
    "ltx2.3": "ltx2",
    "ltx-2.5": "ltx2_5",
    "ltx_2.5": "ltx2_5",
    "ltx2.5": "ltx2_5",
    "ltx-2.5-distilled": "ltx2_5",
    "cosmos": "cosmos",
    "cosmos-1.0": "cosmos",
    "cosmos-predict2": "cosmos",
    "predict2": "cosmos",
    "video2world": "cosmos",
    "hunyuanvideo": "hunyuanvideo",
    "hunyuan-video": "hunyuanvideo",
    "hunyuan_video": "hunyuanvideo",
    "svd": "svd",
    "stable-video-diffusion": "svd",
    "svd-xt": "svd",
    "img2vid-xt": "svd",
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
    "skyreels": "skyreels",
    "skyreels-v3": "skyreels",
    "r2v": "skyreels",
    "v2v": "skyreels",
    "a2v": "skyreels",
    "opensora": "opensora",
    "open-sora": "opensora",
    "open_sora": "opensora",
    "opensora-v2": "opensora",
    "vace": "wan2",
    "wan-vace": "wan2",
    "wan2.1-vace": "wan2",
    "uniworld": "uniworld",
    "uniworld-v1": "uniworld",
    "univa": "uniworld",
    "minimax-h3": "minimax_h3",
    "minimax_h3": "minimax_h3",
    "h3": "minimax_h3",
    "h3-fl2va": "minimax_h3",
    "h3-ref2va": "minimax_h3",
    "fl2va": "minimax_h3",
    "ref2va": "minimax_h3",
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
    "LTX2_5Backend",
    "CosmosBackend",
    "HunyuanVideoBackend",
    "SVDBackend",
    "Wan2Backend",
    "SkyReelsBackend",
    "LegacyLTXBackend",
    "CogVideoBackend",
    "OpenSoraBackend",
    "UniWorldBackend",
    "MiniMaxH3Backend",
]
