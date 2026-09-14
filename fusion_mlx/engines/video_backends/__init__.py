# SPDX-License-Identifier: Apache-2.0
# Video backend registry. VideoGenEngine resolves a concrete backend here.
# LTX2Backend (LTX-2 + LTX-2.3) and Wan2Backend run on vendored pure-MLX ports
# (fusion_mlx.video.ltx2 / wan2, Phases 4/5); ltx_video_legacy is a direct
# pure-MLX impl. No mlx-video runtime dependency remains.
#
# A4: backend classes are lazily imported via __getattr__ (PEP 562) so that
# importing this package (e.g. for validate_params/constraints_for) does not
# eagerly parse all 12 backend modules. The BACKENDS dict is built on first
# access. On lite profile (video disabled), backends are never imported.

from __future__ import annotations

from typing import Any

from .base import (
    VideoBackend,
    VideoConstraints,
    VideoGenParams,
    validate_params,
)

_LAZY_BACKENDS: dict[str, tuple[str, str]] = {
    "LTX2_5Backend": (".ltx2_5", "LTX2_5Backend"),
    "LTX2Backend": (".ltx2", "LTX2Backend"),
    "CosmosBackend": (".cosmos", "CosmosBackend"),
    "SVDBackend": (".svd", "SVDBackend"),
    "Wan2Backend": (".wan2", "Wan2Backend"),
    "SkyReelsBackend": (".skyreels", "SkyReelsBackend"),
    "LegacyLTXBackend": (".ltx_video_legacy", "LegacyLTXBackend"),
    "CogVideoBackend": (".cogvideox", "CogVideoBackend"),
    "HunyuanVideoBackend": (".hunyuanvideo", "HunyuanVideoBackend"),
    "OpenSoraBackend": (".opensora", "OpenSoraBackend"),
    "UniWorldBackend": (".uniworld", "UniWorldBackend"),
    "MiniMaxH3Backend": (".minimax_h3", "MiniMaxH3Backend"),
}

_REGISTRY_KEYS: dict[str, str] = {
    "ltx2_5": "LTX2_5Backend",
    "ltx2": "LTX2Backend",
    "cosmos": "CosmosBackend",
    "svd": "SVDBackend",
    "wan2": "Wan2Backend",
    "skyreels": "SkyReelsBackend",
    "ltx_video_legacy": "LegacyLTXBackend",
    "cogvideo": "CogVideoBackend",
    "hunyuanvideo": "HunyuanVideoBackend",
    "opensora": "OpenSoraBackend",
    "uniworld": "UniWorldBackend",
    "minimax_h3": "MiniMaxH3Backend",
}

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

_BACKENDS_CACHE: dict[str, type[VideoBackend]] | None = None


def _get_backends() -> dict[str, type[VideoBackend]]:
    global _BACKENDS_CACHE
    if _BACKENDS_CACHE is not None:
        return _BACKENDS_CACHE
    import importlib

    _BACKENDS_CACHE = {}
    for key, cls_name in _REGISTRY_KEYS.items():
        submod, attr = _LAZY_BACKENDS[cls_name]
        mod = importlib.import_module(submod, __name__)
        _BACKENDS_CACHE[key] = getattr(mod, attr)
    return _BACKENDS_CACHE


def __getattr__(name: str):
    if name == "BACKENDS":
        return _get_backends()
    entry = _LAZY_BACKENDS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    submod, attr = entry
    import importlib

    mod = importlib.import_module(submod, __name__)
    val = getattr(mod, attr)
    globals()[name] = val
    return val


def __dir__() -> list[str]:
    return sorted(
        list(_LAZY_BACKENDS.keys())
        + [
            "BACKENDS",
            "VideoBackend",
            "VideoConstraints",
            "VideoGenParams",
            "validate_params",
            "resolve_backend",
            "constraints_for",
        ]
    )


def resolve_backend(
    model_name: str,
    *,
    explicit: str | None = None,
    **kwargs: Any,
) -> VideoBackend:
    if explicit:
        key = _ALIASES.get(explicit.lower(), explicit.lower())
        backends = _get_backends()
        cls = backends.get(key)
        if cls is None:
            raise ValueError(f"unknown video backend: {explicit}")
        return cls(model_name, **kwargs)

    backends = _get_backends()
    for cls in backends.values():
        if cls.detect(model_name):
            return cls(model_name, **kwargs)

    return __getattr__("LTX2Backend")(model_name, **kwargs)


def constraints_for(
    model_name: str, *, explicit: str | None = None
) -> VideoConstraints:
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
