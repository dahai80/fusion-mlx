"""fusion_mlx.public_api — 公开稳定 API 入口。

下游（fusion-comfyui 等）应统一用 `from fusion_mlx.public_api import X`，
而非深入内部子模块路径（`fusion_mlx.engines.video` / `fusion_mlx.model_registry`
/ `fusion_mlx.config` / `fusion_mlx.video.*.pipeline`）。后者属内部实现，
重构可能变，不保证稳定。

注意：本模块名 `public_api`，与 `fusion_mlx.api`（server API models 包，
OpenAI/Anthropic pydantic models + routes）不同，勿混。

本模块只 re-export 已被下游实际依赖、承诺稳定对外公开的符号：
- 引擎类（TTSEngine/ImageGenEngine/VideoGenEngine/STTEngine/STSEngine/EmbeddingEngine/RerankerEngine/VLMBatchedEngine）
- 引擎池（EnginePool，sequential offload 核心依赖）
- 配置与注册（get_config/get_registry/list_available_models/ServerConfig/MemoryConfig/MemoryTier）
- 视频 pipeline（LipsyncPipelineMLX/MuseTalkPipeline/PuLIDPipeline，下游已依赖故显式提升为公开）
- 服务入口（Server/create_app/__version__）
"""

import logging
import warnings

from ._version import __version__
from .config import MemoryConfig, MemoryTier, ServerConfig, get_config
from .engines import (
    EmbeddingEngine,
    ImageGenEngine,
    RerankerEngine,
    STSEngine,
    STTEngine,
    TTSEngine,
    VideoGenEngine,
)
from .engines.vlm import VLMBatchedEngine
from .image.sr.config import RealESRGANConfig
from .image.sr.generate import super_resolve
from .image.sr.rrdb import RRDBNet
from .model_registry import get_registry, list_available_models
from .pool.engine_pool import EnginePool
from .profile import ServerProfile, profile_from_config, resolve_profile
from .server import Server, create_app
from .video.latentsync_mlx.pipeline import LipsyncPipelineMLX
from .video.musetalk_mlx import MuseTalkPipeline
from .video.pulid_mlx.pipeline import PuLIDPipeline

logger = logging.getLogger(__name__)

__all__ = [
    "__version__",
    "Server",
    "create_app",
    "get_config",
    "get_registry",
    "list_available_models",
    "ServerConfig",
    "MemoryConfig",
    "MemoryTier",
    "ServerProfile",
    "profile_from_config",
    "resolve_profile",
    "EnginePool",
    "TTSEngine",
    "STTEngine",
    "STSEngine",
    "EmbeddingEngine",
    "RerankerEngine",
    "ImageGenEngine",
    "VideoGenEngine",
    "VLMBatchedEngine",
    "LipsyncPipelineMLX",
    "MuseTalkPipeline",
    "PuLIDPipeline",
    "RealESRGANConfig",
    "RRDBNet",
    "super_resolve",
]


def validate_public_api() -> list[str]:
    """G5 (#0910 audit): verify every symbol in ``__all__`` is importable.

    Returns list of missing symbols (empty = all OK). Called by ``doctor``
    to catch broken re-exports before they reach downstream consumers.
    """
    missing: list[str] = []
    import sys

    mod = sys.modules[__name__]
    for name in __all__:
        if name not in mod.__dict__:
            missing.append(name)
    if missing:
        logger.error(
            "G5: public_api __all__ has %d broken exports: %s", len(missing), missing
        )
    return missing


def __getattr__(name: str):
    # G5 (#0910 audit): warn when downstream imports a symbol not in
    # ``__all__`` — it may work today but is not a stable public symbol and
    # can disappear without a deprecation cycle.
    _public = set(__all__)
    import sys

    mod = sys.modules[__name__]
    # Check __dict__ directly to avoid re-triggering __getattr__ (recursion).
    if name in mod.__dict__:
        val = mod.__dict__[name]
        if name not in _public and not name.startswith("_"):
            warnings.warn(
                f"fusion_mlx.public_api: '{name}' is not in __all__ and is not a "
                f"stable public symbol. It may be removed without a deprecation "
                f"cycle. Use only symbols in public_api.__all__.",
                DeprecationWarning,
                stacklevel=2,
            )
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
