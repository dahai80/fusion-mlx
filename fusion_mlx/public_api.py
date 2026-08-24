"""fusion_mlx.public_api — 公开稳定 API 入口。

下游（fusion-comfyui 等）应统一用 `from fusion_mlx.public_api import X`，
而非深入内部子模块路径（`fusion_mlx.engines.video` / `fusion_mlx.model_registry`
/ `fusion_mlx.config` / `fusion_mlx.video.*.pipeline`）。后者属内部实现，
重构可能变，不保证稳定。

注意：本模块名 `public_api`，与 `fusion_mlx.api`（server API models 包，
OpenAI/Anthropic pydantic models + routes）不同，勿混。

本模块只 re-export 已被下游实际依赖、承诺稳定对外公开的符号：
- 引擎类（TTSEngine/ImageGenEngine/VideoGenEngine/STTEngine/STSEngine/EmbeddingEngine/RerankerEngine）
- 配置与注册（get_config/get_registry/ServerConfig/MemoryConfig/MemoryTier）
- 视频 pipeline（LipsyncPipelineMLX/PuLIDPipeline，下游已依赖故显式提升为公开）
- 服务入口（Server/create_app/__version__）
"""

import logging

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
from .model_registry import get_registry
from .server import Server, create_app
from .video.latentsync_mlx.pipeline import LipsyncPipelineMLX
from .video.pulid_mlx.pipeline import PuLIDPipeline

logger = logging.getLogger(__name__)

__all__ = [
    "__version__",
    "Server",
    "create_app",
    "get_config",
    "get_registry",
    "ServerConfig",
    "MemoryConfig",
    "MemoryTier",
    "TTSEngine",
    "STTEngine",
    "STSEngine",
    "EmbeddingEngine",
    "RerankerEngine",
    "ImageGenEngine",
    "VideoGenEngine",
    "LipsyncPipelineMLX",
    "PuLIDPipeline",
]
