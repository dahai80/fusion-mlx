"""fusion-mlx — unified local model management for Apple Silicon.

Merges the best of omlx (long-context, memory control, multi-model
concurrency) with Rapid-MLX (speculative decoding, multi-modal,
cloud routing) into a single codebase.

Key features:
- EnginePool with LRU eviction, pinning, TTL auto-unload
- 4-tier ProcessMemoryEnforcer (safe/balanced/aggressive/custom)
- Paged KV cache with SSD cold layer
- Block-aware prefix cache with copy-on-write
- Speculative decoding (SuffixDecoding, DFlash, MTP, VLM-MTP)
- OpenAI/Anthropic/Responses API compatibility
- Claude Code, OpenClaw, ComfyUI integrations
"""

from ._version import __version__
from .server import Server, create_app
from .scheduler import Scheduler, SchedulerConfig, SchedulingPolicy, SchedulerOutput
from .engine_core import AsyncEngineCore, EngineConfig
from .config import ServerConfig, MemoryConfig, MemoryTier
from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .engines import (
        BaseEngine,
        BaseNonStreamingEngine,
        GenerationOutput,
        BatchedEngine,
        VLMBatchedEngine,
        EmbeddingEngine,
        RerankerEngine,
        STTEngine,
        TTSEngine,
        STSEngine,
        ImageGenEngine,
)
from .pool import EnginePool, ProcessMemoryEnforcer, MemoryProfile, ModelDiscovery
from .router import RequestRouter, CloudRouter

__all__ = [
        # Version
        "__version__",
        # Server
        "Server",
        "create_app",
        "ServerConfig",
        "MemoryConfig",
        "MemoryTier",
        # Scheduler
        "Scheduler",
        "SchedulerConfig",
        "SchedulingPolicy",
        "SchedulerOutput",
        # Engine core
        "AsyncEngineCore",
        "EngineConfig",
        # Request
        "Request",
        "RequestOutput",
        "RequestStatus",
        "SamplingParams",
        # Engines
        "BaseEngine",
        "BaseNonStreamingEngine",
        "GenerationOutput",
        "BatchedEngine",
        "VLMBatchedEngine",
        "EmbeddingEngine",
        "RerankerEngine",
        "STTEngine",
        "TTSEngine",
        "STSEngine",
        "ImageGenEngine",
        # Pool
        "EnginePool",
        "ProcessMemoryEnforcer",
        "MemoryProfile",
        "ModelDiscovery",
        # Router
        "RequestRouter",
        "CloudRouter",
]
