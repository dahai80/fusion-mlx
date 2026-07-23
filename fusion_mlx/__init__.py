"""fusion-mlx — unified local model management for Apple Silicon.

Merges the best of fusion-mlx (long-context, memory control, multi-model
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
from .config import MemoryConfig, MemoryTier, ServerConfig
from .dispatch import CloudRouter, RequestRouter
from .engine_core import AsyncEngineCore, EngineConfig
from .engines import (
    BaseEngine,
    BaseNonStreamingEngine,
    BatchedEngine,
    EmbeddingEngine,
    GenerationOutput,
    ImageGenEngine,
    RerankerEngine,
    STSEngine,
    STTEngine,
    TTSEngine,
    VLMBatchedEngine,
)
from .pool import EnginePool, MemoryProfile, ModelDiscovery, ProcessMemoryEnforcer
from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .scheduler import Scheduler, SchedulerConfig, SchedulerOutput, SchedulingPolicy
from .server import Server, create_app

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
