"""
Engine abstraction for fusion-mlx inference.

Provides multiple engine implementations:
- BatchedEngine: Continuous batching for multiple concurrent users
- VLMBatchedEngine: Vision-language model engine with image support
- EmbeddingEngine: Batch embedding generation using mlx-embeddings
- RerankerEngine: Document reranking using SequenceClassification models
- DFlashEngine: High-performance speculative decoding with draft models
- ImageGenEngine: Flux 2 image generation (fusion-mlx only)

Also re-exports core engine components for backwards compatibility.
"""

# Re-export from engine_core for backwards compatibility
from ..engine_core import AsyncEngineCore, EngineConfig, EngineCore
from .base import BaseEngine, BaseNonStreamingEngine, GenerationOutput
from .batched import BatchedEngine
from .dflash import DFlashEngine
from .embedding import EmbeddingEngine
from .image_gen import ImageGenEngine
from .reranker import RerankerEngine
from .sts import STSEngine
from .stt import STTEngine
from .tts import TTSEngine
from .vlm import VLMBatchedEngine

__all__ = [
    "BaseEngine",
    "BaseNonStreamingEngine",
    "GenerationOutput",
    "BatchedEngine",
    "DFlashEngine",
    "VLMBatchedEngine",
    "EmbeddingEngine",
    "RerankerEngine",
    "STTEngine",
    "TTSEngine",
    "STSEngine",
    "ImageGenEngine",
    # Core engine components
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
]
