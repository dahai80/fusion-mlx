"""Engine layer — unified inference engines.

Provides BaseEngine, BatchedEngine (LLM), VLMBatchedEngine (vision),
EmbeddingEngine, RerankerEngine, NEREngine, STTEngine, TTSEngine, STSEngine,
and ImageGenEngine (Flux 2 image generation).

A4: engine classes are lazily imported via __getattr__ (PEP 562) so that
``import fusion_mlx`` does not eagerly parse all engine modules + video
backends. Each class is loaded on first access (``from .engines import X``
or ``fusion_mlx.engines.X``). On lite profile, unused engine modules are
never imported — saving parse time and module-level side effects.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "BaseEngine",
    "BaseNonStreamingEngine",
    "GenerationOutput",
    "BatchedEngine",
    "VLMBatchedEngine",
    "EmbeddingEngine",
    "NEREngine",
    "RerankerEngine",
    "STTEngine",
    "TTSEngine",
    "STSEngine",
    "ImageGenEngine",
    "VideoGenEngine",
]

_LAZY_MAP: dict[str, tuple[str, str]] = {
    "BaseEngine": (".base", "BaseEngine"),
    "BaseNonStreamingEngine": (".base", "BaseNonStreamingEngine"),
    "GenerationOutput": (".base", "GenerationOutput"),
    "BatchedEngine": (".batched", "BatchedEngine"),
    "VLMBatchedEngine": (".vlm", "VLMBatchedEngine"),
    "EmbeddingEngine": (".embedding", "EmbeddingEngine"),
    "NEREngine": (".ner", "NEREngine"),
    "RerankerEngine": (".reranker", "RerankerEngine"),
    "STTEngine": (".stt", "STTEngine"),
    "TTSEngine": (".tts", "TTSEngine"),
    "STSEngine": (".sts", "STSEngine"),
    "ImageGenEngine": (".image_gen", "ImageGenEngine"),
    "VideoGenEngine": (".video", "VideoGenEngine"),
}


def __getattr__(name: str):
    entry = _LAZY_MAP.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    submod, attr = entry
    import importlib

    mod = importlib.import_module(submod, __name__)
    val = getattr(mod, attr)
    globals()[name] = val
    logger.debug("lazy-loaded engine class %s from %s", name, submod)
    return val


def __dir__() -> list[str]:
    return sorted(__all__)
