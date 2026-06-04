# SPDX-License-Identifier: Apache-2.0
"""
Request router — dispatches requests to the appropriate engine.

Routing rules:
- Text-only → BatchedEngine (LLM)
- Image/video content → VLMBatchedEngine
- Audio input (STT) → STTEngine
- Text-to-speech → TTSEngine
- Speech-to-speech → STSEngine
- Image generation → ImageGenEngine
- Embedding → EmbeddingEngine
- Reranking → RerankerEngine
- Large uncached context → CloudRouter (optional)
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RequestRouter:
    """Routes incoming requests to the appropriate engine based on content type."""

    def __init__(
        self,
        llm_engine: Any = None,
        vlm_engine: Any = None,
        stt_engine: Any = None,
        tts_engine: Any = None,
        sts_engine: Any = None,
        image_gen_engine: Any = None,
        embedding_engine: Any = None,
        reranker_engine: Any = None,
        cloud_router: Any = None,
    ):
        self.llm_engine = llm_engine
        self.vlm_engine = vlm_engine
        self.stt_engine = stt_engine
        self.tts_engine = tts_engine
        self.sts_engine = sts_engine
        self.image_gen_engine = image_gen_engine
        self.embedding_engine = embedding_engine
        self.reranker_engine = reranker_engine
        self.cloud_router = cloud_router

    def _has_images(self, messages: list[dict[str, Any]]) -> bool:
        """Check if any message contains image content."""
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return True
        return False

    def _has_audio_input(self, request_data: dict[str, Any]) -> bool:
        """Check if request includes audio file input (STT)."""
        return bool(request_data.get("audio") or request_data.get("audio_path"))

    def _is_tts_request(self, request_data: dict[str, Any]) -> bool:
        """Check if request is for text-to-speech."""
        return request_data.get("task") == "tts" or bool(request_data.get("synthesize"))

    def _is_sts_request(self, request_data: dict[str, Any]) -> bool:
        """Check if request is for speech-to-speech."""
        return request_data.get("task") == "sts" or bool(request_data.get("audio_process"))

    def _is_image_gen(self, request_data: dict[str, Any]) -> bool:
        """Check if request is for image generation."""
        return request_data.get("task") == "image_gen" or request_data.get("generate_image")

    def _is_embedding(self, request_data: dict[str, Any]) -> bool:
        """Check if request is for embedding."""
        return request_data.get("task") == "embedding" or bool(request_data.get("embed"))

    def _is_rerank(self, request_data: dict[str, Any]) -> bool:
        """Check if request is for reranking."""
        return request_data.get("task") == "rerank" or bool(request_data.get("rerank"))

    def select_engine(self, messages: list[dict], request_data: dict[str, Any]) -> Any:
        """Select the appropriate engine for the request.

        Returns (engine, engine_type) tuple.
        """
        # Explicit task-based routing first
        if self._is_embedding(request_data) and self.embedding_engine:
            return (self.embedding_engine, "embedding")
        if self._is_rerank(request_data) and self.reranker_engine:
            return (self.reranker_engine, "reranker")
        if self._is_image_gen(request_data) and self.image_gen_engine:
            return (self.image_gen_engine, "image_gen")
        if self._is_tts_request(request_data) and self.tts_engine:
            return (self.tts_engine, "tts")
        if self._is_sts_request(request_data) and self.sts_engine:
            return (self.sts_engine, "sts")
        if self._has_audio_input(request_data) and self.stt_engine:
            return (self.stt_engine, "stt")

        # Content-based routing for chat/completion
        if self._has_images(messages):
            if self.vlm_engine:
                return (self.vlm_engine, "vlm")
            logger.warning("VLM requested but no VLM engine available, falling back to LLM")
        if self.llm_engine:
            return (self.llm_engine, "llm")

        raise RuntimeError("No suitable engine available for request")

    async def route_chat(
        self,
        messages: list[dict],
        request_data: dict[str, Any],
        **kwargs,
    ) -> Any:
        """Route a chat request to the appropriate engine."""
        engine, etype = self.select_engine(messages, request_data)

        # Cloud routing check for large uncached context
        if (
            self.cloud_router
            and etype == "llm"
            and getattr(engine, "prefix_cache_enabled", False)
        ):
            new_tokens = getattr(engine, "count_chat_tokens", lambda m, **_: 0)(messages)
            if self.cloud_router.should_route_to_cloud(new_tokens):
                logger.info(f"Routing {new_tokens}-token request to cloud ({self.cloud_router.cloud_model})")
                return await self.cloud_router.completion(messages, **kwargs)

        return await engine.chat(messages, **kwargs)

    async def route_stream_chat(
        self,
        messages: list[dict],
        request_data: dict[str, Any],
        **kwargs,
    ) -> Any:
        """Route a streaming chat request."""
        engine, etype = self.select_engine(messages, request_data)
        return engine.stream_chat(messages, **kwargs)

    def get_stats(self) -> dict[str, Any]:
        """Return router status."""
        return {
            "has_llm": self.llm_engine is not None,
            "has_vlm": self.vlm_engine is not None,
            "has_stt": self.stt_engine is not None,
            "has_tts": self.tts_engine is not None,
            "has_sts": self.sts_engine is not None,
            "has_image_gen": self.image_gen_engine is not None,
            "has_embedding": self.embedding_engine is not None,
            "has_reranker": self.reranker_engine is not None,
            "has_cloud": self.cloud_router is not None,
        }
