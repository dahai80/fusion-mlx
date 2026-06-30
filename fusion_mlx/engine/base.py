# SPDX-License-Identifier: Apache-2.0
"""
Base engine interface for fusion-mlx inference.
"""

import asyncio
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

import mlx.core as mx

from ..engine_core import get_mlx_executor

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """
    Output from generation.

    Compatible with both simple and batched engines.
    """

    text: str
    tokens: List[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: Optional[str] = "stop"
    # For streaming
    new_text: str = ""
    finished: bool = True
    # For tool calling (Harmony and other models)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    # Prefix cache stats
    cached_tokens: int = 0
    # Optional engine-native throughput stats
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    # Diffusion-specific fields
    diffusion_canvas_tokens: int = 0
    diffusion_denoising_steps: int = 0
    diffusion_work_tokens: int = 0
    diffusion_canvas_tps: float = 0.0
    diffusion_work_tps: float = 0.0


class BaseEngine(ABC):
    """
    Abstract base class for inference engines.

    Both SimpleEngine and BatchedEngine implement this interface,
    allowing the server to use either without code changes.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass

    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> GenerationOutput:
        """Generate a complete response (non-streaming)."""
        pass

    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Stream generation token by token."""
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: Optional[List[dict]] = None,
        **kwargs,
    ) -> GenerationOutput:
        """Chat completion (non-streaming)."""
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: Optional[List[dict]] = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Stream chat completion token by token."""
        pass

    @property
    @abstractmethod
    def model_type(self) -> Optional[str]:
        """Get the model type from config.json (e.g., 'llama', 'qwen2')."""
        pass

    @property
    def grammar_compiler(self):
        """Return the grammar compiler for this engine, or None."""
        return None

    @property
    def prefix_cache_enabled(self) -> bool:
        """Whether automatic prefix caching is active on this engine."""
        return False

    def has_active_requests(self) -> bool:
        """Check if the engine has active in-flight requests."""
        return False

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        pass

    @abstractmethod
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        pass


class BaseNonStreamingEngine(ABC):
    """Base class for non-streaming engines (embedding, reranker).

    These engines compute outputs in a single forward pass and don't
    support streaming or chat completion interfaces.
    """

    def __init__(self):
        self._active_count = 0
        self._active_lock = threading.Lock()
        self._activities: Dict[str, Dict[str, Any]] = {}

    def has_active_requests(self) -> bool:
        """Check if the engine has active in-flight requests."""
        with self._active_lock:
            return self._active_count > 0

    def _reset_activity_tracking(self) -> None:
        """Clear the in-flight activity counter + records on engine teardown."""
        with self._active_lock:
            self._active_count = 0
            self._activities.clear()

    _ACTIVITY_RESERVED_KEYS = {
        "request_id", "kind", "detail",
        "started_at", "last_activity_at", "total_items",
    }

    def _sanitize_activity_metadata(
        self, metadata: Dict[str, Any] | None
    ) -> Dict[str, Any]:
        """Drop reserved activity keys from caller-provided metadata."""
        if not metadata:
            return {}
        return {
            key: value
            for key, value in metadata.items()
            if key not in self._ACTIVITY_RESERVED_KEYS
        }

    def _begin_activity(
        self,
        kind: str,
        detail: str | None = None,
        total_items: int | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        """Track a non-streaming operation for admin visibility."""
        activity_id = str(uuid.uuid4())
        now = time.monotonic()
        with self._active_lock:
            self._active_count += 1
            activity = {
                "request_id": activity_id,
                "kind": kind,
                "detail": detail or kind,
                "started_at": now,
                "last_activity_at": now,
                "total_items": total_items,
            }
            activity.update(self._sanitize_activity_metadata(metadata))
            self._activities[activity_id] = activity
        return activity_id

    def _update_activity(self, activity_id: str, **updates: Any) -> None:
        """Update tracked non-streaming operation metadata."""
        with self._active_lock:
            activity = self._activities.get(activity_id)
            if activity is None:
                return
            activity.update(self._sanitize_activity_metadata(updates))
            activity["last_activity_at"] = time.monotonic()

    def _end_activity(self, activity_id: str) -> None:
        """End an activity."""
        with self._active_lock:
            removed = self._activities.pop(activity_id, None)
            if removed is None:
                raise RuntimeError(
                    f"Activity {activity_id} ended more than once or was never started"
                )
            self._active_count -= 1
            if self._active_count < 0:
                raise RuntimeError("Active request count became negative")

    async def _finish_activity(self, activity_id: str) -> None:
        """End an activity and clear the Metal buffer pool."""
        self._end_activity(activity_id)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(),
            lambda: (mx.synchronize(), mx.clear_cache()),
        )

    def get_activity_snapshot(self) -> Dict[str, Any]:
        """Return active non-streaming operations for admin display."""
        now = time.monotonic()
        with self._active_lock:
            activities = []
            for activity in self._activities.values():
                item = dict(activity)
                started_at = item.pop("started_at", None)
                last_activity_at = item.pop("last_activity_at", None)
                item["elapsed_seconds"] = (
                    max(0.0, now - started_at) if started_at is not None else None
                )
                item["last_activity_age_seconds"] = (
                    max(0.0, now - last_activity_at)
                    if last_activity_at is not None else None
                )
                activities.append(item)
            return {
                "active_requests": self._active_count,
                "activities": activities,
            }

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        pass
