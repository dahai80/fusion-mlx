# SPDX-License-Identifier: Apache-2.0
"""Base engine interface for fusion-mlx inference."""

import asyncio
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

import mlx.core as mx

from ..engine_core import get_executor

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    text: str
    tokens: List[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: Optional[str] = "stop"
    new_text: str = ""
    finished: bool = True
    tool_calls: Optional[List[Dict[str, Any]]] = None
    cached_tokens: int = 0


class BaseEngine(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        pass

    @property
    def is_mllm(self) -> bool:
        pass

    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def stop(self) -> None:
        pass

    @abstractmethod
    async def generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.7,
        top_p: float = 0.9, top_k: int = 0, min_p: float = 0.0,
        repetition_penalty: float = 1.0, presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None, **kwargs,
    ) -> GenerationOutput:
        pass

    @abstractmethod
    async def stream_generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.7,
        top_p: float = 0.9, top_k: int = 0, min_p: float = 0.0,
        repetition_penalty: float = 1.0, presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None, **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        pass

    @abstractmethod
    async def chat(
        self, messages: List[Dict[str, Any]], max_tokens: int = 256,
        temperature: float = 0.7, top_p: float = 0.9, top_k: int = 0,
        min_p: float = 0.0, repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0, tools: Optional[List[dict]] = None, **kwargs,
    ) -> GenerationOutput:
        pass

    @abstractmethod
    async def stream_chat(
        self, messages: List[Dict[str, Any]], max_tokens: int = 256,
        temperature: float = 0.7, top_p: float = 0.9, top_k: int = 0,
        min_p: float = 0.0, repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0, tools: Optional[List[dict]] = None, **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        pass

    @property
    @abstractmethod
    def model_type(self) -> Optional[str]:
        pass

    @property
    def grammar_compiler(self):
        return None

    @property
    def prefix_cache_enabled(self) -> bool:
        return False

    def has_active_requests(self) -> bool:
        return False

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        pass


class BaseNonStreamingEngine(ABC):
    def __init__(self):
        self._active_count = 0
        self._active_lock = threading.Lock()
        self._activities: Dict[str, Dict[str, Any]] = {}

    def has_active_requests(self) -> bool:
        with self._active_lock:
            return self._active_count > 0

    _ACTIVITY_RESERVED_KEYS = {"request_id", "kind", "detail", "started_at", "last_activity_at", "total_items"}

    def _sanitize_activity_metadata(self, metadata: Dict[str, Any] | None) -> Dict[str, Any]:
        if not metadata:
            return {}
        return {k: v for k, v in metadata.items() if k not in self._ACTIVITY_RESERVED_KEYS}

    def _begin_activity(self, kind: str, detail: str | None = None, total_items: int | None = None, metadata: Dict[str, Any] | None = None) -> str:
        activity_id = str(uuid.uuid4())
        now = time.monotonic()
        with self._active_lock:
            self._active_count += 1
            activity = {"request_id": activity_id, "kind": kind, "detail": detail or kind, "started_at": now, "last_activity_at": now, "total_items": total_items}
            activity.update(self._sanitize_activity_metadata(metadata))
            self._activities[activity_id] = activity
        return activity_id

    def _update_activity(self, activity_id: str, **updates: Any) -> None:
        with self._active_lock:
            activity = self._activities.get(activity_id)
            if activity is None:
                return
            activity.update(self._sanitize_activity_metadata(updates))
            activity["last_activity_at"] = time.monotonic()

    def _end_activity(self, activity_id: str) -> None:
        with self._active_lock:
            removed = self._activities.pop(activity_id, None)
            if removed is None:
                raise RuntimeError(f"Activity {activity_id} ended more than once or was never started")
            self._active_count -= 1
            if self._active_count < 0:
                raise RuntimeError("Active request count became negative")

    async def _finish_activity(self, activity_id: str) -> None:
        self._end_activity(activity_id)
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(get_executor("llm"), lambda: (mx.synchronize(), mx.clear_cache())), timeout=5.0)

    def get_activity_snapshot(self) -> Dict[str, Any]:
        now = time.monotonic()
        with self._active_lock:
            activities = []
            for activity in self._activities.values():
                item = dict(activity)
                started_at = item.pop("started_at", None)
                last_activity_at = item.pop("last_activity_at", None)
                item["elapsed_seconds"] = max(0.0, now - started_at) if started_at is not None else None
                item["last_activity_age_seconds"] = max(0.0, now - last_activity_at) if last_activity_at is not None else None
                activities.append(item)
            return {"active_requests": self._active_count, "activities": activities}

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def stop(self) -> None:
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass
