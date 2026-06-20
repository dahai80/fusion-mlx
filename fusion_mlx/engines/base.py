# SPDX-License-Identifier: Apache-2.0
"""Base engine interface for fusion-mlx inference."""

import asyncio
import copy
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EngineStatus(str, Enum):
    """Engine lifecycle states for safe eviction."""
    READY = "ready"
    CLOSING = "closing"
    UNLOADED = "unloaded"


@dataclass
class GenerationOutput:
    text: str
    tokens: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = "stop"
    new_text: str = ""
    finished: bool = True
    tool_calls: list[dict[str, Any]] | None = None
    cached_tokens: int = 0
    kv_state: dict[str, Any] | None = None



def _fallback_parse_tool_calls(gen: GenerationOutput, tokenizer: Any, tools: list[dict]) -> GenerationOutput:
    """Fallback tool call extraction when the scheduler has no parser session."""
    try:
        from ..api.tool_calling import parse_tool_calls
        cleaned, tc_list = parse_tool_calls(gen.text, tokenizer, tools)
        if tc_list:
            tc_dicts = []
            for tc in tc_list:
                tc_dicts.append({
                     "id": tc.id,
                     "type": tc.type,
                     "function": {
                         "name": tc.function.name,
                         "arguments": tc.function.arguments,
                     },
                 })
            gen = copy.deepcopy(gen)
            gen.tool_calls = tc_dicts
            if cleaned.strip() and cleaned.strip() != gen.text.strip():
                gen.text = cleaned
    except Exception as e:
        logger.debug(f"_fallback_parse_tool_calls failed: {e}")
    return gen

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
        self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7,
        top_p: float = 0.9, top_k: int = 0, min_p: float = 0.0,
        repetition_penalty: float = 1.0, presence_penalty: float = 0.0,
        stop: list[str] | None = None, **kwargs,
    ) -> GenerationOutput:
        pass

    @abstractmethod
    async def stream_generate(
        self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7,
        top_p: float = 0.9, top_k: int = 0, min_p: float = 0.0,
        repetition_penalty: float = 1.0, presence_penalty: float = 0.0,
        stop: list[str] | None = None, **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        pass

    @abstractmethod
    async def chat(
        self, messages: list[dict[str, Any]], max_tokens: int = 4096,
        temperature: float = 0.7, top_p: float = 0.9, top_k: int = 0,
        min_p: float = 0.0, repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0, tools: list[dict] | None = None, **kwargs,
    ) -> GenerationOutput:
        pass

    @abstractmethod
    async def stream_chat(
        self, messages: list[dict[str, Any]], max_tokens: int = 4096,
        temperature: float = 0.7, top_p: float = 0.9, top_k: int = 0,
        min_p: float = 0.0, repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0, tools: list[dict] | None = None, **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        pass

    @property
    @abstractmethod
    def model_type(self) -> str | None:
        pass

    @property
    def grammar_compiler(self):
        return None

    @property
    def prefix_cache_enabled(self) -> bool:
        return False

    def __init__(self):
        self._active_streams_count = 0
        self._stream_lock = threading.Lock()
        self._status = EngineStatus.READY

    def _inc_streams(self) -> None:
        with self._stream_lock:
            self._active_streams_count += 1

    def _dec_streams(self) -> None:
        with self._stream_lock:
            self._active_streams_count -= 1

    def has_active_requests(self) -> bool:
        with self._stream_lock:
            return self._active_streams_count > 0

    @property
    def status(self) -> EngineStatus:
        return self._status

    async def safe_evict(self, timeout: float = 30.0) -> None:
        with self._stream_lock:
            self._status = EngineStatus.CLOSING
        remaining = self._active_streams_count
        if remaining > 0:
            logger.info(
                "Engine evicting: draining %d active streams (timeout=%.0fs)",
                remaining, timeout,
            )
        try:
            await asyncio.wait_for(
                self._wait_streams_drained(), timeout=timeout
            )
        except TimeoutError:
            logger.warning(
                "Engine eviction timeout: %d streams still active, forcing stop",
                self._active_streams_count,
            )
        await self.stop()
        with self._stream_lock:
            self._status = EngineStatus.UNLOADED

    async def _wait_streams_drained(self) -> None:
        while True:
            with self._stream_lock:
                if self._active_streams_count <= 0:
                    break
            await asyncio.sleep(0.05)

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_cache_stats(self) -> dict[str, Any] | None:
        pass
class BaseNonStreamingEngine(ABC):
    def __init__(self):
        self._active_count = 0
        self._active_lock = threading.Lock()
        self._activities: dict[str, dict[str, Any]] = {}

    def has_active_requests(self) -> bool:
        with self._active_lock:
            return self._active_count > 0

    _ACTIVITY_RESERVED_KEYS = {"request_id", "kind", "detail", "started_at", "last_activity_at", "total_items"}

    def _sanitize_activity_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        if not metadata:
            return {}
        return {k: v for k, v in metadata.items() if k not in self._ACTIVITY_RESERVED_KEYS}

    def _begin_activity(self, kind: str, detail: str | None = None, total_items: int | None = None, metadata: dict[str, Any] | None = None) -> str:
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

    def get_activity_snapshot(self) -> dict[str, Any]:
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
    def get_stats(self) -> dict[str, Any]:
        pass
