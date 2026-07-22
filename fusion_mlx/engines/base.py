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
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    diffusion_canvas_tokens: int = 0
    diffusion_denoising_steps: int = 0
    diffusion_work_tokens: int = 0
    diffusion_canvas_tps: float = 0.0
    diffusion_work_tps: float = 0.0
    # OutputRouter channel (content/reasoning/tool_call); StreamingPostProcessor branches on it.
    channel: str | None = None
    # Unprocessed output (pre special-token cleanup) for reasoning/wire-leak
    # detection; engines may populate it, else ``or output.text`` falls back.
    raw_text: str | None = None
    # Per-token full-vocab log-softmax vectors (mx.array) for OpenAI
    # logprobs; populated only when the request asked for logprobs.
    # Spec-decode paths (mtp/suffix/dflash/dspark) leave this None.
    logprobs: Any = None
    # Sampled token ids aligned 1:1 with ``logprobs``.
    new_token_ids: list[int] = field(default_factory=list)


def _normalize_tools_to_dicts(tools: list[dict] | None) -> list[dict] | None:
    """Normalize tool definitions to pure dicts for parse_tool_calls.

    The chat route passes request.tools (Pydantic ToolDefinition objects)
    to engine.generate(), and those objects lack .get() which mlx-lm's
    native parsers call unconditionally.  See #191.
    """
    if not tools:
        return tools
    from ..api.tool_calling import convert_tools_for_template

    converted = convert_tools_for_template(tools)
    return converted if converted else tools


def _fallback_parse_tool_calls(
    gen: GenerationOutput, tokenizer: Any, tools: list[dict]
) -> GenerationOutput:
    """Fallback tool call extraction when the scheduler has no parser session."""
    try:
        from ..api.tool_calling import parse_tool_calls

        dict_tools = _normalize_tools_to_dicts(tools)
        cleaned, tc_list = parse_tool_calls(gen.text, tokenizer, dict_tools)
        if tc_list:
            tc_dicts = []
            for tc in tc_list:
                tc_dicts.append(
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )
            gen = copy.deepcopy(gen)
            gen.tool_calls = tc_dicts
            if cleaned.strip() and cleaned.strip() != gen.text.strip():
                gen.text = cleaned
    except Exception as e:
        logger.debug(f"_fallback_parse_tool_calls failed: {e}")
    return gen


def _apply_reasoning_parser(
    gen: "GenerationOutput",
    model_settings: Any | None,
    ct_kwargs: dict | None,
    model_name: str | None = None,
) -> "GenerationOutput":
    """Strip reasoning tags from non-streaming output.

    Runs the configured reasoning parser (model_settings.reasoning_parser)
    over gen.text so Qwen3's chat-template-injected "Here's a thinking
    process:" preamble and thinking blocks go to the reasoning channel
    instead of consuming the content token budget. No-op when no parser
    is configured or extraction yields nothing.

    When ``model_settings.reasoning_parser`` is unset (common on the
    multi-model discovery path, which unlike ``cli_serve --model`` does
    not call ``detect_model_config``), fall back to family inference via
    ``detect_model_config(model_name)`` so Qwen3 models still get their
    parser without requiring persisted per-model settings.
    """
    parser_name = (
        getattr(model_settings, "reasoning_parser", None) if model_settings else None
    )
    if not parser_name and model_name:
        try:
            from ..model_auto_config import detect_model_config

            auto = detect_model_config(model_name)
            if auto is not None:
                parser_name = auto.reasoning_parser
        except Exception as e:
            logger.debug(
                "reasoning parser auto-detect failed for %s: %s", model_name, e
            )
    if not parser_name:
        return gen
    try:
        from ..reasoning import get_parser

        parser_cls = get_parser(parser_name)
    except Exception as e:
        logger.debug("reasoning parser %r unavailable: %s", parser_name, e)
        return gen
    try:
        enable_thinking = (ct_kwargs or {}).get("enable_thinking")
        parser = parser_cls(tokenizer=None)
        reasoning, content = parser.extract_reasoning(
            gen.text, enable_thinking=enable_thinking
        )
        # content is the real answer channel; reasoning is the thinking trace
        # (template-injected or model-emitted). When the run truncated mid-
        # think (content is None), surface empty so the caller knows no real
        # answer was produced — do NOT leak the reasoning preamble back as
        # content (that is the exact bug this fix targets).
        new_text = content if content is not None else ""
        if new_text != gen.text:
            gen = copy.deepcopy(gen)
            gen.text = new_text
    except Exception as e:
        logger.debug("reasoning parser extract failed: %s", e)
    return gen


_warn_scheduler_logged = set()


def _warn_scheduler_unreachable_once(engine, context: str) -> None:
    key = (engine.model_name, context)
    if key not in _warn_scheduler_logged:
        _warn_scheduler_logged.add(key)
        logger.warning(
            "Scheduler unreachable in %s for %s; skipping preflight check",
            context,
            engine.model_name,
        )


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
        return False

    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def stop(self) -> None:
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        pass

    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        pass

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
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
                remaining,
                timeout,
            )
        try:
            await asyncio.wait_for(self._wait_streams_drained(), timeout=timeout)
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
    _non_streaming_engine = True

    def __init__(self):
        self._active_count = 0
        self._active_lock = threading.Lock()
        self._activities: dict[str, dict[str, Any]] = {}

    def has_active_requests(self) -> bool:
        with self._active_lock:
            return self._active_count > 0

    _ACTIVITY_RESERVED_KEYS = {
        "request_id",
        "kind",
        "detail",
        "started_at",
        "last_activity_at",
        "total_items",
    }

    def _sanitize_activity_metadata(
        self, metadata: dict[str, Any] | None
    ) -> dict[str, Any]:
        if not metadata:
            return {}
        return {
            k: v for k, v in metadata.items() if k not in self._ACTIVITY_RESERVED_KEYS
        }

    def _begin_activity(
        self,
        kind: str,
        detail: str | None = None,
        total_items: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
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
                raise RuntimeError(
                    f"Activity {activity_id} ended more than once or was never started"
                )
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
                item["elapsed_seconds"] = (
                    max(0.0, now - started_at) if started_at is not None else None
                )
                item["last_activity_age_seconds"] = (
                    max(0.0, now - last_activity_at)
                    if last_activity_at is not None
                    else None
                )
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
