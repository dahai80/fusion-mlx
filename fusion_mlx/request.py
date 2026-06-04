# SPDX-License-Identifier: Apache-2.0
"""Request management for fusion-mlx continuous batching."""

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


class RequestStatus(enum.IntEnum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finish_reason(status: "RequestStatus") -> Optional[str]:
        if status == RequestStatus.FINISHED_STOPPED:
            return "stop"
        elif status == RequestStatus.FINISHED_LENGTH_CAPPED:
            return "length"
        elif status == RequestStatus.FINISHED_ABORTED:
            return "abort"
        return None


@dataclass
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    min_p: float = 0.0
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.1
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    logprobs: bool = False
    top_logprobs: Optional[int] = None
    thinking_budget: Optional[int] = None
    compiled_grammar: Any = None
    seed: Optional[int] = None

    # SpecPrefill (set per-request by engine)
    _specprefill_enabled: bool = False
    _specprefill_keep_pct: float = 0.8
    _specprefill_threshold: int = 0
    specprefill_system_end: int = 0

    def __post_init__(self):
        if self.stop is None:
            self.stop = []
        if self.stop_token_ids is None:
            self.stop_token_ids = []


@dataclass
class Request:
    request_id: str
    prompt: Union[str, List[int]]
    sampling_params: SamplingParams
    arrival_time: float = field(default_factory=time.monotonic)
    priority: int = 0

    # Set after tokenization
    prompt_token_ids: Optional[List[int]] = None
    num_prompt_tokens: int = 0

    # Generation state
    status: RequestStatus = RequestStatus.WAITING
    num_computed_tokens: int = 0
    output_token_ids: List[int] = field(default_factory=list)
    output_text: str = ""
    generation_started_at: Optional[float] = None
    last_activity_at: Optional[float] = None

    # For BatchGenerator integration
    batch_uid: Optional[int] = None

    # Prefix cache fields
    prompt_cache: Optional[List[Any]] = None
    cached_tokens: int = 0
    remaining_tokens: Optional[List[int]] = None

    # Paged cache fields
    block_table: Optional[Any] = None
    shared_prefix_blocks: int = 0

    # Multimodal content
    images: Optional[List[Any]] = None
    videos: Optional[List[Any]] = None

    # VLM fields
    vlm_inputs_embeds: Optional[Any] = None
    vlm_extra_kwargs: Optional[Dict[str, Any]] = None
    vlm_image_hash: Optional[str] = None
    vlm_cache_key_start: int = 0
    vlm_cache_key_ranges: Optional[List[Tuple[int, str]]] = None
    rope_deltas: float = 0.0

    # SpecPrefill
    specprefill_indices: Optional[Any] = None
    specprefill_total_tokens: int = 0
    specprefill_position_offset: int = 0
    specprefill_system_end: int = 0

    # Cache corruption recovery
    cache_corruption_retries: int = 0

    # Reasoning model support
    needs_think_prefix: bool = False
    think_prefix_sent: bool = False
    is_harmony_model: bool = False

    # Metadata
    finish_reason: Optional[str] = None

    @property
    def vlm_extra_keys_for_cache(self) -> Optional[Tuple[str, ...]]:
        if self.vlm_image_hash:
            return (self.vlm_image_hash,)
        return None

    @property
    def vlm_extra_key_token_start_for_cache(self) -> Optional[int]:
        if self.vlm_image_hash:
            return self.vlm_cache_key_start
        return None

    @property
    def vlm_extra_key_ranges_for_cache(
        self,
    ) -> Optional[List[Tuple[int, Tuple[str, ...]]]]:
        if not self.vlm_cache_key_ranges:
            return None
        return [(start, (h,)) for start, h in self.vlm_cache_key_ranges]

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def max_tokens(self) -> int:
        return self.sampling_params.max_tokens

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finish_reason(self) -> Optional[str]:
        if self.finish_reason:
            return self.finish_reason
        return RequestStatus.get_finish_reason(self.status)

    def append_output_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)
        self.num_computed_tokens += 1

    def set_finished(self, status: RequestStatus, reason: Optional[str] = None) -> None:
        self.status = status
        self.finish_reason = reason or RequestStatus.get_finish_reason(status)

    def __lt__(self, other: "Request") -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.arrival_time < other.arrival_time

    def __hash__(self) -> int:
        return hash(self.request_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Request):
            return False
        return self.request_id == other.request_id


@dataclass
class RequestOutput:
    request_id: str
    new_token_ids: List[int] = field(default_factory=list)
    new_text: str = ""
    output_token_ids: List[int] = field(default_factory=list)
    output_text: str = ""
    finished: bool = False
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: Optional[List[Dict[str, str]]] = None
    cached_tokens: int = 0
    error: Optional[str] = None

    @property
    def usage(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }
