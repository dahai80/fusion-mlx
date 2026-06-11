# SPDX-License-Identifier: Apache-2.0
"""
Scheduler for oMLX continuous batching.

This module provides a Scheduler class that manages request scheduling
using mlx-lm's BatchGenerator for efficient continuous batching.

The scheduler follows vLLM's design with:
- Waiting queue for pending requests
- Running set for active requests
- Continuous batching via BatchGenerator
"""

import concurrent.futures
import copy
import gc
import logging

logger = logging.getLogger(__name__)
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.generate import (
    BatchGenerator,
    GenerationBatch,
    PromptProcessingBatch,
    SequenceStateMachine,
    generation_stream,
)
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_logits_processors

from ..cache.observability import CacheRateTracker
from ..cache.paged_cache import PagedCacheManager
from ..cache.prefix_cache import BlockAwarePrefixCache
from ..exceptions import is_cache_corruption_error
from ..prefill_progress import get_prefill_tracker
from ..prefill_transient_tracker import PrefillTransientTracker
from ..request import Request, RequestOutput, RequestStatus, SamplingParams
from ..speculative.vlm_mtp import VLMMTPDrafter, run_vlm_mtp_decode
from ..utils.proc_memory import get_phys_footprint
from ..utils.sampling import make_sampler as omlx_make_sampler

# Module-level alias so Scheduler.__init__ can fall back to mlx-lm's default
# stream when no per-engine stream is provided.

from .types import (
     _VLMMTPDecodeState, _VLMMTPResponse, _mx_buffer_access_lock,
     _StoreCacheGate, _PrefillAbortedError, _PrefillState,
     _BoundarySnapshotProvider,
)
from .helpers import (
     _sync_and_clear_cache, _safe_sync_stream,
     _prompt_cache_needs_snapshots, _cache_layer_token_count, _cache_base_sizes,
     _vlm_extra_seq_slice, _slice_vlm_extra, _advance_vlm_extra,
     _KNOWN_SLICEABLE_CACHE_TYPES,
)
from .monkeypatches import _default_generation_stream

def _build_state_machine(    self, request: "Request") -> SequenceStateMachine:
    """Build a SequenceStateMachine for per-request stop tokens.

    Combines base stop tokens (EOS, Harmony) with request-specific
    stop_token_ids and tokenized stop strings into a single state
    machine that tells BatchGenerator when to stop generating for
    this request.
    """
    stop_tokens_set = self._get_stop_tokens()
    if request.sampling_params.stop_token_ids:
        stop_tokens_set.update(request.sampling_params.stop_token_ids)

    transitions: dict[str, list] = {
        "normal": [([t], None) for t in stop_tokens_set]
    }

    # Tokenize stop strings into token sequences. mlx-lm's
    # SequenceStateMachine uses Aho-Corasick, so per-token match
    # cost stays O(1) regardless of how many sequences are added.
    # BPE merge edge cases (where a stop string boundary lands
    # mid-token) may miss; that is a known limitation.
    for stop_str in request.sampling_params.stop or []:
        if not isinstance(stop_str, str) or not stop_str:
            continue
        try:
            seq = self.tokenizer.encode(stop_str, add_special_tokens=False)
        except TypeError:
            seq = self.tokenizer.encode(stop_str)
        if seq:
            transitions["normal"].append((list(seq), None))

    if transitions["normal"]:
        return SequenceStateMachine(transitions, initial="normal")
    return SequenceStateMachine({}, initial="normal")

def _emit_prefill_boundary_snapshot(    self,
    request: "Request",
    prompt_cache: list[Any],
    total_tokens: int,
) -> None:
    """Capture boundary snapshot from individual (non-batch) cache.

    During external prefill we have direct access to per-layer cache
    objects (not BatchKVCache). Extract non-sliceable layers for
    boundary snapshot storage.

    Pass ``request_id`` directly. The request is mid-prefill and has
    not been inserted into ``BatchGenerator`` yet, so
    ``request_id_to_uid`` has no entry for it. The earlier shape
    routed through ``self.request_id_to_uid.get(request_id, -1)`` →
    ``uid_to_request_id.get(-1)`` → ``None`` → silent return,
    dropping every snapshot. For ArraysCache / GDN / hybrid models
    that made every non-last block store a placeholder, and the
    next identical-prompt request rejected the cache and re-
    prefilled from scratch.
    """
    snapshot_cache = [
        c if type(c).__name__ not in _KNOWN_SLICEABLE_CACHE_TYPES else None
        for c in prompt_cache
    ]
    self._on_prefill_boundary_snapshot(
        request.request_id,
        snapshot_cache,
        total_tokens,
    )

def _build_sampler_and_processors(
    self, sampling_params: SamplingParams, request: Any = None
) -> tuple[Callable[[mx.array], mx.array], list[Callable]]:
    """Build per-request sampler and logits processors."""
    # Use fusion_mlx.utils.sampling.make_sampler instead of mlx_lm.sample_utils.
    # The mlx-lm version decorates categorical_sampling and apply_* with
    # @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state),
    # which fails to advance the RNG state after the first call in this
    # server environment. Identical prompts then produce identical output
    # even at temperature > 1.
    sampler = omlx_make_sampler(
        temp=sampling_params.temperature,
        top_p=sampling_params.top_p,
        min_p=sampling_params.min_p,
        top_k=sampling_params.top_k,
        xtc_probability=sampling_params.xtc_probability,
        xtc_threshold=sampling_params.xtc_threshold,
        xtc_special_tokens=self._xtc_special_tokens,
    )
    logits_processors = make_logits_processors(
        repetition_penalty=(
            sampling_params.repetition_penalty
            if sampling_params.repetition_penalty != 1.0
            else None
        ),
        presence_penalty=(
            sampling_params.presence_penalty
            if sampling_params.presence_penalty != 0.0
            else None
        ),
        frequency_penalty=(
            sampling_params.frequency_penalty
            if sampling_params.frequency_penalty != 0.0
            else None
        ),
    )

    # Add thinking budget processor for reasoning models
    if (
        sampling_params.thinking_budget is not None
        and request is not None
        and getattr(request, "needs_think_prefix", False)
        and not getattr(request, "is_harmony_model", False)
    ):
        think_end_ids = self._resolve_think_end_token_ids()
        if think_end_ids:
            from .api.thinking import ThinkingBudgetProcessor

            think_start_id = self._get_think_token_id("think_start_id")
            leading_ids, trailing_ids = self._resolve_think_close_pattern()
            processor = ThinkingBudgetProcessor(
                think_end_token_ids=think_end_ids,
                budget=sampling_params.thinking_budget,
                think_start_token_id=think_start_id,
                leading_token_ids=leading_ids,
                trailing_token_ids=trailing_ids,
            )
            logits_processors.append(processor)

    # Add grammar constraint processor for structured output.
    # Phase awareness (thinking vs output) is handled by the compiled
    # grammar itself via xgrammar structural tags, so we don't need
    # think_end_ids here.
    if sampling_params.compiled_grammar is not None:
        try:
            from ..api.grammar import GrammarConstraintProcessor

            vocab_size = self._get_model_vocab_size()
            if vocab_size is not None:
                processor = GrammarConstraintProcessor(
                    compiled_grammar=sampling_params.compiled_grammar,
                    vocab_size=vocab_size,
                )
                logits_processors.append(processor)
            else:
                logger.warning(
                    "Cannot determine vocab_size; skipping grammar constraint"
                )
        except ImportError:
            logger.warning("xgrammar not installed; skipping grammar constraint")

    return sampler, logits_processors

def _get_model_vocab_size(self) -> int | None:
    """Return vocab_size from model config, or None if unavailable."""
    from ..utils.tokenizer import resolve_vocab_size

    return resolve_vocab_size(self.model)

def _get_think_token_id(    self, attr: str) -> int | None:
    """Safely read a think token id from the tokenizer.

    mlx-lm tokenizers expose ``think_start_id`` / ``think_end_id`` as
    properties that may raise ``ValueError`` (multi-token sequence) or
    ``TypeError`` (``_think_start_tokens`` is ``None`` for models without
    thinking support, e.g. context-1 / harmony parser).

    Returns the token id, or ``None`` when unavailable.
    """
    try:
        return getattr(self.tokenizer, attr, None)
    except (ValueError, TypeError):
        return None

def _resolve_think_end_token_ids(self) -> list[int] | None:
    """Resolve token ID(s) for the close-think tag.

    Uses mlx-lm's built-in think_end_id which supports both
    </think> and </longcat_think> automatically.
    """
    # Tier 1: mlx-lm tokenizer attribute (covers all known think variants)
    think_end_id = self._get_think_token_id("think_end_id")
    if think_end_id is not None:
        return [think_end_id]

    # Tier 2: encode the think_end string
    think_end_str = getattr(self.tokenizer, "think_end", "</think>")
    try:
        ids = self.tokenizer.encode(think_end_str, add_special_tokens=False)
        if ids:
            return list(ids)
    except Exception:
        logger.debug("swallowed exception at fusion_mlx/scheduler/sched_thinking.py:255")

        pass

    # Tier 3: direct token lookup
    try:
        tid = self.tokenizer.convert_tokens_to_ids("</think>")
        if tid != getattr(self.tokenizer, "unk_token_id", None):
            return [tid]
    except (AttributeError, KeyError, TypeError):
        pass

    return None

def _resolve_think_close_pattern(self) -> tuple[list[int] | None, list[int] | None]:
    """Detect leading/trailing tokens around </think> from the chat template.

    Different models use different patterns:
    - Qwen3/3.5, MiniMax: ``\\n</think>\\n\\n``
    - DeepSeek V3.2, GLM-5: ``</think>`` (no newlines)
    - GLM-4.6V: ``</think>\\n``
    - Step-3.5-Flash: ``\\n</think>\\n``

    Returns (leading_token_ids, trailing_token_ids) or (None, None).
    """
    import re

    think_end_str = getattr(self.tokenizer, "think_end", "</think>")

    # Try to get the chat template text
    template_text = self._get_chat_template_text()
    if not template_text:
        return None, None

    # Find the close pattern in the template, e.g. \n</think>\n\n
    # Look for the think_end_str surrounded by whitespace/newlines in string literals
    escaped = re.escape(think_end_str)
    # Match patterns like: \n</think>\n\n or </think> in template strings
    match = re.search(
        r"(\\n|\\r|[\n\r])*" + escaped + r"((?:\\n|\\r|[\n\r])*)",
        template_text,
    )
    if not match:
        return None, None

    # Extract raw leading/trailing whitespace, converting \n escapes to actual newlines
    raw_leading = (
        match.group(0)
        .split(think_end_str)[0]
        .replace("\\n", "\n")
        .replace("\\r", "\r")
    )
    raw_trailing = (
        match.group(0)
        .split(think_end_str)[1]
        .replace("\\n", "\n")
        .replace("\\r", "\r")
    )

    # Encode to token IDs
    leading_ids = None
    trailing_ids = None
    if raw_leading:
        try:
            ids = self.tokenizer.encode(raw_leading, add_special_tokens=False)
            if ids:
                leading_ids = list(ids)
        except Exception:
            logger.debug("swallowed exception at fusion_mlx/scheduler/sched_thinking.py:322")

            pass
    if raw_trailing:
        try:
            ids = self.tokenizer.encode(raw_trailing, add_special_tokens=False)
            if ids:
                trailing_ids = list(ids)
        except Exception:
            logger.debug("swallowed exception at fusion_mlx/scheduler/sched_thinking.py:330")

            pass

    return leading_ids, trailing_ids

def _get_chat_template_text(self) -> str | None:
    """Get chat template text from the tokenizer or model directory."""
    # Try tokenizer's chat_template attribute (Jinja string)
    ct = getattr(self.tokenizer, "_chat_template", None)
    if ct:
        return ct if isinstance(ct, str) else str(ct)
    ct = getattr(self.tokenizer, "chat_template", None)
    if ct:
        return ct if isinstance(ct, str) else str(ct)

    # Try reading the .jinja file from model directory
    import os

    model_path = getattr(self.config, "model_name", None) or ""
    jinja_path = os.path.join(model_path, "chat_template.jinja")
    if os.path.isfile(jinja_path):
        try:
            with open(jinja_path, encoding="utf-8") as f:
                return f.read()
        except Exception:
            logger.debug("swallowed exception at fusion_mlx/scheduler/sched_thinking.py:355")

            pass

    return None

def _detect_needs_think_prefix(    self, request: "Request") -> bool:
    """Detect if prompt ends with an open <think> tag (thinking enabled).

    Returns False for disabled-thinking patterns like <think></think>
    where </think> immediately follows <think> in the prompt tail.
    """
    think_start_id = self._get_think_token_id("think_start_id")
    if think_start_id is None:
        try:
            think_start_id = self.tokenizer.convert_tokens_to_ids("<think>")
            if think_start_id == getattr(self.tokenizer, "unk_token_id", None):
                return False
        except (AttributeError, KeyError, TypeError):
            return False

    if not think_start_id or not request.prompt_token_ids:
        return False

    last_tokens = list(request.prompt_token_ids[-3:])
    if think_start_id not in last_tokens:
        return False

    # <think> found. Check if </think> follows it (disabled thinking pattern).
    last_idx = len(last_tokens) - 1 - last_tokens[::-1].index(think_start_id)
    after_start = last_tokens[last_idx + 1 :]

    if after_start:
        think_end_ids = self._resolve_think_end_token_ids()
        if think_end_ids and think_end_ids[0] in after_start:
            return False

    return True

def _ensure_batch_generator(    self, sampling_params: SamplingParams) -> None:
    """Ensure BatchGenerator exists with compatible settings."""
    # Only create once; per-request samplers are passed at insert time.
    if self.batch_generator is None:
        self.batch_generator = self._create_batch_generator(sampling_params)

    # Track latest params for debugging/metrics.
    self._current_sampler_params = (
        sampling_params.temperature,
        sampling_params.top_p,
        sampling_params.min_p,
        sampling_params.top_k,
        sampling_params.repetition_penalty,
    )

def _cache_tree_has_stateful_non_sliceable(    self, cache_obj: Any) -> bool:
    """Detect non-sliceable recurrent cache layers requiring snapshots."""
    # None placeholders from boundary snapshots (sliceable layers replaced).
    if cache_obj is None:
        return False

    # CacheList nests multiple cache objects.
    sub_caches = getattr(cache_obj, "caches", None)
    if isinstance(sub_caches, (list, tuple)):
        return any(
            self._cache_tree_has_stateful_non_sliceable(sub_cache)
            for sub_cache in sub_caches
        )

    class_name = type(cache_obj).__name__

    # Known sliceable cache types — no boundary snapshots needed.
    if class_name in (
        "KVCache",
        "BatchKVCache",
        "QuantizedKVCache",
    ):
        return False

    # Stateful non-sliceable caches require boundary-safe snapshots.
    if class_name in (
        "RotatingKVCache",
        "BatchRotatingKVCache",
        "ArraysCache",
        "SizedArraysCache",
    ):
        return True

    if HAS_CACHE_TYPE_HANDLERS and CacheTypeRegistry is not None:
        handler = CacheTypeRegistry.get_handler_by_class_name(class_name)
        if not handler.supports_block_slicing:
            return True

    # Best-effort fallback for unknown recurrent cache structures.
    state_list = getattr(cache_obj, "cache", None)
    if isinstance(state_list, list):
        return True

    return False
