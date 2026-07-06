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

import logging

logger = logging.getLogger(__name__)
import os
from typing import TYPE_CHECKING, Any, Optional

from mlx_lm.generate import (
    BatchGenerator,
)

# NaiveStreamingDetokenizer was removed/renamed in newer mlx-lm releases.
# Import with fallback so _get_detokenizer guard still works.
try:
    from mlx_lm.generate import NaiveStreamingDetokenizer
except ImportError:
    NaiveStreamingDetokenizer = None  # type: ignore
from mlx_lm.sample_utils import make_logits_processors

from ..prefill_progress import get_prefill_tracker
from ..request import SamplingParams
from ..utils.sampling import make_sampler as omlx_make_sampler

if TYPE_CHECKING:
    from ..parsers.output_parser import OutputParserSession

# Module-level alias so Scheduler.__init__ can fall back to mlx-lm's default
# stream when no per-engine stream is provided.


def _load_generation_config_eos(self) -> set[int] | None:
    """Load EOS token IDs from generation_config.json if available."""
    try:
        model_path = getattr(self.tokenizer, "name_or_path", None)
        if not model_path:
            return None
        import json
        import os

        gc_path = os.path.join(model_path, "generation_config.json")
        if not os.path.exists(gc_path):
            # name_or_path may be a HuggingFace repo ID (e.g. for VLM
            # tokenizers loaded via AutoProcessor).  Try the HF cache.
            try:
                from huggingface_hub import try_to_load_from_cache

                cached = try_to_load_from_cache(model_path, "generation_config.json")
                if cached and isinstance(cached, str):
                    gc_path = cached
                else:
                    return None
            except (ImportError, Exception):
                return None
        with open(gc_path) as f:
            gc = json.load(f)
        eos = gc.get("eos_token_id")
        if eos is None:
            return None
        if isinstance(eos, list):
            result = set(eos)
        else:
            result = {eos}
        # Only return if there are tokens beyond what tokenizer already provides
        tokenizer_eos = getattr(self.tokenizer, "eos_token_id", None)
        if tokenizer_eos is not None:
            existing = (
                {tokenizer_eos}
                if isinstance(tokenizer_eos, int)
                else set(tokenizer_eos)
            )
            extra = result - existing
            if extra:
                logger.info(
                    f"Loaded {len(extra)} additional EOS token(s) from "
                    f"generation_config.json: {extra}"
                )
                return result
        return result
    except Exception as e:
        logger.debug(f"Could not load generation_config.json: {e}")
        return None


def _get_stop_tokens(self) -> set[int]:
    """Get stop token IDs from tokenizer and generation_config."""
    stop_tokens = set()
    if (
        hasattr(self.tokenizer, "eos_token_id")
        and self.tokenizer.eos_token_id is not None
    ):
        if isinstance(self.tokenizer.eos_token_id, list):
            stop_tokens.update(self.tokenizer.eos_token_id)
        else:
            stop_tokens.add(self.tokenizer.eos_token_id)
    if (
        hasattr(self.tokenizer, "eos_token_ids")
        and self.tokenizer.eos_token_ids is not None
    ):
        eos_ids = self.tokenizer.eos_token_ids
        if isinstance(eos_ids, int):
            stop_tokens.add(eos_ids)
        else:
            stop_tokens.update(eos_ids)

    # Read additional EOS tokens from generation_config.json.
    # Some models (e.g. GLM-4.6V) define multiple EOS tokens there
    # that are not reflected in tokenizer.eos_token_id.
    if self._generation_config_eos is not None:
        stop_tokens.update(self._generation_config_eos)

    # Add protocol-specific stop tokens (e.g. Harmony action stops)
    if self._output_parser_factory is not None:
        stop_tokens.update(self._output_parser_factory.stop_token_ids)

    return stop_tokens


# _update_stop_tokens deleted — per-request stop tokens are now
# handled via SequenceStateMachine passed to insert().


def _get_detokenizer(self, request_id: str):
    """Get or create a streaming detokenizer for a request.

    This enables proper UTF-8 handling for multi-byte characters
    (Korean, Chinese, Japanese, etc.) during streaming.

    NOTE: Each request gets a fresh detokenizer instance. Pooling was removed
    because internal state (byte buffers) can leak between requests even after
    finalize()/reset(), causing text corruption (e.g., spaces inserted in paths,
    character swaps like 'features' -> 'featurse').
    """
    if request_id not in self._request_detokenizers:
        # Always create a fresh detokenizer - no pooling to prevent state contamination
        if hasattr(self.tokenizer, "detokenizer"):
            detok = self.tokenizer.detokenizer
        elif NaiveStreamingDetokenizer is not None:
            detok = NaiveStreamingDetokenizer(self.tokenizer)
        else:
            # Fallback: return None, we'll use decode([token])
            return None
        detok.reset()
        self._request_detokenizers[request_id] = detok
    return self._request_detokenizers[request_id]


def _cleanup_detokenizer(self, request_id: str):
    """Clean up detokenizer for a finished request.

    NOTE: Detokenizers are NOT pooled - each request gets a fresh instance
    to prevent state contamination that causes text corruption.
    """
    detok = self._request_detokenizers.pop(request_id, None)
    # Let GC collect - no pooling to prevent state contamination


def _get_output_parser_session(
    self, request_id: str
) -> Optional["OutputParserSession"]:
    """Get or create a protocol-specific output parser session."""
    if self._output_parser_factory is None:
        return None

    if request_id not in self._output_parser_sessions:
        self._output_parser_sessions[request_id] = (
            self._output_parser_factory.create_session(self.tokenizer)
        )
    return self._output_parser_sessions[request_id]


def _cleanup_output_parser_session(self, request_id: str):
    """Remove any per-request protocol parser session."""
    self._output_parser_sessions.pop(request_id, None)


def _get_xtc_special_tokens(self) -> list[int]:
    """Get special tokens to exclude from XTC sampling (newline + EOS).

    Reuses _get_stop_tokens() for EOS coverage (includes generation_config.json
    tokens) so XTC exclusions stay consistent with stop-token logic.
    """
    tokens = self.tokenizer.encode("\n")
    tokens.extend(self._get_stop_tokens())
    return tokens


def _create_batch_generator(self, sampling_params: SamplingParams) -> BatchGenerator:
    """Create a BatchGenerator with the given sampling parameters."""
    sampler = omlx_make_sampler(
        temp=sampling_params.temperature,
        top_p=sampling_params.top_p,
        min_p=sampling_params.min_p,
        top_k=sampling_params.top_k,
        xtc_probability=sampling_params.xtc_probability,
        xtc_threshold=sampling_params.xtc_threshold,
        xtc_special_tokens=self._xtc_special_tokens,
    )

    # Create logits processors for repetition/presence/frequency penalties
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

    # Convert stop tokens from Set[int] to Sequence[Sequence[int]]
    # for the new BatchGenerator API (each stop token is a sequence).
    stop_tokens_set = self._get_stop_tokens()
    if sampling_params.stop_token_ids:
        stop_tokens_set.update(sampling_params.stop_token_ids)
    stop_tokens_seq = [[t] for t in stop_tokens_set] if stop_tokens_set else None
    logger.info(
        "BatchGenerator: max_tokens=%d, stop_tokens=%s",
        sampling_params.max_tokens,
        stop_tokens_seq,
    )

    bg = BatchGenerator(
        model=self.model,
        max_tokens=sampling_params.max_tokens,
        stop_tokens=stop_tokens_seq,
        sampler=sampler,
        logits_processors=logits_processors if logits_processors else None,
        prefill_batch_size=self.config.prefill_batch_size,
        completion_batch_size=self.config.completion_batch_size,
        prefill_step_size=self.config.prefill_step_size,
        stream=self._stream,
    )

    # Attach fused sampler to the model object so GenerationBatch._step
    # can pick it up. The model is shared across all batches, so this
    # works even when new GenerationBatch instances are created via
    # PromptProcessingBatch.generate().
    from .sampler_fast_path import get_or_create_fused_sampler

    fused = get_or_create_fused_sampler(
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        top_k=sampling_params.top_k,
        min_p=sampling_params.min_p,
    )
    if fused is not None:
        self.model._fused_sampler = fused
        logger.debug(
            "Fused sampler attached for temp=%.2f top_p=%.2f top_k=%d",
            sampling_params.temperature,
            sampling_params.top_p,
            sampling_params.top_k,
        )
    else:
        self.model._fused_sampler = None

    return bg


def _on_prompt_progress(self, updates: list[tuple[int, int, int]]) -> None:
    """Callback from BatchGenerator's prefill loop.

    Called once per prefill chunk (default 2048 tokens) with a list of
    (uid, processed_tokens, total_tokens) tuples.  Updates the global
    PrefillProgressTracker so the admin dashboard can display per-request
    prefill progress.  Only touches CPU counters — zero GPU overhead.
    """
    tracker = get_prefill_tracker()
    # model_name is a full path; use basename to match engine_pool model_id.
    model_id = os.path.basename(self.config.model_name.rstrip("/"))
    for uid, processed, total in updates:
        request_id = self.uid_to_request_id.get(uid)
        if request_id is None:
            continue
        tracker.update(
            request_id=request_id,
            processed=processed,
            total=total,
            model_id=model_id,
        )
        logger.info(
            "Prompt processing progress: model=%s, processed=%d, total=%d",
            model_id,
            processed,
            total,
        )


# ------------------------------------------------------------------
# External prefill (composition pattern — replaces _process_prompts)
# ------------------------------------------------------------------


def _apply_turboquant_kv_empty(self, prompt_cache: list[Any]) -> None:
    """Replace KVCache with empty TurboQuantKVCache before prefill.

    merge() blocker resolved via monkeypatches.py (BatchTurboQuantKVCache.merge).

    Tokens are quantized on the fly during update_and_fetch, avoiding
    the peak memory spike from storing full-precision KV then converting.
    Skips the last KVCache layer if turboquant_skip_last is set.
    """
    from mlx_lm.models.cache import CacheList, KVCache
    from mlx_vlm.turboquant import TurboQuantKVCache

    kv_indices = [i for i, c in enumerate(prompt_cache) if isinstance(c, KVCache)]
    skip_last = self._turboquant_skip_last and len(kv_indices) > 1
    last_kv_idx = kv_indices[-1] if skip_last else -1

    converted = 0
    bits = float(self._turboquant_kv_bits)
    mode = getattr(self, "_turboquant_kv_mode", "v4")
    for i, cache_obj in enumerate(prompt_cache):
        if isinstance(cache_obj, KVCache):
            if i == last_kv_idx:
                continue
            prompt_cache[i] = TurboQuantKVCache(bits=bits)
            converted += 1
        elif isinstance(cache_obj, CacheList):
            new_caches = []
            for c in cache_obj.caches:
                if isinstance(c, KVCache):
                    new_caches.append(TurboQuantKVCache(bits=bits))
                    converted += 1
                else:
                    new_caches.append(c)
            cache_obj.caches = tuple(new_caches)
    if converted > 0:
        skip_msg = ", skipped last KVCache layer" if skip_last else ""
        mode_msg = f" (mode={mode})" if mode != "v4" else ""
        logger.info(
            f"TurboQuant: {converted}/{len(prompt_cache)} "
            f"cache layers set to {bits}-bit{skip_msg}{mode_msg}"
        )


def _apply_turboquant_kv_convert(self, prompt_cache: list[Any]) -> None:
    """Convert existing KVCache data to TurboQuantKVCache via from_cache().

    merge() blocker resolved via monkeypatches.py (BatchTurboQuantKVCache.merge).

    Used when an existing cache is provided (e.g. from SSD prefix cache).
    Uses from_cache() to quantize the existing KV data.
    """
    from mlx_lm.models.cache import CacheList, KVCache
    from mlx_vlm.turboquant import TurboQuantKVCache

    kv_indices = [i for i, c in enumerate(prompt_cache) if isinstance(c, KVCache)]
    skip_last = self._turboquant_skip_last and len(kv_indices) > 1
    last_kv_idx = kv_indices[-1] if skip_last else -1

    converted = 0
    bits = float(self._turboquant_kv_bits)
    mode = getattr(self, "_turboquant_kv_mode", "v4")
    for i, cache_obj in enumerate(prompt_cache):
        if isinstance(cache_obj, KVCache):
            if i == last_kv_idx:
                continue
            prompt_cache[i] = TurboQuantKVCache.from_cache(cache_obj, bits=bits)
            converted += 1
        elif isinstance(cache_obj, CacheList):
            new_caches = []
            for c in cache_obj.caches:
                if isinstance(c, KVCache):
                    new_caches.append(TurboQuantKVCache.from_cache(c, bits=bits))
                    converted += 1
                else:
                    new_caches.append(c)
            cache_obj.caches = tuple(new_caches)
    if converted > 0:
        skip_msg = ", skipped last KVCache layer" if skip_last else ""
        mode_msg = f" (mode={mode})" if mode != "v4" else ""
        logger.info(
            f"TurboQuant: converted {converted}/{len(prompt_cache)} "
            f"cache layers to {bits}-bit{skip_msg}{mode_msg}"
        )
