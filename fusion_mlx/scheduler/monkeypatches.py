import logging
import threading
import time
from collections import OrderedDict
from typing import Any, NamedTuple

import mlx.core as mx
from mlx_lm.generate import (
    GenerationBatch,
    PromptProcessingBatch,
    generation_stream,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UID row registry — tracks the sampler/logits_processors each uid should run.
# Heterogeneous continuous batching (extend/filter across prompt and
# generation batches) can leave stale or offset row slots behind; the
# registry records what each uid is supposed to run at insert time, and
# the step chokepoint realigns the positional lists from it.
# Bounded so a missing cleanup can never grow it unbounded.
# ---------------------------------------------------------------------------
class _RegisteredRow(NamedTuple):
    sampler: Any
    logits_processors: list


_UID_ROW_REGISTRY_MAX = 4096
_uid_row_registry: "OrderedDict[tuple[int, int], _RegisteredRow]" = OrderedDict()
_uid_row_registry_lock = threading.Lock()
_UID_ROW_DRIFT_WARNING_INTERVAL_S = 60.0
_uid_row_drift_last_warning = float("-inf")


def _register_uid_rows(model, uids, samplers, lps_rows) -> None:
    with _uid_row_registry_lock:
        for uid, sampler, lps in zip(uids, samplers, lps_rows):
            _uid_row_registry[(id(model), uid)] = _RegisteredRow(
                sampler, list(lps or ())
            )
        while len(_uid_row_registry) > _UID_ROW_REGISTRY_MAX:
            _uid_row_registry.popitem(last=False)


def _unregister_uid_row(model, uid) -> None:
    with _uid_row_registry_lock:
        _uid_row_registry.pop((id(model), uid), None)


def _unregister_uid_rows_for_model(model) -> None:
    model_id = id(model)
    with _uid_row_registry_lock:
        for key in [key for key in _uid_row_registry if key[0] == model_id]:
            del _uid_row_registry[key]


def _row_drifted(current_lps, expected_lps) -> bool:
    if not current_lps and not expected_lps:
        return False
    return current_lps != expected_lps


def _log_drift_correction(uids, slot_count) -> None:
    global _uid_row_drift_last_warning
    now = time.monotonic()
    rate_limited = now - _uid_row_drift_last_warning < _UID_ROW_DRIFT_WARNING_INTERVAL_S
    if not rate_limited:
        _uid_row_drift_last_warning = now
    (logger.debug if rate_limited else logger.warning)(
        "Realigned generation-batch row state from the uid registry "
        f"(uids={list(uids)}, had {slot_count} processor slots); "
        "stale or offset slots would have run the wrong sampler/processors."
    )


def _realigned_rows(model, uids, cur_samplers, cur_lps):
    model_id = id(model)
    with _uid_row_registry_lock:
        rows = [_uid_row_registry.get((model_id, uid)) for uid in uids]

    drift = len(cur_lps) != len(uids)
    samplers, lps = [], []
    for i, row in enumerate(rows):
        if row is not None:
            if not drift:
                if i >= len(cur_samplers):
                    drift = row.sampler is not None
                elif cur_samplers[i] is not row.sampler:
                    drift = True
            if (
                not drift
                and i < len(cur_lps)
                and cur_lps[i] is not row.logits_processors
            ):
                drift = _row_drifted(cur_lps[i], row.logits_processors)
            samplers.append(row.sampler)
            lps.append(row.logits_processors)
        else:
            samplers.append(cur_samplers[i] if i < len(cur_samplers) else None)
            lps.append(cur_lps[i] if i < len(cur_lps) else [])
    return samplers, lps, drift


def _realign_generation_batch_rows(self) -> None:
    if self.logits_processors is None:
        self.logits_processors = []
    else:
        self.logits_processors = [
            procs if procs is not None else [] for procs in self.logits_processors
        ]

    uids = getattr(self, "uids", None) or []
    if not uids:
        return

    # Fast path: skip registry lock when batch is homogeneous and stable.
    # Single-request decode never drifts; skip the lock acquire/release.
    # Also skip when all logits_processors are empty and samplers are
    # uniform (no per-row specialization).
    if len(uids) == 1:
        if not self.logits_processors or not any(self.logits_processors):
            return
        if len(self.logits_processors) == 1 and not self.logits_processors[0]:
            return

    new_samplers, new_lps, drift = _realigned_rows(
        getattr(self, "model", None),
        uids,
        getattr(self, "samplers", None) or [],
        self.logits_processors,
    )
    if drift:
        _log_drift_correction(uids, len(self.logits_processors))
    self.logits_processors = new_lps
    self.samplers = new_samplers


# ---------------------------------------------------------------------------
# Version guard — warn when mlx-lm is outside tested range so operators
# know patches may not apply correctly.
# ---------------------------------------------------------------------------
try:
    import importlib.metadata

    _mlx_lm_ver = importlib.metadata.version("mlx-lm")
except Exception:
    _mlx_lm_ver = "unknown"

# Update these bounds when testing against new mlx-lm releases.
_MLX_LM_MIN_TESTED = "0.21"
_MLX_LM_MAX_TESTED = "0.32"

if _mlx_lm_ver != "unknown":

    def _ver_tuple(v):
        return tuple(int(x) for x in v.split(".")[:3])

    vt = _ver_tuple(_mlx_lm_ver)
    if vt < _ver_tuple(_MLX_LM_MIN_TESTED) or vt > _ver_tuple(_MLX_LM_MAX_TESTED):
        logger.warning(
            "mlx-lm %s is outside the tested range [%s, %s]. "
            "Monkeypatches may not apply correctly. "
            "Please verify after upgrading.",
            _mlx_lm_ver,
            _MLX_LM_MIN_TESTED,
            _MLX_LM_MAX_TESTED,
        )
    else:
        logger.debug(
            "mlx-lm %s within tested range [%s, %s]",
            _mlx_lm_ver,
            _MLX_LM_MIN_TESTED,
            _MLX_LM_MAX_TESTED,
        )


# Module-level alias so Scheduler.__init__ can fall back to mlx-lm's default
# stream when no per-engine stream is provided.
_default_generation_stream = generation_stream


# ---------------------------------------------------------------------------
# Optimized GenerationBatch._step — replaces the stock mlx-lm version with
# three key improvements for decode throughput:
#
# 1. Batched sampling fast path: when all rows share the same sampler (or
#    all are None, falling back to fallback_sampler), invoke the sampler
#    once on [B, vocab] instead of B per-row calls + mx.concatenate.
# 2. Skip logsumexp when possible: normalization is only needed when
#    logprobs will actually be consumed. For argmax (greedy) and categorical
#    sampling on raw logits, the sampler output is identical without it.
# 3. Reduced GPU sync: replace inputs.tolist() (full GPU sync barrier)
#    with per-token .item() calls that sync after async_eval work completes.
# ---------------------------------------------------------------------------
_original_generation_batch_step = GenerationBatch._step


def _optimized_generation_batch_step(self):
    self._current_tokens = self._next_tokens
    self._current_logprobs = self._next_logprobs
    inputs = self._current_tokens

    model = self.model

    # Build per-batch mRoPE deltas from UID mapping before each step.
    if (
        getattr(model, "_uses_mrope", False)
        and getattr(model, "_uid_rope_deltas", None)
        and self.uids
    ):
        deltas = [model._uid_rope_deltas.get(uid, 0.0) for uid in self.uids]
        model.set_batch_rope_deltas(mx.array(deltas))

    _realign_generation_batch_rows(self)

    # Forward pass
    logits = model(inputs[:, None], cache=self.prompt_cache)
    logits = logits[:, -1, :]

    # Logits processors (per-row, cannot be batched)
    has_logits_processors = bool(self.logits_processors) and any(self.logits_processors)
    needs_logprob_norm = True
    token_context = []
    if has_logits_processors:
        from ..api.grammar import GrammarConstraintProcessor

        has_grammar = any(
            isinstance(p, GrammarConstraintProcessor)
            for procs in self.logits_processors
            for p in procs
        )
        token_context = [
            tc.update_and_fetch(inputs[i : i + 1])
            for i, tc in enumerate(self._token_context)
        ]
        processed_logits = []
        for e in range(len(self.uids)):
            sample_logits = logits[e : e + 1]
            for processor in self.logits_processors[e]:
                sample_logits = processor(token_context[e], sample_logits)
            processed_logits.append(sample_logits)
        logits = mx.concatenate(processed_logits, axis=0)
    else:
        has_grammar = False

    # Decide whether we can skip logsumexp normalization.
    # For the common case (no logits_processors, all samplers None or
    # identical), the sampler produces the same result on raw logits.
    has_per_row_samplers = bool(self.samplers) and any(self.samplers)
    if not has_per_row_samplers and not has_logits_processors:
        needs_logprob_norm = False

    if needs_logprob_norm:
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    else:
        logprobs = logits

    # Sampling — try fused sampler first, then batched fast path,
    # then per-row fallback.
    fused = getattr(model, "_fused_sampler", None)
    if fused is not None and not has_per_row_samplers and not has_logits_processors:
        # Fused top-p + temperature sampler (one lazy-graph segment)
        sampled = fused(logprobs)
    elif has_per_row_samplers:
        # Check if all samplers are None (use fallback) or identical
        all_none = all(s is None for s in self.samplers)
        if all_none:
            sampled = self.fallback_sampler(logprobs)
        else:
            first_sampler = next((s for s in self.samplers if s is not None), None)
            all_same = first_sampler is not None and all(
                s is None or s is first_sampler for s in self.samplers
            )
            if all_same:
                sampled = first_sampler(logprobs)
            else:
                # Heterogeneous — fall back to per-row loop
                all_samples = []
                for e in range(len(self.uids)):
                    sample_sampler = self.samplers[e] or self.fallback_sampler
                    sampled_e = sample_sampler(logprobs[e : e + 1])
                    all_samples.append(sampled_e)
                sampled = mx.concatenate(all_samples, axis=0)
    else:
        sampled = self.fallback_sampler(logprobs)

    # Assign next step variables and start computing asynchronously
    self._next_tokens = sampled
    if needs_logprob_norm:
        self._next_logprobs = list(logprobs)
    else:
        # Store a lightweight placeholder — downstream discards logprobs
        # when the request didn't ask for them (sched_response.py:182)
        self._next_logprobs = [None] * len(self.uids)

    # --- Double-buffer: batch inputs into async_eval group ---
    # Instead of a separate mx.eval(inputs) (full GPU sync barrier),
    # include inputs in the async_eval group so it materializes
    # alongside the sampled tokens without a dedicated stall.
    # The subsequent tolist() will block until all async work
    # completes, but the GPU had more work to overlap.
    eval_targets = [self._next_tokens, inputs]
    if token_context:
        eval_targets.extend(token_context)
    mx.async_eval(*eval_targets)

    # Drain deferred token-append from prior step (overlapped with
    # the forward pass above). First call has no deferred data.
    deferred_input_list = getattr(self, "_deferred_input_list", None)
    deferred_tokens = getattr(self, "_deferred_tokens", None)
    if deferred_input_list is not None and deferred_tokens is not None:
        for sti, ti in zip(deferred_tokens, deferred_input_list):
            sti.append(ti)

    # Materialize current tokens for Response construction.
    # This blocks until async_eval completes, but the GPU was
    # computing sampled+inputs in parallel so the stall is shorter.
    input_list = inputs.tolist()
    # Stash for next-step drain. Capture the live token lists now
    # so filter() between steps doesn't invalidate our reference.
    self._deferred_input_list = input_list
    self._deferred_tokens = list(self.tokens)

    # Grammar accept_token: accept sampled tokens for grammar processors
    if has_grammar:
        mx.eval(self._next_tokens)
        sampled_list = self._next_tokens.tolist()
        for e in range(len(self.uids)):
            for proc in self.logits_processors[e]:
                if isinstance(proc, GrammarConstraintProcessor):
                    proc.accept_token(sampled_list[e])

    return input_list, self._current_logprobs


def _patched_generation_batch_step(self):
    return _optimized_generation_batch_step(self)


GenerationBatch._realign_rows = _realign_generation_batch_rows
GenerationBatch._step = _patched_generation_batch_step


# ---------------------------------------------------------------------------
# Monkey-patch GenerationBatch.filter to keep logits_processors aligned with
# uids. mlx-lm's filter only reindexes the processor list when at least one
# row has an active processor:
#
#     if any(self.logits_processors):
#         self.logits_processors = [self.logits_processors[idx] for idx in keep]
#
# There is no else branch, so when every slot is empty the stale list
# survives while uids/tokens shrink. A later extend() then appends the
# next request's processors behind its own row index.
# ---------------------------------------------------------------------------
_original_generation_batch_filter = GenerationBatch.filter


def _patched_generation_batch_filter(self, keep):
    lps = self.logits_processors
    lps_inert = not lps or not any(lps)
    if lps is None:
        self.logits_processors = []
    _original_generation_batch_filter(self, keep)
    if lps_inert:
        self.logits_processors = [[] for _ in keep]


GenerationBatch.filter = _patched_generation_batch_filter


# ---------------------------------------------------------------------------
# Monkey-patch BatchGenerator._next to skip the unconditional
# mx.clear_cache() every 512 steps. The fusion-mlx scheduler already
# performs fragmentation-aware cache clearing with its own cadence
# (decode_clear_interval, mlx_cache_cleanup_interval), so the stock
# clear_cache is redundant and adds an unwanted GPU sync barrier that
# stalls decode throughput.
# ---------------------------------------------------------------------------
try:
    from mlx_lm.generate import BatchGenerator as _BatchGenerator

    _original_batch_generator_next = _BatchGenerator._next

    def _patched_batch_generator_next(self):
        # Stock _next does:
        #   self._steps_counter += 1
        #   if self._steps_counter % 512 == 0: mx.clear_cache()
        # We pre-adjust the counter so that after the increment
        # it's never a multiple of 512, preventing clear_cache.
        # The fusion scheduler's own fragmentation-aware clearing
        # (sched_step.py) handles cache management instead.
        if (self._steps_counter + 1) % 512 == 0:
            self._steps_counter += 1  # skip past the modulo trigger
        return _original_batch_generator_next(self)

    _BatchGenerator._next = _patched_batch_generator_next
    logger.debug("Patched BatchGenerator._next to skip stock clear_cache")

    # -----------------------------------------------------------------------
    # Monkey-patch BatchGenerator.next_generated() to skip the
    # `with mx.stream(self._stream):` context manager. The fusion-mlx
    # scheduler creates the BatchGenerator on the MLX executor thread
    # which already has the correct thread-local stream, so the
    # __enter__/__exit__ overhead (~50-100us per step) is pure waste.
    # -----------------------------------------------------------------------
    _original_next_generated = getattr(_BatchGenerator, "next_generated", None)
    if _original_next_generated is not None:
        import inspect

        try:
            _ng_src = inspect.getsource(_original_next_generated)
        except (TypeError, OSError):
            _ng_src = ""
        # Only patch if the source contains the stream context manager.
        if "mx.stream" in _ng_src:

            def _patched_next_generated(self):
                # Inline the stock logic without the stream context manager.
                # Stock: with mx.stream(self._stream): <body>
                # We just run <body> directly since we're already on the
                # correct stream thread.
                while True:
                    prompt_resp, gen_resp = self._next()
                    if gen_resp or prompt_resp:
                        yield from gen_resp
                        return
                    if not self._prompt_batch and not self._unprocessed_sequences:
                        yield from gen_resp
                        return

            _BatchGenerator.next_generated = _patched_next_generated
            logger.debug(
                "Patched BatchGenerator.next_generated to skip mx.stream context"
            )

    # Also patch BatchGenerator._next to skip its own mx.stream wrapper.
    # The _next method wraps its body in `with mx.stream(self._stream):`
    # which is redundant when already on the correct stream thread.
    _original_inner_next = _BatchGenerator._next

    def _patched_inner_next_no_stream(self):
        # We already patched _next above for clear_cache skipping.
        # That patch calls _original_batch_generator_next which still
        # has the stream context manager. This second-level patch
        # removes it by calling the stock _next's inner logic directly.
        # Since we can't easily unwrap the context manager, we rely on
        # the fact that setting the stream at thread init makes the
        # context manager a no-op in practice (same stream object).
        # The real win is in next_generated() above.
        return _patched_batch_generator_next(self)

    # Don't double-patch; the clear_cache patch is already applied.
    # The stream overhead in _next is smaller than in next_generated
    # since _next is called once per step, not in a while loop.
except ImportError:
    pass


# Monkey-patch TurboQuantKVCache.merge so _merge_caches() works
try:
    from mlx_vlm.turboquant import TurboQuantKVCache as _TQCache

    from ..turboquant_kv import BatchTurboQuantKVCache as _BTQCache

    if not hasattr(_TQCache, "merge"):
        _TQCache.merge = _BTQCache.merge
except ImportError:
    pass


# Monkey-patch ChunkedKVCache for Llama-4 (Scout / Maverick): mlx_lm's
# ChunkedKVCache lacks the batch-aware methods (`merge`, `filter`, `extract`,
# `size`, `extend`) that BatchGenerator's continuous-batching code path
# expects, so any chat completion targeting a Llama-4 model raises
# `Cache corruption not recoverable: <ChunkedKVCache> does not yet support
# batching with history` and returns 500.
#
# Real continuous batching with chunked attention is unimplemented upstream;
# this patch installs batch=1 pass-throughs so serialized requests work.
# Run the server with `--max-concurrent-requests 1` to honor the assumption.
try:
    from mlx_lm.models.cache import ChunkedKVCache as _CKVCache

    _ckvcache_methods_skipped: list[str] = []

    if not hasattr(_CKVCache, "merge"):

        @classmethod
        def _ckvcache_merge_passthrough(cls, caches):
            if len(caches) == 1:
                return caches[0]
            raise NotImplementedError(
                "ChunkedKVCache.merge for batch_size > 1 is not implemented. "
                "Run with --max-concurrent-requests 1 when serving Llama-4."
            )

        _CKVCache.merge = _ckvcache_merge_passthrough
    else:
        _ckvcache_methods_skipped.append("merge")

    if not hasattr(_CKVCache, "filter"):

        def _ckvcache_filter_passthrough(self, batch_indices):
            try:
                n = len(batch_indices)
            except TypeError:
                n = int(getattr(batch_indices, "shape", (0,))[0] or 0)
            if n == 0:
                self.keys = None
                self.values = None
                self.offset = 0
                self.start_position = 0
                return
            if n == 1:
                return
            raise NotImplementedError(
                f"ChunkedKVCache.filter with batch_size={n} > 1 is not "
                "implemented. Run with --max-concurrent-requests 1 when "
                "serving Llama-4."
            )

        _CKVCache.filter = _ckvcache_filter_passthrough
    else:
        _ckvcache_methods_skipped.append("filter")

    if not hasattr(_CKVCache, "extract"):

        def _ckvcache_extract_passthrough(self, idx):
            return self

        _CKVCache.extract = _ckvcache_extract_passthrough
    else:
        _ckvcache_methods_skipped.append("extract")

    if not hasattr(_CKVCache, "size"):

        def _ckvcache_size(self):
            return max(0, self.offset - self.start_position)

        _CKVCache.size = _ckvcache_size
    else:
        _ckvcache_methods_skipped.append("size")

    if not hasattr(_CKVCache, "extend"):

        def _ckvcache_extend_passthrough(self, other):
            if other is None or other.empty():
                return
            if self.empty():
                self.keys = other.keys
                self.values = other.values
                self.offset = other.offset
                self.start_position = other.start_position
                return
            raise NotImplementedError(
                "ChunkedKVCache.extend across non-empty caches is not "
                "supported. Run with --max-concurrent-requests 1."
            )

        _CKVCache.extend = _ckvcache_extend_passthrough
    else:
        _ckvcache_methods_skipped.append("extend")

    if _ckvcache_methods_skipped:
        # Upstream may have landed implementations between mlx_lm upgrades.
        # Surface which ones so a regression in Llama-4 batching is visible
        # to operators without diffing the patch against installed mlx_lm.
        logger.info(
            "ChunkedKVCache patch: methods already present upstream, " "skipped: %s",
            ", ".join(_ckvcache_methods_skipped),
        )
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Monkey-patch PromptProcessingBatch.prompt to set mRoPE deltas before the
# prompt processing loop.  Without this, batched VLM prompt processing
# (e.g. the 1-token final prompt after external prefill) would use
# per-request offsets without rope_deltas, corrupting attention masks
# for concurrent VLM requests.
# ---------------------------------------------------------------------------
_original_ppb_prompt = PromptProcessingBatch.prompt


def _patched_ppb_prompt(self, tokens):
    model = self.model
    if (
        getattr(model, "_uses_mrope", False)
        and getattr(model, "_uid_rope_deltas", None)
        and self.uids
    ):
        deltas = [model._uid_rope_deltas.get(uid, 0.0) for uid in self.uids]
        model.set_batch_rope_deltas(mx.array(deltas))
    return _original_ppb_prompt(self, tokens)


PromptProcessingBatch.prompt = _patched_ppb_prompt
