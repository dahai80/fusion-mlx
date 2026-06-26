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
import time
from typing import Any

logger = logging.getLogger(__name__)


def _log_completion_summary(request, output):
    """Log Ollama-style request completion summary."""
    prompt_tokens = output.prompt_tokens
    completion_tokens = output.completion_tokens
    cached_tokens = output.cached_tokens
    finish_reason = output.finish_reason or "unknown"

    # Calculate timing
    now = time.monotonic()
    total_time = (now - request.arrival_time) if request.arrival_time else 0
    gen_time = (now - request.generation_started_at) if request.generation_started_at else total_time
    ttft = (now - request.first_token_at) if request.first_token_at else 0

    # Token-per-second (Ollama style: ms/token)
    prompt_ms_per_token = (ttft * 1000 / prompt_tokens) if prompt_tokens > 0 else 0
    completion_ms_per_token = ((gen_time - ttft) * 1000 / max(completion_tokens - 1, 1)) if gen_time > ttft else 0
    prompt_tps = (prompt_tokens / ttft) if ttft > 0 else 0
    completion_tps = (completion_tokens / max(gen_time - ttft, 0.001)) if gen_time > ttft else 0

    logger.info(
        "| completed | prompt_tokens=%d | completion_tokens=%d | cached_tokens=%d | "
        "ttft=%.2fs | prompt_eval=%.1fms/token | completion_eval=%.1fms/token | "
        "prompt_tps=%.1f | completion_tps=%.1f | total=%.1fs | finish=%s",
        prompt_tokens,
        completion_tokens,
        cached_tokens,
        ttft,
        prompt_ms_per_token,
        completion_ms_per_token,
        prompt_tps,
        completion_tps,
        total_time,
        finish_reason,
    )

import mlx.core as mx

from ..exceptions import is_cache_corruption_error
from ..prefill_progress import get_prefill_tracker
from ..request import RequestOutput, RequestStatus

# Module-level alias so Scheduler.__init__ can fall back to mlx-lm's default
# stream when no per-engine stream is provided.
from .helpers import (
    _safe_sync_stream,
)


def _process_batch_responses(
    self, responses: list[Any]
) -> tuple[list[RequestOutput], set[str]]:
    """
    Process responses from BatchGenerator.

    Args:
        responses: List of BatchGenerator.Response objects

    Returns:
        Tuple of (outputs, finished_request_ids)
    """
    outputs = []
    finished_ids = set()

    step_now = time.monotonic()
    for response in responses:
        request_id = self.uid_to_request_id.get(response.uid)
        if request_id is None:
            continue

        request = self.running.get(request_id)
        if request is None:
            continue

        request.last_activity_at = step_now

        # Release VLM embeddings after first decode token (prefill is done)
        if request.vlm_inputs_embeds is not None:
            request.vlm_inputs_embeds = None
            request.vlm_extra_kwargs = None

        # Check finish reason first - don't include EOS token in output
        # (following mlx-lm's batch_generate behavior)
        is_stop = response.finish_reason == "stop"
        is_length = response.finish_reason == "length"
        is_finished = response.finish_reason is not None

        # Only append token if not stopping due to EOS token
        new_text = ""

        # Check if this request uses a protocol-specific output parser
        parser_session = self._get_output_parser_session(request_id)

        if parser_session is not None:
            parser_result = parser_session.process_token(response.token)
            new_text = parser_result.stream_text
            if parser_result.visible_text:
                request.output_text += parser_result.visible_text

            # Parser-defined stop token can override finish reason
            if parser_result.is_stop and not is_finished:
                is_finished = True
                is_stop = True

            should_record_token = (
                parser_result.record_token
                if parser_result.record_token is not None
                else not is_stop
            )
            if should_record_token:
                request.append_output_token(response.token)

        elif not is_stop:
            # Standard processing without a protocol parser
            request.append_output_token(response.token)

            # Decode the new token using streaming detokenizer for proper UTF-8 handling
            detokenizer = self._get_detokenizer(request_id)
            if detokenizer is not None:
                detokenizer.add_token(response.token)
                new_text = detokenizer.last_segment
            else:
                # Fallback to single-token decode
                new_text = self.tokenizer.decode([response.token])

            # Text-level stop-string fallback. Catches BPE edge cases
            # where the tokenized stop sequence does not match the
            # model's actual output tokens (e.g. " delta" vs "delta").
            # Only scans the tail to keep cost O(stop_len) per step.
            stop_strs = request.sampling_params.stop or []
            if stop_strs and not is_finished and detokenizer is not None:
                full_text = detokenizer.text
                prev_len = len(full_text) - len(new_text)
                for ss in stop_strs:
                    if not ss:
                        continue
                    scan_start = max(0, prev_len - len(ss) + 1)
                    idx_in_tail = full_text.find(ss, scan_start)
                    if idx_in_tail < 0:
                        continue
                    is_finished = True
                    is_stop = True
                    response.finish_reason = "stop"
                    if idx_in_tail >= prev_len:
                        new_text = new_text[: idx_in_tail - prev_len]
                    else:
                        new_text = ""
                    break

        # Prepend <think> tag for first chunk if this is a reasoning model
        # (skip when a protocol parser already manages reasoning formatting)
        if parser_session is None and getattr(request, "needs_think_prefix", False):
            if not getattr(request, "think_prefix_sent", False):
                think_tag = getattr(self.tokenizer, "think_start", "<think>")
                new_text = think_tag + "\n" + new_text
                request.think_prefix_sent = True

        # Immediately discard logprobs if not requested to free memory (~800KB per response)
        # This prevents accumulation of large MLX arrays during streaming
        if (
            hasattr(response, "logprobs")
            and response.logprobs is not None
            and not request.sampling_params.logprobs
        ):
            response.logprobs = None

        # Track first token arrival time
        if request.num_output_tokens == 1 and request.first_token_at is None:
            request.first_token_at = time.monotonic()

        # Create output
        output = RequestOutput(
            request_id=request_id,
            new_token_ids=[response.token] if not is_stop else [],
            new_text=new_text,
            output_token_ids=list(request.output_token_ids),
            prompt_tokens=request.num_prompt_tokens,
            completion_tokens=request.num_output_tokens,
            cached_tokens=request.cached_tokens,
        )

        if not is_finished:
            self._maybe_capture_boundary_snapshot(request, response.uid)

        # Handle finished requests
        if is_finished:
            if is_stop:
                request.set_finished(RequestStatus.FINISHED_STOPPED)
            elif is_length:
                request.set_finished(RequestStatus.FINISHED_LENGTH_CAPPED)

            output.finished = True
            output.finish_reason = response.finish_reason
            finished_ids.add(request_id)

             # Ollama-style completion summary
            _log_completion_summary(request, output)

            if parser_session is not None:
                final_result = parser_session.finalize()
                if final_result.stream_text:
                    output.new_text += final_result.stream_text
                if final_result.visible_text:
                    request.output_text += final_result.visible_text
                if final_result.output_text_prefix:
                    request.output_text = (
                        final_result.output_text_prefix + request.output_text
                    )
                if final_result.tool_calls:
                    output.tool_calls = final_result.tool_calls
                if final_result.finish_reason:
                    output.finish_reason = final_result.finish_reason
                output.output_text = request.output_text
            else:
                # Standard finalization without a protocol parser
                # Finalize detokenizer to flush any remaining bytes
                detokenizer = self._get_detokenizer(request_id)
                if detokenizer is not None:
                    detokenizer.finalize()
                    final_segment = detokenizer.last_segment
                    if final_segment:
                        output.new_text += final_segment

                # Decode full output
                output.output_text = self.tokenizer.decode(request.output_token_ids)
                request.output_text = output.output_text

                # Trim accumulated output text at the first stop string
                # match so non-streaming responses do not include the
                # stop sequence itself (matches OpenAI semantics).
                if is_stop:
                    stop_strs = request.sampling_params.stop or []
                    for ss in stop_strs:
                        if not ss:
                            continue
                        cut = output.output_text.find(ss)
                        if cut >= 0:
                            output.output_text = output.output_text[:cut]
                            request.output_text = output.output_text
                            break

            # Extract cache for future reuse.
            # In the new API, prompt_cache is a direct value (not callable).
            raw_cache = getattr(response, "prompt_cache", None)
            if raw_cache is not None:
                try:
                    # SpecPrefill: sparse KV data can't be stored in
                    # paged cache (hash mismatch with full token IDs).
                    if request.specprefill_indices is not None:
                        raw_cache = None

                    # For paged cache, extract actual tensor states
                    # This allows cache to survive BatchGenerator recreation
                    elif self.block_aware_cache is not None:
                        extracted_cache, model_cache_config = (
                            self._extract_cache_states(raw_cache)
                        )
                        if extracted_cache:
                            request._extracted_cache = extracted_cache
                            request._model_cache_config = model_cache_config
                            logger.debug(
                                f"Extracted {len(extracted_cache)} layer states "
                                f"for request {request_id}"
                            )
                    else:
                        # Standard cache stores object references
                        request._extracted_cache = raw_cache
                        request._model_cache_config = None
                except Exception as e:
                    logger.debug(f"Failed to extract cache for {request_id}: {e}")

            self.total_completion_tokens += request.num_output_tokens
            self.num_requests_processed += 1

            logger.info(
                "Request %s finished: %s, %d tokens",
                request_id, response.finish_reason, request.num_output_tokens,
                )
            logger.debug(
                f"Request {request_id} finished: {response.finish_reason}, "
                f"{request.num_output_tokens} tokens"
            )
            logger.log(
                5, "Request %s generated text:\n%s", request_id, output.output_text
            )

        outputs.append(output)

    return outputs, finished_ids

def _release_paged_cache_for_request(    self, request_id: str) -> None:
    """Drop a request's paged-cache footprint on rejection paths.

    ``add_request`` routes through ``block_aware_cache.fetch_cache``
    which records the request in ``_request_tables`` and increments
    ref counts on every prefix-matched paged-cache block. The normal
    completion path releases that state in ``_cleanup_finished``;
    the prefill-rejection paths in ``_advance_chunked_prefills`` /
    ``_schedule_waiting`` must do the same or rejected requests
    leak block refs (pinning the paged cache and compounding the
    very memory pressure that triggered the rejection) and orphan
    ``_request_tables`` entries.
    """
    if self.block_aware_cache is not None:
        self.block_aware_cache.release_cache(request_id)
    elif self.paged_cache_manager is not None:
        self.paged_cache_manager.delete_block_table(request_id)

def _cleanup_finished(    self, finished_ids: set[str]) -> None:
    """Clean up finished requests and store caches for reuse."""
    # Synchronize pending engine stream operations before cache storage.
    # store_cache -> mx.save_safetensors triggers implicit mx.eval() which
    # can conflict with async Metal operations on the generation stream.
    if finished_ids:
        with self._phase_timer("cleanup_finished_sync"):
            _safe_sync_stream(self._stream)

    # SpecPrefill: restore original RoPE if active request finished
    for rid in finished_ids:
        self._cleanup_specprefill(rid)

    # Remove finished requests from prefill progress tracker.
    tracker = get_prefill_tracker()
    for rid in finished_ids:
        tracker.remove(rid)

    for request_id in finished_ids:
        request = self.running.get(request_id)

        # Guard: skip if request was already removed from running by another path
        # (e.g. concurrent abort, preemption, or recovery). This prevents
        # double-cleanup and inconsistent state (dict iteration safety fix).
        if request is None:
            logger.warning(
                f"_cleanup_finished: request {request_id} not in running, "
                f"skipping cleanup (likely removed by abort/preemption)"
            )
            continue

        # Store cache for future reuse (G2-async): submit to background
        # executor so the post-finish 28GB+ memcpy doesn't block response
        # streaming. The inference thread does mx.synchronize +
        # boundary merge + a single batched mx.eval here; the worker
        # handles _extract_tensor_bytes (CPU memcpy) + index/queue
        # registration. batch_generator.remove(uid) is deferred and
        # picked up at the next step's _drain_pending_async_removes.
        store_future = None
        if request is not None and request.prompt_token_ids:
            if self.block_aware_cache is not None:
                if (
                    hasattr(request, "_extracted_cache")
                    and request._extracted_cache is not None
                ):
                    try:
                        full_token_sequence = list(request.prompt_token_ids) + list(
                            request.output_token_ids
                        )
                        # For reasoning models, only cache prompt tokens.
                        # Output contains <think> tokens that the API layer
                        # strips before the next turn, so they never match.
                        if getattr(request, "needs_think_prefix", False):
                            cacheable_sequence = list(request.prompt_token_ids)
                        else:
                            cacheable_sequence = full_token_sequence
                        token_sequence_to_store = cacheable_sequence
                        cache_to_store = request._extracted_cache
                        model_cache_config = getattr(
                            request, "_model_cache_config", None
                        )
                        intermediate_snapshots = None

                        # Inference-thread store_cache prep, timed as
                        # three sub-phases (boundary / collect / dispatch)
                        # mirroring boundary_capture_* granularity.
                        # async_eval dispatches KV array materialization
                        # without blocking; the worker calls
                        # mx.synchronize() to wait before extracting
                        # bytes.
                        with mx.stream(self._stream):
                            with self._phase_timer("store_cache_main_boundary"):
                                boundary_override = self._get_boundary_store_override(
                                    request_id,
                                    cacheable_sequence,
                                )
                                if boundary_override is not None:
                                    (
                                        token_sequence_to_store,
                                        boundary_cache,
                                        boundary_model_config,
                                        intermediate_snapshots,
                                    ) = boundary_override
                                    cache_to_store = (
                                        self._merge_boundary_with_full_cache(
                                            boundary_cache, request._extracted_cache
                                        )
                                    )
                                    if boundary_model_config is not None:
                                        model_cache_config = boundary_model_config
                                    logger.info(
                                        f"Using boundary cache snapshot for {request_id}: "
                                        f"storing {len(token_sequence_to_store)}/"
                                        f"{len(full_token_sequence)} tokens "
                                        f"(skipping trailing partial block, "
                                        f"{len(intermediate_snapshots) if intermediate_snapshots else 0} "
                                        f"intermediate snapshots)"
                                    )
                            with self._phase_timer("store_cache_main_collect"):
                                pre_eval_arrays = (
                                    self._collect_arrays_from_extracted_cache(
                                        cache_to_store
                                    )
                                )
                            with self._phase_timer("store_cache_main_dispatch"):
                                if pre_eval_arrays:
                                    mx.async_eval(*pre_eval_arrays)

                        if self._store_cache_executor is not None:
                            # Hand the store-cache write to the background
                            # executor without ever blocking the generation
                            # step. The gate only counts in-flight writes;
                            # backpressure is applied at admission in
                            # _schedule_waiting (in_flight >= cap defers new
                            # prefills) so cache persistence never stalls
                            # token generation (#1496). note_submitted is
                            # called before submit so a fast worker whose
                            # done callback fires immediately still
                            # decrements a counted slot.
                            gate = self._store_cache_gate
                            if gate is not None:
                                gate.note_submitted()
                            try:
                                store_future = self._store_cache_executor.submit(
                                    self._async_store_cache_worker,
                                    request_id,
                                    token_sequence_to_store,
                                    cache_to_store,
                                    model_cache_config,
                                    intermediate_snapshots,
                                    request.vlm_extra_keys_for_cache,
                                    request.vlm_extra_key_token_start_for_cache,
                                    request.vlm_extra_key_ranges_for_cache,
                                )
                            except BaseException:
                                if gate is not None:
                                    gate.note_done()
                                raise
                            if gate is not None:
                                store_future.add_done_callback(
                                    lambda _f, g=gate: g.note_done()
                                )
                            self._inflight_store_futures[request_id] = store_future
                        else:
                            # Executor unavailable — synchronous fallback.
                            self._async_store_cache_worker(
                                request_id,
                                token_sequence_to_store,
                                cache_to_store,
                                model_cache_config,
                                intermediate_snapshots,
                                request.vlm_extra_keys_for_cache,
                                request.vlm_extra_key_token_start_for_cache,
                                request.vlm_extra_key_ranges_for_cache,
                            )
                        logger.debug(
                            f"Submitted async store_cache for {request_id} "
                            f"({len(token_sequence_to_store)} tokens, "
                            f"{len(full_token_sequence)} total: "
                            f"{len(request.prompt_token_ids)} prompt + "
                            f"{len(request.output_token_ids)} output)"
                        )
                    except Exception as e:
                        logger.debug(
                            f"Failed to submit async store for {request_id}: {e}"
                        )
                else:
                    # No extracted_cache to store, but ensure block leak guard.
                    block_table = None
                    if self.paged_cache_manager:
                        block_table = self.paged_cache_manager.get_block_table(
                            request_id
                        )
                        if block_table is None and hasattr(request, "block_table"):
                            block_table = request.block_table
                    if block_table and self.paged_cache_manager:
                        self.paged_cache_manager.release_for_eviction(
                            block_table.block_ids
                        )
                    self.block_aware_cache.clear_request_entry(request_id)

        # Remove from running
        if request_id in self.running:
            del self.running[request_id]

        # batch_generator.remove(uid): defer until the async store_cache
        # worker finishes so the BatchKVCache slot isn't reused while the
        # worker is still reading buffer references via cache_to_store.
        # _drain_pending_async_removes (next step) handles the actual
        # mx.synchronize + remove + uid_maps cleanup. If we have no async
        # store (no extracted_cache, executor missing, fallback fail),
        # fall back to immediate remove for back-compat behavior.
        if request_id in self.request_id_to_uid:
            uid = self.request_id_to_uid[request_id]
            if store_future is not None:
                self._pending_async_removes.append((uid, request_id, store_future))
            else:
                # Synchronize in-flight GPU work before modifying batch state.
                # batch_generator.remove() triggers lazy KV cache array slicing
                # (BatchKVCache.filter) that replaces references to arrays still
                # used by in-flight Metal command buffers from the previous
                # batch_generator.next() call.  Without this barrier the Metal
                # driver can hit 'completeMemory() prepare count underflow'.
                _safe_sync_stream(self._stream)
                self._remove_uid_from_active_batch(uid)
                if hasattr(self.model, "unregister_rope_delta"):
                    self.model.unregister_rope_delta(uid)
                if uid in self.uid_to_request_id:
                    del self.uid_to_request_id[uid]
                del self.request_id_to_uid[request_id]

        # Clean up streaming detokenizer
        self._cleanup_detokenizer(request_id)

        # Clean up protocol-specific output parser session
        self._cleanup_output_parser_session(request_id)

        # Clean up VLM adapter state (position_ids, rope_deltas, pending embeddings)
        if hasattr(self.model, "clear_vlm_position_state"):
            self.model.clear_vlm_position_state()
        if hasattr(self.model, "clear_pending_embeddings"):
            self.model.clear_pending_embeddings()

        # Drop any boundary snapshot for this request. The in-memory
        # dict pop is safe — the async store worker holds its own
        # reference to the snapshot dict via _BoundarySnapshotProvider.
        self._boundary_cache_snapshots.pop(request_id, None)
        # cleanup_request rmtree's the on-disk snapshot directory and
        # races the worker's boundary_snapshot_store.load() calls. If
        # an async store_future is in flight, defer cleanup until the
        # worker finishes (handled in _drain_pending_async_removes).
        if self._boundary_snapshot_store is not None and store_future is None:
            self._boundary_snapshot_store.cleanup_request(request_id)

        # Track as finished
        self.finished_req_ids.add(request_id)

        # Remove from requests dict to prevent memory leak.
        # When async store_cache is in flight, keep _extracted_cache alive
        # until the worker finishes — the worker holds a reference via
        # cache_to_store argument, but request._extracted_cache pointing
        # to the same data is the canonical owner. We pop here only when
        # no future is pending; the future's done callback (or
        # _drain_pending_async_removes) clears the request later.
        if store_future is None:
            req_to_remove = self.requests.pop(request_id, None)
            if req_to_remove is not None:
                req_to_remove._extracted_cache = None
                req_to_remove.prompt_cache = None
        else:
            # Drop request from running but keep in self.requests so the
            # async worker keeps the cache buffers alive via reachability.
            # Cleanup happens in _drain_pending_async_removes.
            pass

    # Emit phase timing diagnostics when accumulated counts are meaningful.
    # Helps diagnose cache-on overhead (boundary capture / store_cache /
    # hot cache eviction). Logged at info level so operators can see it
    # without enabling debug.
    if finished_ids and self._phase_total_ms:
        stats_parts = []
        for phase, total_ms in sorted(self._phase_total_ms.items()):
            count = self._phase_count.get(phase, 0)
            if count == 0:
                continue
            stats_parts.append(f"{phase}={total_ms:.1f}ms/{count}")
        if stats_parts:
            logger.info("Cache phase timings: %s", ", ".join(stats_parts))

    # Schedule deferred Metal cache cleanup after request completion.
    if finished_ids:
        # Schedule deferred Metal cache cleanup instead of clearing immediately.
        # Immediate mx.clear_cache() after request completion races with IOKit's
        # asynchronous completeMemory() callbacks — the kernel-level GPU memory
        # reference counting can still be in-flight even after mx.synchronize()
        # returns, causing 'prepare count underflow' kernel panics (#435).
        # Deferring by _DEFERRED_CLEAR_DELAY generation steps (~10-40 ms) gives
        # IOKit time to process callbacks while still reclaiming buffers fast
        # enough to prevent TTFT spikes from pool bloat (#411).
        #
        # Use max() so that concurrent completions (max_num_seqs > 1) each get
        # a full _DEFERRED_CLEAR_DELAY window counted from *their own* finish
        # step.  The old "only set if None" guard meant the second request's
        # window was anchored to the first request's finish step, allowing the
        # second request's KV cache blocks to be re-allocated before IOKit
        # finished their completeMemory() callbacks (#557).
        target = self._step_counter + self._DEFERRED_CLEAR_DELAY
        if self._deferred_clear_at is None or target > self._deferred_clear_at:
            self._deferred_clear_at = target

def _is_cache_corruption_error(    self, error: Exception) -> bool:
    """Check if an error indicates cache corruption."""
    return is_cache_corruption_error(error)

def _recover_from_cache_error(self) -> None:
    """Recover from cache corruption error."""
    # Clear batch generator (this is the source of the corruption)
    self.batch_generator = None
    self._current_sampler_params = None
    self._boundary_cache_snapshots.clear()
    if self._boundary_snapshot_store is not None:
        self._boundary_snapshot_store.cleanup_all()
    self._boundary_snapshot_required = None

    # Clear stale VLM position state to prevent re-corruption on retry
    if hasattr(self.model, "clear_vlm_position_state"):
        self.model.clear_vlm_position_state()

    # Clear pending VLM embeddings
    if hasattr(self.model, "clear_pending_embeddings"):
        self.model.clear_pending_embeddings()

    # Clear caches
    if self.block_aware_cache is not None:
        self.block_aware_cache.clear()
    self._cache_rate_tracker.clear()

    # Clear UID mappings
    self.request_id_to_uid.clear()
    self.uid_to_request_id.clear()

    # Cancel any pending deferred Metal cache clear
    self._deferred_clear_at = None

    # Clear detokenizer state to prevent contamination after recovery
    self._request_detokenizers.clear()

    # Clear protocol-specific output parser sessions
    self._output_parser_sessions.clear()

    logger.info("Cache recovery completed")

def _reschedule_running_requests(
    self, is_corruption: bool = False, max_corruption_retries: int = 3
) -> list[str]:
    """Move running requests back to waiting queue for retry.

    Args:
        is_corruption: If True, increment corruption retry counter and
            fail requests that exceed max_corruption_retries.
        max_corruption_retries: Max corruption retries before failing a request.

    Returns:
        List of request IDs that exceeded max retries (corruption only).
    """
    failed_ids: list[str] = []
    count = 0
    for request_id, request in list(self.running.items()):
        if is_corruption:
            request.cache_corruption_retries += 1
            if request.cache_corruption_retries > max_corruption_retries:
                failed_ids.append(request_id)
                del self.running[request_id]
                # Clean up from requests dict (prevent memory leak)
                req = self.requests.pop(request_id, None)
                if req is not None:
                    req._extracted_cache = None
                    req.prompt_cache = None
                continue

        # Reset scheduling state
        request.status = RequestStatus.WAITING
        request.batch_uid = None

        # Reset cache state
        request.prompt_cache = None
        request.cached_tokens = 0
        request.remaining_tokens = request.prompt_token_ids
        request.block_table = None
        request.shared_prefix_blocks = 0

        # Reset generation output (prevent duplicate tokens on re-prefill)
        request.output_token_ids = []
        request.output_text = ""
        request.num_computed_tokens = 0

        # Reset extracted cache (prevent GPU memory leak)
        request._extracted_cache = None
        request._model_cache_config = None

        # Reset reasoning model state
        request.think_prefix_sent = False

        # Move to waiting queue (at front for priority)
        self.waiting.appendleft(request)
        del self.running[request_id]
        count += 1

    if count > 0:
        logger.info(f"Rescheduled {count} requests for re-prefill")
    return failed_ids
