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
import gc
import logging
import math
import time

import mlx.core as mx

logger = logging.getLogger(__name__)
from typing import Any

from ..request import Request, RequestOutput, RequestStatus

# Module-level alias so Scheduler.__init__ can fall back to mlx-lm's default
# stream when no per-engine stream is provided.
from .config import SchedulerOutput
from .helpers import (
    _should_clear_on_fragmentation,
    _sync_and_clear_cache,
)
from .monkeypatches import _unregister_uid_rows_for_model
from .types import (
    _PrefillAbortedError,
)


def step(self) -> SchedulerOutput:
    output = SchedulerOutput()
    self._step_counter += 1

    # --- Pure-decode fast path ---
    # When there are no waiting/prefilling requests and only 1 running request,
    # skip all scheduling, memory checks, admin snapshots, and contention
    # detection. The only work is: bg._next() + process responses.
    if (
        not self.waiting
        and not self.prefilling
        and len(self.running) == 1
        and self.batch_generator is not None
        and not self._vlm_mtp_active
        and not self._pending_abort_ids
        and not self._pending_async_removes
    ):
        self._pure_decode_count = getattr(self, "_pure_decode_count", 0) + 1
        if self._pure_decode_count <= 3 or self._pure_decode_count % 50 == 0:
            logger.info(
                "step(%d): pure_decode path (#%d)",
                self._step_counter,
                self._pure_decode_count,
            )
        return self._step_pure_decode(output)

    # --- Full step path ---
    logger.debug(
        "step(%d): waiting=%d, running=%d, prefilling=%d",
        self._step_counter,
        len(self.waiting),
        len(self.running),
        len(self.prefilling),
    )

    self._process_pending_aborts()
    self._drain_pending_async_removes()

    if (
        self.memory_monitor is not None
        and self._step_counter % self.config.memory_check_interval == 0
    ):
        self._check_memory_pressure()

    try:
        # Advance in-flight chunked prefills (one chunk per request).
        # Must run before _schedule_waiting() so that completing prefills
        # are inserted into BatchGenerator before the decode step.
        chunked_scheduled: list[Request] = []
        chunked_rejected: list[RequestOutput] = []
        if self.prefilling:
            self._advance_chunked_prefills(chunked_scheduled, chunked_rejected)

        # Schedule waiting requests
        scheduled, rejected = self._schedule_waiting()
        # Merge chunked-prefill completions into the scheduled list.
        if chunked_scheduled:
            scheduled = chunked_scheduled + scheduled
        output.scheduled_request_ids = [r.request_id for r in scheduled]
        output.num_scheduled_tokens = sum(r.num_prompt_tokens for r in scheduled)
        if chunked_rejected:
            output.outputs.extend(chunked_rejected)
            output.has_work = True
        if rejected:
            output.outputs.extend(rejected)
            output.has_work = True

        # Run generation step if we have running requests.
        # Use next_generated() which returns only GenerationBatch.Response
        # objects (prefill is handled externally before insert).
        if (self.batch_generator is not None or self._vlm_mtp_active) and self.running:
            _decode_t0 = time.perf_counter()
            if self.batch_generator is not None:
                # Fast path for pure decode: when no prompts are pending,
                # _next() returns (prompt_resp=[], gen_resp=[...]) in one
                # call — the while-True loop in next_generated() is wasted
                # overhead. Skip the generator entirely and call _next()
                # directly.
                bg = self.batch_generator
                has_pending = getattr(bg, "_prompt_batch", None) or getattr(
                    bg, "_unprocessed_sequences", None
                )
                if not has_pending:
                    _, responses = bg._next()
                else:
                    responses = list(bg.next_generated())
            else:
                responses = []
            _decode_dt = time.perf_counter() - _decode_t0
            # GPU contention detection: track decode step time in rolling
            # window and compute CV. Bimodal latency (fast ~80ms vs slow
            # ~400ms) indicates competing GPU processes.
            self._step_time_window.append(_decode_dt)
            if len(self._step_time_window) > self._step_time_window_size:
                self._step_time_window.pop(0)
            if len(self._step_time_window) >= 8:
                _times = self._step_time_window
                _mean = sum(_times) / len(_times)
                if _mean > 0:
                    _var = sum((t - _mean) ** 2 for t in _times) / len(_times)
                    _cv = math.sqrt(_var) / _mean
                    self._contention_detected = _cv > self._contention_cv_threshold
                    if self._contention_detected:
                        _log_step_delta = (
                            self._step_counter - self._last_contention_log_step
                        )
                        if _log_step_delta >= self._contention_log_interval:
                            self._last_contention_log_step = self._step_counter
                            logger.warning(
                                "step(%d): GPU contention detected — "
                                "decode CV=%.1f%% (mean=%.1fms, std=%.1fms, "
                                "n=%d). Competing GPU processes may cause "
                                "3-4x slowdown. Consider stopping other "
                                "MLX/GPU workloads.",
                                self._step_counter,
                                _cv * 100,
                                _mean * 1000,
                                math.sqrt(_var) * 1000,
                                len(_times),
                            )
            # Drive vlm_mtp generators alongside BatchGenerator. Order
            # matters only for log determinism; _process_batch_responses
            # is per-uid.
            if self._vlm_mtp_active:
                responses.extend(self._step_vlm_mtp())
            output.has_work = True

            if responses:
                outputs, finished_ids = self._process_batch_responses(responses)
                output.outputs.extend(outputs)
                output.finished_request_ids = finished_ids
                self._cleanup_finished(finished_ids)
                if finished_ids:
                    logger.info(
                        "step(%d): finished=%s", self._step_counter, finished_ids
                    )
            elif self.running and not scheduled:
                # Empty responses with running requests = stale.
                # Model batch cleared them silently (finished length/EOS
                # without returning a final response token). Reschedule
                # so they don't rot in running forever.
                #
                # Skip when we just scheduled requests: the first decode
                # step after insert() may return empty gen_responses
                # because prefill just completed and the batch generator
                # hasn't produced a generation token yet. This is normal,
                # not stale — the next step will produce responses.
                logger.warning(
                    "step(%d): empty responses with %d running requests — "
                    "rescheduling as stale",
                    self._step_counter,
                    len(self.running),
                )
                for rid in list(self.running.keys()):
                    req = self.running.pop(rid)
                    # Clean up UID mapping so re-insert gets fresh state
                    old_uid = self.request_id_to_uid.pop(rid, None)
                    if old_uid is not None:
                        self.uid_to_request_id.pop(old_uid, None)
                    # Reset output state to prevent duplicate tokens on re-prefill
                    req.output_token_ids = []
                    req.output_text = ""
                    req.num_computed_tokens = 0
                    req.prompt_cache = None
                    req.cached_tokens = 0
                    req.remaining_tokens = req.prompt_token_ids
                    req.think_prefix_sent = False
                    req.status = RequestStatus.WAITING
                    req.batch_uid = None
                    self.waiting.appendleft(req)
                    output.outputs.append(
                        RequestOutput(
                            request_id=rid,
                            finished=False,
                            finish_reason=None,
                        )
                    )

                # Periodic Metal allocator cleanup during long decodes.
                # mx.random.categorical inside the sampler allocates a
                # tiny scalar via gumbel → uniform on every call.
                # omlx ships its own non-compiled sampler
                # (omlx/utils/sampling.py) so that RNG state actually
                # advances in the server, but the trade-off is that
                # those scalars accumulate in the IOGPU residency set
                # — macOS aborts at ~4096 entries. Long contexts
                # (50k+) decoding thousands of tokens hit that limit
                # mid-stream. Synchronise the generation stream first
                # so any in-flight Metal command buffer that still
                # references buffers we're about to drop has
                # completed; the allocator only releases pool entries
                # whose ref count is zero, but the sync guarantees
                # there is no race window. Decode-only path —
                # next_generated() returns nothing during prefill, so
                # we never disrupt prefill activation buffers.
                self._tokens_since_clear_cache += len(responses)
                if self._tokens_since_clear_cache >= self.config.decode_clear_interval:
                    frag = getattr(self, "_fragmentation_ratio", 0.0)
                    cache_mem = mx.get_cache_memory()
                    cache_threshold = self._periodic_clear_threshold_bytes()
                    if (
                        _should_clear_on_fragmentation(frag)
                        and cache_mem > cache_threshold
                    ):
                        _sync_and_clear_cache(self._stream)
                    else:
                        logger.debug(
                            "step(%d): skipping clear_cache — frag=%.2f cache_mem=%dMB threshold=%dMB",
                            self._step_counter,
                            frag,
                            cache_mem // (1024 * 1024),
                            cache_threshold // (1024 * 1024),
                        )
                    self._tokens_since_clear_cache = 0

    except _PrefillAbortedError as e:
        # Prefill was interrupted by a pending abort.
        # BatchGenerator is in an inconsistent state (partial
        # prefill), so reset it entirely. Pending aborts will
        # be processed at the start of the next step().
        _unregister_uid_rows_for_model(self.model)
        self.batch_generator = None
        self._current_sampler_params = None
        self._boundary_cache_snapshots.clear()
        if self._boundary_snapshot_store is not None:
            self._boundary_snapshot_store.cleanup_all()
        self._boundary_snapshot_required = None
        # Only reschedule the aborted requests, not the entire
        # batch — innocent requests should keep decoding.
        if e.aborted_uids:
            for uid in e.aborted_uids:
                rid = self.uid_to_request_id.get(uid)
                if rid and rid in self.running:
                    req = self.running.pop(rid)
                    req.status = RequestStatus.WAITING
                    req.batch_uid = None
                    self.waiting.append(req)
                    logger.debug("Rescheduled aborted uid=%d rid=%s", uid, rid)
        else:
            self._reschedule_running_requests()

    except (TypeError, AttributeError) as e:
        if self._is_cache_corruption_error(e):
            import traceback

            logger.warning(
                f"Cache corruption detected: {e}, "
                f"clearing cache and re-prefilling..."
            )
            logger.debug(f"Cache corruption traceback:\n{traceback.format_exc()}")
            # Full reset: clear batch generator, all caches, VLM state
            self._recover_from_cache_error()
            # Reschedule requests for re-prefill from scratch.
            # Requests exceeding max corruption retries are failed.
            failed_ids = self._reschedule_running_requests(is_corruption=True)
            for rid in failed_ids:
                output.outputs.append(
                    RequestOutput(
                        request_id=rid,
                        finished=True,
                        finish_reason="error",
                        error=(
                            f"Cache corruption not recoverable " f"after retries: {e}"
                        ),
                    )
                )
                output.finished_request_ids.add(rid)
        else:
            raise

    except Exception as e:
        import traceback

        logger.error(
            f"Error in batch generation step: {e}\n" f"{traceback.format_exc()}"
        )
        raise

    # Clear finished tracking for next step
    self.finished_req_ids = set()

    # Periodic Metal cache cleanup
    should_clear = self._should_periodic_clear_cache()
    # Deferred post-completion cleanup: fire once the step counter reaches
    # the target set by _cleanup_finished() (#435, #557).
    if (
        self._deferred_clear_at is not None
        and self._step_counter >= self._deferred_clear_at
    ):
        should_clear = True
        self._deferred_clear_at = None
    if should_clear:
        frag = getattr(self, "_fragmentation_ratio", 0.0)
        cache_mem = mx.get_cache_memory()
        cache_threshold = self._periodic_clear_threshold_bytes()
        if _should_clear_on_fragmentation(frag) and cache_mem > cache_threshold:
            _sync_and_clear_cache(self._stream)
        else:
            logger.debug(
                "step(%d): skipping periodic clear_cache — frag=%.2f cache_mem=%dMB threshold=%dMB",
                self._step_counter,
                frag,
                cache_mem // (1024 * 1024),
                cache_threshold // (1024 * 1024),
            )
    if (
        self.config.gc_cleanup_interval > 0
        and self._step_counter % self.config.gc_cleanup_interval == 0
    ):
        gc.collect()

    if self._step_counter % self.config.admin_snapshot_interval == 0:
        self._publish_admin_snapshot()

    return output


def _publish_admin_snapshot(self) -> None:
    """Atomically publish a fresh admin-visible snapshot.

    Called from step() on the engine thread, where running/waiting are
    not concurrently mutated. The admin endpoint reads the reference via
    snapshot_for_admin() and never iterates the live structures.
    """
    self._admin_snapshot = {
        "running_by_id": dict(self.running),
        "waiting": list(self.waiting),
    }


def snapshot_for_admin(self) -> dict[str, Any]:
    """Return the most recently published admin snapshot.

    Reference read is GIL-atomic; the dict itself is no longer mutated
    after publication. May be one step stale, which is fine for dashboard
    polling.
    """
    return self._admin_snapshot


def get_request(self, request_id: str) -> Request | None:
    """Get a request by ID."""
    return self.requests.get(request_id)


def remove_finished_request(self, request_id: str) -> Request | None:
    """Remove a finished request from tracking."""
    return self.requests.pop(request_id, None)


def get_stats(self) -> dict[str, Any]:
    """Get scheduler statistics."""
    stats = {
        "num_waiting": len(self.waiting),
        "num_prefilling": len(self.prefilling),
        "num_running": len(self.running),
        "num_requests_processed": self.num_requests_processed,
        "total_prompt_tokens": self.total_prompt_tokens,
        "total_completion_tokens": self.total_completion_tokens,
        "gpu_contention_detected": self._contention_detected,
    }
    if self._step_time_window:
        _times = self._step_time_window
        _mean = sum(_times) / len(_times)
        if _mean > 0 and len(_times) >= 2:
            _var = sum((t - _mean) ** 2 for t in _times) / len(_times)
            _cv = math.sqrt(_var) / _mean
            stats["decode_step_time_ms"] = round(_mean * 1000, 1)
            stats["decode_step_cv_pct"] = round(_cv * 100, 1)
        else:
            stats["decode_step_time_ms"] = round(_mean * 1000, 1)
            stats["decode_step_cv_pct"] = 0.0
    # Include cache stats
    if self.block_aware_cache is not None:
        stats["ssd_cache"] = self.block_aware_cache.get_stats()
    return stats


def get_cache_stats(self) -> dict[str, Any] | None:
    """Get cache statistics."""
    if self.block_aware_cache is not None:
        return self.block_aware_cache.get_stats()
    return None


def _step_pure_decode(self, output: SchedulerOutput) -> SchedulerOutput:
    """Fast path for single-request pure decode.

    Skips all scheduling, memory checks, admin snapshots, contention
    detection, and periodic cleanup. Only runs the model forward pass
    and processes the response. This path is ~2ms faster than the full
    step() for a typical 90ms decode step.
    """
    bg = self.batch_generator
    # Time the regular decode forward. bg._next() syncs internally (tolist
    # in the patched step), so this is real GPU time, not dispatch time.
    # Consumed by ngram_spec dynamic break-even (T_verify / T_decode).
    _decode_t0 = time.perf_counter()
    with mx.stream(self._stream):
        _, responses = bg._next()
    self._last_decode_dt = time.perf_counter() - _decode_t0

    if not responses:
        return output

    output.has_work = True
    outputs, finished_ids = self._process_batch_responses(responses)
    output.outputs.extend(outputs)
    output.finished_request_ids = finished_ids

    if finished_ids:
        self._cleanup_finished(finished_ids)
        logger.info("step(%d): finished=%s", self._step_counter, finished_ids)
    else:
        # Speculative decode: after regular step produces one token,
        # try to draft and verify additional tokens via n-gram matching
        spec_outputs = self._try_spec_decode(responses, output)
        if spec_outputs:
            output.outputs.extend(spec_outputs)
            # Check if spec decode finished the request
            for so in spec_outputs:
                if so.finished and so.request_id not in finished_ids:
                    finished_ids.add(so.request_id)
            if finished_ids:
                self._cleanup_finished(finished_ids)
                logger.info(
                    "step(%d): spec_finished=%s", self._step_counter, finished_ids
                )

    # Decode-phase cache clear (same interval as full path)
    if self._tokens_since_clear_cache is not None:
        self._tokens_since_clear_cache += len(responses)
        if self._tokens_since_clear_cache >= self.config.decode_clear_interval:
            _sync_and_clear_cache(self._stream)
            self._tokens_since_clear_cache = 0

    return output


def _try_spec_decode(
    self, responses: list, output: SchedulerOutput
) -> list[RequestOutput]:
    """Try speculative decode after a regular decode step.

    Tries n-gram spec decode first (CPU-side, zero GPU overhead),
    then falls back to draft-model spec decode if available.
    """
    if len(self.running) != 1:
        return []
    if len(responses) != 1:
        return []

    if output.finished_request_ids:
        return []

    resp = responses[0]
    if resp.finish_reason is not None:
        return []

    current_token = resp.token
    request_id = self.uid_to_request_id.get(resp.uid)
    if request_id is None:
        return []

    request = self.running.get(request_id)
    if request is None or request.is_finished():
        return []

    if self._vlm_mtp_active:
        return []
    # Standard mtp owns this decode step (verify+accept ran inside
    # bg._next()). Running suffix/dflash/dspark/draft on top would
    # double-spec and corrupt cache state. Bail so the per-step mtp flag
    # drives mtp<->suffix per-request routing when both are loaded.
    from ..patches.mlx_lm_mtp import last_step_was_mtp

    if last_step_was_mtp(self.batch_generator):
        return []
    if request_id in self._output_parser_sessions:
        return []

    if self._pending_abort_ids:
        return []

    # N-gram spec decode (CPU-side, zero GPU overhead for drafting)
    ngram_state = getattr(self, "_ngram_spec_state", None)
    if ngram_state is not None:
        from .ngram_spec import ngram_spec_step

        result = ngram_spec_step(self, output, current_token, request_id)
        if result:
            return result

    # DFlash block-diffusion spec decode (block-size draft + verify)
    if self._dflash_runtime is not None:
        from .spec_decode import dflash_spec_step

        result = dflash_spec_step(self, output, current_token, request_id)
        if result:
            return result

    # DSpark DeepSpec spec decode (self-contained propose-verify)
    if self._dspark_runtime is not None:
        from .spec_decode import dspark_spec_step

        result = dspark_spec_step(self, output, current_token, request_id)
        if result:
            return result

    # Draft-model spec decode (GPU-side, requires loaded draft model)
    if (
        self._spec_decode_state is not None
        and self._spec_decode_state.draft_model is not None
    ):
        from .spec_decode import spec_decode_step

        return spec_decode_step(self, output, current_token, request_id)

    return []


def reset(self) -> None:
    """Reset the scheduler state."""
    # Drain any pending deferred aborts
    self._pending_abort_ids.clear()

    # Abort all requests directly (reset is synchronous)
    for request_id in list(self.requests.keys()):
        self._do_abort_request(request_id)

    self.waiting.clear()
    self.prefilling.clear()
    self._prefill_states.clear()
    self.running.clear()
    self.requests.clear()
    self.finished_req_ids.clear()
    _unregister_uid_rows_for_model(self.model)
    self.request_id_to_uid.clear()
    self.uid_to_request_id.clear()
    # Async store_cache bookkeeping. shutdown() drains these before us,
    # but clear here too so reset() is safe to call standalone (e.g. tests
    # or recovery paths) without leaking Request refs through stale futures.
    self._pending_async_removes.clear()
    self._inflight_store_futures.clear()
    self.batch_generator = None
    self._current_sampler_params = None
    self._boundary_cache_snapshots.clear()
    if self._boundary_snapshot_store is not None:
        self._boundary_snapshot_store.cleanup_all()
    self._boundary_snapshot_required = None

    # Reset GPU contention detection state
    self._step_time_window.clear()
    self._contention_detected = False
    self._last_contention_log_step = 0

    # Clear caches
    if self.block_aware_cache is not None:
        self.block_aware_cache.clear()
    self._cache_rate_tracker.clear()

    # Clear detokenizers
    self._request_detokenizers.clear()

    # Clear protocol-specific output parser sessions
    self._output_parser_sessions.clear()

    # Cancel any pending deferred Metal cache clear
    self._deferred_clear_at = None


def deep_reset(self) -> None:
    """
    Deep reset that clears ALL cache state including model-level caches.

    This is more aggressive than reset() and should be used when
    switching engines or recovering from errors.
    """
    # Standard reset first
    self.reset()

    # Clear any model-level cache state
    # MLX models may have internal cache references
    if hasattr(self.model, "cache"):
        self.model.cache = None

    # Some MLX models store cache in layers
    if hasattr(self.model, "layers"):
        for layer in self.model.layers:
            if hasattr(layer, "cache"):
                layer.cache = None
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "cache"):
                layer.self_attn.cache = None

    # Release model and tokenizer references for GC
    self.model = None
    self.tokenizer = None

    # Release all cache-related references for GC
    self.paged_cache_manager = None
    self.block_aware_cache = None
    self.memory_monitor = None
    self._boundary_snapshot_store = None

    # Force garbage collection of any lingering cache objects
    import gc

    gc.collect()

    logger.info("Deep reset completed - all caches cleared")


def shutdown(self) -> None:
    """
    Graceful shutdown.

    Flushes hot cache to SSD and closes the background writer.
    paged SSD cache files are NOT cleared to allow reuse on reload.
    """
    logger.info("Scheduler shutdown initiated...")
    # The store-cache gate is a non-blocking counter (#1496), so there is
    # no step-thread caller to wake here. Inflight futures are drained
    # below before the executor is joined.
    # Wait for any inflight async store_cache futures + drain pending
    # batch_generator removes so the writer thread / underlying paged SSD
    # cache see all blocks before close().
    if self._store_cache_executor is not None:
        try:
            inflight = list(self._inflight_store_futures.values())
            if inflight:
                logger.info(
                    "Waiting for %d inflight async store_cache future(s)...",
                    len(inflight),
                )
                concurrent.futures.wait(inflight, timeout=30.0)
            self._drain_pending_async_removes()
            self._store_cache_executor.shutdown(wait=True)
            # Final drain after executor join. All workers are now done,
            # so any entries still in _pending_async_removes (skipped by
            # the first drain because their future hadn't completed yet)
            # are guaranteed drainable here. Without this, slow worker
            # finishes between the 30s wait timeout and shutdown(wait=True)
            # would leave KV cache references pinned on Request objects.
            self._drain_pending_async_removes()
        except Exception as e:
            logger.warning(f"Async store_cache shutdown error: {e}")
        self._store_cache_executor = None
        self._store_cache_gate = None
    if self.paged_ssd_cache_manager is not None:
        self.paged_ssd_cache_manager.close()
        self.paged_ssd_cache_manager = None
    logger.info("Scheduler shutdown completed")
