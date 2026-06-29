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
import time

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from ..prefill_progress import get_prefill_tracker
from ..request import Request, RequestOutput, RequestStatus
from ..utils.proc_memory import get_phys_footprint
from .helpers import (
    _sync_and_clear_cache,
)
from .monkeypatches import _register_uid_rows

# Module-level alias so Scheduler.__init__ can fall back to mlx-lm's default
# stream when no per-engine stream is provided.
from .types import (
    _PrefillAbortedError,
)


def _release_multimodal_tensors(request: "Request") -> None:
    """Release multimodal tensors after prefill to free unified memory."""
    if request.images is not None:
        request.images = None
    if request.videos is not None:
        request.videos = None
    if request.vlm_inputs_embeds is not None:
        request.vlm_inputs_embeds = None


def _schedule_waiting(    self,
) -> tuple[list["Request"], list[RequestOutput]]:
    """
    Move requests from waiting queue to running.

    Each request is prefilled externally before being inserted into
    BatchGenerator, so prefill_batch_size=1 is always used. Cache
    status homogeneity tracking is kept for safety since it affects
    how we handle the existing_cache argument.

    Returns:
        Tuple of (scheduled requests, rejected error outputs)
    """
    scheduled = []
    rejected_outputs: list[RequestOutput] = []

    # Track cache status of first scheduled request to ensure homogeneity
    # None = not determined yet, True = has cache, False = no cache
    batch_cache_status: bool | None = None
    # Track VLM status: VLM and text-only requests cannot be in the same prefill batch
    # None = not determined yet, True = VLM request, False = text-only request
    batch_vlm_status: bool | None = None
    # Track SpecPrefill: these requests must be alone (RoPE patching affects whole model)
    batch_specprefill_status: bool | None = None

    while self.waiting and len(self.running) < self.config.max_num_seqs:
         # Token budget guard: max_num_batched_tokens bounds the total
         # tokens (decode + prefill) in a single forward pass.
        batched_tokens = len(self.running) + sum(
            len(
                r.remaining_tokens if r.remaining_tokens is not None
                else r.prompt_token_ids
              )
            for r in scheduled
          )
        if batched_tokens >= self.config.max_num_batched_tokens:
            break

        # Admission pause: set by ProcessMemoryEnforcer when phys
        # crosses soft_threshold. New prefills wait; in-flight requests
        # continue. First request always passes (self.running is empty)
        # so admission can recover by completing the current generation.
        if self._admission_paused and self.running:
            logger.debug(
                "Admission paused by memory pressure, %d running",
                len(self.running),
            )
            break

        # Store-cache backpressure: when the post-completion pipeline is
        # at its in-flight cap, defer admitting new prefills instead of
        # blocking the generation step on the store-cache write (#1496).
        # The cap bounds concurrent extracted-KV copies (the #1383 OOM
        # guard) and shrinks under memory pressure via
        # adjust_store_cache_cap. In-flight requests keep generating;
        # the first request always passes (self.running is empty) so a
        # lone slow SSD write cannot deadlock admission.
        gate = self._store_cache_gate
        if gate is not None and self.running and not gate.has_capacity:
            logger.debug(
                "Admission deferred: store-cache pipeline full "
                "(in_flight=%d cap=%d), %d running",
                gate.in_flight,
                gate.cap,
                len(self.running),
            )
            break

        # Generation memory guard: when requests are already running,
        # defer scheduling if current memory + estimated prefill peak
        # exceeds the soft limit. This prevents admitting new requests
        # when there isn't enough headroom for their KV cache + SDPA
        # temp allocations, avoiding Metal OOM during batch_generator.next().
        # First request always passes (self.running is empty).
        if (
            self._prefill_memory_guard
            and self._memory_limit_bytes > 0
            and self.running
        ):
            current = max(mx.get_active_memory(), get_phys_footprint())
            _next = self.waiting[0]
            new_tokens = max(_next.num_prompt_tokens - (_next.cached_tokens or 0), 0)
            estimated_prefill = self._estimate_prefill_peak(new_tokens)
            if current + estimated_prefill > self._memory_limit_bytes:
                logger.debug(
                    "Generation memory guard: deferring scheduling "
                    "(current=%s + prefill=%s > limit=%s), %d running",
                    current,
                    estimated_prefill,
                    self._memory_limit_bytes,
                    len(self.running),
                )
                break

        request = self.waiting.popleft()

         # SpecPrefill: score remaining tokens on the executor thread
         # (not in add_request, which runs on the FastAPI event loop)
        self._try_specprefill_scoring(request)

        # Ensure we have a batch generator
        self._ensure_batch_generator(request.sampling_params)

        if self.batch_generator is None:
            # Put back and try again later
            self.waiting.appendleft(request)
            break

        # Determine tokens to process and cache to use
        # Note: Don't use `remaining_tokens or prompt_token_ids` because empty list
        # is falsy in Python. For exact cache match, remaining_tokens=[] but we should
        # pass just the last token so BatchGenerator can start generation.
        if (
            request.remaining_tokens is not None
            and len(request.remaining_tokens) == 0
        ):
            # Exact cache match - pass only last token for generation kickoff
            tokens_to_process = request.prompt_token_ids[-1:]
        elif request.remaining_tokens:
            tokens_to_process = request.remaining_tokens
        else:
            tokens_to_process = request.prompt_token_ids
        cache_to_use = request.prompt_cache  # May be None

        # Validate cache before using it
        if cache_to_use is not None and not self._validate_cache(cache_to_use):
            logger.debug(
                f"Request {request.request_id}: invalid cache detected, "
                f"proceeding without cache"
            )
            cache_to_use = None
            request.prompt_cache = None
            request.cached_tokens = 0
            request.remaining_tokens = request.prompt_token_ids
            tokens_to_process = request.prompt_token_ids

        # SpecPrefill requests must be alone in the batch (RoPE patching
        # affects the entire model). Also block scheduling if another
        # specprefill request is already running (offset RoPE active).
        request_is_specprefill = request.specprefill_indices is not None
        if (
            self._specprefill_active_request_id is not None
            and not request_is_specprefill
        ):
            # A specprefill request is running — defer all others until it finishes
            self.waiting.appendleft(request)
            break
        if batch_specprefill_status is None:
            batch_specprefill_status = request_is_specprefill
        elif batch_specprefill_status != request_is_specprefill:
            self.waiting.appendleft(request)
            break
        if request_is_specprefill and len(scheduled) > 0:
            # SpecPrefill request must be alone
            self.waiting.appendleft(request)
            break

        # Check VLM status homogeneity: VLM and text-only requests use
        # different prefill paths (embeddings vs token IDs)
        request_is_vlm = request.vlm_inputs_embeds is not None
        if batch_vlm_status is None:
            batch_vlm_status = request_is_vlm
        elif batch_vlm_status != request_is_vlm:
            # VLM status mismatch - defer this request to next batch
            self.waiting.appendleft(request)
            logger.debug(
                f"Deferring request {request.request_id} to next batch "
                f"(VLM status mismatch: batch={batch_vlm_status}, request={request_is_vlm})"
            )
            break

        # Check cache status homogeneity (kept for consistent prefill behavior)
        request_has_cache = cache_to_use is not None
        if batch_cache_status is None:
            batch_cache_status = request_has_cache
        elif batch_cache_status != request_has_cache:
            # Cache status mismatch - defer this request to next batch
            self.waiting.appendleft(request)
            logger.debug(
                f"Deferring request {request.request_id} to next batch "
                f"(cache status mismatch: batch={batch_cache_status}, request={request_has_cache})"
            )
            break

        # Mark as Harmony model if applicable (before think detection)
        if self._is_harmony_model:
            request.is_harmony_model = True

        # Check if prompt ends with <think> token for reasoning models.
        # Must happen before _build_sampler_and_processors so the thinking
        # budget processor can check needs_think_prefix.
        if self._detect_needs_think_prefix(request):
            request.needs_think_prefix = True

        # Per-request sampler/logits processors to avoid BatchGenerator recreation.
        sampler, logits_processors = self._build_sampler_and_processors(
            request.sampling_params, request
        )

        # Pre-flight memory guard: estimate peak memory for this request
        # and reject if it would exceed the hard limit.
        preflight_rejection = self._preflight_memory_check(request)
        if preflight_rejection is not None:
            logger.warning(
                f"Request {request.request_id} rejected by prefill "
                f"memory guard: {preflight_rejection.message}"
            )
            self._release_paged_cache_for_request(request.request_id)
            self.requests.pop(request.request_id, None)
            rejected_outputs.append(
                RequestOutput(
                    request_id=request.request_id,
                    finished=True,
                    finish_reason="error",
                    error=preflight_rejection.message,
                    error_metadata={
                        "estimated_bytes": preflight_rejection.estimated_bytes,
                        "limit_bytes": preflight_rejection.limit_bytes,
                    },
                )
            )
            continue

        # SpecPrefill: replace tokens with selected subset and pre-fill
        # cache via sparse_prefill before inserting into BatchGenerator.
        #
        # Key design: sparse_prefill processes selected tokens (excluding
        # the last prompt token). BatchGenerator then processes the last
        # prompt token to produce generation logits. This avoids:
        #   - Double-processing the last token (Bug #2)
        #   - Off-by-one RoPE positions (Bug #1)
        #
        # Position math:
        #   sparse_prefill: N' tokens, adjustment = M - N'
        #   We subtract 1: adjustment = M - N' - 1
        #   BatchGenerator last token: pos = N' + (M - N' - 1) = M - 1
        #   First gen token: pos = (N'+1) + (M - N' - 1) = M
        if request.specprefill_indices is not None:
            tracker = get_prefill_tracker()
            model_id = os.path.basename(self.config.model_name.rstrip("/"))
            total_pp = 0
            try:
                from ..patches.specprefill import (
                    _find_attention_layers,
                    _get_attn_module,
                    _OffsetAdjustedRoPE,
                    cleanup_rope,
                    sparse_prefill,
                )

                t0 = time.monotonic()

                sp_cache = make_prompt_cache(self.model)
                all_tokens = tokens_to_process
                sys_count = getattr(request, "_specprefill_system_tokens", 0)

                # Register tracker entry so the dashboard shows the PP
                # indicator throughout sys + sparse prefill. Denominator
                # mirrors the last-token removal applied below so the bar
                # ends cleanly at 100%.
                sel_list_pre = request.specprefill_indices.tolist()
                m_pre = len(all_tokens) - sys_count
                n_eff = len(sel_list_pre) - (
                    1 if (m_pre - 1) in sel_list_pre else 0
                )
                total_pp = sys_count + n_eff
                tracker.update(request.request_id, 0, total_pp, model_id)

                def _check_specprefill_abort(processed: int) -> None:
                    if request.request_id in self._pending_abort_ids:
                        logger.info(
                            f"SpecPrefill interrupted at {processed}/{total_pp} "
                            f"tokens: request aborted"
                        )
                        tracker.remove(request.request_id)
                        self.waiting.appendleft(request)
                        raise _PrefillAbortedError([], processed)

                # Phase 1: system prompt full prefill (if not cached)
                if sys_count > 0:
                    sys_arr = mx.array(all_tokens[:sys_count])
                    step = self.config.prefill_step_size
                    sys_processed = 0
                    spec_sparse_extra = {
                        "prompt_tokens": request.num_prompt_tokens,
                        "system_tokens": request.specprefill_system_end,
                        "conversation_tokens": request.num_prompt_tokens - request.specprefill_system_end,
                        "cached_tokens": request.cached_tokens,
                        "scored_tokens": m_pre,
                        "selected_tokens": n_eff,
                        "keep_percent": round(n_eff / m_pre * 100)
                        if m_pre > 0
                        else 0,
                    }
                    while sys_arr.size > step:
                        _check_specprefill_abort(sys_processed)
                        tracker.update(
                            request.request_id,
                            sys_processed,
                            total_pp,
                            model_id,
                            phase="specprefill_system",
                            detail="system prompt prefill",
                            extra=spec_sparse_extra,
                        )
                        self.model(sys_arr[:step][None], cache=sp_cache)
                        mx.eval([c.state for c in sp_cache])
                        sys_processed += step
                        _check_specprefill_abort(sys_processed)
                        tracker.update(
                            request.request_id,
                            min(sys_processed, total_pp - 1),
                            total_pp,
                            model_id,
                            phase="specprefill_system",
                            detail="system prompt prefill",
                            extra=spec_sparse_extra,
                        )
                        sys_arr = sys_arr[step:]
                        # Use _sync_and_clear_cache() instead of bare
                        # mx.clear_cache() to flush the engine stream
                        # before releasing Metal buffers.  A bare call here
                        # can race with in-flight command buffers submitted
                        # by the preceding mx.eval(), triggering the same
                        # 'completeMemory() prepare count underflow' kernel
                        # panic that #435 fixed elsewhere (#557).
                        _sync_and_clear_cache(self._stream)
                    if sys_arr.size > 0:
                        _check_specprefill_abort(sys_processed)
                        final_sys = int(sys_arr.size)
                        tracker.update(
                            request.request_id,
                            sys_processed,
                            total_pp,
                            model_id,
                            phase="specprefill_system",
                            detail="system prompt prefill",
                            extra=spec_sparse_extra,
                        )
                        self.model(sys_arr[None], cache=sp_cache)
                        mx.eval([c.state for c in sp_cache])
                        sys_processed += final_sys
                        _check_specprefill_abort(sys_processed)
                        tracker.update(
                            request.request_id,
                            min(sys_processed, total_pp - 1),
                            total_pp,
                            model_id,
                            phase="specprefill_system",
                            detail="system prompt prefill",
                            extra=spec_sparse_extra,
                        )
                    logger.info(
                        f"SpecPrefill: system prompt {sys_count} tokens full prefill"
                    )

                # Phase 2: conversation sparse prefill
                conv_tokens = all_tokens[sys_count:]
                selected = request.specprefill_indices
                M = len(conv_tokens)
                pos_offset = request.specprefill_position_offset
                last_idx = M - 1

                # Remove last token from selected set — BatchGenerator
                # will process it separately for generation kickoff.
                selected_list = selected.tolist()
                if last_idx in selected_list:
                    selected_list.remove(last_idx)
                    selected = mx.array(sorted(selected_list))

                def _sparse_progress(processed: int, total: int) -> None:
                    _check_specprefill_abort(sys_count + processed)
                    tracker.update(
                        request.request_id,
                        min(sys_count + processed, total_pp - 1),
                        total_pp,
                        model_id,
                        phase="specprefill_sparse",
                        detail="sparse target prefill",
                        extra={
                            "scored_tokens": M,
                            "selected_tokens": int(selected.shape[0]),
                            "keep_percent": round(int(selected.shape[0]) / M * 100)
                            if M > 0
                            else 0,
                            "prompt_tokens": request.num_prompt_tokens,
                            "system_tokens": request.specprefill_system_end,
                            "conversation_tokens": request.num_prompt_tokens - request.specprefill_system_end,
                            "cached_tokens": request.cached_tokens,
                        },
                    )

                sparse_prefill(
                    self.model,
                    conv_tokens,
                    selected,
                    sp_cache,
                    step_size=self.config.prefill_step_size,
                    position_offset=pos_offset,
                    progress_callback=_sparse_progress,
                )
                # sparse_prefill installs _OffsetAdjustedRoPE with
                # adjustment = M - N'. Subtract 1 to account for the
                # extra token BatchGenerator will process.
                for _, layer in _find_attention_layers(self.model):
                    attn = _get_attn_module(layer)
                    if (
                        attn
                        and hasattr(attn, "rope")
                        and isinstance(attn.rope, _OffsetAdjustedRoPE)
                    ):
                        attn.rope._adjustment -= 1

                N = int(selected.shape[0])
                t_prefill = time.monotonic() - t0
                total_prompt = request.num_prompt_tokens
                cached = request.cached_tokens
                logger.info(
                    f"SpecPrefill: sparse prefill {N}/{M} conv tokens in {t_prefill:.1f}s "
                    f"(total {total_prompt}, cached {cached}, "
                    f"system {sys_count} full, conv {M} sparse)"
                )

                # Set up request as if we had a prefix cache hit
                cache_to_use = sp_cache
                # Last token for generation kickoff
                tokens_to_process = all_tokens[-1:]
                self._specprefill_active_request_id = request.request_id

                # Mark spec-prefill complete (auto-removes tracker entry).
                tracker.update(request.request_id, total_pp, total_pp, model_id)

            except _PrefillAbortedError:
                cleanup_rope(self.model)
                request.specprefill_indices = None
                tracker.remove(request.request_id)
                raise
            except Exception as e:
                logger.error(f"SpecPrefill sparse prefill failed: {e}")
                cleanup_rope(self.model)
                request.specprefill_indices = None
                tracker.remove(request.request_id)
                # Fall through to normal prefill

        # External prefill: process tokens[0:N-1] outside BatchGenerator.
        # Only the last token goes to insert() for the first decode step.
        # SpecPrefill already handled its own prefill above, so skip for those.
        if request.specprefill_indices is None and len(tokens_to_process) > 1:
            vlm_embeds = None
            if request.vlm_inputs_embeds is not None:
                vlm_embeds = (
                    request.vlm_inputs_embeds,
                    request.vlm_extra_kwargs or {},
                    request.cached_tokens,
                )

            # Chunked prefill: non-VLM prompts longer than one step are
            # spread across multiple step() calls. The first chunk is run
            # here; subsequent chunks run in _advance_chunked_prefills().
            if (
                self.config.chunked_prefill
                and vlm_embeds is None
                and len(tokens_to_process) > self.config.prefill_step_size + 1
            ):
                sm = self._build_state_machine(request)
                per_row_lps = list(logits_processors) if logits_processors else []
                state = self._begin_prefill(request, tokens_to_process, cache_to_use)
                state.sampler = sampler
                state.sm = sm
                state.per_row_lps = per_row_lps

                try:
                    done = self._step_prefill_chunk(state)
                except _PrefillAbortedError:
                    raise
                except RuntimeError as e:
                    # Hard memory limit hit on the first chunk.
                    # _step_prefill_chunk updates the PrefillProgressTracker
                    # before the limit check, so without this catch the
                    # tracker entry leaks and stays in the dashboard
                    # forever (#1405). Mirrors the cleanup in
                    # _advance_chunked_prefills (d736bfd).
                    logger.error(
                        "Chunked prefill (first chunk) failed for %s: %s",
                        request.request_id,
                        e,
                    )
                    self._release_paged_cache_for_request(request.request_id)
                    self.requests.pop(request.request_id, None)
                    get_prefill_tracker().remove(request.request_id)
                    # Drop Metal cache pool buffers held by the aborted
                    # first chunk's forward / mx.eval transients.
                    _sync_and_clear_cache()
                    rejected_outputs.append(
                        RequestOutput(
                            request_id=request.request_id,
                            finished=True,
                            finish_reason="error",
                            error=str(e),
                        )
                    )
                    continue

                if done:
                    self._emit_final_boundary_if_needed(state)
                    _sync_and_clear_cache(self._stream)
                    get_prefill_tracker().remove(request.request_id)
                    self._insert_prefilled_request(request, state, scheduled)
                else:
                    self.prefilling.append(request)
                    self._prefill_states[request.request_id] = state
                continue  # Skip normal prefill + insert path

            # Normal (non-chunked) full prefill path.
            # Assign a temporary UID so progress callbacks can map
            # uid→request_id during external prefill. Replaced by the
            # real UID returned from insert().
            temp_uid = id(request)  # unique, won't collide with BatchGenerator UIDs
            self.request_id_to_uid[request.request_id] = temp_uid
            self.uid_to_request_id[temp_uid] = request.request_id

            try:
                prefilled_cache, last_token = self._do_external_prefill(
                    request,
                    tokens_to_process,
                    cache_to_use,
                    vlm_embeds=vlm_embeds,
                )
            except RuntimeError as e:
                # Hard memory limit hit during external prefill. Without
                # this catch, the exception bubbles up to step() and then
                # engine_core's fail_all_requests(), which pops
                # self.requests but cannot reach the PrefillProgressTracker
                # singleton, so the dashboard entry leaks across model
                # reload (#1405). Mirrors the cleanup in
                # _advance_chunked_prefills (d736bfd).
                logger.error("Prefill failed for %s: %s", request.request_id, e)
                self.uid_to_request_id.pop(temp_uid, None)
                self.request_id_to_uid.pop(request.request_id, None)
                self._release_paged_cache_for_request(request.request_id)
                self.requests.pop(request.request_id, None)
                get_prefill_tracker().remove(request.request_id)
                # Drop Metal cache pool buffers held by the aborted
                # chunk's forward / mx.eval transients.
                _sync_and_clear_cache()
                rejected_outputs.append(
                    RequestOutput(
                        request_id=request.request_id,
                        finished=True,
                        finish_reason="error",
                        error=str(e),
                    )
                )
                continue

             # Clean up temp UID mapping (use .pop() because
             # fail_all_requests may have already cleared the maps
             # if a timeout fired while _do_external_prefill was running)
            self.uid_to_request_id.pop(temp_uid, None)
            self.request_id_to_uid.pop(request.request_id, None)
            # Prefill complete: remove from progress tracker so dashboard
            # shows "generating" instead of "PP" during decode.
            get_prefill_tracker().remove(request.request_id)

            cache_to_use = prefilled_cache
            tokens_to_process = last_token

        # Capture per-request mRoPE rope_deltas for decode.
        # Prefer _captured_rope_deltas from per-request extra_kwargs
        # (set during get_input_embeddings), since the global
        # _rope_deltas may be stale when explicit position_ids are used.
        if request.vlm_inputs_embeds is not None:
            extra = request.vlm_extra_kwargs or {}
            captured = extra.get("_captured_rope_deltas")
            if captured is not None:
                if hasattr(captured, "item"):
                    request.rope_deltas = float(captured.item())
                else:
                    request.rope_deltas = float(captured)
            elif hasattr(self.model, "get_last_rope_deltas"):
                request.rope_deltas = self.model.get_last_rope_deltas()

        # Build per-request state machine for stop tokens
        sm = self._build_state_machine(request)

        # NOTE: TurboQuant KV conversion is not applied during prefill.
        # See _do_external_prefill() comment for rationale (#771).

        # VLM MTP routing: if a gemma4_assistant drafter is attached, run
        # an extra last-token forward to capture hidden + shared_kv_states,
        # sample the first bonus, and hand the request to a vlm_mtp
        # generator instead of BatchGenerator. Falls through on any
        # eligibility issue so other speculative paths stay intact.
        if self._vlm_mtp_drafter is not None and cache_to_use is not None:
            vlm_mtp_uid = self._route_to_vlm_mtp(
                request, cache_to_use, tokens_to_process, sampler, sm
            )
            if vlm_mtp_uid is not None:
                self.request_id_to_uid[request.request_id] = vlm_mtp_uid
                self.uid_to_request_id[vlm_mtp_uid] = request.request_id
                now = time.monotonic()
                request.batch_uid = vlm_mtp_uid
                request.status = RequestStatus.RUNNING
                _release_multimodal_tensors(request)
                request.generation_started_at = now
                request.last_activity_at = now
                self.running[request.request_id] = request
                scheduled.append(request)
                self.total_prompt_tokens += request.num_prompt_tokens
                logger.debug(
                    f"Scheduled request {request.request_id} via vlm_mtp "
                    f"(uid={vlm_mtp_uid}, {request.num_prompt_tokens} prompt tokens)"
                )
                continue

         # If _route_to_vlm_mtp returned None but the request is in the
        # pending queue (batch not full yet), skip the BatchGenerator
        # fallthrough. The request will be flushed by _vlm_mtp_drain_pending
        # at the end of this _schedule_waiting loop.
        pending_queue = getattr(self, "_vlm_mtp_pending_queue", None)
        if pending_queue and any(
             p["request"].request_id == request.request_id
            for p in pending_queue
         ):
            continue

        # Insert into BatchGenerator with pre-filled cache + last token.
        # BatchGenerator only handles decode from here.
        #
        # IMPORTANT: ``logits_processors`` MUST be passed as a per-row
        # list (possibly empty), never None.  mlx-lm's
        # GenerationBatch._step does ``for p in self.logits_processors[e]``
        # in any branch where ``any(self.logits_processors)`` is True
        # (e.g., heterogeneous merge with another row that has a
        # processor).  A None slot crashes that loop with
        # ``TypeError: 'NoneType' object is not iterable``, which then
        # bubbles into the engine retry loop and presents as a hang.
        # See vllm-mlx-patched commit 8d4052b for the same root cause
        # in a sibling project, and #934 for the user-visible symptom.
        per_row_lps = list(logits_processors) if logits_processors else []
        uids = self.batch_generator.insert(
            [tokens_to_process],
            max_tokens=[request.sampling_params.max_tokens],
            caches=[cache_to_use] if cache_to_use else None,
            samplers=[sampler],
            logits_processors=[per_row_lps],
            state_machines=[sm],
        )

        if uids:
            _register_uid_rows(self.model, uids, [sampler], [per_row_lps])
            uid = uids[0]
            self.request_id_to_uid[request.request_id] = uid
            self.uid_to_request_id[uid] = request.request_id
            now = time.monotonic()
            request.batch_uid = uid
            request.status = RequestStatus.RUNNING
            request.generation_started_at = now
            request.last_activity_at = now
            self.running[request.request_id] = request
            scheduled.append(request)

            # Register per-UID rope_delta for mRoPE decode.
            if hasattr(self.model, "register_rope_delta"):
                self.model.register_rope_delta(uid, request.rope_deltas)

            self.total_prompt_tokens += request.num_prompt_tokens
            cache_info = (
                f", {request.cached_tokens} cached"
                if request.cached_tokens > 0
                else ""
            )
            cache_used = "with cache" if cache_to_use else "no cache"
            logger.debug(
                f"Scheduled request {request.request_id} (uid={uid}) "
                f"with {len(tokens_to_process)} tokens to process "
                f"({request.num_prompt_tokens} total){cache_info}, {cache_used}"
            )

      # Drain any pending vlm_mtp requests (batch not full but should flush)
    if self._vlm_mtp_drafter is not None:
        from .sched_vlm_mtp_batched import _vlm_mtp_drain_pending
        lm = getattr(self.model, "_language_model", None)
        if lm is not None:
            for row in _vlm_mtp_drain_pending(self, lm, self._vlm_mtp_drafter):
                rid = row.request.request_id
                self.request_id_to_uid[rid] = row.uid
                self.uid_to_request_id[row.uid] = rid
                now = time.monotonic()
                row.request.batch_uid = row.uid
                row.request.status = RequestStatus.RUNNING
                row.request.generation_started_at = now
                row.request.last_activity_at = now
                self.running[rid] = row.request
                scheduled.append(row.request)
                self.total_prompt_tokens += row.request.num_prompt_tokens


    return scheduled, rejected_outputs
