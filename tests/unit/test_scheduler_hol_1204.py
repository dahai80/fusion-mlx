# SPDX-License-Identifier: Apache-2.0
"""Tests for #1204: scheduler head-of-line blocking behind one oversized
chunked-prefill prompt.

Root cause (log evidence): an 83,254-token prompt occupied chunked
prefill for 55+ min at ~25 tok/s while waiting=3, running=0 — every
waiting request starved because the admission token-budget check
counted the in-flight prefill's FULL remaining prompt (83K) against
max_num_batched_tokens, so nothing else could be admitted until the
whole prefill finished.

Fixes under test:
1. batched_tokens counts only the NEXT chunk (prefill_step_size) of
   each in-flight prefilling request, matching what a forward pass
   actually contains (_advance_chunked_prefills advances every
   prefilling request by exactly one chunk per step).
2. The adaptive throttle floors at 512 tokens — 64-token chunks made
   per-step fixed overhead dominate (~25 tok/s on the 83K prompt).
"""

from unittest.mock import MagicMock

import mlx.core as mx

from fusion_mlx.request import Request, SamplingParams
from fusion_mlx.scheduler import Scheduler, SchedulerConfig, _PrefillState


def _make_scheduler(
    max_num_batched_tokens: int = 64, prefill_step_size: int = 8
) -> Scheduler:
    model = MagicMock()
    model.layers = []
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    config = SchedulerConfig(
        max_num_seqs=8,
        prefill_step_size=prefill_step_size,
        chunked_prefill=True,
        paged_cache_block_size=0,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    scheduler = Scheduler(model=model, tokenizer=tokenizer, config=config)
    mock_bg = MagicMock()
    mock_bg.insert.return_value = [42]
    mock_bg.next_generated.return_value = iter([])
    scheduler.batch_generator = mock_bg
    scheduler._current_sampler_params = ()
    # Keep the memory/store admission guards out of the way — the
    # token-budget accounting is what #1204 tests.
    scheduler._prefill_memory_guard = False
    scheduler._memory_limit_bytes = 0
    return scheduler


def _make_request(request_id: str, n_tokens: int) -> Request:
    req = Request(
        request_id=request_id,
        prompt=list(range(n_tokens)),
        sampling_params=SamplingParams(max_tokens=32),
    )
    req.prompt_token_ids = list(range(n_tokens))
    req.num_prompt_tokens = n_tokens
    req.remaining_tokens = list(range(n_tokens))
    return req


def _make_prefill_state(request: Request, n_remaining: int) -> _PrefillState:
    return _PrefillState(
        request=request,
        cache=[],
        tokens_remaining=mx.zeros((1, n_remaining), dtype=mx.int32),
        last_token=[99],
        tokens_processed=0,
        base_size=0,
        emitted_boundaries={},
        boundary_enabled=False,
        block_size=0,
        total_length=n_remaining + 1,
        sampler=MagicMock(),
        sm=MagicMock(),
        per_row_lps=[],
    )


class TestNextChunkTokenLoad:
    def test_counts_one_chunk_not_full_remaining(self):
        scheduler = _make_scheduler(prefill_step_size=8)
        req = _make_request("hol-1", n_tokens=83254)
        scheduler._prefill_states["hol-1"] = _make_prefill_state(req, n_remaining=83254)
        # 83K-token prompt in flight: the next forward pass carries ONE
        # 8-token chunk, not 83K tokens.
        assert scheduler._next_chunk_token_load(req) == 8

    def test_last_chunk_clamped_to_remaining(self):
        scheduler = _make_scheduler(prefill_step_size=8)
        req = _make_request("hol-2", n_tokens=20)
        scheduler._prefill_states["hol-2"] = _make_prefill_state(req, n_remaining=3)
        assert scheduler._next_chunk_token_load(req) == 3

    def test_falls_back_to_prompt_len_without_state(self):
        scheduler = _make_scheduler()
        req = _make_request("hol-3", n_tokens=50)
        assert scheduler._next_chunk_token_load(req) == 50


class TestAdmissionNotStarved:
    def test_waiting_admitted_while_oversized_prefill_in_flight(self):
        # The #1204 scenario: 200-token prompt mid-prefill (192 still
        # remaining), a 10-token request waiting, budget 64. The old
        # full-remaining accounting (192 + 10 >= 64) starved the waiter
        # for the entire prefill; per-chunk accounting (8 + 10 < 64)
        # admits it.
        scheduler = _make_scheduler(max_num_batched_tokens=64, prefill_step_size=8)
        big = _make_request("big", n_tokens=200)
        scheduler.prefilling.append(big)
        scheduler.requests["big"] = big
        scheduler._prefill_states["big"] = _make_prefill_state(big, n_remaining=192)
        small = _make_request("small", n_tokens=10)
        scheduler.waiting.append(small)
        scheduler.requests["small"] = small

        scheduled, rejected = scheduler._schedule_waiting()
        assert rejected == []
        admitted = small in scheduled or small in scheduler.prefilling
        assert admitted, (
            "waiting request must be admitted while an oversized chunked "
            "prefill is in flight (head-of-line block, #1204)"
        )

    def test_still_respects_budget_at_real_chunk_scale(self):
        # Guard the guard: with enough in-flight chunk load that one more
        # chunk would exceed the budget, admission still defers.
        scheduler = _make_scheduler(max_num_batched_tokens=16, prefill_step_size=8)
        a = _make_request("a", n_tokens=100)
        b = _make_request("b", n_tokens=100)
        for r in (a, b):
            scheduler.prefilling.append(r)
            scheduler.requests[r.request_id] = r
            scheduler._prefill_states[r.request_id] = _make_prefill_state(
                r, n_remaining=100
            )
        # 2 in-flight chunks = 16 tokens; a 20-token admit would exceed
        # the budget of 16, so admission defers (old behavior kept for
        # genuine budget pressure).
        c = _make_request("c", n_tokens=20)
        scheduler.waiting.append(c)
        scheduler.requests["c"] = c
        scheduled, rejected = scheduler._schedule_waiting()
        assert c not in scheduled and c in scheduler.waiting


class TestThrottleFloor:
    def test_adaptive_chunk_never_below_512(self, monkeypatch):
        from fusion_mlx.scheduler import sched_batch

        scheduler = _make_scheduler(prefill_step_size=4096)
        scheduler._memory_limit_bytes = 10 * 1024**3
        scheduler._memory_hard_limit_bytes = 10 * 1024**3
        scheduler._prefill_safe_zone_ratio = 0.80
        scheduler._prefill_min_chunk_tokens = 32

        # Deep in the soft→hard band (75%+ → tier 128), the floor wins.
        current = 10 * 1024**3 - 64 * 1024**2
        monkeypatch.setattr(sched_batch.mx, "get_active_memory", lambda: current)
        monkeypatch.setattr(sched_batch, "get_phys_footprint", lambda: current)
        n = scheduler._adaptive_chunk_size(
            4096, request_id="floor-1", loop_label="chunked_step"
        )
        assert n >= 512, "throttled chunk must never drop below 512 (#1204)"

    def test_floor_constant(self):
        from fusion_mlx.scheduler.sched_batch import (
            _PREFILL_THROTTLE_FLOOR_TOKENS,
        )

        assert _PREFILL_THROTTLE_FLOOR_TOKENS == 512
