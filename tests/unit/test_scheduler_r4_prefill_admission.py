# SPDX-License-Identifier: Apache-2.0
"""R-4 (#811 audit 0906): admission pause gate + token budget must count
in-flight chunked-prefill requests (self.prefilling), not just running.

These tests exercise the two R-4 guards via the real ``_schedule_waiting``
loop. They assert the BLOCK cases (pause holds / budget blocks), which
break BEFORE the deep admit path runs, so no model wiring is needed. The
admit (positive) cases are covered by the existing scheduler suite — the
fix only tightens the block conditions."""

from collections import deque
from unittest.mock import MagicMock

import pytest

from fusion_mlx.scheduler import Scheduler


def _bare_scheduler(max_batched_tokens=65536, max_num_seqs=8):
    s = Scheduler.__new__(Scheduler)
    s.config = MagicMock(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_batched_tokens,
    )
    s.waiting = deque()
    s.requests = {}
    s.prefilling = deque()
    s.running = {}
    s._admission_paused = False
    # Downstream guards — disable so ONLY the R-4 guards are exercised.
    s._store_cache_gate = None
    s._pending_async_removes = []
    s._prefill_memory_guard = False
    s._memory_limit_bytes = 0
    s._memory_hard_limit_bytes = 0
    s._serialize_llama4_requests = False
    s._generation_overflow_recovery_ids = set()
    # Post-loop drain attrs (the break cases reach the post-loop drain).
    s._vlm_mtp_drafter = None
    s._specprefill_active_request_id = None
    s.model = MagicMock()
    s._pending_vlm_mtp = deque()
    s._should_defer_for_cache_freshness = MagicMock(return_value=False)
    s._stream = MagicMock()
    return s


def _req(rid, n_tokens):
    r = MagicMock()
    r.request_id = rid
    r.prompt = "x"
    r.prompt_token_ids = list(range(n_tokens))
    r.remaining_tokens = list(range(n_tokens))
    r.num_prompt_tokens = n_tokens
    r.cached_tokens = 0
    r.sampling_params = MagicMock()
    r.is_vlm = False
    r.is_specprefill = False
    return r


class TestR4PrefillPauseGate:
    def test_pause_holds_when_only_prefilling_in_flight(self):
        # R-4: prefilling non-empty + running empty + paused -> pause MUST
        # hold (admit zero). Pre-fix the gate was `(running or scheduled)`
        # = False, so it admitted a fresh burst under memory pressure -> OOM.
        s = _bare_scheduler(max_num_seqs=8)
        s._admission_paused = True
        # One in-flight prefill chunk (under the 8 cap so the while-loop cap
        # itself does not block admission — only the pause gate should).
        s.prefilling.append(_req("pf", 100))
        s.waiting.append(_req("w1", 10))
        scheduled, _ = s._schedule_waiting()
        assert scheduled == []
        assert len(s.waiting) == 1  # not popped

    def test_pause_holds_with_running_empty_scheduled_empty_prefilling_only(self):
        # Explicit mirror of the audit failure: 3 chunked-prefill chunks
        # in flight, running empty, memory pressure set. Pre-fix admitted.
        s = _bare_scheduler(max_num_seqs=8)
        s._admission_paused = True
        for i in range(3):
            s.prefilling.append(_req(f"pf{i}", 8000))
        s.waiting.append(_req("fresh", 500))
        scheduled, _ = s._schedule_waiting()
        assert scheduled == []
        assert len(s.waiting) == 1


class TestR4TokenBudgetCountsPrefilling:
    def test_budget_blocks_when_prefilling_alone_exceeds_cap(self):
        # R-4: in-flight prefill chunk tokens MUST count against
        # max_num_batched_tokens. Two ~3k prefill chunks alone (6000) exceed a
        # 4k cap -> no fresh admit. Pre-fix the budget ignored prefilling
        # entirely, so these chunks + a new admit silently exceeded the cap.
        s = _bare_scheduler(max_batched_tokens=4000, max_num_seqs=8)
        s.prefilling.append(_req("pf1", 3000))
        s.prefilling.append(_req("pf2", 3000))
        s.waiting.append(_req("w1", 10))
        scheduled, _ = s._schedule_waiting()
        assert scheduled == []
        assert len(s.waiting) == 1  # not popped

    def test_budget_admits_when_prefilling_under_cap(self):
        # Positive control: prefilling under the cap does not block.
        s = _bare_scheduler(max_batched_tokens=4000, max_num_seqs=8)
        s.prefilling.append(_req("pf1", 100))
        s.waiting.append(_req("w1", 10))
        # Budget (100) < cap (4000) -> guard does NOT break at the budget
        # check. The request is popped from waiting (admitted past the R-4
        # guards); the deep admit path may raise on bare-stub attrs, so we
        # only assert the R-4 budget guard did not block (waiting emptied).
        try:
            scheduled, _ = s._schedule_waiting()
        except AttributeError:
            # Deep admit path needs model wiring we don't stub; the R-4
            # budget guard already passed (request was popped).
            pass
        assert len(s.waiting) == 0
