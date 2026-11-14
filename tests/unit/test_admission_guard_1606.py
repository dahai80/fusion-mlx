# SPDX-License-Identifier: Apache-2.0
"""Tests for the #1606 prefix-cache metastability admission guard.

Covers: shed during low hit-rate EMA + deep queue, recovery after the EMA
rises, warm-prefix bypass, explicit request bypass, kill switch,
per-request bounded retry, and the per-episode deferred-depth cap.
No real models are loaded; the prefix cache is a MagicMock.
"""

from unittest.mock import MagicMock

import pytest

from fusion_mlx.request import Request, SamplingParams
from fusion_mlx.scheduler import Scheduler


def _make_scheduler(mock_model, mock_tokenizer, hit_rate=0.05, **knobs):
    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    scheduler.block_aware_cache = MagicMock()
    scheduler.block_aware_cache.get_stats_dict.return_value = {"hit_rate": hit_rate}
    cfg = scheduler.config
    cfg.admission_guard_enabled = knobs.get("enabled", True)
    cfg.admission_guard_hit_rate_floor = knobs.get("floor", 0.3)
    cfg.admission_guard_queue_depth = knobs.get("queue_depth", 2)
    cfg.admission_guard_ema_alpha = knobs.get("alpha", 0.1)
    cfg.admission_guard_max_defers = knobs.get("max_defers", 8)
    cfg.admission_guard_max_shed = knobs.get("max_shed", 64)
    return scheduler


def _make_request(
    request_id="req-cold",
    cached_tokens=0,
    prompt_cache=None,
    bypass=False,
):
    return Request(
        request_id=request_id,
        prompt="hello",
        sampling_params=SamplingParams(max_tokens=4),
        prompt_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
        prompt_cache=prompt_cache,
        cached_tokens=cached_tokens,
        bypass_admission_guard=bypass,
    )


def _queue(scheduler, request):
    scheduler.waiting.append(request)
    scheduler.requests[request.request_id] = request


def _fill(scheduler, count, prefix="filler"):
    # Depth is measured on the waiting queue AFTER the decided request is
    # popped, so shed tests need filler requests behind the target.
    for i in range(count):
        _queue(scheduler, _make_request(f"{prefix}-{i}"))


def _pop_and_decide(scheduler, request):
    scheduler.waiting.remove(request)
    return scheduler._admission_guard_should_shed(request)


class TestAdmissionGuardEMA:
    def test_first_sample_seeds_ema(self, mock_model, mock_tokenizer):
        scheduler = _make_scheduler(mock_model, mock_tokenizer, hit_rate=0.05)
        scheduler._admission_guard_note_hit_rate()
        assert scheduler._admission_guard_ema == pytest.approx(0.05)

    def test_ema_moves_toward_new_sample(self, mock_model, mock_tokenizer):
        scheduler = _make_scheduler(mock_model, mock_tokenizer, hit_rate=0.05)
        scheduler._admission_guard_note_hit_rate()
        scheduler.block_aware_cache.get_stats_dict.return_value = {"hit_rate": 0.45}
        scheduler._admission_guard_note_hit_rate()
        assert scheduler._admission_guard_ema == pytest.approx(0.05 + 0.1 * 0.4)

    def test_missing_stats_leave_ema_none(self, mock_model, mock_tokenizer):
        scheduler = _make_scheduler(mock_model, mock_tokenizer)
        scheduler.block_aware_cache.get_stats_dict.return_value = None
        scheduler._admission_guard_note_hit_rate()
        assert scheduler._admission_guard_ema is None

    def test_missing_hit_rate_key_leaves_ema_none(self, mock_model, mock_tokenizer):
        scheduler = _make_scheduler(mock_model, mock_tokenizer)
        scheduler.block_aware_cache.get_stats_dict.return_value = {"hits": 1}
        scheduler._admission_guard_note_hit_rate()
        assert scheduler._admission_guard_ema is None

    def test_no_cache_is_safe(self, mock_model, mock_tokenizer):
        scheduler = _make_scheduler(mock_model, mock_tokenizer)
        scheduler.block_aware_cache = None
        scheduler._admission_guard_note_hit_rate()
        assert scheduler._admission_guard_ema is None
        request = _make_request()
        _queue(scheduler, request)
        assert _pop_and_decide(scheduler, request) is False


class TestAdmissionGuardShed:
    def test_shed_cold_admissions_during_low_ema_deep_queue(
        self, mock_model, mock_tokenizer
    ):
        scheduler = _make_scheduler(
            mock_model,
            mock_tokenizer,
            hit_rate=0.05,
            queue_depth=2,
            max_defers=100,
            max_shed=100,
        )
        scheduler._ensure_batch_generator = MagicMock()
        cold = [_make_request(f"cold-{i}") for i in range(3)]
        for request in cold:
            _queue(scheduler, request)

        scheduled, rejected = scheduler._schedule_waiting()

        assert scheduled == []
        assert rejected == []
        assert len(list(scheduler.waiting)) == 3
        assert set(scheduler.waiting) == set(cold)
        assert scheduler._admission_guard_shed_total == 3
        assert scheduler._admission_guard_active is True
        scheduler._ensure_batch_generator.assert_not_called()

    def test_no_shed_above_floor_even_with_deep_queue(self, mock_model, mock_tokenizer):
        scheduler = _make_scheduler(
            mock_model,
            mock_tokenizer,
            hit_rate=0.9,
            queue_depth=2,
        )
        request = _make_request("target")
        _queue(scheduler, request)
        _fill(scheduler, 2)
        assert _pop_and_decide(scheduler, request) is False
        assert scheduler._admission_guard_active is False

    def test_no_shed_below_floor_with_shallow_queue(self, mock_model, mock_tokenizer):
        scheduler = _make_scheduler(
            mock_model,
            mock_tokenizer,
            hit_rate=0.05,
            queue_depth=2,
        )
        request = _make_request("target")
        _queue(scheduler, request)
        _fill(scheduler, 1)
        assert _pop_and_decide(scheduler, request) is False
        # Degraded EMA alone still marks the guard active (visible state),
        # but nothing is shed while the backlog is shallow.
        assert scheduler._admission_guard_active is True
        assert scheduler._admission_guard_shed_total == 0

    def test_recovery_admits_again_after_ema_rises(self, mock_model, mock_tokenizer):
        scheduler = _make_scheduler(
            mock_model,
            mock_tokenizer,
            hit_rate=0.05,
            queue_depth=2,
            max_defers=100,
            max_shed=100,
        )
        requests = [_make_request(f"cold-{i}") for i in range(3)]
        for request in requests:
            _queue(scheduler, request)
        first = scheduler.waiting.popleft()
        assert scheduler._admission_guard_should_shed(first) is True
        assert scheduler._admission_guard_active is True

        scheduler.block_aware_cache.get_stats_dict.return_value = {"hit_rate": 1.0}
        scheduler.waiting.append(first)
        # alpha=0.1 from 0.05: EMA crosses the 0.3 floor after 3 samples.
        shed = True
        for _ in range(10):
            request = scheduler.waiting.popleft()
            shed = scheduler._admission_guard_should_shed(request)
            scheduler.waiting.append(request)
            if not shed:
                break
        assert shed is False
        assert scheduler._admission_guard_active is False
        assert scheduler._admission_guard_ema >= 0.3

    def test_warm_request_bypasses_shed(self, mock_model, mock_tokenizer):
        scheduler = _make_scheduler(
            mock_model,
            mock_tokenizer,
            hit_rate=0.05,
            queue_depth=2,
        )
        warm_tokens = _make_request("warm-tokens", cached_tokens=64)
        _queue(scheduler, warm_tokens)
        _fill(scheduler, 2)
        assert _pop_and_decide(scheduler, warm_tokens) is False

        warm_cache = _make_request("warm-cache", prompt_cache=[object()])
        _queue(scheduler, warm_cache)
        _fill(scheduler, 2, prefix="filler2")
        assert _pop_and_decide(scheduler, warm_cache) is False
        assert scheduler._admission_guard_shed_total == 0

    def test_explicit_request_bypass_flag(self, mock_model, mock_tokenizer):
        scheduler = _make_scheduler(
            mock_model,
            mock_tokenizer,
            hit_rate=0.05,
            queue_depth=2,
        )
        request = _make_request("bypassed", bypass=True)
        _queue(scheduler, request)
        _fill(scheduler, 2)
        assert _pop_and_decide(scheduler, request) is False
        assert scheduler._admission_guard_shed_total == 0

    def test_kill_switch_disables_guard(self, mock_model, mock_tokenizer):
        scheduler = _make_scheduler(
            mock_model,
            mock_tokenizer,
            hit_rate=0.05,
            queue_depth=2,
            enabled=False,
        )
        request = _make_request("target")
        _queue(scheduler, request)
        _fill(scheduler, 2)
        calls = scheduler.block_aware_cache.get_stats_dict.call_count
        assert _pop_and_decide(scheduler, request) is False
        # Guard off = zero overhead: no stats sampling either.
        assert scheduler.block_aware_cache.get_stats_dict.call_count == calls
        assert scheduler._admission_guard_active is False

    def test_per_request_bounded_retry_fails_open(self, mock_model, mock_tokenizer):
        scheduler = _make_scheduler(
            mock_model,
            mock_tokenizer,
            hit_rate=0.05,
            queue_depth=2,
            max_defers=2,
        )
        request = _make_request("retry")
        _queue(scheduler, request)
        _fill(scheduler, 3)
        assert _pop_and_decide(scheduler, request) is True
        _queue(scheduler, request)
        assert _pop_and_decide(scheduler, request) is True
        _queue(scheduler, request)
        assert _pop_and_decide(scheduler, request) is False
        assert scheduler._admission_guard_shed_total == 2

    def test_episode_deferred_depth_cap_fails_open(self, mock_model, mock_tokenizer):
        scheduler = _make_scheduler(
            mock_model,
            mock_tokenizer,
            hit_rate=0.05,
            queue_depth=1,
            max_defers=100,
            max_shed=2,
        )
        requests = [_make_request(f"cold-{i}") for i in range(4)]
        for request in requests:
            _queue(scheduler, request)
        assert _pop_and_decide(scheduler, requests[0]) is True
        assert _pop_and_decide(scheduler, requests[1]) is True
        # Episode cap reached: the third cold request admits (fail open)
        # even though the queue is still deep enough to shed.
        assert _pop_and_decide(scheduler, requests[2]) is False
        assert scheduler._admission_guard_episode_sheds == 2
        assert scheduler._admission_guard_shed_total == 2

    def test_cap_resets_on_recovery(self, mock_model, mock_tokenizer):
        scheduler = _make_scheduler(
            mock_model,
            mock_tokenizer,
            hit_rate=0.05,
            queue_depth=1,
            max_defers=100,
            max_shed=1,
        )
        first = _make_request("first")
        _queue(scheduler, first)
        _fill(scheduler, 1)
        assert _pop_and_decide(scheduler, first) is True
        second = _make_request("second")
        _queue(scheduler, second)
        # Cap hit while the queue is still deep enough to shed.
        assert _pop_and_decide(scheduler, second) is False

        # Recover (EMA above floor), then re-enter a degraded episode: the
        # shed cap must be reset with the new episode.
        scheduler.block_aware_cache.get_stats_dict.return_value = {"hit_rate": 1.0}
        for _ in range(10):
            scheduler._admission_guard_note_hit_rate()
        assert scheduler._admission_guard_ema >= 0.3
        scheduler._admission_guard_set_active(False)
        assert scheduler._admission_guard_episode_sheds == 0
        scheduler.block_aware_cache.get_stats_dict.return_value = {"hit_rate": 0.05}
        for _ in range(30):
            scheduler._admission_guard_note_hit_rate()
        _queue(scheduler, _make_request("refill"))
        _queue(scheduler, second)
        assert _pop_and_decide(scheduler, second) is True


class TestAdmissionGuardStats:
    def test_get_stats_exports_guard_state(self, mock_model, mock_tokenizer):
        scheduler = _make_scheduler(
            mock_model,
            mock_tokenizer,
            hit_rate=0.05,
            queue_depth=2,
        )
        request = _make_request("target")
        _queue(scheduler, request)
        _fill(scheduler, 2)
        assert _pop_and_decide(scheduler, request) is True
        _queue(scheduler, request)

        stats = scheduler.get_stats()
        assert stats["admission_guard_shed_total"] == 1
        guard = stats["admission_guard"]
        assert guard["enabled"] is True
        assert guard["active"] is True
        assert guard["shed_total"] == 1
        assert guard["episode_sheds"] == 1
        assert guard["hit_rate_ema"] == pytest.approx(0.05)

    def test_disabled_guard_reports_state(self, mock_model, mock_tokenizer):
        scheduler = _make_scheduler(mock_model, mock_tokenizer, enabled=False)
        stats = scheduler.get_stats()
        assert stats["admission_guard"]["enabled"] is False
        assert stats["admission_guard"]["active"] is False
        assert stats["admission_guard_shed_total"] == 0
