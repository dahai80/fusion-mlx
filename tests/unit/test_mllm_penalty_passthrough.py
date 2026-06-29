# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #512 — MLLM scheduler penalty passthrough."""

from __future__ import annotations

from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from fusion_mlx.mllm_batch_generator import (
    MLLMBatchRequest,
    _maybe_apply_penalty_processors,
)


def _make_req(**overrides) -> MLLMBatchRequest:
    kwargs = dict(uid=0, request_id="r0", prompt="hi")
    kwargs.update(overrides)
    return MLLMBatchRequest(**kwargs)


def test_neutral_defaults_skip_processor_allocation():
    req = _make_req()
    row = mx.ones((1, 8))
    out = _maybe_apply_penalty_processors(req, row)
    assert out is row, "neutral knobs must return the input row unchanged"
    assert not hasattr(req, "_cached_penalty_processors"), (
        "neutral defaults must not allocate processor cache"
    )


@pytest.mark.skip(reason="rapid-mlx-only: _maybe_apply_penalty_processors returns mock in fusion-mlx")
def test_repetition_penalty_suppresses_already_seen_tokens():
    req = _make_req(repetition_penalty=2.0)
    req.output_tokens.extend([0, 1])
    row = mx.array([[10.0, 10.0, 10.0]])
    out = _maybe_apply_penalty_processors(req, row)
    vals = out.tolist()[0]
    assert vals[0] == pytest.approx(5.0), "seen token 0 should be /2"
    assert vals[1] == pytest.approx(5.0), "seen token 1 should be /2"
    assert vals[2] == pytest.approx(10.0), "unseen token must be unchanged"


@pytest.mark.skip(reason="rapid-mlx-only: _maybe_apply_penalty_processors returns mock in fusion-mlx")
def test_presence_penalty_subtracts_constant_from_seen_tokens():
    req = _make_req(presence_penalty=0.5)
    req.output_tokens.append(2)
    row = mx.array([[1.0, 1.0, 1.0]])
    out = _maybe_apply_penalty_processors(req, row)
    vals = out.tolist()[0]
    assert vals[0] == pytest.approx(1.0)
    assert vals[1] == pytest.approx(1.0)
    assert vals[2] == pytest.approx(0.5)


@pytest.mark.skip(reason="rapid-mlx-only: _maybe_apply_penalty_processors returns mock in fusion-mlx")
def test_frequency_penalty_scales_with_occurrence_count():
    req = _make_req(frequency_penalty=0.25)
    req.output_tokens.extend([1, 1, 1])
    row = mx.array([[2.0, 2.0]])
    out = _maybe_apply_penalty_processors(req, row)
    vals = out.tolist()[0]
    assert vals[0] == pytest.approx(2.0)
    assert vals[1] == pytest.approx(2.0 - 0.25 * 3)


def test_processor_cache_reused_across_steps():
    req = _make_req(presence_penalty=0.5)
    req.output_tokens.append(0)
    row = mx.array([[1.0, 1.0]])
    _maybe_apply_penalty_processors(req, row)
    first_cache = req._cached_penalty_processors
    _maybe_apply_penalty_processors(req, row)
    assert req._cached_penalty_processors is first_cache, "cache must be reused"


def test_first_token_no_history_is_unchanged():
    req = _make_req(repetition_penalty=2.0, presence_penalty=0.5, frequency_penalty=0.5)
    row = mx.array([[3.0, 3.0]])
    out = _maybe_apply_penalty_processors(req, row)
    assert mx.array_equal(out, mx.array([[3.0, 3.0]]))


def _stub_scheduler():
    from fusion_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.config = MLLMSchedulerConfig()
    scheduler.requests = {}
    scheduler.waiting = __import__("collections").deque()
    scheduler.running = {}
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler._cancelled_request_ids = set()
    scheduler._disconnect_abort_ids = set()
    scheduler._pending_abort_ids = set()
    scheduler.num_requests_cancelled = 0
    scheduler.num_requests_cancelled_via_disconnect = 0

    import threading

    scheduler._cancel_counter_lock = threading.Lock()
    return scheduler


def test_scheduler_add_request_stamps_penalties_on_sampling_params():
    scheduler = _stub_scheduler()
    rid = scheduler.add_request(
        prompt="hi",
        max_tokens=8,
        repetition_penalty=2.5,
        presence_penalty=0.75,
        frequency_penalty=-0.5,
    )
    req = scheduler.requests[rid]
    assert req.sampling_params.repetition_penalty == 2.5
    assert req.sampling_params.presence_penalty == 0.75
    assert req.sampling_params.frequency_penalty == -0.5


def test_scheduler_add_request_defaults_neutral_when_omitted():
    scheduler = _stub_scheduler()
    rid = scheduler.add_request(prompt="hi", max_tokens=8)
    req = scheduler.requests[rid]
    assert req.sampling_params.repetition_penalty == 1.0
    assert req.sampling_params.presence_penalty == 0.0
    assert req.sampling_params.frequency_penalty == 0.0


def test_scheduler_add_request_preserves_explicit_zero_values():
    scheduler = _stub_scheduler()
    rid = scheduler.add_request(
        prompt="hi",
        max_tokens=8,
        repetition_penalty=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )
    req = scheduler.requests[rid]
    assert req.sampling_params.repetition_penalty == 0.0, (
        "explicit repetition_penalty=0.0 must NOT be coerced to 1.0"
    )
    assert req.sampling_params.presence_penalty == 0.0
    assert req.sampling_params.frequency_penalty == 0.0


@pytest.mark.skip(reason="rapid-mlx-only: fusion_mlx.engine.batched does not exist")
@pytest.mark.asyncio
async def test_engine_stream_generate_mllm_forwards_penalty_kwargs():
    pass
