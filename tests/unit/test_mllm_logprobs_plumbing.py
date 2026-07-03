# SPDX-License-Identifier: Apache-2.0
"""Tests for ``logprobs`` plumbing through the MLLM scheduler."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fusion_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
from fusion_mlx.request import RequestStatus


def _make_scheduler() -> MLLMScheduler:
    model = MagicMock()
    processor = MagicMock()
    processor.tokenizer = MagicMock()
    processor.tokenizer.eos_token_id = 0
    processor.tokenizer.eos_token_ids = None
    processor.tokenizer._eos_token_ids = None
    processor.tokenizer.decode = lambda toks: "hello world"
    config = MLLMSchedulerConfig(max_num_seqs=2)
    scheduler = MLLMScheduler(model=model, processor=processor, config=config)
    return scheduler


def _make_mllm_request(scheduler: MLLMScheduler, rid: str):
    from fusion_mlx.mllm_scheduler import MLLMRequest

    req = MLLMRequest(
        request_id=rid,
        prompt="hi",
        num_prompt_tokens=4,
        stop=[],
    )
    req.status = RequestStatus.RUNNING
    scheduler.running[rid] = req
    return req


def test_mllm_response_logprobs_reach_request_output():
    scheduler = _make_scheduler()
    _make_mllm_request(scheduler, "r1")
    scheduler.uid_to_request_id[0] = "r1"

    fake_logprobs = "<sentinel-logprobs>"
    response = MagicMock()
    response.uid = 0
    response.token = 42
    response.finish_reason = None
    response.logprobs = fake_logprobs

    outputs, _finished = scheduler._process_batch_responses([response])
    assert len(outputs) == 1
    assert outputs[0].logprobs is fake_logprobs


def test_mllm_logprobs_field_present_even_when_response_lacks_attr():
    scheduler = _make_scheduler()
    _make_mllm_request(scheduler, "r2")
    scheduler.uid_to_request_id[0] = "r2"

    response = MagicMock(spec=["uid", "token", "finish_reason"])
    response.uid = 0
    response.token = 7
    response.finish_reason = None

    outputs, _finished = scheduler._process_batch_responses([response])
    assert len(outputs) == 1
    assert outputs[0].logprobs is None


class _RecordingVLMModel:

    def __init__(self):
        self.language_model = object()


def test_mllm_batch_generator_init_does_not_call_new_stream(monkeypatch):
    import mlx.core as mx

    from fusion_mlx.mllm_batch_generator import MLLMBatchGenerator

    monkeypatch.setattr(MLLMBatchGenerator, "_stream", None)

    new_stream_calls: list[object] = []

    def _trap_new_stream(device):
        new_stream_calls.append(device)
        raise AssertionError(
            "MLLMBatchGenerator.__init__ called mx.new_stream — under "
            "mlx-lm 0.31.3+ the resulting stream is bound to the "
            "constructing thread and the logprobs mx.array crashes the "
            "route-handler thread on cross-thread np.array(...). "
            "Use mx.default_stream(mx.default_device()) instead."
        )

    monkeypatch.setattr(mx, "new_stream", _trap_new_stream)

    gen = MLLMBatchGenerator(
        model=_RecordingVLMModel(),
        processor=object(),
        mm_processor=None,
        enable_vision_cache=False,
    )

    try:
        assert new_stream_calls == [], (
            "Construction must not allocate a new mx.stream. "
            "Saw mx.new_stream calls: " + repr(new_stream_calls)
        )
        expected = mx.default_stream(mx.default_device())
        assert MLLMBatchGenerator._stream is not None
        assert MLLMBatchGenerator._stream == expected, (
            "MLLMBatchGenerator._stream must be the worker default "
            "stream (process-wide, materialisable from any thread). "
            f"Got: {MLLMBatchGenerator._stream!r}, expected: {expected!r}"
        )
    finally:
        MLLMBatchGenerator._stream = None


def test_mllm_next_evals_outgoing_logprobs_before_response(monkeypatch):
    import mlx.core as mx

    from fusion_mlx.mllm_batch_generator import (
        MLLMBatch,
        MLLMBatchGenerator,
        MLLMBatchRequest,
        MLLMBatchStats,
    )

    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._stats = MLLMBatchStats()
    gen.stop_tokens = set()
    gen.unprocessed_requests = []
    gen._shared_batch_sampler = None
    gen.completion_batch_size = 16
    gen.prefill_batch_size = 4
    gen.prefill_step_size = 1024
    gen.sampler = lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)

    next_step_logprobs = [mx.zeros((4,)) for _ in range(2)]

    def _fake_step(input_tokens, cache, requests):
        return (
            mx.zeros((input_tokens.shape[0],), dtype=mx.uint32),
            next_step_logprobs,
        )

    gen._step = _fake_step

    sentinel_prev_logprobs = [mx.zeros((4,)), mx.zeros((4,))]
    request_a = MLLMBatchRequest(uid=0, request_id="ra", prompt="x", max_tokens=8)
    request_b = MLLMBatchRequest(uid=1, request_id="rb", prompt="y", max_tokens=8)
    gen.active_batch = MLLMBatch(
        uids=[0, 1],
        request_ids=["ra", "rb"],
        y=mx.zeros((2,), dtype=mx.uint32),
        logprobs=sentinel_prev_logprobs,
        max_tokens=[8, 8],
        num_tokens=[0, 0],
        cache=[],
        requests=[request_a, request_b],
    )

    eval_args: list[tuple] = []
    orig_eval = mx.eval

    def _record_eval(*args, **kwargs):
        eval_args.append(args)
        return orig_eval(*args, **kwargs)

    monkeypatch.setattr(mx, "eval", _record_eval)

    responses = gen._next()

    assert eval_args, (
        "_next() did not invoke mx.eval at all — the fix that forces "
        "the outgoing per-step logprobs array to materialise on the "
        "worker thread is missing."
    )
    matched = any(
        len(args) == 1 and args[0] is sentinel_prev_logprobs for args in eval_args
    )
    assert matched, (
        "_next() called mx.eval but NOT on the outgoing logprobs array "
        "that gets sliced into MLLMBatchResponse. The regression "
        "target is the cross-thread crash on the exact per-step "
        "logprob slice — evaluating a different variable does not "
        "close the bug. Recorded mx.eval call args: "
        + repr([[type(a).__name__ for a in args] for args in eval_args])
    )

    assert len(responses) == 2
    assert responses[0].logprobs is sentinel_prev_logprobs[0]
    assert responses[1].logprobs is sentinel_prev_logprobs[1]
