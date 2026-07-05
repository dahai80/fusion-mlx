# SPDX-License-Identifier: Apache-2.0
"""Tests for ``prompt_tokens`` plumbing through the MLLM scheduler."""

from __future__ import annotations

from unittest.mock import MagicMock

import mlx.core as mx

from fusion_mlx.mllm_batch_generator import (
    MLLMBatch,
    MLLMBatchGenerator,
    MLLMBatchRequest,
    MLLMBatchResponse,
    MLLMBatchStats,
)
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
    return MLLMScheduler(model=model, processor=processor, config=config)


def _make_mllm_request(scheduler: MLLMScheduler, rid: str):
    from fusion_mlx.mllm_scheduler import MLLMRequest

    req = MLLMRequest(
        request_id=rid,
        prompt="hi",
        num_prompt_tokens=0,
        stop=[],
    )
    req.status = RequestStatus.RUNNING
    scheduler.running[rid] = req
    return req


def test_scheduler_memoises_prompt_tokens_from_first_response():
    scheduler = _make_scheduler()
    req = _make_mllm_request(scheduler, "r1")
    scheduler.uid_to_request_id[0] = "r1"

    assert req.num_prompt_tokens == 0

    response = MagicMock(spec=MLLMBatchResponse)
    response.uid = 0
    response.token = 42
    response.finish_reason = None
    response.logprobs = None
    response.prompt_tokens = 137

    outputs, _finished = scheduler._process_batch_responses([response])
    assert len(outputs) == 1
    assert outputs[0].prompt_tokens == 137, (
        f"RequestOutput.prompt_tokens must reflect the value the batch "
        f"generator stamped on MLLMBatchResponse; got {outputs[0].prompt_tokens}."
    )
    assert req.num_prompt_tokens == 137


def test_scheduler_does_not_overwrite_existing_count():
    scheduler = _make_scheduler()
    req = _make_mllm_request(scheduler, "r1")
    req.num_prompt_tokens = 137
    scheduler.uid_to_request_id[0] = "r1"

    response = MagicMock(spec=MLLMBatchResponse)
    response.uid = 0
    response.token = 7
    response.finish_reason = None
    response.logprobs = None
    response.prompt_tokens = 0

    outputs, _finished = scheduler._process_batch_responses([response])
    assert len(outputs) == 1
    assert outputs[0].prompt_tokens == 137, "must NOT overwrite memoised count with 0"
    assert req.num_prompt_tokens == 137


def test_scheduler_handles_response_without_prompt_tokens_attr():
    scheduler = _make_scheduler()
    req = _make_mllm_request(scheduler, "r2")
    scheduler.uid_to_request_id[0] = "r2"

    response = MagicMock(spec=["uid", "token", "finish_reason", "logprobs"])
    response.uid = 0
    response.token = 5
    response.finish_reason = None
    response.logprobs = None

    outputs, _finished = scheduler._process_batch_responses([response])
    assert len(outputs) == 1
    assert outputs[0].prompt_tokens == 0
    assert req.num_prompt_tokens == 0


def test_next_stamps_prompt_tokens_from_request(monkeypatch):
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._stats = MLLMBatchStats()
    gen.stop_tokens = set()
    gen.unprocessed_requests = []
    gen._shared_batch_sampler = None
    gen.completion_batch_size = 16
    gen.prefill_batch_size = 4
    gen.prefill_step_size = 1024
    gen.sampler = lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)

    def _fake_step(input_tokens, cache, requests):
        return (
            mx.zeros((input_tokens.shape[0],), dtype=mx.uint32),
            [mx.zeros((4,)) for _ in range(input_tokens.shape[0])],
        )

    gen._step = _fake_step

    request_a = MLLMBatchRequest(
        uid=0, request_id="ra", prompt="x", max_tokens=8, num_prompt_tokens=259
    )
    request_b = MLLMBatchRequest(
        uid=1, request_id="rb", prompt="y", max_tokens=8, num_prompt_tokens=12
    )

    gen.active_batch = MLLMBatch(
        uids=[0, 1],
        request_ids=["ra", "rb"],
        y=mx.zeros((2,), dtype=mx.uint32),
        logprobs=[mx.zeros((4,)), mx.zeros((4,))],
        max_tokens=[8, 8],
        num_tokens=[0, 0],
        cache=[],
        requests=[request_a, request_b],
    )

    responses = gen._next()
    assert len(responses) == 2
    assert responses[0].prompt_tokens == 259, (
        f"First response must carry request_a's prompt_tokens=259; "
        f"got {responses[0].prompt_tokens}."
    )
    assert responses[1].prompt_tokens == 12, (
        f"Second response must carry request_b's prompt_tokens=12; "
        f"got {responses[1].prompt_tokens}."
    )


def test_dataclass_fields_present():
    resp = MLLMBatchResponse(uid=1, request_id="x", token=5, logprobs=None)
    assert hasattr(resp, "prompt_tokens")
    assert resp.prompt_tokens == 0

    req = MLLMBatchRequest(uid=1, request_id="x", prompt="hi")
    assert hasattr(req, "num_prompt_tokens")
    assert req.num_prompt_tokens == 0
