# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-model memory plan (L2) + KV-token-ceiling clamp +
the preflight-rejection 413 error_code bridge.

Covers:
- ``compute_model_memory_plan`` produces a plan with a sane kv_token_ceiling.
- ``clamp_max_tokens_to_kv_ceiling`` clamps an over-budget max_tokens and
  leaves a fitting request untouched.
- ``engine_core._raise_request_output_error`` raises
  ``PrefillMemoryExceededError`` iff the RequestOutput carries
  ``error_code="prefill_memory_exceeded"`` — the bridge the
  ``_schedule_waiting`` preflight-rejection stamp relies on to surface 413
  instead of a generic 500.
"""

from unittest.mock import MagicMock

import pytest

from fusion_mlx.memory_plan import compute_model_memory_plan
from fusion_mlx.request import Request, RequestOutput, SamplingParams
from fusion_mlx.scheduler.sched_query import (
    clamp_max_tokens_to_kv_ceiling,
    get_memory_plan,
)


def _bind_real_methods(sched):
    sched.get_memory_plan = get_memory_plan.__get__(sched)
    sched.clamp_max_tokens_to_kv_ceiling = clamp_max_tokens_to_kv_ceiling.__get__(sched)
    sched._memory_plan = None


def _make_sched_with_monitor(*, hard_limit, kv_bytes_per_token, max_pos=4096):
    sched = MagicMock()
    sched._memory_hard_limit_bytes = hard_limit
    sched._memory_plan_weights_bytes = 0
    sched._last_mlx_active_memory_bytes = 0
    mm = MagicMock()
    mm.has_model_info.return_value = True
    mm.estimate_decode_kv_bytes.return_value = kv_bytes_per_token
    sched.memory_monitor = mm
    cfg = MagicMock()
    cfg.max_position_embeddings = max_pos
    cfg.text_config = None
    cfg.language_config = None
    cfg.llm_config = None
    model = MagicMock()
    model.config = cfg
    sched.model = model
    sched._stream = None
    _bind_real_methods(sched)
    return sched


def test_compute_plan_returns_kv_token_ceiling(monkeypatch):
    hard_limit = 10 * 1024**3
    kv_bpt = 256 * 1024
    sched = _make_sched_with_monitor(
        hard_limit=hard_limit, kv_bytes_per_token=kv_bpt, max_pos=32768
    )
    monkeypatch.setattr("fusion_mlx.memory_plan._current_usage_bytes", lambda s: 0)

    plan = compute_model_memory_plan(sched)
    assert plan is not None
    assert plan.kv_bytes_per_token == kv_bpt
    expected_ceiling = hard_limit // kv_bpt
    assert plan.kv_token_ceiling == expected_ceiling
    assert plan.max_context_per_request == min(32768, expected_ceiling)


def test_compute_plan_returns_none_without_monitor():
    sched = MagicMock()
    sched.memory_monitor = None
    assert compute_model_memory_plan(sched) is None


def test_clamp_reduces_over_budget_max_tokens(monkeypatch):
    hard_limit = 10 * 1024**3
    kv_bpt = 256 * 1024
    sched = _make_sched_with_monitor(
        hard_limit=hard_limit, kv_bytes_per_token=kv_bpt, max_pos=200000
    )
    monkeypatch.setattr("fusion_mlx.memory_plan._current_usage_bytes", lambda s: 0)

    ceiling = hard_limit // kv_bpt
    req = Request(
        request_id="r1",
        prompt="hi",
        prompt_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
        sampling_params=SamplingParams(max_tokens=ceiling + 1000),
    )
    clamped = sched.clamp_max_tokens_to_kv_ceiling(req)
    assert clamped is True
    assert req.sampling_params.max_tokens == ceiling - 3


def test_clamp_leaves_fitting_request_untouched(monkeypatch):
    hard_limit = 10 * 1024**3
    kv_bpt = 256 * 1024
    sched = _make_sched_with_monitor(hard_limit=hard_limit, kv_bytes_per_token=kv_bpt)
    monkeypatch.setattr("fusion_mlx.memory_plan._current_usage_bytes", lambda s: 0)

    ceiling = hard_limit // kv_bpt
    req = Request(
        request_id="r2",
        prompt="hi",
        prompt_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
        sampling_params=SamplingParams(max_tokens=min(500, ceiling - 3)),
    )
    clamped = sched.clamp_max_tokens_to_kv_ceiling(req)
    assert clamped is False


def test_clamp_noop_when_plan_unavailable():
    sched = MagicMock()
    sched.get_memory_plan = MagicMock(return_value=None)
    sched.clamp_max_tokens_to_kv_ceiling = clamp_max_tokens_to_kv_ceiling.__get__(sched)
    req = Request(
        request_id="r3",
        prompt="hi",
        prompt_token_ids=[1],
        num_prompt_tokens=1,
        sampling_params=SamplingParams(max_tokens=99999),
    )
    assert sched.clamp_max_tokens_to_kv_ceiling(req) is False


def test_raise_request_output_error_bridge_raises_on_error_code():
    from fusion_mlx.engine_core import _raise_request_output_error
    from fusion_mlx.exceptions import PrefillMemoryExceededError

    output = RequestOutput(
        request_id="bridge-1",
        finished=True,
        finish_reason="error",
        error="prefill memory exceeded",
        error_code="prefill_memory_exceeded",
        error_metadata={"estimated_bytes": 999, "limit_bytes": 100},
    )
    with pytest.raises(PrefillMemoryExceededError):
        _raise_request_output_error(output)


def test_raise_request_output_error_bridge_no_code_does_not_raise_prefill():
    from fusion_mlx.engine_core import _raise_request_output_error
    from fusion_mlx.exceptions import PrefillMemoryExceededError

    output = RequestOutput(
        request_id="bridge-2",
        finished=True,
        finish_reason="error",
        error="some other error",
        error_code=None,
    )
    with pytest.raises(Exception) as exc_info:
        _raise_request_output_error(output)
    assert not isinstance(exc_info.value, PrefillMemoryExceededError)


def test_schedule_waiting_rejection_stamps_error_code(monkeypatch):
    """The _schedule_waiting preflight-rejection path must stamp
    error_code='prefill_memory_exceeded' so engine_core raises
    PrefillMemoryExceededError -> HTTP 413 instead of a generic 500.
    """
    from fusion_mlx.scheduler import Scheduler, SchedulerConfig
    from fusion_mlx.scheduler.sched_query import _PreflightRejection

    model = MagicMock()
    model.layers = []
    model.config = MagicMock(max_position_embeddings=4096)
    del model.make_cache
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    config = SchedulerConfig(max_num_seqs=8, prefill_step_size=2048)
    sched = Scheduler(model=model, tokenizer=tokenizer, config=config)
    sched._prefill_memory_guard = True
    sched._memory_hard_limit_bytes = 1

    monkeypatch.setattr(
        "fusion_mlx.scheduler.sched_query.mx", MagicMock(get_active_memory=lambda: 0)
    )
    monkeypatch.setattr(
        "fusion_mlx.scheduler.sched_query.get_phys_footprint", lambda *a, **k: 0
    )
    monkeypatch.setattr(
        sched,
        "_preflight_memory_check",
        lambda req: _PreflightRejection(
            message="boom", estimated_bytes=999, limit_bytes=1
        ),
    )
    monkeypatch.setattr(
        sched, "_release_paged_cache_for_request", lambda *a, **kw: None
    )
    monkeypatch.setattr(sched, "clamp_max_tokens_to_kv_ceiling", lambda req: False)

    req = Request(
        request_id="stamp-1",
        prompt="hi",
        prompt_token_ids=[1, 2],
        num_prompt_tokens=2,
        sampling_params=SamplingParams(max_tokens=10),
    )
    sched.add_request(req)
    result = sched.step()
    outputs = (
        getattr(result, "outputs", result) if not isinstance(result, list) else result
    )
    rejected = [o for o in outputs if o.request_id == "stamp-1" and o.finished]
    assert rejected, "expected a rejected output"
    assert rejected[0].error_code == "prefill_memory_exceeded"
