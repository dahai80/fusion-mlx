# SPDX-License-Identifier: Apache-2.0
"""#1480 item 2b: cooperative training-job abort. The registry
(training/abort.py) tracks active in-process training jobs; the memory
enforcer requests an abort when sustained soft pressure has no idle AND no
busy non-pinned victim (the training job itself is the pressure source).
The train loops check between iterations and fail the job.
"""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest

from fusion_mlx.training import abort as ta
from fusion_mlx.training.abort import (
    TrainingAbortedError,
    abort_all_active,
    abort_enabled,
    abort_reason,
    active_jobs,
    check_abort,
    deregister_job,
    register_job,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    with ta._lock:
        ta._active.clear()
    yield
    with ta._lock:
        ta._active.clear()


def _import_memory_enforcer():
    from fusion_mlx.pool import memory_enforcer as me

    return me


class _NoVictimPool:
    # No loaded entries at all: no idle victim, no busy victim — the
    # training-job-is-the-pressure-source shape.

    def __init__(self):
        self._lock = asyncio.Lock()
        self._entries = {}

    def _find_lru_victim(self):
        return None

    def _find_lru_busy_non_pinned_victim_locked(self):
        return None

    def _find_pending_unload_ready_locked(self):
        return None

    def _entry_has_active_requests(self, entry):
        return False


def _make_enforcer(me, pool):
    enforcer = me.ProcessMemoryEnforcer.__new__(me.ProcessMemoryEnforcer)
    enforcer._engine_pool = pool
    enforcer._soft_threshold = 0.85
    enforcer._hard_threshold = 0.95
    enforcer._pressure_level = "ok"
    enforcer._over_ceiling_polls = 0
    enforcer._unrecoverable_polls = 0
    enforcer._no_models_loaded_cycles = 0
    enforcer._soft_busy_cycles = 0
    enforcer._train_abort_fired = False
    enforcer._hard_busy_victim = None
    enforcer._hard_busy_cycles = 0
    enforcer._eviction_marked = set()
    enforcer._deferred_victim_pick = False
    enforcer._state_lock = threading.Lock()
    enforcer._last_evicted_at = {}
    enforcer._drain_flipped = False
    enforcer._propagate_memory_limit = lambda: None
    enforcer._walk_store_cache_caps = lambda: None
    enforcer._shrink_hot_cache_for_pressure = lambda current, target: 0
    enforcer.get_media_reservation_bytes = lambda: 0
    # 100GB ceiling, usage 90GB: soft (85-95GB), never hard.
    enforcer._get_hard_limit_bytes = lambda: 100 * 1024**3
    enforcer._current_usage_bytes = lambda: 90 * 1024**3
    return enforcer


def _run_check(enforcer):
    async def _go():
        enforcer._engine_pool._lock = asyncio.Lock()
        await enforcer._check_and_enforce()

    asyncio.run(_go())


# =========================================================================
# Registry unit tests
# =========================================================================


def test_register_check_abort_raises_with_reason():
    register_job("job-1", "dpo")
    assert active_jobs() == [("job-1", "dpo")]
    check_abort("job-1")
    assert abort_all_active("test reason") == 1
    assert abort_reason("job-1") == "test reason"
    with pytest.raises(TrainingAbortedError, match="test reason"):
        check_abort("job-1")
    deregister_job("job-1")
    # Deregistered: check is a no-op even if the id is reused.
    check_abort("job-1")


def test_abort_all_active_only_pending():
    register_job("job-a", "dpo")
    register_job("job-b", "grpo")
    assert ta.request_abort("job-a", "first") is True
    # Already-requested job is not re-counted.
    assert ta.request_abort("job-a", "second") is False
    assert abort_all_active("second wave") == 1
    assert abort_reason("job-a") == "first"
    assert abort_reason("job-b") == "second wave"
    with pytest.raises(TrainingAbortedError, match="second wave"):
        check_abort("job-b")


def test_abort_all_active_empty_registry():
    assert abort_all_active("nothing running") == 0


def test_abort_enabled_knob(monkeypatch):
    monkeypatch.delenv("FUSION_MLX_ENFORCER_ABORT_TRAINING", raising=False)
    assert abort_enabled() is True
    for off in ("0", "false", "off", "no", "OFF"):
        monkeypatch.setenv("FUSION_MLX_ENFORCER_ABORT_TRAINING", off)
        assert abort_enabled() is False
    monkeypatch.setenv("FUSION_MLX_ENFORCER_ABORT_TRAINING", "1")
    assert abort_enabled() is True


# =========================================================================
# Enforcer trigger
# =========================================================================


@pytest.fixture()
def abort_env(monkeypatch):
    me = _import_memory_enforcer()
    monkeypatch.setattr(me, "_SOFT_BUSY_EVICT_POLLS", 2)
    monkeypatch.delenv("FUSION_MLX_ENFORCER_ABORT_TRAINING", raising=False)
    return me


def test_enforcer_requests_training_abort_at_threshold(abort_env, caplog):
    me = abort_env
    enforcer = _make_enforcer(me, _NoVictimPool())
    register_job("train-job", "dpo")
    with caplog.at_level(logging.CRITICAL, logger="fusion_mlx.pool.memory_enforcer"):
        _run_check(enforcer)
        _run_check(enforcer)
    assert abort_reason("train-job") is not None
    assert "#1480" in abort_reason("train-job")
    assert enforcer._train_abort_fired is True
    critical = [r for r in caplog.records if "#1480 item 2b" in r.getMessage()]
    assert len(critical) == 1, "CRITICAL abort log fires once per episode"
    assert "train-job" in critical[0].getMessage()


def test_enforcer_abort_not_repeated_within_episode(abort_env):
    me = abort_env
    enforcer = _make_enforcer(me, _NoVictimPool())
    register_job("train-job", "grpo")
    _run_check(enforcer)
    _run_check(enforcer)
    assert enforcer._train_abort_fired is True
    # More sustained pressure: no new request (job already signalled).
    _run_check(enforcer)
    assert active_jobs() == [("train-job", "grpo")]
    with pytest.raises(TrainingAbortedError):
        check_abort("train-job")


def test_enforcer_abort_skipped_without_training_jobs(abort_env, caplog):
    me = abort_env
    enforcer = _make_enforcer(me, _NoVictimPool())
    with caplog.at_level(logging.CRITICAL, logger="fusion_mlx.pool.memory_enforcer"):
        _run_check(enforcer)
        _run_check(enforcer)
        _run_check(enforcer)
    assert enforcer._train_abort_fired is False
    assert not [
        r for r in caplog.records if "#1480 item 2b" in r.getMessage()
    ], "no active training job — no abort request, no CRITICAL spam"


def test_enforcer_abort_disabled_by_knob(abort_env, monkeypatch, caplog):
    me = abort_env
    monkeypatch.setenv("FUSION_MLX_ENFORCER_ABORT_TRAINING", "0")
    enforcer = _make_enforcer(me, _NoVictimPool())
    register_job("train-job", "rft")
    with caplog.at_level(logging.CRITICAL, logger="fusion_mlx.pool.memory_enforcer"):
        _run_check(enforcer)
        _run_check(enforcer)
    assert abort_reason("train-job") is None
    assert enforcer._train_abort_fired is False
    assert not [r for r in caplog.records if "#1480 item 2b" in r.getMessage()]


def test_enforcer_abort_flag_resets_on_ok(abort_env):
    me = abort_env
    enforcer = _make_enforcer(me, _NoVictimPool())
    register_job("train-job", "dpo")
    _run_check(enforcer)
    _run_check(enforcer)
    assert enforcer._train_abort_fired is True
    # Pressure clears: the ok path re-arms the once-per-episode flag.
    enforcer._pressure_level = "soft"  # current level != ok path entry
    enforcer._current_usage_bytes = lambda: 50 * 1024**3
    _run_check(enforcer)
    assert enforcer._train_abort_fired is False
    assert enforcer._soft_busy_cycles == 0


# =========================================================================
# Service-loop wiring (source-order assertions — the execute paths load
# real MLX models, too heavy for unit tests; AST-free substring order
# checks keep the wiring honest without mocks)
# =========================================================================


def _source(path):
    import fusion_mlx

    root = fusion_mlx.__file__.rsplit("/fusion_mlx/", 1)[0]
    with open(f"{root}/fusion_mlx/{path}") as f:
        return f.read()


def test_dpo_service_abort_wiring():
    src = _source("training/dpo_service.py")
    assert src.index("register_job(job.job_id") < src.index("self._run_dpo_inner(")
    assert src.index("self._run_dpo_inner(") < src.index("deregister_job(job.job_id)")
    assert "check_abort(job.job_id)" in src
    # Check sits inside the train loop, before the step.
    loop_start = src.index("for it in range(cfg.iters):")
    check = src.index("check_abort(job.job_id)")
    step = src.index("result = trainer.train_step(batch)", loop_start)
    assert loop_start < check < step


def test_grpo_service_abort_wiring():
    src = _source("training/grpo_service.py")
    assert src.index("register_job(job.job_id") < src.index("self._run_grpo(job)")
    assert src.index("self._run_grpo(job)") < src.index("deregister_job(job.job_id)")
    loop_start = src.index("for it in range(cfg.iters):")
    check = src.index("check_abort(job.job_id)")
    step = src.index("result = trainer.train_step(batch)", loop_start)
    assert loop_start < check < step


def test_rft_service_abort_wiring():
    src = _source("training/rft_service.py")
    assert src.index("register_job(job.job_id") < src.index("self._run_rft(job)")
    assert src.index("self._run_rft(job)") < src.index("deregister_job(job.job_id)")
    loop_start = src.index("for it in range(cfg.iters):")
    check = src.index("check_abort(job.job_id)")
    step = src.index("result = trainer.train_step(batch)", loop_start)
    assert loop_start < check < step


def test_reward_service_abort_wiring():
    src = _source("training/reward_service.py")
    assert src.index("register_job(job.job_id") < src.index("self._run_reward_inner(")
    assert src.index("self._run_reward_inner(") < src.index(
        "deregister_job(job.job_id)"
    )
    loop_start = src.index("for it in range(cfg.iters):")
    check = src.index("check_abort(job.job_id)")
    step = src.index("result = trainer.train_step(batch)", loop_start)
    assert loop_start < check < step


def test_sft_service_abort_wiring():
    src = _source("training/service.py")
    # Thread path registers around the training body...
    assert src.index("register_job(job.job_id") < src.index(
        "self._run_training_thread("
    )
    assert src.index("self._run_training_thread(") < src.index(
        "deregister_job(job.job_id)"
    )
    # ...and the progress callback checks on every event.
    push = src.index("def _push(self, event: dict):")
    assert src.index("check_abort(self._job.job_id)", push) < src.index(
        "self._job.events.append(event)", push
    )


def test_vlm_thread_mode_not_registered():
    src = _source("training/vlm_service.py")
    assert "register_job" not in src, (
        "VLM thread mode must stay OUT of the abort registry: mlx_vlm's "
        "train() is an opaque single call — registering would promise an "
        "abort that can never fire (see the #1480 note in _execute_training)."
    )
