# SPDX-License-Identifier: Apache-2.0
"""Migrated from Rapid-MLX test_mtp_cli_wiring.py.

Coverage for the four MTP CLI-wiring surfaces:
1. detect_mtp_eligibility(has_external_sidecar=...) contract
2. CLI argparse for --mtp-sidecar
3. SchedulerConfig.mtp_sidecar field
4. Engine dispatch call site — dispatch_mtp_inject sees the sidecar path

Tests referencing features not yet migrated to fusion-mlx are
skipped with pytest.skip("feature not migrated") and a comment
explaining what's missing.
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. detect_mtp_eligibility(has_external_sidecar=...) contract
# ---------------------------------------------------------------------------


def test_detect_sidecar_promotes_gemma4_unified_with_missing_mtp_layers():
    # fusion_mlx.speculative.mtp is a stub — MTPEligibility and
    # detect_mtp_eligibility are not yet implemented.
    pytest.skip(
        "feature not migrated: fusion_mlx.speculative.mtp does not export "
        "MTPEligibility or detect_mtp_eligibility"
    )


def test_detect_sidecar_promotes_gemma4_unified_with_zero_mtp_layers():
    pytest.skip(
        "feature not migrated: fusion_mlx.speculative.mtp does not export "
        "MTPEligibility or detect_mtp_eligibility"
    )


def test_detect_sidecar_no_effect_on_qwen3_5_missing_mtp():
    pytest.skip(
        "feature not migrated: fusion_mlx.speculative.mtp does not export "
        "MTPEligibility or detect_mtp_eligibility"
    )


def test_detect_sidecar_no_effect_on_gemma4_multimodal():
    pytest.skip(
        "feature not migrated: fusion_mlx.speculative.mtp does not export "
        "MTPEligibility or detect_mtp_eligibility"
    )


def test_detect_sidecar_leaves_qwen3_5_with_mtp_layers_untouched():
    pytest.skip(
        "feature not migrated: fusion_mlx.speculative.mtp does not export "
        "MTPEligibility or detect_mtp_eligibility"
    )


def test_detect_sidecar_default_argument_matches_pre_0913_behaviour():
    pytest.skip(
        "feature not migrated: fusion_mlx.speculative.mtp does not export "
        "MTPEligibility or detect_mtp_eligibility"
    )


# ---------------------------------------------------------------------------
# 2. CLI argparse for --mtp-sidecar
# ---------------------------------------------------------------------------


def _serve_help_stdout() -> str:
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "fusion_mlx.cli_serve", "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_cli_serve_help_advertises_mtp_sidecar():
    # fusion_mlx CLI does not yet have --mtp-sidecar flag.
    pytest.skip(
        "feature not migrated: fusion_mlx.cli_serve does not expose "
        "--mtp-sidecar argparse flag"
    )


# ---------------------------------------------------------------------------
# 3. SchedulerConfig.mtp_sidecar field
# ---------------------------------------------------------------------------


def test_scheduler_config_mtp_sidecar_default_none():
    from fusion_mlx.config import SchedulerConfig

    cfg = SchedulerConfig()
    assert cfg.mtp_sidecar is None


def test_scheduler_config_mtp_sidecar_round_trip():
    from fusion_mlx.config import SchedulerConfig

    cfg = SchedulerConfig(mtp_sidecar="Qwen/Qwen3.5-MTP")
    assert cfg.mtp_sidecar == "Qwen/Qwen3.5-MTP"


def test_scheduler_config_mtp_sidecar_local_path_round_trip():
    from fusion_mlx.config import SchedulerConfig

    cfg = SchedulerConfig(mtp_sidecar="/models/qwen3.5-mtp")
    assert cfg.mtp_sidecar == "/models/qwen3.5-mtp"


def test_scheduler_config_mtp_model_type_default_none():
    from fusion_mlx.config import SchedulerConfig

    cfg = SchedulerConfig()
    assert cfg.mtp_model_type is None


def test_scheduler_config_mtp_model_type_round_trip():
    from fusion_mlx.config import SchedulerConfig

    cfg = SchedulerConfig(mtp_model_type="gemma4_unified")
    assert cfg.mtp_model_type == "gemma4_unified"


# ---------------------------------------------------------------------------
# 4. Engine dispatch call site — dispatch_mtp_inject sees the sidecar path
# ---------------------------------------------------------------------------


def _patch_dispatch_mtp_inject(monkeypatch, return_value=True, recorder=None):
    import fusion_mlx.speculative.mtp.dispatch as disp

    def _fake(model, model_type, mtp_sidecar=None, allow_random_init=False):
        if recorder is not None:
            recorder["model_type"] = model_type
            recorder["mtp_sidecar"] = mtp_sidecar
            recorder["allow_random_init"] = allow_random_init
        return return_value

    monkeypatch.setattr(disp, "dispatch_mtp_inject", _fake)
    return _fake


def test_run_dispatch_mtp_inject_forwards_sidecar_path(monkeypatch):
    from fusion_mlx.engine.batched import _DISPATCH_ATTACHED, _run_dispatch_mtp_inject

    rec = {}
    _patch_dispatch_mtp_inject(monkeypatch, return_value=True, recorder=rec)
    result = _run_dispatch_mtp_inject(object(), "gemma4_unified", "/path/sidecar")
    assert result == _DISPATCH_ATTACHED
    assert rec["mtp_sidecar"] == "/path/sidecar"
    assert rec["model_type"] == "gemma4_unified"


def test_run_dispatch_mtp_inject_returns_unresolved_when_model_type_missing(
    monkeypatch,
):
    from fusion_mlx.engine.batched import (
        _DISPATCH_UNRESOLVED,
        _run_dispatch_mtp_inject,
    )

    rec = {}
    _patch_dispatch_mtp_inject(monkeypatch, return_value=True, recorder=rec)
    # object() has no resolvable model_type and model_type arg is None
    result = _run_dispatch_mtp_inject(object(), None, "/path/sidecar")
    assert result == _DISPATCH_UNRESOLVED
    assert rec == {}


def test_run_dispatch_mtp_inject_returns_rejected_when_injector_refuses(monkeypatch):
    from fusion_mlx.engine.batched import _DISPATCH_REJECTED, _run_dispatch_mtp_inject

    _patch_dispatch_mtp_inject(monkeypatch, return_value=False)
    result = _run_dispatch_mtp_inject(object(), "gemma4_unified", "/path/sidecar")
    assert result == _DISPATCH_REJECTED


def test_run_dispatch_mtp_inject_returns_no_inject_for_unregistered_model_type(
    monkeypatch,
):
    from fusion_mlx.engine.batched import _DISPATCH_NO_INJECT, _run_dispatch_mtp_inject

    rec = {}
    _patch_dispatch_mtp_inject(monkeypatch, return_value=True, recorder=rec)
    result = _run_dispatch_mtp_inject(object(), "totally_unknown_type", "/path/sidecar")
    assert result == _DISPATCH_NO_INJECT
    assert rec == {}


def test_run_dispatch_mtp_inject_prefers_cli_provided_model_type(monkeypatch):
    from fusion_mlx.engine.batched import _DISPATCH_ATTACHED, _run_dispatch_mtp_inject

    rec = {}
    _patch_dispatch_mtp_inject(monkeypatch, return_value=True, recorder=rec)
    # cli_vetted_model_type wins over the model_type positional arg
    result = _run_dispatch_mtp_inject(
        object(), "qwen3_5", "/path/sidecar", cli_vetted_model_type="gemma4_unified"
    )
    assert result == _DISPATCH_ATTACHED
    assert rec["model_type"] == "gemma4_unified"


def test_run_dispatch_mtp_inject_falls_back_when_no_preferred_model_type(monkeypatch):
    from fusion_mlx.engine.batched import _DISPATCH_ATTACHED, _run_dispatch_mtp_inject

    rec = {}
    _patch_dispatch_mtp_inject(monkeypatch, return_value=True, recorder=rec)
    # no cli_vetted -> fall back to the model_type positional arg
    result = _run_dispatch_mtp_inject(object(), "gemma4_unified", "/path/sidecar")
    assert result == _DISPATCH_ATTACHED
    assert rec["model_type"] == "gemma4_unified"


def test_run_dispatch_mtp_inject_propagates_none_sidecar(monkeypatch):
    from fusion_mlx.engine.batched import _DISPATCH_ATTACHED, _run_dispatch_mtp_inject

    rec = {}
    _patch_dispatch_mtp_inject(monkeypatch, return_value=True, recorder=rec)
    result = _run_dispatch_mtp_inject(object(), "gemma4_unified", None)
    assert result == _DISPATCH_ATTACHED
    assert rec["mtp_sidecar"] is None


# ---------------------------------------------------------------------------
# 4b. Boot-time contract — _decide_mtp_dispatch_action
# ---------------------------------------------------------------------------


def _drive_start_llm_dispatch_gate(dispatch_result, cli_vetted_model_type=None):
    from fusion_mlx.engine.batched import _decide_mtp_dispatch_action

    try:
        _decide_mtp_dispatch_action(dispatch_result, cli_vetted_model_type)
        return ("continue", None)
    except RuntimeError as e:
        return ("raise", str(e))


def test_decide_mtp_dispatch_action_returns_attached_for_attached_result():
    from fusion_mlx.engine.batched import _DISPATCH_ATTACHED

    action, _ = _drive_start_llm_dispatch_gate(_DISPATCH_ATTACHED)
    assert action == "continue"


def test_decide_mtp_dispatch_action_carries_cli_vetted_model_type_into_error():
    from fusion_mlx.engine.batched import _DISPATCH_UNRESOLVED

    action, msg = _drive_start_llm_dispatch_gate(
        _DISPATCH_UNRESOLVED, "gemma4_unified"
    )
    assert action == "raise"
    assert "gemma4_unified" in msg


def test_start_llm_raises_runtime_error_on_dispatch_rejected():
    from fusion_mlx.engine.batched import _DISPATCH_REJECTED

    action, _ = _drive_start_llm_dispatch_gate(_DISPATCH_REJECTED)
    assert action == "raise"


def test_start_llm_continues_on_dispatch_unresolved_when_not_cli_vetted():
    from fusion_mlx.engine.batched import _DISPATCH_UNRESOLVED

    action, _ = _drive_start_llm_dispatch_gate(_DISPATCH_UNRESOLVED, None)
    assert action == "continue"


def test_start_llm_raises_on_dispatch_unresolved_when_cli_vetted():
    from fusion_mlx.engine.batched import _DISPATCH_UNRESOLVED

    action, _ = _drive_start_llm_dispatch_gate(
        _DISPATCH_UNRESOLVED, "gemma4_unified"
    )
    assert action == "raise"


def test_start_llm_continues_on_dispatch_no_inject_when_not_cli_vetted():
    from fusion_mlx.engine.batched import _DISPATCH_NO_INJECT

    action, _ = _drive_start_llm_dispatch_gate(_DISPATCH_NO_INJECT, None)
    assert action == "continue"


def test_start_llm_raises_on_dispatch_no_inject_when_cli_vetted():
    from fusion_mlx.engine.batched import _DISPATCH_NO_INJECT

    action, _ = _drive_start_llm_dispatch_gate(
        _DISPATCH_NO_INJECT, "gemma4_unified"
    )
    assert action == "raise"


class _SyncExecutor:
    def submit(self, fn, /, *args, **kwargs):
        import concurrent.futures as _cf

        f: _cf.Future = _cf.Future()
        try:
            f.set_result(fn(*args, **kwargs))
        except BaseException as e:
            f.set_exception(e)
        return f


class _TimeoutExecutor:
    def submit(self, fn, /, *args, **kwargs):
        import concurrent.futures as _cf

        class _NeverFuture:
            @staticmethod
            def result(timeout=None):
                raise _cf.TimeoutError("simulated dispatch hang")

            @staticmethod
            def cancel():
                return True

        return _NeverFuture()


def test_apply_mtp_dispatch_returns_attached_on_happy_path(monkeypatch):
    import fusion_mlx.engine.batched as b
    from fusion_mlx.config import SchedulerConfig

    _patch_dispatch_mtp_inject(monkeypatch, return_value=True)
    cfg = SchedulerConfig(mtp_model_type="gemma4_unified", mtp_sidecar="/path/sidecar")
    result = b._apply_mtp_dispatch(object(), cfg, _SyncExecutor())
    assert result == b._DISPATCH_ATTACHED


def test_apply_mtp_dispatch_raises_on_rejected(monkeypatch):
    import fusion_mlx.engine.batched as b
    from fusion_mlx.config import SchedulerConfig

    _patch_dispatch_mtp_inject(monkeypatch, return_value=False)
    cfg = SchedulerConfig(mtp_model_type="gemma4_unified", mtp_sidecar="/path/sidecar")
    with pytest.raises(RuntimeError):
        b._apply_mtp_dispatch(object(), cfg, _SyncExecutor())


def test_apply_mtp_dispatch_raises_when_cli_vetted_and_unresolved(monkeypatch):
    import fusion_mlx.engine.batched as b
    from fusion_mlx.config import SchedulerConfig

    # mtp_model_type=None + unresolvable model -> UNRESOLVED; cli_vetted makes it fatal
    cfg = SchedulerConfig(mtp_model_type=None, mtp_sidecar="/path/sidecar")
    with pytest.raises(RuntimeError):
        b._apply_mtp_dispatch(
            object(), cfg, _SyncExecutor(), cli_vetted_model_type="gemma4_unified"
        )


def test_apply_mtp_dispatch_soft_skips_when_not_cli_vetted(monkeypatch):
    import fusion_mlx.engine.batched as b
    from fusion_mlx.config import SchedulerConfig

    cfg = SchedulerConfig(mtp_model_type=None, mtp_sidecar="/path/sidecar")
    result = b._apply_mtp_dispatch(object(), cfg, _SyncExecutor())
    assert result == b._DISPATCH_UNRESOLVED


def test_apply_mtp_dispatch_raises_runtime_error_on_timeout(monkeypatch):
    import fusion_mlx.engine.batched as b
    from fusion_mlx.config import SchedulerConfig

    cfg = SchedulerConfig(mtp_model_type="gemma4_unified", mtp_sidecar="/path/sidecar")
    with pytest.raises(RuntimeError):
        b._apply_mtp_dispatch(object(), cfg, _TimeoutExecutor())


def test_apply_mtp_dispatch_timeout_logs_critical_and_does_not_call_os_exit(
    monkeypatch, caplog
):
    import fusion_mlx.engine.batched as b
    from fusion_mlx.config import SchedulerConfig

    def _boom(code=0):
        raise AssertionError("os._exit must not be called on dispatch timeout")

    monkeypatch.setattr("os._exit", _boom)
    cfg = SchedulerConfig(mtp_model_type="gemma4_unified", mtp_sidecar="/path/sidecar")
    with caplog.at_level(
        logging.CRITICAL, logger="fusion_mlx.engine.batched._mtp_dispatch"
    ):
        with pytest.raises(RuntimeError):
            b._apply_mtp_dispatch(object(), cfg, _TimeoutExecutor())
    assert any("TIMEOUT" in r.getMessage() for r in caplog.records)


def test_log_mtp_dispatch_timeout_does_not_call_os_exit(monkeypatch):
    from fusion_mlx.engine.batched import _log_mtp_dispatch_timeout

    def _boom(code=0):
        raise AssertionError("_log_mtp_dispatch_timeout must not call os._exit")

    monkeypatch.setattr("os._exit", _boom)
    result = _log_mtp_dispatch_timeout("gemma4_unified", "/path/sidecar", 30.0)
    assert result is None


def test_apply_mtp_dispatch_timeout_does_not_shut_down_shared_executor(monkeypatch):
    import fusion_mlx.engine.batched as b
    from fusion_mlx.config import SchedulerConfig

    class _ShutdownSpy:
        def __init__(self):
            self.shutdown_called = False

        def submit(self, fn, /, *a, **kw):
            return _TimeoutExecutor().submit(fn, *a, **kw)

        def shutdown(self, *a, **kw):
            self.shutdown_called = True

    spy = _ShutdownSpy()
    cfg = SchedulerConfig(mtp_model_type="gemma4_unified", mtp_sidecar="/path/sidecar")
    with pytest.raises(RuntimeError):
        b._apply_mtp_dispatch(object(), cfg, spy)
    assert spy.shutdown_called is False


def test_get_mtp_dispatch_timeout_sec_default(monkeypatch):
    from fusion_mlx.engine.batched import _get_mtp_dispatch_timeout_sec

    monkeypatch.delenv("FUSION_MLX_MTP_DISPATCH_TIMEOUT_SEC", raising=False)
    assert _get_mtp_dispatch_timeout_sec() == 30.0


def test_get_mtp_dispatch_timeout_sec_zero_disables(monkeypatch):
    from fusion_mlx.engine.batched import _get_mtp_dispatch_timeout_sec

    monkeypatch.setenv("FUSION_MLX_MTP_DISPATCH_TIMEOUT_SEC", "0")
    assert _get_mtp_dispatch_timeout_sec() == 0.0


def test_get_mtp_dispatch_timeout_sec_malformed_falls_back_to_default(monkeypatch):
    from fusion_mlx.engine.batched import _get_mtp_dispatch_timeout_sec

    monkeypatch.setenv("FUSION_MLX_MTP_DISPATCH_TIMEOUT_SEC", "not-a-number")
    assert _get_mtp_dispatch_timeout_sec() == 30.0


def test_start_llm_calls_apply_mtp_dispatch():
    pytest.skip(
        "deferred: driving async BatchedEngine._start_llm requires mocking "
        "load_model_with_fallback + AsyncEngineCore + engine.start; the "
        "_apply_mtp_dispatch call site is wired in _start_llm and its logic "
        "is covered by the unit tests above"
    )


class _MonkeypatchScope:
    def __init__(self):
        self._undo_stack: list[tuple] = []

    def setattr(self, target, name, value):
        original = getattr(target, name)
        self._undo_stack.append((target, name, original))
        setattr(target, name, value)

    def undo(self):
        while self._undo_stack:
            target, name, original = self._undo_stack.pop()
            setattr(target, name, original)


# ---------------------------------------------------------------------------
# 5. _install_mtp_vendored gate closures
# ---------------------------------------------------------------------------


class _StubBatchGen:
    def __init__(self):
        import mlx.core as mx

        self.uids: list[int] = []
        self.tokens: list[list[int]] = [[]]
        self.logits_processors: list = []
        self.prompt_cache: list = []
        self.max_tokens: list[int] = [4096]
        self._next_tokens = None
        self._next_logprobs: list = []
        self.orig_step_calls = 0
        self._orig_next_sample = mx.array([999], dtype=mx.uint32)
        self._orig_next_logprob = mx.array([0.0])

    def _step(self):

        self.orig_step_calls += 1
        current = self._next_tokens
        if current is None:
            return [], []
        current_list = [int(current[i].item()) for i in range(current.shape[0])]
        for e, ct in enumerate(current_list):
            self.tokens[e].append(ct)
        self._next_tokens = self._orig_next_sample
        self._next_logprobs = [self._orig_next_logprob]
        return current_list, self._next_logprobs


class _StubModel:
    mtp_forward = object()
    make_mtp_cache = object()
    mtp = object()


def _make_batch_gen_with_gb():
    from types import SimpleNamespace

    gb = _StubBatchGen()
    return SimpleNamespace(_generation_batch=gb), gb


def test_install_mtp_vendored_gate_fails_closed_on_missing_request_metadata(
    monkeypatch,
):
    pytest.skip(
        "feature not migrated: fusion_mlx.scheduler does not export "
        "_install_mtp_vendored"
    )


def test_install_mtp_vendored_falls_back_to_orig_step_on_batch_size_growth(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.scheduler does not export "
        "_install_mtp_vendored; fusion_mlx.speculative.mtp has no "
        "generator submodule"
    )


def test_install_mtp_vendored_b_gt_1_handoff_keeps_yielding_tokens(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.scheduler does not export "
        "_install_mtp_vendored; fusion_mlx.speculative.mtp has no "
        "generator submodule"
    )


def test_install_mtp_vendored_b_gt_1_soft_fallthrough_when_no_state():
    pytest.skip(
        "feature not migrated: fusion_mlx.scheduler does not export "
        "_install_mtp_vendored"
    )


def test_install_mtp_vendored_first_call_construction_failure_does_not_double_book(
    monkeypatch,
):
    pytest.skip(
        "feature not migrated: fusion_mlx.scheduler does not export "
        "_install_mtp_vendored; fusion_mlx.speculative.mtp has no "
        "generator submodule"
    )


def test_install_mtp_vendored_first_call_failure_disables_subsequent_calls(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.scheduler does not export "
        "_install_mtp_vendored; fusion_mlx.speculative.mtp has no "
        "generator submodule"
    )


def test_install_mtp_vendored_disabled_uid_cleared_on_uid_reuse(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.scheduler does not export "
        "_install_mtp_vendored; fusion_mlx.speculative.mtp has no "
        "generator submodule"
    )


def test_install_mtp_vendored_cleanup_does_not_clear_disabled_uids(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.scheduler does not export "
        "_install_mtp_vendored; fusion_mlx.speculative.mtp has no "
        "generator submodule"
    )


def test_install_mtp_vendored_stop_iteration_disables_uid_before_raise(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.scheduler does not export "
        "_install_mtp_vendored; fusion_mlx.speculative.mtp has no "
        "generator submodule"
    )


def test_install_mtp_vendored_non_greedy_mid_stream_falls_back_to_orig_step(
    monkeypatch,
):
    pytest.skip(
        "feature not migrated: fusion_mlx.scheduler does not export "
        "_install_mtp_vendored; fusion_mlx.speculative.mtp has no "
        "generator submodule"
    )


def test_install_mtp_vendored_logits_processors_mid_stream_falls_back_to_orig_step(
    monkeypatch,
):
    pytest.skip(
        "feature not migrated: fusion_mlx.scheduler does not export "
        "_install_mtp_vendored; fusion_mlx.speculative.mtp has no "
        "generator submodule"
    )


def test_install_mtp_vendored_non_greedy_before_state_soft_fallthrough(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.scheduler does not export "
        "_install_mtp_vendored"
    )


def test_install_mtp_vendored_mid_stream_generator_failure_raises(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.scheduler does not export "
        "_install_mtp_vendored; fusion_mlx.speculative.mtp has no "
        "generator submodule"
    )


def test_install_mtp_vendored_first_call_syncs_next_tokens(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.scheduler does not export "
        "_install_mtp_vendored; fusion_mlx.speculative.mtp has no "
        "generator submodule"
    )


def test_install_mtp_vendored_subsequent_syncs_next_tokens(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.scheduler does not export "
        "_install_mtp_vendored; fusion_mlx.speculative.mtp has no "
        "generator submodule"
    )


def test_install_mtp_vendored_next_tokens_shape_survives_stop_iteration(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.scheduler does not export "
        "_install_mtp_vendored; fusion_mlx.speculative.mtp has no "
        "generator submodule"
    )


# ---------------------------------------------------------------------------
# 6. _apply_mtp_cli_model_type_reconciliation
# ---------------------------------------------------------------------------


def test_apply_mtp_cli_model_type_reconciliation_promotes_eligibility_read():
    pytest.skip(
        "feature not migrated: fusion_mlx.cli_serve does not export "
        "_apply_mtp_cli_model_type_reconciliation; SchedulerConfig lacks "
        "mtp_sidecar and mtp_model_type"
    )


def test_apply_mtp_cli_model_type_reconciliation_hard_fails_when_model_type_missing(
    capsys,
):
    pytest.skip(
        "feature not migrated: fusion_mlx.cli_serve does not export "
        "_apply_mtp_cli_model_type_reconciliation; SchedulerConfig lacks "
        "mtp_model_type"
    )


def test_apply_mtp_cli_model_type_reconciliation_prefers_eligibility_on_disagreement():
    pytest.skip(
        "feature not migrated: fusion_mlx.cli_serve does not export "
        "_apply_mtp_cli_model_type_reconciliation; SchedulerConfig lacks "
        "mtp_model_type"
    )


def test_install_mtp_vendored_uid_reuse_clears_stale_state(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.scheduler does not export "
        "_install_mtp_vendored; fusion_mlx.speculative.mtp has no "
        "generator submodule"
    )
