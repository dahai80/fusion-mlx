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
    # SchedulerConfig in fusion_mlx.config does not have mtp_sidecar field.
    pytest.skip(
        "feature not migrated: fusion_mlx.config.SchedulerConfig does not "
        "have mtp_sidecar field"
    )


def test_scheduler_config_mtp_sidecar_round_trip():
    pytest.skip(
        "feature not migrated: fusion_mlx.config.SchedulerConfig does not "
        "have mtp_sidecar field"
    )


def test_scheduler_config_mtp_sidecar_local_path_round_trip():
    pytest.skip(
        "feature not migrated: fusion_mlx.config.SchedulerConfig does not "
        "have mtp_sidecar field"
    )


def test_scheduler_config_mtp_model_type_default_none():
    pytest.skip(
        "feature not migrated: fusion_mlx.config.SchedulerConfig does not "
        "have mtp_model_type field"
    )


def test_scheduler_config_mtp_model_type_round_trip():
    pytest.skip(
        "feature not migrated: fusion_mlx.config.SchedulerConfig does not "
        "have mtp_model_type field"
    )


# ---------------------------------------------------------------------------
# 4. Engine dispatch call site — dispatch_mtp_inject sees the sidecar path
# ---------------------------------------------------------------------------


def test_run_dispatch_mtp_inject_forwards_sidecar_path(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_run_dispatch_mtp_inject, _resolve_hf_model_type, _DISPATCH_ATTACHED, "
        "or dispatch_mtp_inject"
    )


def test_run_dispatch_mtp_inject_returns_unresolved_when_model_type_missing(
    monkeypatch,
):
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_run_dispatch_mtp_inject, _DISPATCH_UNRESOLVED, _DISPATCH_REJECTED, "
        "_resolve_hf_model_type"
    )


def test_run_dispatch_mtp_inject_returns_rejected_when_injector_refuses(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_run_dispatch_mtp_inject, _DISPATCH_REJECTED, _resolve_hf_model_type"
    )


def test_run_dispatch_mtp_inject_returns_no_inject_for_unregistered_model_type(
    monkeypatch,
):
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_run_dispatch_mtp_inject, _DISPATCH_NO_INJECT, _resolve_hf_model_type"
    )


def test_run_dispatch_mtp_inject_prefers_cli_provided_model_type(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_run_dispatch_mtp_inject, _DISPATCH_ATTACHED, _resolve_hf_model_type"
    )


def test_run_dispatch_mtp_inject_falls_back_when_no_preferred_model_type(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_run_dispatch_mtp_inject, _DISPATCH_ATTACHED, _resolve_hf_model_type"
    )


def test_run_dispatch_mtp_inject_propagates_none_sidecar(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_run_dispatch_mtp_inject, _DISPATCH_ATTACHED, _resolve_hf_model_type"
    )


# ---------------------------------------------------------------------------
# 4b. Boot-time contract — _decide_mtp_dispatch_action
# ---------------------------------------------------------------------------


def _drive_start_llm_dispatch_gate(dispatch_result, cli_vetted_model_type=None):
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_decide_mtp_dispatch_action"
    )


def test_decide_mtp_dispatch_action_returns_attached_for_attached_result():
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_decide_mtp_dispatch_action or _DISPATCH_ATTACHED"
    )


def test_decide_mtp_dispatch_action_carries_cli_vetted_model_type_into_error():
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_decide_mtp_dispatch_action or _DISPATCH_UNRESOLVED"
    )


def test_start_llm_raises_runtime_error_on_dispatch_rejected():
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_decide_mtp_dispatch_action or _DISPATCH_REJECTED"
    )


def test_start_llm_continues_on_dispatch_unresolved_when_not_cli_vetted():
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_decide_mtp_dispatch_action or _DISPATCH_UNRESOLVED"
    )


def test_start_llm_raises_on_dispatch_unresolved_when_cli_vetted():
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_decide_mtp_dispatch_action or _DISPATCH_UNRESOLVED"
    )


def test_start_llm_continues_on_dispatch_no_inject_when_not_cli_vetted():
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_decide_mtp_dispatch_action or _DISPATCH_NO_INJECT"
    )


def test_start_llm_raises_on_dispatch_no_inject_when_cli_vetted():
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_decide_mtp_dispatch_action or _DISPATCH_NO_INJECT"
    )


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
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_apply_mtp_dispatch, _DISPATCH_ATTACHED; SchedulerConfig lacks "
        "mtp_model_type"
    )


def test_apply_mtp_dispatch_raises_on_rejected(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_apply_mtp_dispatch, _DISPATCH_REJECTED; SchedulerConfig lacks "
        "mtp_sidecar"
    )


def test_apply_mtp_dispatch_raises_when_cli_vetted_and_unresolved(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_apply_mtp_dispatch, _DISPATCH_UNRESOLVED; SchedulerConfig lacks "
        "mtp_model_type"
    )


def test_apply_mtp_dispatch_soft_skips_when_not_cli_vetted(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_apply_mtp_dispatch, _DISPATCH_UNRESOLVED"
    )


def test_apply_mtp_dispatch_raises_runtime_error_on_timeout(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_apply_mtp_dispatch; SchedulerConfig lacks mtp_model_type; "
        "FUSION_MLX_MTP_DISPATCH_TIMEOUT_SEC env var not wired"
    )


def test_apply_mtp_dispatch_timeout_logs_critical_and_does_not_call_os_exit(
    monkeypatch,
):
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_apply_mtp_dispatch, _log_mtp_dispatch_timeout; SchedulerConfig "
        "lacks mtp_model_type"
    )


def test_log_mtp_dispatch_timeout_does_not_call_os_exit(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_log_mtp_dispatch_timeout"
    )


def test_apply_mtp_dispatch_timeout_does_not_shut_down_shared_executor(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_apply_mtp_dispatch; SchedulerConfig lacks mtp_model_type"
    )


def test_get_mtp_dispatch_timeout_sec_default(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_get_mtp_dispatch_timeout_sec; FUSION_MLX_MTP_DISPATCH_TIMEOUT_SEC "
        "env var not wired"
    )


def test_get_mtp_dispatch_timeout_sec_zero_disables(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_get_mtp_dispatch_timeout_sec"
    )


def test_get_mtp_dispatch_timeout_sec_malformed_falls_back_to_default(monkeypatch):
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_get_mtp_dispatch_timeout_sec"
    )


def test_start_llm_calls_apply_mtp_dispatch():
    pytest.skip(
        "feature not migrated: fusion_mlx.engine.batched does not export "
        "_apply_mtp_dispatch; fusion_mlx.utils.tokenizer lacks "
        "load_model_with_fallback; SchedulerConfig lacks mtp_model_type"
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
        import mlx.core as mx

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
