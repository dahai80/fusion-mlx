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
    """Base Gemma 4 unified checkpoint (no MTP head) + sidecar -> TREE.

    The stock ``mlx-community/gemma-4-12B-it-4bit/config.json`` reports
    ``model_type: gemma4_unified`` with no ``mtp_num_hidden_layers``
    key. Without ``--mtp-sidecar``, detection collapses to NONE. With
    ``--mtp-sidecar``, fusion-mlx's detector returns TREE
    unconditionally - the external assistant drafter overrides
    eligibility. Model-type scoping (gemma4_unified backbone only)
    lives in the CLI reconciliation layer, not in detect (see the
    Section 3 reconciliation tests).
    """
    from fusion_mlx.speculative.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "gemma4_unified"}  # no mtp_num_hidden_layers
    assert detect_mtp_eligibility(config) is MTPEligibility.NONE
    assert (
        detect_mtp_eligibility(config, has_external_sidecar=True) is MTPEligibility.TREE
    )


def test_detect_sidecar_promotes_gemma4_unified_with_zero_mtp_layers():
    """Explicit ``mtp_num_hidden_layers: 0`` + sidecar -> TREE too.

    Same shape as the base 12B checkpoint after someone hand-edited
    the config to stamp a zero on it. Sidecar-mode still returns TREE
    - the assistant weights come from the external path.
    """
    from fusion_mlx.speculative.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "gemma4_unified", "mtp_num_hidden_layers": 0}
    assert detect_mtp_eligibility(config) is MTPEligibility.NONE
    assert (
        detect_mtp_eligibility(config, has_external_sidecar=True) is MTPEligibility.TREE
    )


def test_detect_sidecar_unconditional_for_qwen3_5_family():
    """Sidecar -> TREE regardless of model_type in fusion-mlx.

    Diverges from Rapid-MLX, which scoped the sidecar promotion to
    ``gemma4_unified`` only (qwen3.5 stayed NONE). fusion-mlx moved
    the model-type gate OUT of detect and into the CLI reconciliation
    layer (``_apply_mtp_cli_model_type_reconciliation`` exits 2 for a
    non-gemma4_unified sidecar backbone). detect itself stays
    unconditional so the eligibility signal is a pure "external
    drafter present" flag. This pins that contract.
    """
    from fusion_mlx.speculative.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "qwen3_5", "mtp_num_hidden_layers": 0}
    assert (
        detect_mtp_eligibility(config, has_external_sidecar=True) is MTPEligibility.TREE
    )
    config_moe = {"model_type": "qwen3_5_moe", "mtp_num_hidden_layers": 0}
    assert (
        detect_mtp_eligibility(config_moe, has_external_sidecar=True)
        is MTPEligibility.TREE
    )


def test_detect_sidecar_unconditional_for_gemma4_multimodal():
    """Multimodal ``gemma4`` + sidecar -> TREE in fusion-mlx detect.

    Same divergence as the qwen3.5 case: Rapid-MLX kept multimodal
    ``gemma4`` at NONE because only ``gemma4_unified`` had a verified
    external drafter. fusion-mlx's detect returns TREE unconditionally;
    the backbone allowlist is enforced downstream in reconciliation.
    """
    from fusion_mlx.speculative.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "gemma4", "mtp_num_hidden_layers": 0}
    assert (
        detect_mtp_eligibility(config, has_external_sidecar=True) is MTPEligibility.TREE
    )


def test_detect_sidecar_overrides_baked_in_mtp_to_tree():
    """Qwen3.5 with baked-in MTP layers + sidecar -> TREE (overrides CHAIN).

    Without sidecar, ``qwen3_5`` with ``mtp_num_hidden_layers >= 1``
    returns CHAIN (baked-in MTP). With ``has_external_sidecar=True``
    the external drafter takes over and detect returns TREE. This is
    NOT additive (it can change CHAIN -> TREE); the sidecar flag
    signals "use the external drafter, not the baked-in one".
    """
    from fusion_mlx.speculative.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "qwen3_5", "mtp_num_hidden_layers": 1}
    assert (
        detect_mtp_eligibility(config, has_external_sidecar=False)
        is MTPEligibility.CHAIN
    )
    assert (
        detect_mtp_eligibility(config, has_external_sidecar=True) is MTPEligibility.TREE
    )


def test_detect_sidecar_default_argument_matches_pre_0913_behaviour():
    """``has_external_sidecar`` defaults to False, preserving the
    pre-0.9.13 rejection contract for every non-CLI caller.

    Regression guard against a future refactor that flips the default
    to True - bench scripts, unit tests, and the CLI eligibility gate
    all rely on the no-argument case being identical to the old NONE
    shape when MTP layers are missing.
    """
    from fusion_mlx.speculative.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "gemma4_unified", "mtp_num_hidden_layers": 0}
    assert detect_mtp_eligibility(config) is MTPEligibility.NONE
    assert (
        detect_mtp_eligibility(config, has_external_sidecar=False)
        is MTPEligibility.NONE
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
    stdout = _serve_help_stdout()
    assert "--mtp-sidecar" in stdout


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

    action, msg = _drive_start_llm_dispatch_gate(_DISPATCH_UNRESOLVED, "gemma4_unified")
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

    action, _ = _drive_start_llm_dispatch_gate(_DISPATCH_UNRESOLVED, "gemma4_unified")
    assert action == "raise"


def test_start_llm_continues_on_dispatch_no_inject_when_not_cli_vetted():
    from fusion_mlx.engine.batched import _DISPATCH_NO_INJECT

    action, _ = _drive_start_llm_dispatch_gate(_DISPATCH_NO_INJECT, None)
    assert action == "continue"


def test_start_llm_raises_on_dispatch_no_inject_when_cli_vetted():
    from fusion_mlx.engine.batched import _DISPATCH_NO_INJECT

    action, _ = _drive_start_llm_dispatch_gate(_DISPATCH_NO_INJECT, "gemma4_unified")
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
    """Codex round-A blocker #1 regression guard.

    Prior revision returned ``True`` from ``_is_greedy_for_uid`` when
    ``requests`` / ``uid_to_request_id`` were unresolvable — that
    silently applied greedy sampling to any request whose bookkeeping
    had just been evicted. The fix flips the default to ``False`` so
    the caller falls through to ``_orig_step()`` (which reads the real
    sampler).

    We can't easily exercise the closure directly (it's local to
    ``_install_mtp_vendored``). But we CAN observe the outer contract:
    when ``requests=None`` and there's a single-uid batch, the patched
    ``_step`` MUST fall through to ``_orig_step()`` — not enter the
    MTP construction path — because the gate now returns False.
    """
    from fusion_mlx.scheduler import _install_mtp_vendored

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [42]  # single uid — passes the B==1 gate

    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests=None,
        uid_to_request_id=None,
    )
    assert ok is True

    # Fire the patched _step. With requests=None, _is_greedy_for_uid
    # must return False → fallthrough to _orig_step. Pre-fix the gate
    # returned True and we would have entered the mtp_generate_step
    # construction path.
    gb._step()
    stats = batch_gen._mtp_vendored_stats
    assert stats["fallthrough_steps"] >= 1
    assert stats["ft_non_greedy"] >= 1, (
        "codex round-A blocker #1 regression: gate did not fall closed "
        "when request bookkeeping is unresolvable"
    )
    assert gb.orig_step_calls == 1


def test_install_mtp_vendored_falls_back_to_orig_step_on_batch_size_growth(monkeypatch):
    """Codex round-A blocker #3 + round-L BLOCKING #2 regression
    guard.

    Two contracts under test:

    * Round-A: a uid that ran MTP for a while then transitions to a
      B>1 batch closes its generator (side-effect observable).

    * Round-L: prior round-H revision raised ``RuntimeError`` when
      B>1 arrived after MTP had emitted tokens, killing the request.
      That is hostile to a multi-request server where B>1 is the
      norm. Round-L flips the behavior: the MTP generator is
      closed, the uid is disabled, and the wrapper delegates to
      ``_orig_step()`` — the request continues on plain decode with
      a bounded stream artifact (see :func:`_log_mtp_mid_stream_
      handoff_once` for the rationale).

    The historical B>1 raise from round-H is intentionally gone.
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from fusion_mlx.scheduler import _install_mtp_vendored
    from fusion_mlx.speculative.mtp import generator as _gen_mod

    fake_gen_calls = {"constructed": 0, "closed": 0}

    class _FakeGen:
        def __init__(self):
            fake_gen_calls["constructed"] += 1
            self._n = 0

        def __iter__(self):
            return self

        def __next__(self):
            self._n += 1
            return (self._n + 1000, mx.array([0.0]), False)

        def close(self):
            fake_gen_calls["closed"] += 1

    def _fake_mtp_generate_step(*args, **kwargs):
        return _FakeGen()

    monkeypatch.setattr(_gen_mod, "mtp_generate_step", _fake_mtp_generate_step)

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [7]
    request_stub = SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0))
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests={"req-7": request_stub},
        uid_to_request_id={7: "req-7"},
    )
    assert ok is True

    gb._next_tokens = mx.array([500], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    # First call — construct the fake generator and populate _state[7].
    gb._step()
    assert fake_gen_calls["constructed"] == 1
    assert fake_gen_calls["closed"] == 0

    # Second call in the SAME warm state — draining the queue.
    gb._step()
    assert fake_gen_calls["closed"] == 0

    # Now transition to B=2. Round-L BLOCKING #2: uid=7 has state,
    # but the wrapper MUST NOT raise. It MUST close the stale
    # generator (round-A), delegate to _orig_step (round-L), and
    # increment the mid-stream handoff counter (operator-visible
    # via stats).
    orig_step_before = gb.orig_step_calls
    gb.uids = [1, 2]
    result = gb._step()

    # Round-L: fall through to _orig_step, not raise.
    assert result is not None, (
        "codex round-L BLOCKING #2 regression: B>1 mid-stream must "
        "return _orig_step()'s tuple, not None. The wrapper is "
        "expected to hand off silently, not abort."
    )
    assert gb.orig_step_calls == orig_step_before + 1, (
        "codex round-L BLOCKING #2 regression: B>1 mid-stream did "
        "NOT delegate to _orig_step. The request would have been "
        "killed by a RuntimeError under the round-H invariant that "
        "round-L relaxed."
    )
    stats = batch_gen._mtp_vendored_stats
    assert stats["ft_batch_size"] >= 1
    assert stats["ft_mid_stream_handoff"] >= 1, (
        "codex round-L BLOCKING #2 regression: mid-stream handoff "
        "counter did not fire. Operator loses observability of the "
        "MTP → plain decode transition."
    )
    assert fake_gen_calls["closed"] >= 1, (
        "codex round-A blocker #3 regression: B>1 handoff path did "
        "not close the stale generator on the way out."
    )


def test_install_mtp_vendored_b_gt_1_handoff_keeps_yielding_tokens(monkeypatch):
    """Codex round-L BLOCKING #2 positive test.

    Once the mid-stream B>1 handoff has fired, subsequent _step
    calls (still under B>1) MUST keep calling _orig_step — the
    request stays on plain decode until it completes. The disable
    marker on the affected uid ensures we don't accidentally re-arm
    MTP mid-request.
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from fusion_mlx.scheduler import _install_mtp_vendored
    from fusion_mlx.speculative.mtp import generator as _gen_mod

    fake_gen_calls = {"constructed": 0, "closed": 0}

    class _FakeGen:
        def __init__(self):
            fake_gen_calls["constructed"] += 1
            self._n = 0

        def __iter__(self):
            return self

        def __next__(self):
            self._n += 1
            return (self._n + 1000, mx.array([0.0]), False)

        def close(self):
            fake_gen_calls["closed"] += 1

    monkeypatch.setattr(_gen_mod, "mtp_generate_step", lambda *a, **kw: _FakeGen())

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [7]
    request_stub = SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0))
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests={"req-7": request_stub},
        uid_to_request_id={7: "req-7"},
    )
    assert ok is True

    gb._next_tokens = mx.array([500], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    # Prime MTP with one successful call, then trigger B>1 handoff.
    gb._step()  # FIRST-call: MTP primed
    gb.uids = [1, 2]
    gb._step()  # handoff fires
    # Handoff happened via _record_terminal_disable, so uid=7 is
    # now in _disabled_uids (accessible via the state map keyed by
    # uid). But the wrapper is now installed at B>1, and the
    # disable gate only fires when len(gb.uids)==1. The B>1 gate
    # in _mtp_step should keep firing for every subsequent step
    # while B>1 — the request never re-enters MTP even if the
    # batch later returns to B=1 for THIS uid because it's
    # disabled.

    orig_before = gb.orig_step_calls
    for _ in range(5):
        gb._step()
    assert gb.orig_step_calls == orig_before + 5, (
        "codex round-L BLOCKING #2 regression: post-handoff _step "
        "calls did not consistently delegate to _orig_step. The "
        "request must continue on plain decode after the handoff."
    )
    stats = batch_gen._mtp_vendored_stats
    assert stats["ft_batch_size"] >= 5


def test_install_mtp_vendored_b_gt_1_soft_fallthrough_when_no_state():
    """Codex round-H BLOCKING #1 companion: the B>1 fallthrough
    remains a soft skip when NO uid has in-flight MTP state.

    This is the "batch legitimately started with B>1" case — the
    wrapper never got a chance to prime any generator, so
    ``gb._next_tokens`` is the fresh baseline sample.
    ``_orig_step()`` here is safe.
    """
    import mlx.core as mx

    from fusion_mlx.scheduler import _install_mtp_vendored

    batch_gen, gb = _make_batch_gen_with_gb()
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests=None,
        uid_to_request_id=None,
    )
    assert ok is True

    gb.uids = [1, 2]
    gb._next_tokens = mx.array([100], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]
    # No state populated; safe soft-fall-through.
    gb._step()
    stats = batch_gen._mtp_vendored_stats
    assert stats["ft_batch_size"] >= 1
    assert gb.orig_step_calls == 1


def test_install_mtp_vendored_first_call_construction_failure_does_not_double_book(
    monkeypatch,
):
    """Codex round-A blocker #2 regression guard.

    Prior revision appended the first token to ``gb.tokens[0]`` BEFORE
    constructing the generator. When ``mtp_generate_step(...)`` raised
    (missing dep, weight-shape mismatch, etc.), the fallthrough path
    then called ``_orig_step()`` which appends the SAME token again,
    double-booking bookkeeping and duplicating the token in the emitted
    stream.

    Fix: construct the generator first, only mutate ``gb.tokens`` on
    success. On construction failure the fallthrough path calls
    ``_orig_step`` on a clean ``tokens`` list.

    Implementation note: ``mtp_generate_step`` is imported lazily
    inside ``_install_mtp_vendored`` via a ``from … import …`` and is
    then captured by the closure that patches ``_step``. Any patch has
    to be installed on the source module BEFORE the install call runs
    so the from-import picks up the fake; a post-install monkeypatch
    would target the module attribute but not the closure's local
    binding.
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from fusion_mlx.scheduler import _install_mtp_vendored
    from fusion_mlx.speculative.mtp import generator as _gen_mod

    def _raising_generator(*args, **kwargs):
        raise RuntimeError("simulated generator construction failure")

    monkeypatch.setattr(_gen_mod, "mtp_generate_step", _raising_generator)

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [99]

    # Provide a sampling_params.temperature=0.0 stub so the greedy
    # gate passes (we want to reach the first-call construction path).
    request_stub = SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0))
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests={"req-99": request_stub},
        uid_to_request_id={99: "req-99"},
    )
    assert ok is True

    # Simulate mlx-lm's original _step having primed the first token
    # into ``_next_tokens`` — a 1-D mx.array of length 1 with a real
    # int payload. The realistic stub (_StubBatchGen._step) mirrors
    # mlx-lm's real _step in ``gb.tokens[0].append(int(inputs[0]))``,
    # so the exact double-book bug the codex round-A fix addressed
    # would manifest as a length-2 tokens list with 12345 repeated.
    gb._next_tokens = mx.array([12345], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    gb._step()

    # Fallthrough happened → _orig_step ran exactly once, which does
    # ONE ``tokens[0].append(first_tok)`` per mlx-lm's real shape.
    # Under the round-A pre-fix, our wrapper would ALSO have appended
    # first_tok before construction — leaving gb.tokens[0] == [first,
    # first]. Codex round-B blocker #3: this assertion now runs
    # against the mlx-lm-shaped stub, so it can actually observe the
    # double-book.
    assert gb.orig_step_calls == 1
    assert gb.tokens[0] == [12345], (
        f"codex round-A blocker #2 regression: gb.tokens[0] = "
        f"{gb.tokens[0]!r} (expected [12345] — one append from "
        "_orig_step, none from our wrapper's pre-construction append)."
    )
    stats = batch_gen._mtp_vendored_stats
    assert stats["fallthrough_steps"] >= 1


def test_install_mtp_vendored_first_call_failure_disables_subsequent_calls(monkeypatch):
    """Codex round-D blocker #2 regression guard.

    Under a deterministic first-call construction failure (bad sidecar,
    weight-shape mismatch, etc.), the wrapper's original
    ``state is None`` branch would re-run the failing ``try/except``
    every step — one construction attempt per token, effectively DoSing
    the request while never getting any MTP benefit.

    Fix: track ``_disabled_uids`` and short-circuit to ``_orig_step``
    once construction has failed for a given uid. This test drives
    two ``_step()`` calls under a deterministically-failing generator
    constructor and asserts:

    * The first call attempts construction (raises internally → falls
      through to ``_orig_step``).
    * The second call does NOT re-attempt construction — the
      ``mtp_generate_step`` monkeypatch's counter stays at 1.
    * Both calls advance ``_orig_step`` correctly (no double-book).
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from fusion_mlx.scheduler import _install_mtp_vendored
    from fusion_mlx.speculative.mtp import generator as _gen_mod

    construction_attempts = {"n": 0}

    def _raising_generator(*args, **kwargs):
        construction_attempts["n"] += 1
        raise RuntimeError("simulated persistent construction failure")

    monkeypatch.setattr(_gen_mod, "mtp_generate_step", _raising_generator)

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [77]
    request_stub = SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0))
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests={"req-77": request_stub},
        uid_to_request_id={77: "req-77"},
    )
    assert ok is True

    gb._next_tokens = mx.array([500], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]
    gb._orig_next_sample = mx.array([501], dtype=mx.uint32)

    # First call — construction is attempted, fails, fall through.
    gb._step()
    assert construction_attempts["n"] == 1
    stats = batch_gen._mtp_vendored_stats
    assert stats["fallthrough_steps"] >= 1

    # Second call — must short-circuit via the disabled-uid path.
    # No new construction attempt.
    gb._orig_next_sample = mx.array([502], dtype=mx.uint32)
    gb._step()
    assert construction_attempts["n"] == 1, (
        "codex round-D blocker #2 regression: wrapper retried "
        f"construction after a first-call failure "
        f"(attempts={construction_attempts['n']!r}). It must mark the "
        "uid as disabled and delegate directly to _orig_step for the "
        "rest of the request."
    )
    stats = batch_gen._mtp_vendored_stats
    assert stats.get("ft_disabled", 0) >= 1, (
        "codex round-D blocker #2 regression: the second _step call did "
        "not hit the disabled-uid short-circuit — check the "
        "_disabled_uids gate ordering vs. _is_greedy_for_uid."
    )
    # And _orig_step ran twice — once per _step() call.
    assert gb.orig_step_calls == 2


def test_install_mtp_vendored_disabled_uid_cleared_on_uid_reuse(monkeypatch):
    """Codex round-E blocker #1 regression guard.

    mlx-lm reuses uid ints once a request completes. The round-D
    ``_disabled_uids`` fix keyed disable state by uid alone; that
    let a bad-sidecar disable from request N silently apply to
    request N+1, N+2, ... if they happened to draw the same uid,
    permanently disabling MTP after a single bad request.

    Fix: store the request_id at disable time. When the same uid
    shows up with a DIFFERENT request_id, the disable is stale —
    clear it and re-enter the normal MTP path.

    This test:
      1. Drives request A (uid=42, req-A) through a first-call
         construction failure — uid=42 lands in _disabled_uids.
      2. Simulates uid=42 being reused for request B (req-B) with
         a working generator constructor.
      3. Verifies that the wrapper does NOT stay in the disabled
         short-circuit — it re-enters the FIRST-call path and
         successfully seeds a fresh generator for request B.
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from fusion_mlx.scheduler import _install_mtp_vendored
    from fusion_mlx.speculative.mtp import generator as _gen_mod

    class _RecoveringCtor:
        """First construction raises; subsequent calls yield a fake
        generator. Simulates "request A had a bad sidecar path,
        request B was retargeted at a working path."
        """

        def __init__(self):
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("simulated request-A sidecar failure")
            return _FakeGen()

    class _FakeGen:
        def __init__(self):
            self._n = 0

        def __iter__(self):
            return self

        def __next__(self):
            self._n += 1
            return (5000 + self._n, mx.array([0.0]), False)

        def close(self):
            pass

    ctor = _RecoveringCtor()
    monkeypatch.setattr(_gen_mod, "mtp_generate_step", ctor)

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [42]
    uid_to_request_id: dict[int, str] = {42: "req-A"}
    requests: dict = {
        "req-A": SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0)),
    }
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests=requests,
        uid_to_request_id=uid_to_request_id,
    )
    assert ok is True

    gb._next_tokens = mx.array([1], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    # Request A step 1 — construction fails, uid=42 goes into _disabled_uids
    # keyed by req-A.
    gb._step()
    assert ctor.calls == 1
    stats = batch_gen._mtp_vendored_stats
    assert stats["fallthrough_steps"] >= 1

    # Request A step 2 — still req-A, so the disabled short-circuit
    # fires; ctor is NOT called again.
    gb._orig_next_sample = mx.array([2], dtype=mx.uint32)
    gb._step()
    assert ctor.calls == 1
    assert stats.get("ft_disabled", 0) >= 1

    # Now simulate request A completing and uid=42 being reused for
    # request B. mlx-lm would update uid_to_request_id to the new
    # request's ID.
    uid_to_request_id[42] = "req-B"
    requests["req-B"] = SimpleNamespace(
        sampling_params=SimpleNamespace(temperature=0.0)
    )
    gb._next_tokens = mx.array([100], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    # Request B step 1 — request_id changed, disabled state MUST be
    # cleared and the wrapper MUST re-enter the FIRST-call path.
    gb._step()
    assert ctor.calls == 2, (
        "codex round-E blocker #1 regression: uid=42 was reused for "
        f"a new request (req-B), but the wrapper stayed in the "
        "disabled short-circuit and did not attempt fresh MTP "
        f"construction (ctor.calls={ctor.calls!r}). This lets one "
        "bad-sidecar disable permanently downgrade every subsequent "
        "request that draws the same uid."
    )


def test_install_mtp_vendored_cleanup_does_not_clear_disabled_uids(monkeypatch):
    """Codex round-G BLOCKING #1 regression guard.

    Earlier revision's ``_cleanup_uid`` unconditionally popped
    ``_disabled_uids[uid]``, which meant any fallthrough branch (B>1
    transition, non-greedy switch, logits-processors override) that
    called ``_cleanup_uid`` would silently un-disable a uid — the
    next single-uid greedy call would then retry MTP construction
    and hit the same broken path all over again, one construction
    attempt per token.

    Fix: ``_cleanup_uid`` no longer touches ``_disabled_uids``.
    The disable state is a per-REQUEST marker cleared only by
    (a) uid reuse detection with a new request_id, or (b) explicit
    delete in the reuse-gate branch. State (the generator + queue)
    is still cleaned by ``_cleanup_uid`` — that's per-generator
    lifecycle, not per-request.

    This test:
      1. Drives a first-call construction failure → uid=99 lands
         in ``_disabled_uids`` keyed by req-A.
      2. Triggers a B>1 fallthrough (which calls ``_cleanup_uid``
         for stale uids in ``_state``).
      3. Returns to B=1 single-uid and drives another step.
      4. Asserts that MTP construction is NOT retried — the
         disable marker survived the cleanup.
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from fusion_mlx.scheduler import _install_mtp_vendored
    from fusion_mlx.speculative.mtp import generator as _gen_mod

    construction_attempts = {"n": 0}

    def _raising_ctor(*args, **kwargs):
        construction_attempts["n"] += 1
        raise RuntimeError("simulated persistent construction failure")

    monkeypatch.setattr(_gen_mod, "mtp_generate_step", _raising_ctor)

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [99]
    request_stub = SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0))
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests={"req-99": request_stub},
        uid_to_request_id={99: "req-99"},
    )
    assert ok is True

    gb._next_tokens = mx.array([100], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    # Step 1 — construction fails, uid=99 disabled.
    gb._step()
    assert construction_attempts["n"] == 1

    # Force a B>1 fallthrough — this calls _cleanup_uid for every
    # uid in _state. Under the round-G BLOCKING #1 pre-fix this
    # would ALSO have popped _disabled_uids[99].
    gb.uids = [99, 100]
    gb._step()
    stats = batch_gen._mtp_vendored_stats
    assert stats.get("ft_batch_size", 0) >= 1

    # Return to B=1 same uid; if _cleanup_uid cleared the disable
    # (pre-fix), the wrapper would retry construction here. Post-
    # fix, the disable marker is intact and we short-circuit.
    gb.uids = [99]
    gb._next_tokens = mx.array([200], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]
    gb._step()
    assert construction_attempts["n"] == 1, (
        "codex round-G BLOCKING #1 regression: _cleanup_uid cleared "
        "_disabled_uids on a B>1 fallthrough. Next single-uid step "
        "retried MTP construction "
        f"(attempts={construction_attempts['n']!r})."
    )


def test_install_mtp_vendored_stop_iteration_disables_uid_before_raise(monkeypatch):
    """Codex round-G BLOCKING #2 regression guard (StopIteration branch).

    On ``StopIteration`` mid-stream, the wrapper must:
    (a) record the current request_id in ``_disabled_uids`` so any
        retry short-circuits to plain decode; and
    (b) raise ``RuntimeError`` so mlx-lm surfaces the failure.

    Earlier revision called ``_cleanup_uid`` which cleared the
    disable, meaning a retry on the same uid+request_id would re-
    enter FIRST-call construction and hit the same bug.
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from fusion_mlx.scheduler import _install_mtp_vendored
    from fusion_mlx.speculative.mtp import generator as _gen_mod

    class _EmptyGen:
        """Yields nothing — first next() call raises StopIteration."""

        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def close(self):
            pass

    monkeypatch.setattr(_gen_mod, "mtp_generate_step", lambda *a, **kw: _EmptyGen())

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [88]
    request_stub = SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0))
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests={"req-88": request_stub},
        uid_to_request_id={88: "req-88"},
    )
    assert ok is True

    gb._next_tokens = mx.array([777], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    # First call — construct + emit first_gen_tok = 777, populates
    # _state[88].
    gb._step()

    # Second call — draining the queue is empty, pulls from _EmptyGen
    # which raises StopIteration. Wrapper must record 88 in
    # _disabled_uids (with req-88 as the marker) before raising.
    try:
        gb._step()
    except RuntimeError as e:
        assert (
            "generator exhausted" in str(e).lower()
            or "stopiteration" in str(e).lower()
            or "before mlx-lm hit" in str(e).lower()
        )
        # Simulate a retry: if mlx-lm re-enters _mtp_step with the
        # same uid+request_id, the disable marker MUST fire and
        # short-circuit to _orig_step (not re-enter construction).
        # This can happen if the caller uses the exception as
        # "back off then retry" rather than propagating.
        gb._next_tokens = mx.array([500], dtype=mx.uint32)
        gb._next_logprobs = [mx.array([0.0])]
        gb.uids = [88]
        pre_retry_orig_step_calls = gb.orig_step_calls
        gb._step()
        # The wrapper hit the disable short-circuit and called
        # _orig_step. NOT a fresh construction attempt.
        assert gb.orig_step_calls == pre_retry_orig_step_calls + 1, (
            "codex round-G BLOCKING #2 regression: retry on the same "
            "uid+request_id after a StopIteration failure did NOT hit "
            "the disable short-circuit."
        )
        stats = batch_gen._mtp_vendored_stats
        assert stats.get("ft_disabled", 0) >= 1
        return
    raise AssertionError(
        "codex round-G BLOCKING #2 regression: wrapper did NOT raise "
        "RuntimeError on internal generator StopIteration."
    )


def test_install_mtp_vendored_non_greedy_mid_stream_falls_back_to_orig_step(
    monkeypatch,
):
    """Codex round-L BLOCKING #3 regression guard.

    Prior round-H revision raised ``RuntimeError`` when sampling
    switched to non-greedy after MTP had already emitted tokens.
    That killed the request whenever an operator adjusted sampling
    params mid-stream.

    Round-L flip: the wrapper closes the MTP generator, marks the
    uid disabled, delegates to ``_orig_step()``, and logs a WARN
    for the operator. Subsequent steps stay on plain decode via
    the disable short-circuit. Same bounded stream-artifact
    tradeoff as the B>1 handoff (see :func:`_log_mtp_mid_stream_
    handoff_once`).
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from fusion_mlx.scheduler import _install_mtp_vendored
    from fusion_mlx.speculative.mtp import generator as _gen_mod

    fake_gen_calls = {"closed": 0}

    class _FakeGen:
        def __init__(self):
            self._n = 0

        def __iter__(self):
            return self

        def __next__(self):
            self._n += 1
            return (self._n + 1000, mx.array([0.0]), False)

        def close(self):
            fake_gen_calls["closed"] += 1

    monkeypatch.setattr(_gen_mod, "mtp_generate_step", lambda *a, **kw: _FakeGen())

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [55]
    # Start greedy so MTP primes the generator.
    sp = SimpleNamespace(temperature=0.0)
    request_stub = SimpleNamespace(sampling_params=sp)
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests={"req-55": request_stub},
        uid_to_request_id={55: "req-55"},
    )
    assert ok is True

    gb._next_tokens = mx.array([300], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    # First call — MTP primed. _state[55] populated.
    gb._step()

    # Mid-stream switch to temp > 0 — round-L handoff branch.
    orig_before = gb.orig_step_calls
    sp.temperature = 0.7
    result = gb._step()

    assert result is not None, (
        "codex round-L BLOCKING #3 regression: non-greedy mid-stream "
        "must delegate to _orig_step, not raise. The wrapper hands "
        "off silently and lets the request continue on plain decode."
    )
    assert gb.orig_step_calls == orig_before + 1, (
        "codex round-L BLOCKING #3 regression: non-greedy mid-stream "
        "did NOT delegate to _orig_step. Under round-H the request "
        "would have been killed by a RuntimeError; round-L relaxes "
        "that to a fallback."
    )
    stats = batch_gen._mtp_vendored_stats
    assert stats["ft_non_greedy"] >= 1
    assert stats["ft_mid_stream_handoff"] >= 1, (
        "codex round-L BLOCKING #3 regression: mid-stream handoff "
        "counter did not fire on non-greedy transition."
    )
    assert fake_gen_calls["closed"] >= 1, (
        "codex round-L BLOCKING #3: non-greedy handoff MUST close "
        "the stale MTP generator so nothing dangles across the "
        "request tail."
    )

    # Subsequent steps stay on plain decode (uid=55 is now disabled).
    gb._next_tokens = mx.array([301], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]
    pre = gb.orig_step_calls
    gb._step()
    assert gb.orig_step_calls == pre + 1, (
        "codex round-L BLOCKING #3 regression: post-handoff retry "
        "did not hit the disable short-circuit; a new MTP generator "
        "would be constructed and the fallback design regressed."
    )


def test_install_mtp_vendored_logits_processors_mid_stream_falls_back_to_orig_step(
    monkeypatch,
):
    """Codex round-L BLOCKING #4 regression guard.

    Prior round-H revision raised ``RuntimeError`` when a logits
    processor was added after MTP had already emitted. That killed
    the request whenever an operator wired a guided-decoding
    grammar (or similar per-request processor) mid-stream.

    Round-L flip: close the MTP generator, mark uid disabled,
    delegate to ``_orig_step`` and log a WARN. Subsequent steps
    stay on plain decode via the disable short-circuit.
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from fusion_mlx.scheduler import _install_mtp_vendored
    from fusion_mlx.speculative.mtp import generator as _gen_mod

    fake_gen_calls = {"closed": 0}

    class _FakeGen:
        def __init__(self):
            self._n = 0

        def __iter__(self):
            return self

        def __next__(self):
            self._n += 1
            return (self._n + 1000, mx.array([0.0]), False)

        def close(self):
            fake_gen_calls["closed"] += 1

    monkeypatch.setattr(_gen_mod, "mtp_generate_step", lambda *a, **kw: _FakeGen())

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [33]
    request_stub = SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0))
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests={"req-33": request_stub},
        uid_to_request_id={33: "req-33"},
    )
    assert ok is True

    gb._next_tokens = mx.array([400], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    # First call — MTP primed.
    gb._step()

    # Mid-stream: install a truthy logits processor — round-L handoff branch.
    gb.logits_processors = [[lambda tokens, logits: logits]]
    orig_before = gb.orig_step_calls
    result = gb._step()

    assert result is not None, (
        "codex round-L BLOCKING #4 regression: mid-stream logits "
        "processor MUST delegate to _orig_step, not raise."
    )
    assert gb.orig_step_calls == orig_before + 1, (
        "codex round-L BLOCKING #4 regression: logits-processor "
        "mid-stream did NOT delegate to _orig_step. The request "
        "would have been killed by a RuntimeError under the "
        "round-H invariant that round-L relaxed."
    )
    stats = batch_gen._mtp_vendored_stats
    assert stats["ft_logits_processors"] >= 1
    assert stats["ft_mid_stream_handoff"] >= 1, (
        "codex round-L BLOCKING #4 regression: mid-stream handoff "
        "counter did not fire on lp transition."
    )
    assert fake_gen_calls["closed"] >= 1, (
        "codex round-L BLOCKING #4: lp handoff MUST close the "
        "stale MTP generator on the way out."
    )

    # Subsequent step stays on plain decode (uid=33 disabled).
    gb._next_tokens = mx.array([401], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]
    pre = gb.orig_step_calls
    gb._step()
    assert gb.orig_step_calls == pre + 1, (
        "codex round-L BLOCKING #4 regression: post-handoff retry "
        "did not hit the disable short-circuit."
    )


def test_install_mtp_vendored_non_greedy_before_state_soft_fallthrough(monkeypatch):
    """Companion to round-H BLOCKING #2: when the request starts
    non-greedy (never populated ``_state``), the wrapper soft-falls
    through to ``_orig_step()`` and marks the uid as disabled to
    prevent re-entry on the next step.

    This preserves the round-A "bench harness with temp>0" path
    working under the round-H tightening.
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from fusion_mlx.scheduler import _install_mtp_vendored

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [11]
    # temp > 0 from the start — MTP never primes.
    request_stub = SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.7))
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests={"req-11": request_stub},
        uid_to_request_id={11: "req-11"},
    )
    assert ok is True

    gb._next_tokens = mx.array([200], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    # Should soft-fall-through, not raise.
    gb._step()
    stats = batch_gen._mtp_vendored_stats
    assert stats["ft_non_greedy"] >= 1
    assert gb.orig_step_calls == 1


def test_install_mtp_vendored_mid_stream_generator_failure_raises(monkeypatch):
    """Codex round-D blocker #3 regression guard.

    Mid-stream failure of the internal ``mtp_generate_step`` generator
    cannot fall back to plain ``_orig_step`` because the wrapper never
    updates ``gb._next_tokens`` — it still holds ``first_gen_tok`` from
    the priming ``_step``. A silent fallback would emit
    ``first_gen_tok`` AGAIN, corrupting the output stream.

    Fix: re-raise as ``RuntimeError`` so mlx-lm surfaces the failure
    to the caller cleanly.

    This test constructs a generator that yields once (the first
    subsequent-call sample) and then raises on the second ``next()``,
    then asserts the wrapper propagates the failure instead of
    delegating to ``_orig_step``.
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from fusion_mlx.scheduler import _install_mtp_vendored
    from fusion_mlx.speculative.mtp import generator as _gen_mod

    class _MidStreamFailingGen:
        def __init__(self):
            self._n = 0

        def __iter__(self):
            return self

        def __next__(self):
            self._n += 1
            if self._n <= 1:
                return (2001, mx.array([0.0]), False)
            raise RuntimeError("simulated mid-stream generator failure")

        def close(self):
            pass

    def _mid_stream_failing_ctor(*args, **kwargs):
        return _MidStreamFailingGen()

    monkeypatch.setattr(_gen_mod, "mtp_generate_step", _mid_stream_failing_ctor)

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [55]
    request_stub = SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0))
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests={"req-55": request_stub},
        uid_to_request_id={55: "req-55"},
    )
    assert ok is True

    gb._next_tokens = mx.array([1000], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    # First call — construct, emit first_gen_tok = 1000.
    gb._step()

    # Second call — pulls from generator, yields 2001.
    gb._step()

    # Third call — generator raises. MUST propagate as RuntimeError
    # rather than falling back to _orig_step (which would emit 1000
    # again and duplicate the token stream).
    orig_step_calls_before = gb.orig_step_calls
    try:
        gb._step()
    except RuntimeError as e:
        assert "mid-stream" in str(e).lower() or "generator raised" in str(e).lower()
        # _orig_step must NOT have been called on the failure branch.
        assert gb.orig_step_calls == orig_step_calls_before, (
            "codex round-D blocker #3 regression: wrapper delegated to "
            "_orig_step on mid-stream generator failure, which duplicates "
            f"first_gen_tok in the output stream "
            f"(orig_step_calls: {orig_step_calls_before} -> "
            f"{gb.orig_step_calls})."
        )
        stats = batch_gen._mtp_vendored_stats
        assert stats.get("gen_raised", 0) >= 1
        return
    raise AssertionError(
        "codex round-D blocker #3 regression: wrapper did NOT raise on "
        "mid-stream generator failure. Falling back to _orig_step here "
        "would emit first_gen_tok twice (duplicated) because _next_"
        "tokens is stale relative to what the vendored path already "
        "emitted."
    )


def test_install_mtp_vendored_first_call_syncs_next_tokens(monkeypatch):
    """Codex round-I BLOCKING #2 + round-J BLOCKING #2/#3 regression
    guard (FIRST-call branch).

    Contract: after ``_step`` returns, ``gb._next_tokens`` must hold
    a coherent-shape ``mx.array([tok], dtype=uint32)`` so
    ``.filter(keep)`` slicing / ``.extend(batch)`` concatenation
    don't blow up on the frozen ``first_gen_tok`` from the priming
    step or a torn shape.

    Round-J review: the initial fix drove the MTP generator one
    step ahead (a "prefetch") to publish the NEXT to-be-emitted
    token here, but that advanced ``prompt_cache`` behind
    mlx-lm's bookkeeping. Round-J directed us to avoid the
    prefetch and stash a coherent shape from the JUST-emitted
    token instead. The "stale value" is safe because round-H
    tightened every ``_orig_step()`` fallthrough branch to raise
    terminally once ``_state[uid]`` is populated — no downstream
    reader consumes the placeholder as a model input.

    Verify:
      * ``_next_tokens`` is not None after the emit.
      * Its value equals the just-emitted token (stale placeholder,
        not a prefetched next token).
      * Shape / dtype are (1,) / uint32 as mlx-lm expects.
      * The MTP generator was NOT driven ahead — only ONE
        ``next()`` call happens per wrapper step, and that
        happens in the SUBSEQUENT branch, not here.
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from fusion_mlx.scheduler import _install_mtp_vendored
    from fusion_mlx.speculative.mtp import generator as _gen_mod

    class _CountingGen:
        def __init__(self):
            self._n = 0

        def __iter__(self):
            return self

        def __next__(self):
            self._n += 1
            return (2000 + self._n, mx.array([0.1 * self._n]), False)

        def close(self):
            pass

    counting_gen = _CountingGen()
    monkeypatch.setattr(_gen_mod, "mtp_generate_step", lambda *a, **kw: counting_gen)

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [7]
    request_stub = SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0))
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests={"req-7": request_stub},
        uid_to_request_id={7: "req-7"},
    )
    assert ok is True

    # Priming step sets _next_tokens = first_gen_tok = 1000.
    gb._next_tokens = mx.array([1000], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    # Step 1 (FIRST-call). Should emit 1000 and update _next_tokens
    # to a coherent placeholder (1000, same as just-emitted). The
    # MTP generator MUST NOT be driven ahead here (round-J BLOCKING
    # #2 — that would advance prompt_cache behind mlx-lm's
    # bookkeeping).
    tokens, logprobs = gb._step()
    assert tokens == [1000]
    assert gb._next_tokens is not None, (
        "codex round-I BLOCKING #2 regression: _next_tokens is None "
        "after successful FIRST-call emission."
    )
    _next_tok_val = int(gb._next_tokens[0].item())
    assert _next_tok_val == 1000, (
        f"codex round-J BLOCKING #2 regression: FIRST-call branch "
        f"published a value ({_next_tok_val}) other than the just-"
        "emitted token. The round-J-approved contract is 'stash the "
        "just-emitted token as a coherent-shape placeholder'; any "
        "other value would imply a prefetch that advances "
        "prompt_cache behind mlx-lm's bookkeeping."
    )
    assert gb._next_tokens.dtype == mx.uint32
    assert gb._next_tokens.shape == (1,)
    assert len(gb._next_logprobs) == 1
    # Round-J BLOCKING #2: verify the generator was NOT driven ahead
    # by the FIRST-call sync. counting_gen.__next__ should not have
    # been invoked yet — the generator's first next() call happens
    # in the SUBSEQUENT branch (Step 2 below).
    assert counting_gen._n == 0, (
        f"codex round-J BLOCKING #2 regression: the wrapper drove "
        f"the MTP generator {counting_gen._n} step(s) ahead in the "
        "FIRST-call branch. This advances prompt_cache behind "
        "GenerationBatch's bookkeeping and was flagged as unsafe."
    )


def test_install_mtp_vendored_subsequent_syncs_next_tokens(monkeypatch):
    """Codex round-I BLOCKING #2 + round-J BLOCKING #2 regression
    guard (SUBSEQUENT branch).

    Same coherent-shape contract as the FIRST-call variant. Verify
    ``_next_tokens`` after each SUBSEQUENT emission holds the
    just-emitted token — not a prefetched next token — and the
    MTP generator advances EXACTLY once per SUBSEQUENT call
    (not once for emit + once for prefetch).
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from fusion_mlx.scheduler import _install_mtp_vendored
    from fusion_mlx.speculative.mtp import generator as _gen_mod

    class _CountingGen:
        def __init__(self):
            self._n = 0

        def __iter__(self):
            return self

        def __next__(self):
            self._n += 1
            return (3000 + self._n, mx.array([0.1 * self._n]), False)

        def close(self):
            pass

    counting_gen = _CountingGen()
    monkeypatch.setattr(_gen_mod, "mtp_generate_step", lambda *a, **kw: counting_gen)

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [9]
    request_stub = SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0))
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests={"req-9": request_stub},
        uid_to_request_id={9: "req-9"},
    )
    assert ok is True

    gb._next_tokens = mx.array([500], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    # Step 1 — FIRST-call, emits 500. Generator NOT touched.
    gb._step()
    assert int(gb._next_tokens[0].item()) == 500
    assert counting_gen._n == 0

    # Step 2 — SUBSEQUENT branch. Pulls one from generator (yields
    # 3001), emits 3001, syncs _next_tokens=3001.
    tokens, _ = gb._step()
    assert tokens == [3001]
    _val_after_step2 = int(gb._next_tokens[0].item())
    assert _val_after_step2 == 3001, (
        "codex round-I BLOCKING #2 regression: SUBSEQUENT branch did "
        f"NOT sync _next_tokens with the just-emitted token (got "
        f"{_val_after_step2}, expected 3001)."
    )
    assert counting_gen._n == 1, (
        f"codex round-J BLOCKING #2 regression: SUBSEQUENT branch "
        f"advanced the generator {counting_gen._n} steps ahead of "
        "the emission — a prefetch was reintroduced."
    )

    # Step 3 — SUBSEQUENT branch again. Pulls once, emits 3002.
    tokens, _ = gb._step()
    assert tokens == [3002]
    assert int(gb._next_tokens[0].item()) == 3002
    assert counting_gen._n == 2


def test_install_mtp_vendored_next_tokens_shape_survives_stop_iteration(
    monkeypatch,
):
    """Codex round-I BLOCKING #2 + round-J BLOCKING #3 regression
    guard.

    Round-J correctly flagged that swallowing a generator
    ``StopIteration`` inside a "prefetch" helper delays the
    terminal-raise. The no-prefetch design has no swallow: the
    generator is only consumed inside the SUBSEQUENT branch's
    queue-empty path, and any exception there terminal-raises
    IMMEDIATELY. Between FIRST-call emit and the SUBSEQUENT
    terminal-raise, ``_next_tokens`` must still be shape-coherent.
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from fusion_mlx.scheduler import _install_mtp_vendored
    from fusion_mlx.speculative.mtp import generator as _gen_mod

    class _OneShotGen:
        """Yields nothing — first ``next()`` raises StopIteration."""

        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def close(self):
            pass

    monkeypatch.setattr(_gen_mod, "mtp_generate_step", lambda *a, **kw: _OneShotGen())

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [13]
    request_stub = SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0))
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests={"req-13": request_stub},
        uid_to_request_id={13: "req-13"},
    )
    assert ok is True

    gb._next_tokens = mx.array([42], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    # Step 1 — FIRST-call, emits 42. No generator prefetch, so no
    # exception surfaces here. _next_tokens is a coherent-shape
    # placeholder (just-emitted token).
    tokens, _ = gb._step()
    assert tokens == [42]
    assert gb._next_tokens is not None
    assert gb._next_tokens.dtype == mx.uint32
    assert gb._next_tokens.shape == (1,)

    # Step 2 — SUBSEQUENT branch. queue empty, generator raises
    # StopIteration IMMEDIATELY. Terminal-raise fires with the
    # real error trace; no swallowing, no delay.
    try:
        gb._step()
    except RuntimeError as e:
        assert (
            "generator exhausted" in str(e).lower()
            or "before mlx-lm hit" in str(e).lower()
        )
        return
    raise AssertionError(
        "codex round-J BLOCKING #3 regression: SUBSEQUENT branch did "
        "NOT terminal-raise on generator StopIteration. Under the "
        "no-prefetch design there is no exception to swallow, and the "
        "raise must fire IMMEDIATELY on the very next _step() call."
    )


# ---------------------------------------------------------------------------
# 6. _apply_mtp_cli_model_type_reconciliation
# ---------------------------------------------------------------------------


def test_apply_mtp_cli_model_type_reconciliation_promotes_eligibility_read():
    from fusion_mlx.cli_serve import _apply_mtp_cli_model_type_reconciliation
    from fusion_mlx.config import SchedulerConfig

    cfg = SchedulerConfig()
    assert cfg.mtp_model_type is None
    hf_cfg = {"model_type": "qwen3_5", "mtp_num_hidden_layers": 1}
    _apply_mtp_cli_model_type_reconciliation(cfg, hf_cfg)
    assert cfg.mtp_model_type == "qwen3_5"


def test_apply_mtp_cli_model_type_reconciliation_hard_fails_when_model_type_missing(
    capsys,
):
    from fusion_mlx.cli_serve import _apply_mtp_cli_model_type_reconciliation
    from fusion_mlx.config import SchedulerConfig

    cfg = SchedulerConfig()
    # MTP layers present but no model_type -> detection returns NONE.
    hf_cfg = {"mtp_num_hidden_layers": 1}
    with pytest.raises(SystemExit):
        _apply_mtp_cli_model_type_reconciliation(cfg, hf_cfg)
    err = capsys.readouterr().err
    assert "not MTP-eligible" in err


def test_apply_mtp_cli_model_type_reconciliation_prefers_eligibility_on_disagreement():
    from fusion_mlx.cli_serve import _apply_mtp_cli_model_type_reconciliation
    from fusion_mlx.config import SchedulerConfig

    cfg = SchedulerConfig()
    cfg.mtp_model_type = "qwen3_5_moe"  # stale operator-set value
    hf_cfg = {"model_type": "qwen3_5", "mtp_num_hidden_layers": 2}
    _apply_mtp_cli_model_type_reconciliation(cfg, hf_cfg)
    # Eligibility read wins over the stale operator value.
    assert cfg.mtp_model_type == "qwen3_5"


def test_install_mtp_vendored_uid_reuse_clears_stale_state(monkeypatch):
    """Codex round-K BLOCKING #1 regression guard.

    mlx-lm reuses uid ints when a request completes and a new one
    joins the batch. Pre-round-K the wrapper's ``_state`` map was
    keyed on uid alone with NO request_id validation (unlike
    ``_disabled_uids`` which stores the owning request_id since
    round-E). Under uid reuse, the wrapper would resume the OLD
    request's generator on the NEW request's first _step call —
    a data corruption bug because the SUBSEQUENT branch pulls
    from the STALE generator (built for the old prompt +
    prompt_cache) and appends stale tokens to gb.tokens[0].

    Verify:
      1. Request A drives one FIRST-call emission and populates
         ``_state[uid=X]`` with request_id=req-A.
      2. Under uid reuse (uid=X → req-B without any
         ``_cleanup_uid``), the wrapper's uid-reuse gate MUST fire
         and treat the entry as stale: close the OLD generator,
         drop ``_state[uid=X]``, and re-enter the FIRST-call
         construction path for req-B.
      3. The new construction ATTEMPT happens (visible via ctor
         call count) — proving the reuse gate cleared the state
         rather than resuming the old generator.
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from fusion_mlx.scheduler import _install_mtp_vendored
    from fusion_mlx.speculative.mtp import generator as _gen_mod

    class _FakeGen:
        def __init__(self, tag):
            self._n = 0
            self._tag = tag

        def __iter__(self):
            return self

        def __next__(self):
            self._n += 1
            # Encode the tag into the emitted token so the test can
            # tell which generator produced the token.
            return (10_000 + 100 * self._tag + self._n, mx.array([0.0]), False)

        def close(self):
            pass

    generators_built: list[int] = []

    def _tagged_ctor(*args, **kwargs):
        tag = len(generators_built) + 1
        generators_built.append(tag)
        return _FakeGen(tag)

    monkeypatch.setattr(_gen_mod, "mtp_generate_step", _tagged_ctor)

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [77]
    uid_to_request_id: dict[int, str] = {77: "req-A"}
    requests: dict = {
        "req-A": SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0)),
    }
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests=requests,
        uid_to_request_id=uid_to_request_id,
    )
    assert ok is True

    gb._next_tokens = mx.array([1000], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    # Step 1 for req-A — FIRST-call construction, generator #1
    # built. State populated with request_id=req-A.
    gb._step()
    assert (
        len(generators_built) == 1
    ), f"expected exactly one generator built for req-A, got {generators_built!r}"

    # Simulate mlx-lm's request completion + uid reuse: same uid,
    # new request_id. No _cleanup_uid call — this exactly mirrors
    # what happens between .filter(keep) removing req-A and
    # .extend(new_batch) adding req-B on the same uid.
    uid_to_request_id[77] = "req-B"
    requests["req-B"] = SimpleNamespace(
        sampling_params=SimpleNamespace(temperature=0.0)
    )
    gb._next_tokens = mx.array([2000], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    # Step 1 for req-B — the uid-reuse gate MUST fire, close the
    # OLD generator, and re-enter FIRST-call construction. A NEW
    # generator (#2) is built. If the round-K fix regressed, the
    # SUBSEQUENT branch of the wrapper would pull the next token
    # from the OLD generator (tag=1) and emit a stale token.
    tokens, _ = gb._step()
    assert len(generators_built) == 2, (
        "codex round-K BLOCKING #1 regression: uid reuse for a new "
        "request did NOT trigger fresh MTP construction. Generators "
        f"built: {generators_built!r}. The stale OLD generator "
        "would emit tokens from the previous request's context."
    )
    # The FIRST-call emission for req-B is the priming step's
    # sample (2000, which we set on gb._next_tokens above).
    assert tokens == [2000], (
        "req-B's FIRST-call did NOT emit the priming-step sample "
        f"(got {tokens!r}, expected [2000]). This suggests the "
        "wrapper resumed the OLD generator's queue / iteration state."
    )
