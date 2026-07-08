# SPDX-License-Identifier: Apache-2.0
# #431 Step 2: router-driven per-request spec-decode dispatch.
#
# Validates that _try_spec_decode runs ONLY the method the router chose for
# the request (no ngram->dflash->dspark fall-through), with draft-model kept
# as a fallback; and that _step_pure_decode decides the method once at the
# first pure-decode step and suppresses mtp on the forward pass for non-mtp
# choices. Self-contained (SimpleNamespace mocks); not in debt_modules.txt.

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _make_request(method):
    from fusion_mlx.request import Request, RequestStatus, SamplingParams

    r = Request("r1", [1, 2, 3], SamplingParams())
    r.status = RequestStatus.RUNNING
    r._active_spec_method = method
    return r


def _make_sched(request, *, ngram=None, dflash=None, dspark=None, draft=None):
    return SimpleNamespace(
        running={"r1": request},
        uid_to_request_id={0: "r1"},
        _vlm_mtp_active=False,
        batch_generator=SimpleNamespace(),
        _output_parser_sessions=set(),
        _pending_abort_ids=set(),
        _ngram_spec_state=ngram,
        _dflash_runtime=dflash,
        _dspark_runtime=dspark,
        _spec_decode_state=(
            SimpleNamespace(draft_model=draft) if draft is not None else None
        ),
    )


def _resp():
    return SimpleNamespace(uid=0, token=42, finish_reason=None)


class TestTrySpecDecodeDispatch:
    # Only the router-chosen method runs; no fall-through among heuristics.

    def test_ngram_chosen_runs_only_ngram(self, monkeypatch):
        import fusion_mlx.scheduler.ngram_spec as ns
        import fusion_mlx.scheduler.spec_decode as sd
        from fusion_mlx.scheduler.config import SchedulerOutput
        from fusion_mlx.scheduler.sched_step import _try_spec_decode
        from fusion_mlx.speculative.auto_router import METHOD_NGRAM

        calls = []
        monkeypatch.setattr(
            ns, "ngram_spec_step", lambda *a, **k: calls.append("ngram") or ["out"]
        )
        monkeypatch.setattr(
            sd, "dflash_spec_step", lambda *a, **k: calls.append("dflash") or ["x"]
        )
        monkeypatch.setattr(
            sd, "dspark_spec_step", lambda *a, **k: calls.append("dspark") or ["x"]
        )
        monkeypatch.setattr(
            sd, "spec_decode_step", lambda *a, **k: calls.append("draft") or ["x"]
        )

        req = _make_request(METHOD_NGRAM)
        sched = _make_sched(req, ngram=object())
        result = _try_spec_decode(sched, [_resp()], SchedulerOutput())
        assert result == ["out"]
        assert calls == ["ngram"]

    def test_dflash_chosen_runs_only_dflash(self, monkeypatch):
        import fusion_mlx.scheduler.spec_decode as sd
        from fusion_mlx.scheduler.config import SchedulerOutput
        from fusion_mlx.scheduler.sched_step import _try_spec_decode
        from fusion_mlx.speculative.auto_router import METHOD_DFLASH

        calls = []
        monkeypatch.setattr(
            sd,
            "dflash_spec_step",
            lambda *a, **k: calls.append("dflash") or ["out"],
        )
        monkeypatch.setattr(
            sd, "dspark_spec_step", lambda *a, **k: calls.append("dspark") or ["x"]
        )
        monkeypatch.setattr(
            sd, "spec_decode_step", lambda *a, **k: calls.append("draft") or ["x"]
        )

        req = _make_request(METHOD_DFLASH)
        sched = _make_sched(req, dflash=object())
        result = _try_spec_decode(sched, [_resp()], SchedulerOutput())
        assert result == ["out"]
        assert calls == ["dflash"]

    def test_dspark_chosen_runs_only_dspark(self, monkeypatch):
        import fusion_mlx.scheduler.spec_decode as sd
        from fusion_mlx.scheduler.config import SchedulerOutput
        from fusion_mlx.scheduler.sched_step import _try_spec_decode
        from fusion_mlx.speculative.auto_router import METHOD_DSPARK

        calls = []
        monkeypatch.setattr(
            sd,
            "dspark_spec_step",
            lambda *a, **k: calls.append("dspark") or ["out"],
        )
        monkeypatch.setattr(
            sd, "spec_decode_step", lambda *a, **k: calls.append("draft") or ["x"]
        )

        req = _make_request(METHOD_DSPARK)
        sched = _make_sched(req, dspark=object())
        result = _try_spec_decode(sched, [_resp()], SchedulerOutput())
        assert result == ["out"]
        assert calls == ["dspark"]

    def test_mtp_chosen_falls_through_no_heuristic(self, monkeypatch):
        import fusion_mlx.scheduler.ngram_spec as ns
        import fusion_mlx.scheduler.spec_decode as sd
        from fusion_mlx.scheduler.config import SchedulerOutput
        from fusion_mlx.scheduler.sched_step import _try_spec_decode
        from fusion_mlx.speculative.auto_router import METHOD_MTP

        calls = []
        monkeypatch.setattr(
            ns, "ngram_spec_step", lambda *a, **k: calls.append("ngram") or ["x"]
        )
        monkeypatch.setattr(
            sd, "dflash_spec_step", lambda *a, **k: calls.append("dflash") or ["x"]
        )
        monkeypatch.setattr(
            sd, "spec_decode_step", lambda *a, **k: calls.append("draft") or ["x"]
        )

        req = _make_request(METHOD_MTP)
        # heuristics loaded but mtp chosen -> none run, no draft -> []
        sched = _make_sched(req, ngram=object(), dflash=object())
        result = _try_spec_decode(sched, [_resp()], SchedulerOutput())
        assert result == []
        assert calls == []

    def test_empty_method_falls_through(self, monkeypatch):
        import fusion_mlx.scheduler.ngram_spec as ns
        from fusion_mlx.scheduler.config import SchedulerOutput
        from fusion_mlx.scheduler.sched_step import _try_spec_decode

        calls = []
        monkeypatch.setattr(
            ns, "ngram_spec_step", lambda *a, **k: calls.append("ngram") or ["x"]
        )

        req = _make_request("")
        sched = _make_sched(req, ngram=object())
        result = _try_spec_decode(sched, [_resp()], SchedulerOutput())
        assert result == []
        assert calls == []

    def test_draft_fallback_when_chosen_heuristic_empty(self, monkeypatch):
        import fusion_mlx.scheduler.ngram_spec as ns
        import fusion_mlx.scheduler.spec_decode as sd
        from fusion_mlx.scheduler.config import SchedulerOutput
        from fusion_mlx.scheduler.sched_step import _try_spec_decode
        from fusion_mlx.speculative.auto_router import METHOD_NGRAM

        calls = []
        monkeypatch.setattr(
            ns, "ngram_spec_step", lambda *a, **k: calls.append("ngram") or []
        )
        monkeypatch.setattr(
            sd,
            "spec_decode_step",
            lambda *a, **k: calls.append("draft") or ["draft-out"],
        )

        req = _make_request(METHOD_NGRAM)
        sched = _make_sched(req, ngram=object(), draft=object())
        result = _try_spec_decode(sched, [_resp()], SchedulerOutput())
        assert result == ["draft-out"]
        assert calls == ["ngram", "draft"]

    def test_chosen_method_not_loaded_falls_through(self):
        # ngram chosen but _ngram_spec_state is None -> skip ngram, no draft
        from fusion_mlx.scheduler.config import SchedulerOutput
        from fusion_mlx.scheduler.sched_step import _try_spec_decode
        from fusion_mlx.speculative.auto_router import METHOD_NGRAM

        req = _make_request(METHOD_NGRAM)
        sched = _make_sched(req, ngram=None)
        result = _try_spec_decode(sched, [_resp()], SchedulerOutput())
        assert result == []


class TestStepPureDecodeDecision:
    # _step_pure_decode decides the method once and suppresses mtp for non-mtp
    # choices; the suppress flag is reset in finally even on exception.

    @pytest.fixture(autouse=True)
    def _stub_mx_stream(self, monkeypatch):
        # _step_pure_decode wraps bg._next() in `with mx.stream(self._stream)`.
        # Stub the stream contextmanager so the test never touches a real Metal
        # stream (works under both real mlx and the CI MagicMock stub).
        import contextlib

        import fusion_mlx.scheduler.sched_step as ss

        monkeypatch.setattr(ss.mx, "stream", lambda s: contextlib.nullcontext())

    def _make_sched(self, req, model, decide, bg_next):
        bg = SimpleNamespace(_next=bg_next)
        return SimpleNamespace(
            batch_generator=bg,
            running={"r1": req},
            model=model,
            _stream=None,
            _decide_spec_method=decide,
            _last_decode_dt=0.0,
        )

    def test_decides_and_suppresses_for_non_mtp(self):
        from fusion_mlx.request import Request, RequestStatus, SamplingParams
        from fusion_mlx.scheduler.config import SchedulerOutput
        from fusion_mlx.scheduler.sched_step import _step_pure_decode
        from fusion_mlx.speculative.auto_router import METHOD_NGRAM

        req = Request("r1", [1, 2, 3], SamplingParams())
        req.status = RequestStatus.RUNNING
        assert req._active_spec_method is None

        model = SimpleNamespace()
        seen = {}

        def _next():
            seen["suppress"] = getattr(model, "_fusion_mlx_mtp_suppressed", None)
            return ([], [])

        sched = self._make_sched(req, model, lambda r: METHOD_NGRAM, _next)
        _step_pure_decode(sched, SchedulerOutput())
        assert req._active_spec_method == METHOD_NGRAM
        assert seen["suppress"] is True
        assert model._fusion_mlx_mtp_suppressed is False

    def test_no_suppress_for_mtp_choice(self):
        from fusion_mlx.request import Request, RequestStatus, SamplingParams
        from fusion_mlx.scheduler.config import SchedulerOutput
        from fusion_mlx.scheduler.sched_step import _step_pure_decode
        from fusion_mlx.speculative.auto_router import METHOD_MTP

        req = Request("r1", [1, 2, 3], SamplingParams())
        req.status = RequestStatus.RUNNING

        model = SimpleNamespace()
        seen = {}

        def _next():
            seen["suppress"] = getattr(model, "_fusion_mlx_mtp_suppressed", None)
            return ([], [])

        sched = self._make_sched(req, model, lambda r: METHOD_MTP, _next)
        _step_pure_decode(sched, SchedulerOutput())
        assert req._active_spec_method == METHOD_MTP
        assert seen["suppress"] is False
        assert model._fusion_mlx_mtp_suppressed is False

    def test_decision_cached_not_re_decided(self):
        from fusion_mlx.request import Request, RequestStatus, SamplingParams
        from fusion_mlx.scheduler.config import SchedulerOutput
        from fusion_mlx.scheduler.sched_step import _step_pure_decode
        from fusion_mlx.speculative.auto_router import METHOD_NGRAM

        req = Request("r1", [1, 2, 3], SamplingParams())
        req.status = RequestStatus.RUNNING
        req._active_spec_method = METHOD_NGRAM  # already decided

        decide_calls = []

        def _decide(r):
            decide_calls.append(1)
            return METHOD_NGRAM

        sched = self._make_sched(req, SimpleNamespace(), _decide, lambda: ([], []))
        _step_pure_decode(sched, SchedulerOutput())
        assert decide_calls == []

    def test_suppress_reset_on_exception(self):
        from fusion_mlx.request import Request, RequestStatus, SamplingParams
        from fusion_mlx.scheduler.config import SchedulerOutput
        from fusion_mlx.scheduler.sched_step import _step_pure_decode
        from fusion_mlx.speculative.auto_router import METHOD_NGRAM

        req = Request("r1", [1, 2, 3], SamplingParams())
        req.status = RequestStatus.RUNNING

        model = SimpleNamespace()

        def _boom():
            raise RuntimeError("next failed")

        sched = self._make_sched(req, model, lambda r: METHOD_NGRAM, _boom)
        with pytest.raises(RuntimeError):
            _step_pure_decode(sched, SchedulerOutput())
        assert model._fusion_mlx_mtp_suppressed is False
