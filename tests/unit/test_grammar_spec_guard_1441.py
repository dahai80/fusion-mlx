# SPDX-License-Identifier: Apache-2.0
# #1441: heuristic speculative decode must not run for grammar-constrained
# requests. Every heuristic path verifies drafts with a raw argmax over the
# target's unmasked logits and emits accepted tokens without running
# logits_processors / GrammarConstraintProcessor.accept_token(), so the
# grammar bitmap desyncs from the emitted stream (invented schema fields,
# fused keys, degenerate loops).

from __future__ import annotations

from types import SimpleNamespace

from fusion_mlx.request import Request, RequestStatus, SamplingParams
from fusion_mlx.scheduler.config import SchedulerOutput
from fusion_mlx.scheduler.sched_step import _decide_spec_method

SENTINEL = "spec-disabled"


def _grammar():
    return SimpleNamespace(kind="llguidance", allow_whitespace=True)


def _make_request(grammar=None, method=None):
    params = SamplingParams(compiled_grammar=grammar)
    req = Request("r1", [1, 2, 3], params)
    req.status = RequestStatus.RUNNING
    req._active_spec_method = method
    return req


def _make_sched(request, *, ngram=None, draft=None):
    return SimpleNamespace(
        running={"r1": request},
        uid_to_request_id={0: "r1"},
        _vlm_mtp_active=False,
        batch_generator=SimpleNamespace(),
        _output_parser_sessions=set(),
        _pending_abort_ids=set(),
        _ngram_spec_state=ngram,
        _dflash_runtime=None,
        _dflash2_runtime=None,
        _dspark_runtime=None,
        _spec_decode_state=(SimpleNamespace(draft_model=draft) if draft else None),
        model=SimpleNamespace(),
        config=SimpleNamespace(model_name="test"),
    )


def _resp():
    return SimpleNamespace(uid=0, token=42, finish_reason=None)


class TestDecideSpecMethodGrammar:
    def test_grammar_request_gets_sentinel(self):
        from fusion_mlx.scheduler.sched_step import _decide_spec_method

        req = _make_request(_grammar())
        sched = SimpleNamespace(config=SimpleNamespace(model_name="test"))
        assert _decide_spec_method(sched, req) == SENTINEL

    def test_router_not_consulted_for_grammar(self, monkeypatch):
        import fusion_mlx.scheduler.sched_step as ss
        import fusion_mlx.speculative.per_request_route as route

        calls = []
        monkeypatch.setattr(
            route, "select_active_method", lambda *a, **k: calls.append(1) or "ngram"
        )
        monkeypatch.setattr(
            ss, "_loaded_spec_methods", lambda self: calls.append("loaded") or {}
        )

        req = _make_request(_grammar())
        sched = SimpleNamespace(config=SimpleNamespace(model_name="test"))
        ss._decide_spec_method(sched, req)
        assert calls == []

    def test_plain_request_still_routes(self, monkeypatch):
        import fusion_mlx.speculative.per_request_route as route

        monkeypatch.setattr(route, "select_active_method", lambda *a, **k: "ngram")

        req = _make_request(None)
        sched = _make_sched(req, ngram=object())
        sched._loaded_spec_methods = lambda: {
            "suffix": True,
            "eagle3": False,
            "dflash": False,
            "dflash2": False,
            "dspark": False,
            "mtp": False,
            "medusa": False,
        }
        assert _decide_spec_method(sched, req) == "ngram"


class TestTrySpecDecodeGrammar:
    def test_ngram_skipped_for_grammar_request(self, monkeypatch):
        import fusion_mlx.scheduler.ngram_spec as ns
        from fusion_mlx.scheduler.sched_step import _try_spec_decode
        from fusion_mlx.speculative.auto_router import METHOD_NGRAM

        calls = []
        monkeypatch.setattr(
            ns, "ngram_spec_step", lambda *a, **k: calls.append("ngram") or ["x"]
        )

        req = _make_request(_grammar(), method=METHOD_NGRAM)
        sched = _make_sched(req, ngram=object())
        assert _try_spec_decode(sched, [_resp()], SchedulerOutput()) == []
        assert calls == []

    def test_draft_fallback_skipped_for_grammar_request(self, monkeypatch):
        import fusion_mlx.scheduler.spec_decode as sd
        from fusion_mlx.scheduler.sched_step import _try_spec_decode

        calls = []
        monkeypatch.setattr(
            sd, "spec_decode_step", lambda *a, **k: calls.append("draft") or ["x"]
        )

        req = _make_request(_grammar(), method=SENTINEL)
        sched = _make_sched(req, draft=object())
        assert _try_spec_decode(sched, [_resp()], SchedulerOutput()) == []
        assert calls == []

    def test_plain_request_still_spec_decodes(self, monkeypatch):
        import fusion_mlx.scheduler.ngram_spec as ns
        from fusion_mlx.scheduler.sched_step import _try_spec_decode
        from fusion_mlx.speculative.auto_router import METHOD_NGRAM

        calls = []
        monkeypatch.setattr(
            ns, "ngram_spec_step", lambda *a, **k: calls.append("ngram") or ["out"]
        )

        req = _make_request(None, method=METHOD_NGRAM)
        sched = _make_sched(req, ngram=object())
        assert _try_spec_decode(sched, [_resp()], SchedulerOutput()) == ["out"]
        assert calls == ["ngram"]
