# SPDX-License-Identifier: Apache-2.0
import logging
from types import SimpleNamespace

from fusion_mlx.speculative.auto_router import (
    METHOD_DFLASH,
    METHOD_DFLASH2,
    METHOD_DSPARK,
    METHOD_DSPINE,
    METHOD_EAGLE3,
    METHOD_MEDUSA,
    METHOD_MTP,
    METHOD_NGRAM,
    SpecAutoRouter,
)
from fusion_mlx.speculative.per_request_route import (
    loaded_methods,
    select_active_method,
)


class TestSelectActiveMethod:
    def test_no_method_loaded_returns_none(self, caplog):
        caplog.set_level(
            logging.DEBUG, logger="fusion_mlx.speculative.per_request_route"
        )
        result = select_active_method(
            prompt_token_count=100,
            loaded=loaded_methods(),
        )
        assert result is None
        assert any("spec disabled" in r.message for r in caplog.records)

    def test_only_suffix_loaded_picks_suffix_regardless_of_prompt(self):
        loaded = loaded_methods(suffix=True)
        assert select_active_method(100, loaded) == METHOD_NGRAM
        # long-doc threshold met but dflash not loaded -> stays suffix
        assert select_active_method(8192, loaded) == METHOD_NGRAM

    def test_suffix_and_dflash_long_doc_routes_to_dflash(self):
        loaded = loaded_methods(suffix=True, dflash=True)
        router = SpecAutoRouter(long_doc_threshold=4096)
        assert select_active_method(8192, loaded, router=router) == METHOD_DFLASH

    def test_suffix_and_dflash_short_doc_routes_to_suffix(self):
        loaded = loaded_methods(suffix=True, dflash=True)
        router = SpecAutoRouter(long_doc_threshold=4096)
        assert select_active_method(100, loaded, router=router) == METHOD_NGRAM

    def test_mtp_loaded_and_eligible_picks_mtp(self):
        loaded = loaded_methods(suffix=True, mtp=True)
        assert select_active_method(100, loaded, has_mtp=True) == METHOD_MTP

    def test_mtp_loaded_but_not_eligible_falls_back_to_suffix(self):
        loaded = loaded_methods(suffix=True, mtp=True)
        assert select_active_method(100, loaded, has_mtp=False) == METHOD_NGRAM

    def test_has_mtp_true_but_mtp_not_loaded_ignores_mtp(self):
        loaded = loaded_methods(suffix=True)
        # has_mtp=True but mtp not in loaded -> must not pick mtp
        assert select_active_method(100, loaded, has_mtp=True) == METHOD_NGRAM

    def test_hysteresis_keeps_working_current_method(self):
        loaded = loaded_methods(suffix=True, dflash=True)
        router = SpecAutoRouter(long_doc_threshold=4096, keep_accept=0.40)
        # current=dflash, acceptance healthy -> keep dflash even on short prompt
        # (short prompt would otherwise route to suffix)
        result = select_active_method(
            100,
            loaded,
            router=router,
            current_method=METHOD_DFLASH,
            recent_accept_rate=0.60,
        )
        assert result == METHOD_DFLASH

    def test_hysteresis_abandons_failing_current_method(self):
        loaded = loaded_methods(suffix=True, dflash=True)
        router = SpecAutoRouter(
            long_doc_threshold=4096, abandon_accept=0.20, keep_accept=0.40
        )
        # current=dflash but acceptance terrible -> abandon, re-pick. Long doc
        # would re-pick dflash, so use short doc to force a different method.
        result = select_active_method(
            100,
            loaded,
            router=router,
            current_method=METHOD_DFLASH,
            recent_accept_rate=0.05,
        )
        assert result == METHOD_NGRAM

    def test_current_method_not_in_loaded_is_ignored(self):
        loaded = loaded_methods(suffix=True)
        # current=dflash but dflash not loaded -> current treated as None,
        # fresh selection picks suffix
        result = select_active_method(
            100,
            loaded,
            current_method=METHOD_DFLASH,
            recent_accept_rate=0.60,
        )
        assert result == METHOD_NGRAM

    def test_only_dflash_loaded_and_abandoned_returns_none(self):
        loaded = loaded_methods(dflash=True)
        router = SpecAutoRouter(
            long_doc_threshold=4096, abandon_accept=0.20, keep_accept=0.40
        )
        # only dflash loaded, it's abandoned+excluded -> degenerate fallback
        # yields a method not in available -> None (spec disabled)
        result = select_active_method(
            100,
            loaded,
            router=router,
            current_method=METHOD_DFLASH,
            recent_accept_rate=0.05,
        )
        assert result is None

    def test_all_four_loaded_long_doc_picks_dflash(self):
        loaded = loaded_methods(suffix=True, dflash=True, dspark=True, mtp=True)
        router = SpecAutoRouter(long_doc_threshold=4096)
        # long doc + dflash available -> dflash (even with mtp available)
        assert (
            select_active_method(5000, loaded, router=router, has_mtp=True)
            == METHOD_DFLASH
        )

    def test_all_four_loaded_short_doc_mtp_eligible_picks_mtp(self):
        loaded = loaded_methods(suffix=True, dflash=True, dspark=True, mtp=True)
        router = SpecAutoRouter(long_doc_threshold=4096)
        assert (
            select_active_method(100, loaded, router=router, has_mtp=True) == METHOD_MTP
        )

    def test_dspark_alone_loaded_picks_dspark(self):
        loaded = loaded_methods(dspark=True)
        result = select_active_method(100, loaded)
        assert result == METHOD_DSPARK

    def test_medusa_loaded_qwen3_picks_medusa(self):
        loaded = loaded_methods(medusa=True, suffix=True)
        result = select_active_method(100, loaded, model_family="qwen3", has_mtp=False)
        assert result == METHOD_MEDUSA

    def test_medusa_loaded_non_qwen3_falls_through(self):
        loaded = loaded_methods(medusa=True, suffix=True)
        result = select_active_method(100, loaded, model_family="llama3", has_mtp=False)
        assert result == METHOD_NGRAM


class TestLoadedMethods:
    def test_default_all_false(self):
        assert loaded_methods() == {
            METHOD_NGRAM: False,
            METHOD_EAGLE3: False,
            METHOD_DFLASH: False,
            METHOD_DFLASH2: False,
            METHOD_DSPARK: False,
            METHOD_DSPINE: False,
            METHOD_MTP: False,
            METHOD_MEDUSA: False,
        }

    def test_selective_true(self):
        lm = loaded_methods(suffix=True, dspark=True)
        assert lm[METHOD_NGRAM] is True
        assert lm[METHOD_DSPARK] is True
        assert lm[METHOD_DFLASH] is False
        assert lm[METHOD_MTP] is False


# --- scheduler-side assembly + decision (#431 Step 2) ---


class TestSchedulerLoadedAssembly:
    # scheduler._loaded_spec_methods assembles the boot-time loaded dict from
    # scheduler attrs (_ngram_spec_state/_dflash_runtime/_dspark_runtime +
    # model._fusion_mlx_mtp_decode_enabled), NOT from the registry whose
    # config_enabled is hardcoded True and does not reflect actual loading.

    def test_all_unloaded_returns_all_false(self):
        from fusion_mlx.scheduler.sched_step import _loaded_spec_methods

        sched = SimpleNamespace(
            _ngram_spec_state=None,
            _spec_decode_state=None,
            _dflash_runtime=None,
            _dspark_runtime=None,
            model=SimpleNamespace(),
        )
        assert _loaded_spec_methods(sched) == {
            METHOD_NGRAM: False,
            METHOD_EAGLE3: False,
            METHOD_DFLASH: False,
            METHOD_DFLASH2: False,
            METHOD_DSPARK: False,
            METHOD_DSPINE: False,
            METHOD_MTP: False,
            METHOD_MEDUSA: False,
        }

    def test_assembles_from_scheduler_attrs(self):
        from fusion_mlx.scheduler.sched_step import _loaded_spec_methods

        sched = SimpleNamespace(
            _ngram_spec_state=object(),
            _spec_decode_state=None,
            _dflash_runtime=object(),
            _dspark_runtime=None,
            model=SimpleNamespace(_fusion_mlx_mtp_decode_enabled=True),
        )
        lm = _loaded_spec_methods(sched)
        assert lm[METHOD_NGRAM] is True
        assert lm[METHOD_DFLASH] is True
        assert lm[METHOD_DSPARK] is False
        assert lm[METHOD_MTP] is True

    def test_mtp_reads_model_decode_enabled(self):
        from fusion_mlx.scheduler.sched_step import _loaded_spec_methods

        sched = SimpleNamespace(
            _ngram_spec_state=None,
            _spec_decode_state=None,
            _dflash_runtime=None,
            _dspark_runtime=None,
            model=SimpleNamespace(_fusion_mlx_mtp_decode_enabled=False),
        )
        assert _loaded_spec_methods(sched)[METHOD_MTP] is False


class TestDecideSpecMethod:
    # scheduler._decide_spec_method picks a METHOD_* (or "" for no-method) via
    # select_active_method; the caller caches the result per-request. The
    # scheduler supplies _loaded_spec_methods (bound module fn) so the decision
    # uses the real boot-time assembly, not a hardcoded dict.

    def _sched(self, *, ngram=False, dflash=False, dspark=False, mtp=False):
        from fusion_mlx.scheduler.sched_step import _loaded_spec_methods as _lsm

        sched = SimpleNamespace(
            _ngram_spec_state=object() if ngram else None,
            _spec_decode_state=None,
            _dflash_runtime=object() if dflash else None,
            _dspark_runtime=object() if dspark else None,
            model=SimpleNamespace(_fusion_mlx_mtp_decode_enabled=mtp),
            config=SimpleNamespace(model_name="test"),
        )
        sched._loaded_spec_methods = lambda: _lsm(sched)
        return sched

    def test_short_prompt_ngram_only_picks_ngram(self):
        from fusion_mlx.scheduler.sched_step import _decide_spec_method

        sched = self._sched(ngram=True)
        request = SimpleNamespace(num_prompt_tokens=100)
        assert _decide_spec_method(sched, request) == METHOD_NGRAM

    def test_long_prompt_routes_to_dflash_when_loaded(self):
        from fusion_mlx.scheduler.sched_step import _decide_spec_method

        sched = self._sched(ngram=True, dflash=True)
        # >= 4096 long_doc_threshold -> dflash wins fresh selection
        request = SimpleNamespace(num_prompt_tokens=5000)
        assert _decide_spec_method(sched, request) == METHOD_DFLASH

    def test_mtp_loaded_short_prompt_picks_mtp(self):
        from fusion_mlx.scheduler.sched_step import _decide_spec_method

        sched = self._sched(mtp=True)
        request = SimpleNamespace(num_prompt_tokens=100)
        assert _decide_spec_method(sched, request) == METHOD_MTP

    def test_nothing_loaded_returns_empty_string(self):
        from fusion_mlx.scheduler.sched_step import _decide_spec_method

        sched = self._sched()
        request = SimpleNamespace(num_prompt_tokens=100)
        assert _decide_spec_method(sched, request) == ""


class TestMaxTokensSignal:
    def test_select_active_method_passes_max_tokens_to_router(self, monkeypatch):
        from fusion_mlx.speculative import auto_router
        from fusion_mlx.speculative.per_request_route import (
            select_active_method as _sam,
        )

        seen = {}
        real_decide = auto_router.SpecAutoRouter.decide

        def spy(self, signals):
            seen["max_tokens"] = signals.max_tokens
            return real_decide(self, signals)

        monkeypatch.setattr(auto_router.SpecAutoRouter, "decide", spy)
        _sam(9000, loaded_methods(dflash2=True), max_tokens=512)
        assert seen["max_tokens"] == 512

    def test_dspine_state_pop_request_stats(self):
        from fusion_mlx.scheduler.spec_decode import DSpineSpecState
        from fusion_mlx.speculative.dspine.runtime import DSpineRuntime

        class _Drafter:
            def reset_cache(self):
                pass

        st = DSpineSpecState(
            DSpineRuntime(drafter=_Drafter(), draft_repo="x", block_size=16)
        )
        st.on_new_request("r1")
        st.record_result(3, 16)
        st.record_result(2, 16)
        assert st.pop_request_stats("r1") == {
            "spec_draft_proposed": 32,
            "spec_draft_accepted": 5,
        }
        # Counters reset after pop; totals stay cumulative.
        assert st.pop_request_stats("r1") == {}
        assert st.total_draft_proposed == 32 and st.total_draft_accepted == 5

    def test_pop_request_stats_wrong_request_empty(self):
        from fusion_mlx.scheduler.spec_decode import DSpineSpecState
        from fusion_mlx.speculative.dspine.runtime import DSpineRuntime

        class _Drafter:
            def reset_cache(self):
                pass

        st = DSpineSpecState(
            DSpineRuntime(drafter=_Drafter(), draft_repo="x", block_size=16)
        )
        st.on_new_request("r1")
        st.record_result(3, 16)
        assert st.pop_request_stats("r-other") == {}


class TestShortOutputGate:
    def test_short_output_routes_to_suffix_when_available(self):
        from fusion_mlx.speculative.per_request_route import (
            select_active_method as _sam,
        )

        assert _sam(100, loaded_methods(dspine=True, suffix=True), max_tokens=16) == METHOD_NGRAM

    def test_short_output_disables_spec_without_suffix(self):
        from fusion_mlx.speculative.per_request_route import (
            select_active_method as _sam,
        )

        assert _sam(100, loaded_methods(dspine=True), max_tokens=16) is None

    def test_no_max_tokens_never_gates(self):
        from fusion_mlx.speculative.per_request_route import (
            select_active_method as _sam,
        )

        assert _sam(100, loaded_methods(dspine=True), max_tokens=None) == METHOD_DSPINE

    def test_long_output_not_gated(self):
        from fusion_mlx.speculative.per_request_route import (
            select_active_method as _sam,
        )

        assert _sam(100, loaded_methods(dspine=True), max_tokens=200) == METHOD_DSPINE

    def test_short_output_recorded_with_reason(self, monkeypatch):
        from fusion_mlx.speculative import auto_router

        records = []
        monkeypatch.setattr(
            auto_router,
            "_append_spec_route_decision",
            lambda d: records.append(d),
        )
        router = auto_router.SpecAutoRouter()
        signals = auto_router.RouteSignals(
            prompt_token_count=100,
            max_tokens=8,
            available=frozenset({auto_router.METHOD_DSPINE, auto_router.METHOD_NGRAM}),
        )
        assert router.decide(signals) == auto_router.METHOD_NGRAM
        assert records[0]["reason"] == "short_output floor"
        assert records[0]["method"] == auto_router.METHOD_NGRAM


class TestEagleAndNgramRequestStats:
    def test_spec_decode_state_pop_request_stats(self):
        from fusion_mlx.scheduler.spec_decode import SpecDecodeState

        st = SpecDecodeState()
        st.on_new_request("r1", [])
        st.record_accepted(4, 8)
        st.record_accepted(2, 8)
        assert st.pop_request_stats("r1") == {
            "spec_draft_proposed": 16,
            "spec_draft_accepted": 6,
        }
        assert st.pop_request_stats("r1") == {}
        assert st.total_draft_proposed == 16 and st.total_draft_accepted == 6

    def test_spec_decode_state_pop_wrong_request_empty(self):
        from fusion_mlx.scheduler.spec_decode import SpecDecodeState

        st = SpecDecodeState()
        st.on_new_request("r1", [])
        st.record_accepted(1, 8)
        assert st.pop_request_stats("other") == {}

    def test_ngram_state_pop_request_stats(self):
        from fusion_mlx.scheduler.ngram_spec import NGramSpecState

        st = NGramSpecState()
        st.on_new_request("r1")
        st.record_result(2, 4)
        assert st.pop_request_stats("r1") == {
            "spec_draft_proposed": 4,
            "spec_draft_accepted": 2,
        }
        assert st.pop_request_stats("r1") == {}
