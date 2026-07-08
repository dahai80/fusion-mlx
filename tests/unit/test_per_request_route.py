# SPDX-License-Identifier: Apache-2.0
import logging
from types import SimpleNamespace

from fusion_mlx.speculative.auto_router import (
    METHOD_DFLASH,
    METHOD_DSPARK,
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
        # no suffix/dflash/mtp -> degenerate path returns dspark (in available)
        result = select_active_method(100, loaded)
        assert result == METHOD_DSPARK


class TestLoadedMethods:
    def test_default_all_false(self):
        assert loaded_methods() == {
            METHOD_NGRAM: False,
            METHOD_DFLASH: False,
            METHOD_DSPARK: False,
            METHOD_MTP: False,
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
            _dflash_runtime=None,
            _dspark_runtime=None,
            model=SimpleNamespace(),
        )
        assert _loaded_spec_methods(sched) == {
            METHOD_NGRAM: False,
            METHOD_DFLASH: False,
            METHOD_DSPARK: False,
            METHOD_MTP: False,
        }

    def test_assembles_from_scheduler_attrs(self):
        from fusion_mlx.scheduler.sched_step import _loaded_spec_methods

        sched = SimpleNamespace(
            _ngram_spec_state=object(),
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
            _dflash_runtime=object() if dflash else None,
            _dspark_runtime=object() if dspark else None,
            model=SimpleNamespace(_fusion_mlx_mtp_decode_enabled=mtp),
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
