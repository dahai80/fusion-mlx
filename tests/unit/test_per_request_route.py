# SPDX-License-Identifier: Apache-2.0
import logging

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
