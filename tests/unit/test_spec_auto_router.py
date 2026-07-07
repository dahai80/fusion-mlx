# SPDX-License-Identifier: Apache-2.0
"""Tests for the speculative-decoding auto-router (FR-005 --spec-route auto)."""

from fusion_mlx.speculative.auto_router import (
    DEFAULT_AVAILABLE,
    METHOD_DFLASH,
    METHOD_MTP,
    METHOD_NGRAM,
    RouteSignals,
    SpecAutoRouter,
    auto_route,
    available_methods,
)

ALL = frozenset({METHOD_NGRAM, METHOD_DFLASH, METHOD_MTP, "dspark"})


def sig(
    prompt=128,
    has_mtp=False,
    rate=None,
    current=None,
    available=ALL,
):
    return RouteSignals(
        prompt_token_count=prompt,
        has_mtp=has_mtp,
        recent_accept_rate=rate,
        current_method=current,
        available=available,
    )


class TestFreshSelection:
    def test_long_doc_routes_to_dflash(self):
        assert SpecAutoRouter().decide(sig(prompt=8192)) == METHOD_DFLASH

    def test_short_prompt_with_mtp_routes_to_mtp(self):
        assert SpecAutoRouter().decide(sig(prompt=128, has_mtp=True)) == METHOD_MTP

    def test_short_prompt_no_mtp_routes_to_ngram(self):
        assert SpecAutoRouter().decide(sig(prompt=128)) == METHOD_NGRAM

    def test_mtp_beats_long_doc_only_when_available(self):
        # Long doc, no MTP -> dflash; with MTP -> still dflash (long-doc wins).
        r = SpecAutoRouter()
        assert r.decide(sig(prompt=8192, has_mtp=False)) == METHOD_DFLASH
        assert r.decide(sig(prompt=8192, has_mtp=True)) == METHOD_DFLASH

    def test_dflash_not_available_long_doc_falls_through(self):
        avail = frozenset({METHOD_NGRAM, METHOD_MTP})
        assert (
            SpecAutoRouter().decide(sig(prompt=8192, has_mtp=True, available=avail))
            == METHOD_MTP
        )
        assert (
            SpecAutoRouter().decide(sig(prompt=8192, has_mtp=False, available=avail))
            == METHOD_NGRAM
        )


class TestHysteresis:
    def test_keeps_working_current_method(self):
        # dflash on a short prompt would normally lose to ngram/mtp, but
        # hysteresis keeps it because acceptance is healthy.
        r = SpecAutoRouter()
        out = r.decide(sig(prompt=128, has_mtp=True, current=METHOD_DFLASH, rate=0.55))
        assert out == METHOD_DFLASH

    def test_hysteresis_respects_keep_threshold_boundary(self):
        r = SpecAutoRouter()  # keep_accept = 0.40
        assert (
            r.decide(sig(prompt=128, current=METHOD_DFLASH, rate=0.40)) == METHOD_DFLASH
        )
        # Just below keep but above abandon -> no hysteresis, fresh selection.
        assert (
            r.decide(sig(prompt=128, current=METHOD_DFLASH, rate=0.39)) == METHOD_NGRAM
        )


class TestAbandon:
    def test_abandons_failing_method_to_ngram(self):
        r = SpecAutoRouter()  # abandon_accept = 0.20
        out = r.decide(sig(prompt=128, current=METHOD_DFLASH, rate=0.10))
        assert out == METHOD_NGRAM

    def test_abandon_excludes_method_from_fresh_selection(self):
        # Long doc would re-pick dflash, but dflash was just abandoned.
        r = SpecAutoRouter()
        out = r.decide(sig(prompt=8192, current=METHOD_DFLASH, rate=0.05))
        assert out != METHOD_DFLASH
        # No MTP -> falls to ngram.
        assert out == METHOD_NGRAM

    def test_abandon_then_mtp_if_available(self):
        r = SpecAutoRouter()
        out = r.decide(sig(prompt=8192, has_mtp=True, current=METHOD_DFLASH, rate=0.05))
        assert out == METHOD_MTP

    def test_abandon_boundary(self):
        r = SpecAutoRouter()  # abandon = 0.20, keep = 0.40
        # On a long doc, the abandon boundary decides whether dflash is
        # re-selectable. At exactly 0.20 (strict <) it is NOT abandoned, so
        # fresh selection re-picks dflash. Just below (0.19) it IS excluded.
        s_kept = sig(prompt=8192, current=METHOD_DFLASH, rate=0.20)
        assert r.decide(s_kept) == METHOD_DFLASH
        s_abandoned = sig(prompt=8192, current=METHOD_DFLASH, rate=0.19)
        assert r.decide(s_abandoned) != METHOD_DFLASH


class TestDegenerate:
    def test_empty_available_returns_ngram_sentinel(self):
        assert SpecAutoRouter().decide(sig(available=frozenset())) == METHOD_NGRAM

    def test_only_unknown_method_available(self):
        out = SpecAutoRouter().decide(sig(available=frozenset({"dspark"})))
        assert out == "dspark"

    def test_default_available_constant_matches_registry_canonicals(self):
        assert METHOD_DFLASH == "ddtree"
        assert METHOD_NGRAM == "suffix"
        assert "ddtree" in DEFAULT_AVAILABLE
        assert "suffix" in DEFAULT_AVAILABLE


class TestDeterminism:
    def test_same_inputs_same_output(self):
        s = sig(prompt=4096, has_mtp=True)
        r = SpecAutoRouter()
        assert r.decide(s) == r.decide(s) == r.decide(s)

    def test_auto_route_uses_default_router(self):
        assert auto_route(sig(prompt=128)) == METHOD_NGRAM


class TestThresholdTuning:
    def test_custom_long_doc_threshold(self):
        r = SpecAutoRouter(long_doc_threshold=100)
        # 200-token prompt now counts as "long".
        assert r.decide(sig(prompt=200)) == METHOD_DFLASH

    def test_custom_abandon_keep(self):
        r = SpecAutoRouter(abandon_accept=0.5, keep_accept=0.8)
        # 0.6 acceptance: below keep(0.8) so no hysteresis, above abandon(0.5)
        # so not excluded -> fresh selection on short prompt -> ngram.
        assert (
            r.decide(sig(prompt=128, current=METHOD_DFLASH, rate=0.6)) == METHOD_NGRAM
        )


class TestRegistryIntegration:
    def test_available_methods_yields_registry_canonicals(self):
        methods = available_methods()
        # The four canonical methods the router reasons about are all
        # registered and config-enabled by default.
        assert {METHOD_NGRAM, METHOD_DFLASH, METHOD_MTP, "dspark"} <= methods
        # Aliases are NOT canonical methods.
        assert "dflash" not in methods
        assert "ngram" not in methods

    def test_router_decision_stays_within_available_methods(self):
        methods = available_methods()
        r = SpecAutoRouter()
        for prompt in (8, 128, 4096, 32768):
            out = r.decide(
                RouteSignals(
                    prompt_token_count=prompt,
                    has_mtp=True,
                    available=methods,
                )
            )
            assert out in methods
