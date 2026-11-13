# SPDX-License-Identifier: Apache-2.0
"""Tests for bucketed spec-decode acceptance telemetry (#1610)."""

import pytest

from fusion_mlx.speculative.accept_buckets import (
    DEFAULT_BUCKET_EDGES,
    DEFAULT_WINDOW,
    SpecAcceptBuckets,
    bucket_index,
    bucket_label,
)
from fusion_mlx.speculative.auto_router import (
    METHOD_MTP,
    METHOD_NGRAM,
    RouteSignals,
    SpecAutoRouter,
)

ALL = frozenset({METHOD_NGRAM, METHOD_MTP})


@pytest.fixture(autouse=True)
def _no_route_log(monkeypatch):
    monkeypatch.setenv("FUSION_MLX_SPEC_ROUTE_LOG", "0")


class TestBucketEdges:
    def test_default_edges_assignment(self):
        edges = DEFAULT_BUCKET_EDGES
        assert bucket_index(0, edges) == 0
        assert bucket_index(511, edges) == 0
        assert bucket_index(512, edges) == 1
        assert bucket_index(2047, edges) == 1
        assert bucket_index(2048, edges) == 2
        assert bucket_index(8191, edges) == 2
        assert bucket_index(8192, edges) == 3
        assert bucket_index(100000, edges) == 3

    def test_custom_edges(self):
        assert bucket_index(99, (100,)) == 0
        assert bucket_index(100, (100,)) == 1

    def test_labels(self):
        assert bucket_label(0) == "<512"
        assert bucket_label(3) == "8k+"
        assert bucket_label(9) == "bucket9"

    def test_record_routes_into_bucket(self):
        b = SpecAcceptBuckets()
        b.record("mtp", 300, 10, 8)
        b.record("mtp", 900, 10, 2)
        assert b.rate("mtp", 0) == 0.8
        assert b.rate("mtp", 1) == 0.2
        assert b.rate("mtp", 2) is None


class TestRollingWindow:
    def test_window_bounds_memory(self):
        b = SpecAcceptBuckets(window=4)
        for i in range(10):
            b.record("suffix", 100, 10, 10 if i >= 6 else 0)
        cell = b._cells[("suffix", 0)]
        assert len(cell) == 4
        # Only the last 4 samples (i=6..9, all accepted) count.
        assert b.rate("suffix", 0) == 1.0

    def test_window_rejects_invalid(self):
        with pytest.raises(ValueError):
            SpecAcceptBuckets(window=0)
        with pytest.raises(ValueError):
            SpecAcceptBuckets(edges=(512, 512))
        with pytest.raises(ValueError):
            SpecAcceptBuckets(edges=(0,))

    def test_record_drops_invalid_input(self):
        b = SpecAcceptBuckets()
        b.record(None, 100, 10, 5)
        b.record("suffix", None, 10, 5)
        b.record("suffix", 100, 0, 0)
        b.record("suffix", -1, 10, 5)
        assert b.export()["buckets"] == {}


class TestRateAggregation:
    def test_rate_is_token_weighted(self):
        b = SpecAcceptBuckets()
        b.record("suffix", 100, 10, 0)
        b.record("suffix", 200, 30, 30)
        # sum/sum = 30/40, not mean of per-request rates.
        assert b.rate("suffix", 0) == pytest.approx(0.75)

    def test_route_bucket_prefers_expected_then_last(self):
        b = SpecAcceptBuckets()
        assert b.route_bucket(None) is None
        b.record("mtp", 9000, 10, 5)
        assert b.route_bucket(None) == 3
        assert b.route_bucket(100) == 0

    def test_route_rate_empty_cell(self):
        b = SpecAcceptBuckets()
        rate, idx = b.route_rate("mtp", 300)
        assert rate is None
        assert idx == 0
        rate, idx = b.route_rate(None, None)
        assert rate is None and idx is None


class TestExportShape:
    def test_export_compact_shape(self):
        b = SpecAcceptBuckets(edges=(512, 2048, 8192), window=8)
        b.record("mtp", 300, 10, 8)
        b.record("mtp", 600, 10, 4)
        b.record("suffix", 9000, 20, 10)
        out = b.export()
        assert out["edges"] == [512, 2048, 8192]
        assert out["window"] == 8
        assert out["last_bucket"] == "8k+"
        assert out["buckets"]["mtp"]["<512"] == {
            "accepted": 8,
            "proposed": 10,
            "samples": 1,
            "rate": 0.8,
        }
        assert out["buckets"]["mtp"]["512-2k"]["rate"] == 0.4
        assert out["buckets"]["suffix"]["8k+"]["rate"] == 0.5
        # Untouched buckets are absent, not zero-filled.
        assert "2k-8k" not in out["buckets"]["mtp"]

    def test_export_empty(self):
        out = SpecAcceptBuckets().export()
        assert out == {
            "edges": [512, 2048, 8192],
            "window": DEFAULT_WINDOW,
            "last_bucket": None,
            "buckets": {},
        }


def _sig(rate=None, current=None, bucket_rate=None, bucket=None, max_tokens=None):
    return RouteSignals(
        prompt_token_count=128,
        max_tokens=max_tokens,
        recent_accept_rate=rate,
        current_method=current,
        available=ALL,
        bucket_accept_rate=bucket_rate,
        output_bucket=bucket,
    )


class TestRouterBucketFeature:
    def test_bucket_rate_overrides_scalar_for_abandon(self):
        # Scalar says keep (0.9) but the 8k+ bucket says the method fails
        # there -> abandon and re-select from the table.
        r = SpecAutoRouter()
        decided = r.decide(
            _sig(
                rate=0.9,
                current=METHOD_MTP,
                bucket_rate=0.05,
                bucket=3,
                max_tokens=9000,
            )
        )
        assert decided == METHOD_NGRAM
        assert METHOD_MTP in r._abandon_cooldowns

    def test_bucket_rate_keeps_when_scalar_low(self):
        # Scalar 0.10 would abandon, but the bucket-local rate is healthy.
        r = SpecAutoRouter()
        assert (
            r.decide(_sig(rate=0.10, current=METHOD_MTP, bucket_rate=0.6, bucket=0))
            == METHOD_MTP
        )

    def test_empty_bucket_falls_back_to_scalar(self):
        r = SpecAutoRouter()
        # bucket_rate None -> scalar 0.10 abandons mtp -> ngram fallback.
        assert (
            r.decide(_sig(rate=0.10, current=METHOD_MTP, bucket_rate=None, bucket=2))
            == METHOD_NGRAM
        )
        # scalar 0.90 keeps mtp unchanged.
        assert (
            r.decide(_sig(rate=0.90, current=METHOD_MTP, bucket_rate=None, bucket=2))
            == METHOD_MTP
        )

    def test_bucket_rate_ignored_without_current_method(self):
        # Bucket stats are only meaningful for the method that produced them;
        # with no current method the router goes straight to table selection.
        r = SpecAutoRouter()
        assert r.decide(_sig(rate=None, current=None, bucket_rate=0.05, bucket=3)) == (
            METHOD_NGRAM
        )

    def test_backward_compatible_default_signals(self):
        # Existing callers constructing RouteSignals without the new fields
        # keep the exact pre-#1610 behavior.
        r = SpecAutoRouter()
        assert r.decide(_sig(rate=0.90, current=METHOD_MTP)) == METHOD_MTP
        assert r.decide(_sig(rate=0.10, current=METHOD_MTP)) == METHOD_NGRAM

    def test_record_then_route_end_to_end(self):
        b = SpecAcceptBuckets()
        b.record("mtp", 9000, 100, 2)
        rate, idx = b.route_rate("mtp", 9000)
        assert rate == pytest.approx(0.02)
        assert idx == 3
        # Router abandons mtp for this bucket (rate < 0.20 abandon threshold).
        assert (
            SpecAutoRouter().decide(
                _sig(rate=0.9, current=METHOD_MTP, bucket_rate=rate, bucket=idx)
            )
            == METHOD_NGRAM
        )
