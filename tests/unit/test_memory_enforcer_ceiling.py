# SPDX-License-Identifier: Apache-2.0
"""Unit tests for memory_enforcer.py ceiling calculation.

Covers:
- _STATIC_RESERVE_LARGE tier behavior (safe/balanced/aggressive/custom)
- small system (<24 GB) special handling
- _format_gb formatting
- _PREFILL_ABORT_MARGIN tier values
"""

from __future__ import annotations

from fusion_mlx.pool.memory_enforcer import (
    _ACTIVE_RECLAIM_RATIO,
    _EMERGENCY_OVER_CEILING_MARGIN_BYTES,
    _EMERGENCY_OVER_CEILING_POLLS,
    _HOT_CACHE_RESERVATION_SLACK_BYTES,
    _PREFILL_ABORT_MARGIN,
    _SMALL_SYSTEM_RESERVE,
    _SMALL_SYSTEM_THRESHOLD,
    _STATIC_RESERVE_LARGE,
    _format_gb,
)


class TestStaticReserveTiers:
    """Static reserve values per tier."""

    def test_safe_reserve_is_8gb(self):
        assert _STATIC_RESERVE_LARGE["safe"] == 8 * 1024**3

    def test_balanced_reserve_is_6gb(self):
        assert _STATIC_RESERVE_LARGE["balanced"] == 6 * 1024**3

    def test_aggressive_reserve_is_4gb(self):
        assert _STATIC_RESERVE_LARGE["aggressive"] == 4 * 1024**3

    def test_custom_has_entry(self):
        assert "custom" in _STATIC_RESERVE_LARGE

    def test_all_tiers_positive(self):
        for tier, reserve in _STATIC_RESERVE_LARGE.items():
            assert reserve > 0, f"Tier {tier} has non-positive reserve"


class TestSmallSystemThreshold:
    """Small system (<24 GB) handling."""

    def test_small_system_reserve_is_4gb(self):
        assert _SMALL_SYSTEM_RESERVE == 4 * 1024**3

    def test_small_system_threshold_is_24gb(self):
        assert _SMALL_SYSTEM_THRESHOLD == 24 * 1024**3


class TestActiveReclaimRatios:
    """Active page reclaim ratios per tier."""

    def test_safe_reclaim_is_20_percent(self):
        assert _ACTIVE_RECLAIM_RATIO["safe"] == 0.2

    def test_balanced_reclaim_is_50_percent(self):
        assert _ACTIVE_RECLAIM_RATIO["balanced"] == 0.5

    def test_aggressive_reclaim_is_80_percent(self):
        assert _ACTIVE_RECLAIM_RATIO["aggressive"] == 0.8

    def test_custom_not_in_reclaim(self):
        assert "custom" not in _ACTIVE_RECLAIM_RATIO


class TestPrefillAbortMargin:
    """Pre-chunk prediction guard margins."""

    def test_safe_and_balanced_margin_is_90_percent(self):
        assert _PREFILL_ABORT_MARGIN["safe"] == 0.90
        assert _PREFILL_ABORT_MARGIN["balanced"] == 0.90

    def test_aggressive_and_custom_margin_is_95_percent(self):
        assert _PREFILL_ABORT_MARGIN["aggressive"] == 0.95
        assert _PREFILL_ABORT_MARGIN["custom"] == 0.95

    def test_all_tiers_covered(self):
        for tier in _STATIC_RESERVE_LARGE:
            assert tier in _PREFILL_ABORT_MARGIN


class TestEmergencyConstants:
    """Emergency over-ceiling constants."""

    def test_over_ceiling_margin_is_2gb(self):
        assert _EMERGENCY_OVER_CEILING_MARGIN_BYTES == 2 * 1024**3

    def test_over_ceiling_polls_is_2(self):
        assert _EMERGENCY_OVER_CEILING_POLLS == 2

    def test_hot_cache_slack_is_512mb(self):
        assert _HOT_CACHE_RESERVATION_SLACK_BYTES == 512 * 1024**2


class TestFormatGb:
    """_format_gb helper."""

    def test_format_gb_rounding(self):
        result = _format_gb(8 * 1024**3)
        assert result == "8.0GB"

    def test_format_gb_fraction(self):
        result = _format_gb(6 * 1024**3 + 512 * 1024**2)
        # 6.5 GB
        assert "6.5" in result
        assert result.endswith("GB")

    def test_format_gb_zero(self):
        result = _format_gb(0)
        assert result == "0.0GB"

    def test_format_gb_large(self):
        result = _format_gb(256 * 1024**3)
        assert result == "256.0GB"
