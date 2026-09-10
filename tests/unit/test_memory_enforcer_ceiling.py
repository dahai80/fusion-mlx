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
    _LARGE_SYSTEM_CEILING_FRACTION,
    _LARGE_SYSTEM_THRESHOLD,
    _MLX_CACHE_LIMIT_BYTES,
    _PHYSICAL_RAM_WIRED_CAP_FRACTION,
    _PREFILL_ABORT_MARGIN,
    _SMALL_SYSTEM_RESERVE,
    _SMALL_SYSTEM_THRESHOLD,
    _STATIC_RESERVE_LARGE,
    _WIRED_LIMIT_BUFFER_BYTES,
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


class TestLargeSystemCeilingFraction:
    """Ceiling fraction cap for large-memory systems (>= 64GB).

    On a 128GB machine, balanced without the cap gives 122GB ceiling —
    too close to jetsam. G1 (#0909 audit) lowered the fraction to 0.5625
    = 72GB ceiling, leaving 56GB for OS/prefill/compile cache.
    """

    def test_threshold_is_64gb(self):
        assert _LARGE_SYSTEM_THRESHOLD == 64 * 1024**3

    def test_balanced_fraction_is_5625(self):
        assert _LARGE_SYSTEM_CEILING_FRACTION["balanced"] == 0.5625

    def test_safe_fraction_is_50_percent(self):
        assert _LARGE_SYSTEM_CEILING_FRACTION["safe"] == 0.50

    def test_aggressive_fraction_is_75_percent(self):
        assert _LARGE_SYSTEM_CEILING_FRACTION["aggressive"] == 0.75

    def test_custom_fraction_is_80_percent(self):
        assert _LARGE_SYSTEM_CEILING_FRACTION["custom"] == 0.80

    def test_all_tiers_covered(self):
        for tier in _STATIC_RESERVE_LARGE:
            assert tier in _LARGE_SYSTEM_CEILING_FRACTION

    def test_balanced_128gb_ceiling_capped(self):
        """128GB balanced: reserve=6GB → base=122GB, but capped at 56.25%=72GB (G1)."""
        from unittest.mock import patch

        from fusion_mlx.pool.memory_enforcer import ProcessMemoryEnforcer

        enforcer = ProcessMemoryEnforcer.__new__(ProcessMemoryEnforcer)
        enforcer._memory_guard_tier = "balanced"
        enforcer._prefill_memory_guard = True
        with patch(
            "fusion_mlx.pool.settings.get_system_memory",
            return_value=128 * 1024**3,
        ):
            ceiling = enforcer._get_static_ceiling()
        expected = int(128 * 1024**3 * 0.5625)
        assert (
            ceiling == expected
        ), f"128GB balanced ceiling should be capped at {expected}, got {ceiling}"

    def test_balanced_32gb_ceiling_not_capped(self):
        """32GB balanced: below 64GB threshold, no fraction cap applied."""
        from unittest.mock import patch

        from fusion_mlx.pool.memory_enforcer import ProcessMemoryEnforcer

        enforcer = ProcessMemoryEnforcer.__new__(ProcessMemoryEnforcer)
        enforcer._memory_guard_tier = "balanced"
        enforcer._prefill_memory_guard = True
        with patch(
            "fusion_mlx.pool.settings.get_system_memory",
            return_value=32 * 1024**3,
        ):
            ceiling = enforcer._get_static_ceiling()
        expected = 32 * 1024**3 - 6 * 1024**3
        assert (
            ceiling == expected
        ), f"32GB balanced ceiling should be {expected} (no cap), got {ceiling}"


class TestWiredLimitTarget:
    """Metal wired limit target = ceiling + buffer, capped at 80% RAM."""

    def test_buffer_is_25gb(self):
        assert _WIRED_LIMIT_BUFFER_BYTES == 25 * 1024**3

    def test_physical_cap_fraction_is_80_percent(self):
        assert _PHYSICAL_RAM_WIRED_CAP_FRACTION == 0.80

    def test_mlx_cache_limit_is_1gb(self):
        assert _MLX_CACHE_LIMIT_BYTES == 1 * 1024**3

    def test_128gb_balanced_wired_target(self):
        """128GB balanced: ceiling=72GB (G1), target=min(72+25, 102.4)=97GB."""
        from unittest.mock import patch

        from fusion_mlx.pool.memory_enforcer import ProcessMemoryEnforcer

        enforcer = ProcessMemoryEnforcer.__new__(ProcessMemoryEnforcer)
        with patch(
            "fusion_mlx.pool.settings.get_system_memory",
            return_value=128 * 1024**3,
        ):
            target = enforcer._get_wired_limit_target(int(128 * 1024**3 * 0.5625))
        physical_cap = int(128 * 1024**3 * 0.80)
        expected = min(int(128 * 1024**3 * 0.5625) + 25 * 1024**3, physical_cap)
        assert target == expected
        assert target < physical_cap, "Wired target must be below 80% RAM"
        assert target > int(
            128 * 1024**3 * 0.5625
        ), "Wired target must be above enforcer ceiling (buffer zone)"
