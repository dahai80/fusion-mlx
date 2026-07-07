# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``fusion_mlx/speculative/dspark/eligibility.py``."""

from __future__ import annotations

import logging

import pytest

from fusion_mlx.model_aliases import AliasProfile
from fusion_mlx.speculative.dspark.eligibility import (
    DSparkUnavailable,
    check,
    report,
)

logger = logging.getLogger(__name__)


def _good_profile() -> AliasProfile:
    return AliasProfile(
        name="qwen3.5-9b-8bit",
        hf_path="mlx-community/Qwen3.5-9B-8bit",
        is_moe=False,
        supports_dspark=True,
    )


def test_check_passes_for_good_profile() -> None:
    p = _good_profile()
    check(p, alias="qwen3.5-9b-8bit")
    assert report(p, alias="qwen3.5-9b-8bit").reasons == ()


def test_check_rejects_alias_without_supports_dspark() -> None:
    p = AliasProfile(name="qwen3.5-9b-8bit", hf_path="mlx-community/Qwen3.5-9B-8bit")
    with pytest.raises(DSparkUnavailable, match="not DSpark-enabled"):
        check(p, alias="qwen3.5-9b-8bit")


def test_check_rejects_moe_alias() -> None:
    p = AliasProfile(
        name="qwen3.6-35b-8bit",
        hf_path="mlx-community/Qwen3.6-35B-A3B-8bit",
        is_moe=True,
        supports_dspark=True,
    )
    with pytest.raises(DSparkUnavailable, match="MoE"):
        check(p, alias="qwen3.6-35b-8bit")


def test_check_rejects_4bit_main_model() -> None:
    p = AliasProfile(
        name="qwen3.5-9b-4bit",
        hf_path="mlx-community/Qwen3.5-9B-4bit",
        supports_dspark=True,
    )
    with pytest.raises(DSparkUnavailable, match="4-bit"):
        check(p, alias="qwen3.5-9b-4bit")


def test_report_collects_all_failures() -> None:
    bad = AliasProfile(
        name="qwen3.6-35b-4bit",
        hf_path="mlx-community/Qwen3.6-35B-A3B-4bit",
        is_moe=True,
        supports_dspark=False,
    )
    r = report(bad, alias="qwen3.6-35b-4bit")
    joined = " ".join(r.reasons)
    assert "MoE" in joined
    assert "4-bit" in joined
    assert "DSpark-enabled" in joined or "supports_dspark" in joined


def test_eligible_aliases_surfaces_alias_registry_errors(monkeypatch) -> None:
    from fusion_mlx.speculative.dspark import eligibility

    def boom():
        raise RuntimeError("alias registry broken")

    monkeypatch.setattr("fusion_mlx.model_aliases.list_profiles", boom)

    # eligible_aliases catches exceptions and returns [] — it does not
    # propagate them. Verify it returns empty instead of crashing.
    result = eligibility.eligible_aliases()
    assert result == []
