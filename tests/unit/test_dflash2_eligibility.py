# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``fusion_mlx/speculative/dflash2/eligibility.py``."""

from __future__ import annotations

import logging

import pytest

from fusion_mlx.model_aliases import AliasProfile
from fusion_mlx.speculative.dflash2.eligibility import (
    DFlash2Unavailable,
    check,
    report,
)

logger = logging.getLogger(__name__)


def _good_profile() -> AliasProfile:
    return AliasProfile(
        name="qwen3.8-27b-4bit",
        hf_path="mlx-community/Qwen3.8-27B-4bit",
        is_moe=False,
        supports_dflash2=True,
    )


def test_check_passes_for_good_profile() -> None:
    p = _good_profile()
    check(p, alias="qwen3.8-27b-4bit")
    assert report(p, alias="qwen3.8-27b-4bit").reasons == ()


def test_check_passes_for_4bit_target() -> None:
    p = AliasProfile(
        name="qwen3.8-27b-4bit",
        hf_path="mlx-community/Qwen3.8-27B-4bit",
        supports_dflash2=True,
    )
    check(p, alias="qwen3.8-27b-4bit")
    r = report(p, alias="qwen3.8-27b-4bit")
    assert r.is_4bit is True
    assert r.reasons == ()


def test_check_rejects_alias_without_supports_dflash2() -> None:
    p = AliasProfile(name="qwen3.8-27b-4bit", hf_path="mlx-community/Qwen3.8-27B-4bit")
    with pytest.raises(DFlash2Unavailable, match="not DFlash2-enabled"):
        check(p, alias="qwen3.8-27b-4bit")


def test_check_rejects_moe_alias() -> None:
    p = AliasProfile(
        name="qwen3.6-35b-8bit",
        hf_path="mlx-community/Qwen3.6-35B-A3B-8bit",
        is_moe=True,
        supports_dflash2=True,
    )
    with pytest.raises(DFlash2Unavailable, match="MoE"):
        check(p, alias="qwen3.6-35b-8bit")


def test_report_collects_all_failures() -> None:
    bad = AliasProfile(
        name="qwen3.6-35b-4bit",
        hf_path="mlx-community/Qwen3.6-35B-A3B-4bit",
        is_moe=True,
        supports_dflash2=False,
    )
    r = report(bad, alias="qwen3.6-35b-4bit")
    joined = " ".join(r.reasons)
    assert "MoE" in joined
    assert "DFlash2-enabled" in joined


def test_eligible_aliases_surfaces_alias_registry_errors(monkeypatch) -> None:
    from fusion_mlx.speculative.dflash2 import eligibility

    def boom():
        raise RuntimeError("alias registry broken")

    monkeypatch.setattr("fusion_mlx.model_aliases.list_profiles", boom)
    result = eligibility.eligible_aliases()
    assert result == []


def test_have_runtime_probe_does_not_raise() -> None:
    from fusion_mlx.speculative.dflash2.eligibility import have_runtime

    assert isinstance(have_runtime(), bool)
