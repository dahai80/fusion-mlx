# SPDX-License-Identifier: Apache-2.0
# DSpark eligibility checks — gate the feature behind validated combos.
#
# Mirrors the DFlash eligibility chokepoint but for DeepSeek's DeepSpec
# block drafter (dspark-metal). DSpark is lossless on Qwen3 4B/8B/14B
# bf16 targets; it REGRESSES on 4-bit (0.84x measured) and on MoE. Gates:
#   - alias must declare supports_dspark=True (explicit opt-in)
#   - alias must NOT be is_moe=True (MoE routing churn kills acceptance)
#   - main model must be 8-bit+ (detected from HF path naming)
# No drafter gate here: the DSpark draft is a LOCAL converted MLX
# artifact (dspark-metal-convert output), not an HF repo, so it is
# operator-supplied via --dspark-drafter-path and validated at server
# boot. This avoids the DFlash path's per-alias drafter-field coupling.
#
# ENG-13 (#0909 audit): template methods delegated to BaseEligibilityChecker.

from __future__ import annotations

import logging
from dataclasses import dataclass

from fusion_mlx.model_aliases import AliasProfile
from fusion_mlx.quant_detect import looks_like_4bit as _looks_like_4bit

from ..base_eligibility import BaseEligibilityChecker

logger = logging.getLogger(__name__)


class DSparkUnavailable(RuntimeError):  # noqa: N818 — domain-specific error name
    pass


@dataclass(frozen=True)
class EligibilityReport:
    alias: str | None
    supports_dspark: bool
    is_moe: bool
    is_4bit: bool
    reasons: tuple[str, ...]


class _DSparkChecker(BaseEligibilityChecker):
    _STRATEGY_LABEL = "DSpark"

    def _unavailable_exc(self) -> type[RuntimeError]:
        return DSparkUnavailable

    def _runtime_module(self) -> str:
        return "fusion_mlx.speculative.dspark.engine"

    def _empty_eligible_suffix(self) -> str:
        return (
            "No aliases currently pass every DSpark gate. DSpark targets "
            "Qwen3 4B/8B/14B bf16 — pass a bf16 Qwen3 repo directly, e.g. "
            "`fusion-mlx serve --enable-dspark mlx-community/Qwen3-8B-bf16 "
            "--dspark-drafter-path <converted-mlx-draft>`."
        )

    def report(
        self, profile: AliasProfile, alias: str | None = None
    ) -> EligibilityReport:
        reasons: list[str] = []
        if not profile.supports_dspark:
            reasons.append(
                "alias is not DSpark-enabled (set supports_dspark=true in "
                "aliases.json after benching to validate the speedup"
            )
        if profile.is_moe:
            reasons.append(
                "alias is MoE (is_moe=true) — DSpark acceptance floors on "
                "expert-routing churn; use a dense target"
            )
        is_4bit = _looks_like_4bit(profile.hf_path)
        if is_4bit:
            reasons.append(
                f"main model hf_path={profile.hf_path!r} is 4-bit quantized; "
                "DSpark regresses on 4-bit (use a bf16/8-bit+ Qwen3 variant)"
            )
        return EligibilityReport(
            alias=alias,
            supports_dspark=profile.supports_dspark,
            is_moe=profile.is_moe,
            is_4bit=is_4bit,
            reasons=tuple(reasons),
        )


_checker = _DSparkChecker()
report = _checker.report


def check(profile, alias=None):
    return _checker.check(profile, alias=alias, _eligible_fn=eligible_aliases)


eligible_aliases = _checker.eligible_aliases
have_runtime = _checker.have_runtime
