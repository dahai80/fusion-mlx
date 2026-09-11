# SPDX-License-Identifier: Apache-2.0
# DFlash2 eligibility checks — gate the feature behind validated combos.
#
# Mirrors the DSpark/DFlash eligibility chokepoint. DFlash2 (official
# dflash pkg) is lossless at temperature=0 (greedy argmax verify). Key
# constraint: MLX quantized matmul efficiency drops at large verify
# widths, so block_size MUST be <= 5 on 4-bit targets (see
# architecture/fusion-mlx-dflash2.md §2.3). Unlike DSpark, 4-bit is
# SUPPORTED (DFlash2 targets Qwen3.8-27B-4bit) — the gate only warns,
# the block_size guard is enforced in the CLI/runtime. Gates:
#   - alias must declare supports_dflash2=True (explicit opt-in)
#   - alias must NOT be is_moe=True (MoE routing churn floors acceptance)
# No 4-bit rejection: Qwen3.8-27B-4bit is the primary target. The
# drafter repo is operator-supplied via --dflash2-drafter-path (same as
# DSpark), not a per-alias registry field.
#
# ENG-13 (#0909 audit): template methods delegated to BaseEligibilityChecker.

from __future__ import annotations

import logging
from dataclasses import dataclass

from fusion_mlx.model_aliases import AliasProfile
from fusion_mlx.quant_detect import looks_like_4bit as _looks_like_4bit

from ..base_eligibility import BaseEligibilityChecker

logger = logging.getLogger(__name__)


class DFlash2Unavailable(RuntimeError):  # noqa: N818 — domain-specific error name
    pass


@dataclass(frozen=True)
class EligibilityReport:
    alias: str | None
    supports_dflash2: bool
    is_moe: bool
    is_4bit: bool
    reasons: tuple[str, ...]


class _DFlash2Checker(BaseEligibilityChecker):
    _STRATEGY_LABEL = "DFlash2"

    def _unavailable_exc(self) -> type[RuntimeError]:
        return DFlash2Unavailable

    def _runtime_module(self) -> str:
        return "dflash"

    def _empty_eligible_suffix(self) -> str:
        return (
            "No aliases currently pass every DFlash2 gate. DFlash2 targets "
            "Qwen3.8-27B dense — pass a dense Qwen3.8 repo, e.g. "
            "`fusion-mlx serve --enable-dflash2 mlx-community/Qwen3.8-27B-4bit "
            "--dflash2-drafter-path z-lab/Qwen3.8-27B-DFlash2 --block-size 5`."
        )

    def report(
        self, profile: AliasProfile, alias: str | None = None
    ) -> EligibilityReport:
        reasons: list[str] = []
        if not profile.supports_dflash2:
            reasons.append(
                "alias is not DFlash2-enabled (set supports_dflash2=true in "
                "aliases.json after validating the speedup)"
            )
        if profile.is_moe:
            reasons.append(
                "alias is MoE (is_moe=true) — DFlash2 acceptance floors on "
                "expert-routing churn; use a dense target"
            )
        is_4bit = _looks_like_4bit(profile.hf_path)
        return EligibilityReport(
            alias=alias,
            supports_dflash2=profile.supports_dflash2,
            is_moe=profile.is_moe,
            is_4bit=is_4bit,
            reasons=tuple(reasons),
        )


_checker = _DFlash2Checker()
report = _checker.report


def check(profile, alias=None):
    return _checker.check(profile, alias=alias, _eligible_fn=eligible_aliases)


eligible_aliases = _checker.eligible_aliases
have_runtime = _checker.have_runtime
