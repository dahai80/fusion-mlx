# SPDX-License-Identifier: Apache-2.0
"""DFly eligibility checks — gate the feature behind validated combos.

DFly is the Hunyuan (Hy3) native block-parallel drafter (DFlash +
hidden-correction).  Gates:
  - model_family must be "hunyuan" (AliasProfile.model_family or
    auto-detected from config/hf_path)
  - alias must NOT be is_moe=True (MoE routing churn kills acceptance)
  - main model must be 8-bit+ (4-bit regresses)
  - drafter HF path must be reachable or operator-supplied

ENG-13 (#0909 audit): template methods delegated to BaseEligibilityChecker.
Unlike the supports_* strategies, DFly replaces the opt-in gate with a
model_family=="hunyuan" gate (overridden in ``report`` below).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fusion_mlx.model_aliases import AliasProfile
from fusion_mlx.quant_detect import looks_like_4bit as _looks_like_4bit

from ..base_eligibility import BaseEligibilityChecker

logger = logging.getLogger(__name__)


class DFlyUnavailable(RuntimeError):  # noqa: N818
    """Raised when a model fails DFly eligibility gates."""


@dataclass(frozen=True)
class EligibilityReport:
    alias: str | None
    model_family: str
    is_moe: bool
    is_4bit: bool
    has_drafter: bool
    reasons: tuple[str, ...]


def _detect_model_family(profile: AliasProfile) -> str:
    family = getattr(profile, "model_family", "")
    if family:
        return family
    hf_lower = profile.hf_path.lower()
    if "hunyuan" in hf_lower or "hy3" in hf_lower:
        return "hunyuan"
    return ""


class _DFlyChecker(BaseEligibilityChecker):
    _STRATEGY_LABEL = "DFly"

    def _unavailable_exc(self) -> type[RuntimeError]:
        return DFlyUnavailable

    def _runtime_module(self) -> str:
        return "fusion_mlx.speculative.dfly.drafter"

    def _empty_eligible_suffix(self) -> str:
        return (
            "No aliases currently pass every DFly gate. DFly targets "
            "Hunyuan (Hy3) bf16/8-bit models — pass an Hy3 repo, e.g. "
            "`fusion-mlx serve --enable-dfly <hy3-model> "
            "--dfly-drafter-path AngelSlim/Hy3-DFly-Block8`."
        )

    def report(
        self, profile: AliasProfile, alias: str | None = None
    ) -> EligibilityReport:
        reasons: list[str] = []
        family = _detect_model_family(profile)
        if family != "hunyuan":
            reasons.append(
                "DFly is a Hunyuan (Hy3)-native drafter; model_family={!r} "
                "is not 'hunyuan'. Use dfly only with Hy3 models.".format(
                    family or "unknown"
                )
            )
        if profile.is_moe:
            reasons.append(
                "alias is MoE (is_moe=true) — DFly acceptance floors on "
                "expert-routing churn; use a dense Hy3 target"
            )
        is_4bit = _looks_like_4bit(profile.hf_path)
        if is_4bit:
            reasons.append(
                f"main model hf_path={profile.hf_path!r} is 4-bit quantized; "
                "DFly regresses on 4-bit (use a bf16/8-bit+ Hy3 variant)"
            )
        has_drafter = bool(
            getattr(profile, "dfly_draft_model", None)
            or getattr(profile, "drafter_hf_path", None)
        )
        return EligibilityReport(
            alias=alias,
            model_family=family,
            is_moe=profile.is_moe,
            is_4bit=is_4bit,
            has_drafter=has_drafter,
            reasons=tuple(reasons),
        )


_checker = _DFlyChecker()
report = _checker.report


def check(profile, alias=None):
    return _checker.check(profile, alias=alias, _eligible_fn=eligible_aliases)


eligible_aliases = _checker.eligible_aliases
have_runtime = _checker.have_runtime
