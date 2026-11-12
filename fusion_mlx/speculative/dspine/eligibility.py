# SPDX-License-Identifier: Apache-2.0
"""DSpine eligibility gates (issue #1549).

DSpine is a full-attention Qwen3-family block-parallel drafter. Gates:
  - model_family must be qwen3 (dense full-attention; hybrid GDN targets
    use the DSpark/ReplaySSM line — Gate C/Gate M economics do not apply)
  - model must NOT be MoE
  - drafter checkpoint must be supplied (model path in spec config)

The bf16 lab measurement (M5 Max, Qwen3-8B) saw 3.6-8.3x; 4-bit targets
are allowed but must re-verify acceptance in serve mode before enabling
by default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_FAMILIES = ("qwen3",)


class DSpineUnavailable(RuntimeError):  # noqa: N818
    """Raised when a model fails DSpine eligibility gates."""


@dataclass(frozen=True)
class DSpineEligibilityReport:
    alias: str | None
    model_family: str
    is_moe: bool
    draft_path: str | None
    eligible: bool
    reasons: tuple[str, ...]


def check(
    model_family: str,
    is_moe: bool = False,
    draft_path: str | None = None,
    alias: str | None = None,
) -> DSpineEligibilityReport:
    reasons: list[str] = []
    family = (model_family or "").lower()
    if family not in SUPPORTED_FAMILIES:
        reasons.append(
            f"model_family {model_family!r} not in {SUPPORTED_FAMILIES} "
            "(hybrid GDN targets need the DSpark/ReplaySSM line)"
        )
    if is_moe:
        reasons.append("MoE routing churn kills acceptance; dense models only")
    if not draft_path:
        reasons.append("no drafter checkpoint configured (spec config 'model')")
    elif not Path(draft_path).exists():
        reasons.append(f"drafter path not found: {draft_path}")
    report = DSpineEligibilityReport(
        alias=alias,
        model_family=model_family,
        is_moe=is_moe,
        draft_path=draft_path,
        eligible=not reasons,
        reasons=tuple(reasons),
    )
    if reasons:
        logger.info("DSpine eligibility: %s -> %s", alias, reasons)
    return report
