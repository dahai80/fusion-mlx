# SPDX-License-Identifier: Apache-2.0
"""DFlash eligibility checks — gate the feature behind validated combos.

This is the single chokepoint between user intent (``--enable-dflash`` on
the CLI) and the runtime hook. Failures here surface as actionable error
messages at server-start, never as silent regressions at request time.

Gates derived from PoC bench data (see issue #264):
- Alias must declare ``supports_dflash=True`` (explicit opt-in)
- Alias must NOT be ``is_moe=True`` (MoE acceptance floors at ~1.5)
- Main model must be 8-bit or higher; detected from the HF path
    naming convention (``-4bit``/``mxfp4``/``nvfp4`` suffixes used by
    mlx-community). A custom-named 4-bit repo would slip through this
    heuristic — for v1 we accept that risk since every supported alias
    is curated; load-time quant-config inspection is a phase-2 item.
- Drafter HF path must be reachable (no auth-gated repo without token)

ENG-13 (#0909 audit): the identical ``eligible_aliases`` / ``check`` /
``have_runtime`` template lives in ``speculative.base_eligibility``;
only ``report`` (the per-strategy gate logic) and three small hooks
(label, exception, runtime module) stay here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fusion_mlx.model_aliases import AliasProfile
from fusion_mlx.quant_detect import looks_like_4bit as _looks_like_4bit

from ..base_eligibility import BaseEligibilityChecker

logger = logging.getLogger(__name__)


class DFlashUnavailable(RuntimeError):  # noqa: N818 — domain-specific error name
    """Raised when an alias fails a DFlash eligibility gate."""


@dataclass(frozen=True)
class EligibilityReport:
    """Structured eligibility result."""

    alias: str | None
    supports_dflash: bool
    is_moe: bool
    is_4bit: bool
    has_drafter: bool
    reasons: tuple[str, ...]


class _DFlashChecker(BaseEligibilityChecker):
    _STRATEGY_LABEL = "DFlash"

    def _unavailable_exc(self) -> type[RuntimeError]:
        return DFlashUnavailable

    def _runtime_module(self) -> str:
        return "mlx_vlm.speculative.drafters"

    def report(
        self, profile: AliasProfile, alias: str | None = None
    ) -> EligibilityReport:
        reasons: list[str] = []
        if not profile.supports_dflash:
            reasons.append(
                "alias is not DFlash-enabled (set supports_dflash=true in "
                "model-config.json after benching to validate ≥1.3× speedup)"
            )
        if profile.is_moe:
            reasons.append(
                "alias is MoE (is_moe=true) — DFlash acceptance floors at "
                "~1.5 tokens/round on expert-routing churn; regression "
                "measured on Qwen3.6-35B-A3B"
            )
        is_4bit = _looks_like_4bit(profile.hf_path)
        if is_4bit:
            reasons.append(
                f"main model hf_path={profile.hf_path!r} is 4-bit quantized; "
                "DFlash regresses on 4-bit (use an 8-bit or higher variant)"
            )
        has_drafter = bool(
            getattr(profile, "dflash_draft_model", None)
            or getattr(profile, "drafter_hf_path", None)
        )
        if profile.supports_dflash and not has_drafter:
            reasons.append("supports_dflash is set but dflash_draft_model is empty")
        return EligibilityReport(
            alias=alias,
            supports_dflash=profile.supports_dflash,
            is_moe=profile.is_moe,
            is_4bit=is_4bit,
            has_drafter=has_drafter,
            reasons=tuple(reasons),
        )


_checker = _DFlashChecker()
report = _checker.report


def check(profile, alias=None):
    return _checker.check(profile, alias=alias, _eligible_fn=eligible_aliases)


eligible_aliases = _checker.eligible_aliases
have_runtime = _checker.have_runtime
