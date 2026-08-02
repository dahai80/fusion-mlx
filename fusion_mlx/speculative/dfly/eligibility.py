# SPDX-License-Identifier: Apache-2.0
"""DFly eligibility checks — gate the feature behind validated combos.

DFly is the Hunyuan (Hy3) native block-parallel drafter (DFlash +
hidden-correction).  Gates:
  - model_family must be "hunyuan" (AliasProfile.model_family or
    auto-detected from config/hf_path)
  - alias must NOT be is_moe=True (MoE routing churn kills acceptance)
  - main model must be 8-bit+ (4-bit regresses)
  - drafter HF path must be reachable or operator-supplied
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fusion_mlx.model_aliases import AliasProfile

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


def _looks_like_4bit(hf_path: str) -> bool:
    lowered = hf_path.lower()
    if "-4bit" in lowered:
        return True
    if "mxfp4" in lowered or "nvfp4" in lowered:
        return True
    return False


def _detect_model_family(profile: AliasProfile) -> str:
    family = getattr(profile, "model_family", "")
    if family:
        return family
    hf_lower = profile.hf_path.lower()
    if "hunyuan" in hf_lower or "hy3" in hf_lower:
        return "hunyuan"
    return ""


def report(profile: AliasProfile, alias: str | None = None) -> EligibilityReport:
    reasons: list[str] = []
    family = _detect_model_family(profile)
    if family != "hunyuan":
        reasons.append(
            "DFly is a Hunyuan (Hy3)-native drafter; model_family={!r} "
            "is not 'hunyuan'. Use dfly only with Hy3 models.".format(family or "unknown")
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
            "DFly regresses on 4-bit (use bf16/8-bit+ Hy3 variant)"
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


def eligible_aliases() -> list[str]:
    try:
        from fusion_mlx.model_aliases import list_profiles

        return sorted(
            name
            for name, profile in list_profiles().items()
            if not report(profile).reasons
        )
    except Exception:  # noqa: BLE001
        return []


def check(profile: AliasProfile, alias: str | None = None) -> None:
    r = report(profile, alias=alias)
    if not r.reasons:
        return
    header = f"DFly unavailable for {alias!r}" if alias else "DFly unavailable"
    bullet = "\n  - ".join(r.reasons)
    eligible = eligible_aliases()
    if eligible:
        suffix = (
            f"Eligible aliases today: {', '.join(eligible)}. Run "
            "`fusion-mlx info <alias>` to inspect per-alias DFly status."
        )
    else:
        suffix = (
            "No aliases currently pass every DFly gate. DFly targets "
            "Hunyuan (Hy3) bf16/8-bit models — pass an Hy3 repo, e.g. "
            "`fusion-mlx serve --enable-dfly <hy3-model> "
            "--dfly-drafter-path AngelSlim/Hy3-DFly-Block8`."
        )
    raise DFlyUnavailable(f"{header}:\n  - {bullet}\n\n{suffix}")


def have_runtime() -> bool:
    try:
        import importlib

        spec = importlib.util.find_spec("fusion_mlx.speculative.dfly.drafter")
        return spec is not None
    except (ImportError, AttributeError, ModuleNotFoundError):
        return False
