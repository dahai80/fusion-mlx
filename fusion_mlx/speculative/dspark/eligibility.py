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

from __future__ import annotations

import logging
from dataclasses import dataclass

from fusion_mlx.model_aliases import AliasProfile

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


def _looks_like_4bit(hf_path: str) -> bool:
    # mlx-community publishes quants as -4bit / -mxfp4 / -nvfp4 suffixes.
    lowered = hf_path.lower()
    if "-4bit" in lowered:
        return True
    if "mxfp4" in lowered or "nvfp4" in lowered:
        return True
    return False


def report(profile: AliasProfile, alias: str | None = None) -> EligibilityReport:
    reasons: list[str] = []
    if not profile.supports_dspark:
        reasons.append(
            "alias is not DSpark-enabled (set supports_dspark=true in "
            "aliases.json after benching to validate the speedup)"
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


def eligible_aliases() -> list[str]:
    # list_profiles() returns a list[AliasProfile] (not a dict); iterate
    # directly. Tolerant: any registry error returns [] rather than raising
    # since this only enriches error text.
    try:
        from fusion_mlx.model_aliases import list_profiles

        return sorted(p.name for p in list_profiles() if not report(p).reasons)
    except Exception as e:  # noqa: BLE001 — diagnostic helper, never fatal
        logger.debug("eligible_aliases failed: %s", e)
        return []


def check(profile: AliasProfile, alias: str | None = None) -> None:
    r = report(profile, alias=alias)
    if not r.reasons:
        return
    header = f"DSpark unavailable for {alias!r}" if alias else "DSpark unavailable"
    bullet = "\n  - ".join(r.reasons)
    eligible = eligible_aliases()
    if eligible:
        suffix = (
            f"Eligible aliases today: {', '.join(eligible)}. Run "
            "`fusion-mlx info <alias>` to inspect per-alias DSpark status."
        )
    else:
        suffix = (
            "No aliases currently pass every DSpark gate. DSpark targets "
            "Qwen3 4B/8B/14B bf16 — pass a bf16 Qwen3 repo directly, e.g. "
            "`fusion-mlx serve --enable-dspark mlx-community/Qwen3-8B-bf16 "
            "--dspark-drafter-path <converted-mlx-draft>`."
        )
    raise DSparkUnavailable(f"{header}:\n  - {bullet}\n\n{suffix}")


def have_runtime() -> bool:
    # Probe dspark-metal without importing it (cheap on the hot CLI path).
    # A partial install would have the package missing entirely; the
    # DSparkGenerator symbol is checked at load_runtime time.
    try:
        import importlib

        spec = importlib.util.find_spec("dspark_metal")
        return spec is not None
    except (ImportError, AttributeError):
        return False
