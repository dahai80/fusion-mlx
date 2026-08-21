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

from __future__ import annotations

import logging
from dataclasses import dataclass

from fusion_mlx.model_aliases import AliasProfile

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


def _looks_like_4bit(hf_path: str) -> bool:
    lowered = hf_path.lower()
    if "-4bit" in lowered:
        return True
    if "mxfp4" in lowered or "nvfp4" in lowered:
        return True
    return False


def report(profile: AliasProfile, alias: str | None = None) -> EligibilityReport:
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


def eligible_aliases() -> list[str]:
    try:
        from fusion_mlx.model_aliases import list_profiles

        return sorted(p.name for p in list_profiles().values() if not report(p).reasons)
    except Exception as e:  # noqa: BLE001 — diagnostic helper, never fatal
        logger.debug("eligible_aliases failed: %s", e)
        return []


def check(profile: AliasProfile, alias: str | None = None) -> None:
    r = report(profile, alias=alias)
    if not r.reasons:
        return
    header = f"DFlash2 unavailable for {alias!r}" if alias else "DFlash2 unavailable"
    bullet = "\n  - ".join(r.reasons)
    eligible = eligible_aliases()
    if eligible:
        suffix = (
            f"Eligible aliases today: {', '.join(eligible)}. Run "
            "`fusion-mlx info <alias>` to inspect per-alias DFlash2 status."
        )
    else:
        suffix = (
            "No aliases currently pass every DFlash2 gate. DFlash2 targets "
            "Qwen3.8-27B dense — pass a dense Qwen3.8 repo, e.g. "
            "`fusion-mlx serve --enable-dflash2 mlx-community/Qwen3.8-27B-4bit "
            "--dflash2-drafter-path z-lab/Qwen3.8-27B-DFlash2 --block-size 5`."
        )
    raise DFlash2Unavailable(f"{header}:\n  - {bullet}\n\n{suffix}")


def have_runtime() -> bool:
    # The official dflash pkg is an external pip dependency (not vendored,
    # unlike DSpark). Probe importability cheaply without importing the
    # heavy mlx stack. DFlash2Generator existence is checked at load time.
    try:
        import importlib

        spec = importlib.util.find_spec("dflash")
        return spec is not None
    except (ImportError, AttributeError, ModuleNotFoundError):
        return False
