# SPDX-License-Identifier: Apache-2.0
#
# Boot-time resolver for ``--spec-decode auto``. Wires SpecAutoRouter into
# the CLI: when the operator passes --spec-decode auto, resolve_spec_auto()
# inspects the loaded model's config to decide between the zero-config
# methods (n-gram suffix for everyone, MTP for MTP-eligible Qwen3.5/3.6
# checkpoints). Drafter-backed methods (dflash/dspark/dfly) stay operator-
# selected — they need a bound drafter and model-specific eligibility
# checks that already run on their explicit flags.
#
# Phase 2: resolve_spec_auto() now accepts model_family, is_moe, quant_bits
# to build richer RouteSignals. The routing table in auto_router provides
# method priorities per family.
import logging
from dataclasses import dataclass

from .auto_router import (
    METHOD_DFLY,
    METHOD_MTP,
    METHOD_NGRAM,
    RouteSignals,
    SpecAutoRouter,
    _SPEC_ROUTING_TABLE,
)
from .mtp import MTPEligibility, detect_mtp_eligibility

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoResolution:
    method: str
    cli_target: str
    reason: str
    model_family: str | None = None


def _family_methods(family: str | None) -> set[str]:
    if family is None:
        return set()
    for entry in _SPEC_ROUTING_TABLE:
        if entry.family == family:
            return set(entry.methods)
    return set()


def resolve_spec_auto(
    hf_config: dict | None,
    *,
    model_family: str | None = None,
    is_moe: bool = False,
    quant_bits: int | None = None,
    router: SpecAutoRouter | None = None,
) -> AutoResolution:
    router = router or SpecAutoRouter()
    available = {METHOD_NGRAM}
    try:
        mtp_elig = detect_mtp_eligibility(hf_config)
        has_mtp = mtp_elig is not MTPEligibility.NONE
    except Exception as exc:
        logger.warning(
            "spec-auto: mtp eligibility probe failed (%s); treating as non-MTP",
            exc,
        )
        has_mtp = False
    if has_mtp:
        available.add(METHOD_MTP)

    family_methods = _family_methods(model_family)
    if family_methods:
        available.update(family_methods)

    method = router.decide(
        RouteSignals(
            prompt_token_count=0,
            has_mtp=has_mtp,
            available=frozenset(available),
            model_family=model_family,
            is_moe=is_moe,
            quant_bits=quant_bits,
        )
    )
    cli_target, reason = _describe(method, has_mtp, model_family)
    logger.info(
        "spec-auto: selected %s (has_mtp=%s, family=%s, available=%s)",
        method,
        has_mtp,
        model_family,
        sorted(available),
    )
    return AutoResolution(
        method=method,
        cli_target=cli_target,
        reason=reason,
        model_family=model_family,
    )


def apply_resolution(args, resolution: AutoResolution) -> None:
    if resolution.method == METHOD_MTP:
        args.spec_decode = "mtp"
        args.suffix_decoding = False
    elif resolution.method == METHOD_DFLY:
        args.spec_decode = "dfly"
        args.suffix_decoding = False
    else:
        args.spec_decode = "none"
        args.suffix_decoding = True


def _describe(
    method: str, has_mtp: bool, model_family: str | None
) -> tuple[str, str]:
    if method == METHOD_DFLY:
        return (
            "dfly",
            f"DFly block-parallel drafter (family={model_family})",
        )
    if method == METHOD_MTP:
        return "mtp", "model is MTP-eligible (mtp_num_hidden_layers >= 1)"
    return "suffix", "n-gram suffix decoding (safe default, zero GPU cost)"
