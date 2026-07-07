# SPDX-License-Identifier: Apache-2.0
#
# Boot-time resolver for ``--spec-decode auto``. Wires SpecAutoRouter into
# the CLI: when the operator passes --spec-decode auto, resolve_spec_auto()
# inspects the loaded model's config to decide between the zero-config
# methods (n-gram suffix for everyone, MTP for MTP-eligible Qwen3.5/3.6
# checkpoints). Drafter-backed methods (dflash/dspark) stay operator-
# selected — they need a bound drafter and model-specific eligibility
# checks that already run on their explicit flags.
#
# Boot-time vs per-request: the router's long_doc_threshold and acceptance
# hysteresis need request-time signals (prompt length, observed accept
# rate). At boot we only know model shape, so prompt_token_count is 0 and
# the long-doc branch never fires. Auto at boot is "pick the safe default
# for this model"; per-request routing is engine work tracked separately.
import logging
from dataclasses import dataclass

from .auto_router import METHOD_MTP, METHOD_NGRAM, RouteSignals, SpecAutoRouter
from .mtp import MTPEligibility, detect_mtp_eligibility

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoResolution:
    # method: auto_router canonical name (suffix or mtp at boot).
    # cli_target: human-readable target for the boot banner.
    # reason: one-line why, shown to the operator.
    method: str
    cli_target: str
    reason: str


def resolve_spec_auto(
    hf_config: dict | None,
    *,
    router: SpecAutoRouter | None = None,
) -> AutoResolution:
    # Pick a zero-config spec-decode method for --spec-decode auto at
    # boot. Never raises — a probe failure narrows the choice to suffix.
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

    method = router.decide(
        RouteSignals(
            prompt_token_count=0,
            has_mtp=has_mtp,
            available=frozenset(available),
        )
    )
    cli_target, reason = _describe(method, has_mtp)
    logger.info(
        "spec-auto: selected %s (has_mtp=%s, available=%s)",
        method,
        has_mtp,
        sorted(available),
    )
    return AutoResolution(method=method, cli_target=cli_target, reason=reason)


def apply_resolution(args, resolution: AutoResolution) -> None:
    # Map a router decision onto args. mtp rides the spec_decode choice
    # slot (the eligibility check below validates it); suffix runs via
    # the suffix_decoding flag. spec_decode is normalized to "none" for
    # suffix so it doesn't trip the mtp-only validation branch.
    if resolution.method == METHOD_MTP:
        args.spec_decode = "mtp"
        args.suffix_decoding = False
    else:
        args.spec_decode = "none"
        args.suffix_decoding = True


def _describe(method: str, has_mtp: bool) -> tuple[str, str]:
    if method == METHOD_MTP:
        return "mtp", "model is MTP-eligible (mtp_num_hidden_layers >= 1)"
    return "suffix", "n-gram suffix decoding (safe default, zero GPU cost)"
