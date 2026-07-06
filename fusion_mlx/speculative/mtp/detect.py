# SPDX-License-Identifier: Apache-2.0
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MTPEligibility(Enum):
    NONE = "none"
    CHAIN = "chain"
    TREE = "tree"


_SUPPORTED_MODEL_TYPES: frozenset[str] = frozenset({
    "qwen3_5",
    "qwen3_5_moe",
    "gemma4_unified",
})


@dataclass(frozen=True)
class _DetectionResult:
    eligibility: MTPEligibility
    model_type: str | None
    num_mtp_layers: int
    reason: str


def detect_mtp_eligibility(
    config: object,
    has_external_sidecar: bool = False,
) -> MTPEligibility:
    r = _detect_mtp_eligibility_verbose(config, has_external_sidecar)
    return r.eligibility


def _detect_mtp_eligibility_verbose(
    config: object,
    has_external_sidecar: bool = False,
) -> _DetectionResult:
    if has_external_sidecar:
        return _DetectionResult(
            eligibility=MTPEligibility.TREE,
            model_type=None,
            num_mtp_layers=0,
            reason="external sidecar overrides eligibility",
        )
    if config is None:
        return _DetectionResult(
            eligibility=MTPEligibility.NONE,
            model_type=None,
            num_mtp_layers=0,
            reason="config is None",
        )
    config_dict: dict = {}
    if isinstance(config, dict):
        config_dict = config
    else:
        model_type = getattr(config, "model_type", None)
        if model_type is None:
            return _DetectionResult(
                eligibility=MTPEligibility.NONE,
                model_type=None,
                num_mtp_layers=0,
                reason="config has no model_type",
            )
        config_dict = {"model_type": model_type}
        for attr in ("mtp_num_hidden_layers", "num_mtp_layers"):
            val = getattr(config, attr, None)
            if val is not None:
                config_dict[attr] = val
    model_type = config_dict.get("model_type", "")
    if model_type not in _SUPPORTED_MODEL_TYPES:
        return _DetectionResult(
            eligibility=MTPEligibility.NONE,
            model_type=model_type,
            num_mtp_layers=0,
            reason=f"model_type {model_type!r} not in supported set",
        )
    num_layers = (
        config_dict.get("mtp_num_hidden_layers")
        or config_dict.get("num_mtp_layers")
        or 0
    )
    if not isinstance(num_layers, int) or num_layers < 1:
        return _DetectionResult(
            eligibility=MTPEligibility.NONE,
            model_type=model_type,
            num_mtp_layers=0,
            reason=f"no MTP layers found (mtp_num_hidden_layers={num_layers})",
        )
    return _DetectionResult(
        eligibility=MTPEligibility.CHAIN,
        model_type=model_type,
        num_mtp_layers=num_layers,
        reason=f"{model_type} with {num_layers} MTP layer(s)",
    )
