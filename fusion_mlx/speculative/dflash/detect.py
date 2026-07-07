# SPDX-License-Identifier: Apache-2.0
import enum
from dataclasses import dataclass
from typing import Any


class DFlashEligibility(str, enum.Enum):
    NONE = "none"
    READY = "ready"


_SUPPORTED_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "qwen3_5",
        "qwen3_5_moe",
    }
)


@dataclass(frozen=True)
class _DetectionResult:
    eligibility: DFlashEligibility
    model_type: str | None
    drafter_path: str | None
    reason: str


def detect_dflash_eligibility(
    config: dict[str, Any] | None,
    *,
    alias: str | None = None,
    drafter_override: str | None = None,
) -> DFlashEligibility:
    r = _detect_dflash_eligibility_verbose(
        config, alias=alias, drafter_override=drafter_override
    )
    return r.eligibility


def _detect_dflash_eligibility_verbose(
    config: dict[str, Any] | None,
    *,
    alias: str | None = None,
    drafter_override: str | None = None,
) -> _DetectionResult:
    if not isinstance(config, dict):
        return _DetectionResult(
            DFlashEligibility.NONE, None, None, "config is not a dict"
        )
    model_type = config.get("model_type")
    if not isinstance(model_type, str):
        return _DetectionResult(
            DFlashEligibility.NONE,
            None,
            None,
            "model_type missing or not a string",
        )
    if model_type not in _SUPPORTED_MODEL_TYPES:
        return _DetectionResult(
            DFlashEligibility.NONE,
            model_type,
            None,
            f"model_type {model_type!r} not in DFlash allowlist",
        )
    drafter_path: str | None = None
    if drafter_override:
        drafter_path = drafter_override
    elif alias:
        from .drafter_registry import get_dflash_drafter_path

        drafter_path = get_dflash_drafter_path(alias)
    if not drafter_path:
        return _DetectionResult(
            DFlashEligibility.NONE,
            model_type,
            None,
            f"no drafter bound for alias {alias!r}",
        )
    return _DetectionResult(
        DFlashEligibility.READY,
        model_type,
        drafter_path,
        f"Qwen3.5/3.6 + drafter {drafter_path!r} bound",
    )
