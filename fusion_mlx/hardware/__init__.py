"""Hardware detection and model compatibility for fusion-mlx."""

from .apple import (
    CHIP_TIER_FULL,
    CHIP_TIER_LITE,
    CHIP_TIER_STANDARD,
    classify_chip_tier,
    detect_chip_tier,
)
from .types import GPUInfo, HardwareInfo

__all__ = [
    "GPUInfo",
    "HardwareInfo",
    "CHIP_TIER_FULL",
    "CHIP_TIER_LITE",
    "CHIP_TIER_STANDARD",
    "classify_chip_tier",
    "detect_chip_tier",
]
