# SPDX-License-Identifier: Apache-2.0
"""MLX device/stream封装 for SkyReels-V3.

剔除全部 CUDA 硬编码 (`.to("cuda")`, `.cuda()`, `torch.cuda.is_available()`),
封装统一 MLX Metal 流接口, 自动判断 M5 Max 启用分级 Tile.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from functools import lru_cache

import mlx.core as mx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device generation detection
# ---------------------------------------------------------------------------
class DeviceGeneration:
    """Apple Silicon GPU generation coarse enum."""

    UNKNOWN = 0
    M1 = 13
    M2 = 14
    M3 = 15
    M4 = 16
    M5 = 17


@lru_cache(maxsize=1)
def detect_generation() -> int:
    """Detect Apple Silicon GPU generation via sysctl + mlx metal info.

    Returns one of :class:`DeviceGeneration` values (UNKNOWN if undetermined).
    """
    # 1) Prefer mlx.metal.device_info() if available (most reliable)
    try:
        if hasattr(mx, "metal") and hasattr(mx.metal, "device_info"):
            raw = mx.metal.device_info()
            if isinstance(raw, dict):
                name = str(raw.get("name", "")).lower()
                if "m5" in name or "m 5" in name:
                    return DeviceGeneration.M5
                if "m4" in name or "m 4" in name:
                    return DeviceGeneration.M4
                if "m3" in name or "m 3" in name:
                    return DeviceGeneration.M3
                if "m2" in name or "m 2" in name:
                    return DeviceGeneration.M2
                if "m1" in name or "m 1" in name:
                    return DeviceGeneration.M1
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("mx.metal.device_info() failed: %s", exc)

    # 2) Fallback: sysctl machdep.cpu.brand_string
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=1,
        )
        brand = result.stdout.lower()
        if "m5" in brand:
            return DeviceGeneration.M5
        if "m4" in brand:
            return DeviceGeneration.M4
        if "m3" in brand:
            return DeviceGeneration.M3
        if "m2" in brand:
            return DeviceGeneration.M2
        if "m1" in brand:
            return DeviceGeneration.M1
        if "apple" in brand:
            return DeviceGeneration.UNKNOWN  # Apple Silicon but unknown gen
    except Exception as exc:  # pragma: no cover
        logger.debug("sysctl brand_string failed: %s", exc)

    # 3) Last resort: platform.processor() == "arm"
    if platform.processor() == "arm" and platform.system() == "Darwin":
        return DeviceGeneration.UNKNOWN

    return DeviceGeneration.UNKNOWN


def is_m5() -> bool:
    """Return True if running on M5-series Apple Silicon."""
    return detect_generation() == DeviceGeneration.M5


def is_apple_silicon() -> bool:
    """Return True if running on any Apple Silicon (arm64 Darwin)."""
    return detect_generation() != DeviceGeneration.UNKNOWN or (
        platform.processor() == "arm" and platform.system() == "Darwin"
    )


# ---------------------------------------------------------------------------
# Stream management
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_stream() -> mx.Stream:
    """Return a dedicated Metal stream for SkyReels inference.

    On M5 Max this enables async double-buffering with the engine's
    async_eval harness (see fusion_mlx/engine_core.py).
    """
    try:
        return mx.new_stream(mx.gpu)
    except Exception as exc:  # pragma: no cover - CPU-only test envs
        logger.warning("Failed to create GPU stream: %s", exc)
        return mx.new_stream(mx.cpu)


@lru_cache(maxsize=1)
def get_compute_stream() -> mx.Stream:
    """Alias for get_stream() - the compute (DiT) stream."""
    return get_stream()


# ---------------------------------------------------------------------------
# Compile policy
# ---------------------------------------------------------------------------
def should_compile() -> bool:
    """M5 Max defaults to mx.compile enabled for DiT blocks + samplers."""
    # On CPU test envs compilation may fail; gate by apple silicon.
    return is_apple_silicon()


def get_tile_block_size() -> tuple[int, int]:
    """分级 Tile 自适应: 根据设备 L2 缓存大小动态选 block 尺寸.

    M5 Max has 16MB L2 per GPU cluster; default 128x128 tile.
    Older generations fall back to 64x64.
    """
    gen = detect_generation()
    if gen == DeviceGeneration.M5:
        return (128, 128)
    if gen == DeviceGeneration.M4:
        return (96, 96)
    return (64, 64)


def get_window_size_default() -> int:
    """默认时序注意力滑动窗口 (M5 上更大窗口).

    Returns window size in frames; -1 means global attention.
    """
    if is_m5():
        return 32  # M5 can afford larger temporal window
    return 16


# ---------------------------------------------------------------------------
# Device constants
# ---------------------------------------------------------------------------
DEFAULT_DTYPE: mx.Dtype = mx.bfloat16
COMPUTE_DTYPE: mx.Dtype = mx.bfloat16  # DiT主干
MODULATION_DTYPE: mx.Dtype = mx.float32  # AdaLN-Zero modulation 保精度
ROPE_DTYPE: mx.Dtype = mx.float32  # 原版fp64, MLX无fp64用fp32替代
VAE_DTYPE: mx.Dtype = mx.float16  # VAE解码器

__all__ = [
    "DeviceGeneration",
    "detect_generation",
    "is_m5",
    "is_apple_silicon",
    "get_stream",
    "get_compute_stream",
    "should_compile",
    "get_tile_block_size",
    "get_window_size_default",
    "DEFAULT_DTYPE",
    "COMPUTE_DTYPE",
    "MODULATION_DTYPE",
    "ROPE_DTYPE",
    "VAE_DTYPE",
]
