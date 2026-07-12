# SPDX-License-Identifier: Apache-2.0
# MFA dispatch policy - auto-select best attention kernel for Apple Silicon.

from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass
from enum import Enum, auto

import mlx.core as mx

logger = logging.getLogger(__name__)


class AttentionBackend(Enum):
    NAX = auto()
    STEEL = auto()
    STEEL_DSPLIT = auto()
    TURBOQUANT = auto()
    MLX_SDPA = auto()


class DeviceGeneration(Enum):
    UNKNOWN = 0
    M1 = 13
    M2 = 14
    M3 = 15
    M4 = 16
    M5 = 17


@dataclass(frozen=True)
class DeviceInfo:
    generation: DeviceGeneration
    gpu_core_count: int
    has_nax: bool
    is_apple_silicon: bool
    device_name: str


@dataclass(frozen=True)
class DispatchDecision:
    backend: AttentionBackend
    reason: str
    supports_backward: bool = True
    block_size: tuple[int, int] = (64, 64)


_DEVICE_INFO: DeviceInfo | None = None


def _detect_device_info() -> DeviceInfo:
    is_apple_silicon = False
    try:
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=1,
        )
        is_apple_silicon = "Apple" in result.stdout
    except Exception:
        is_apple_silicon = (
            platform.processor() == "arm" and platform.system() == "Darwin"
        )
    gen = DeviceGeneration.UNKNOWN
    gpu_cores = 8
    has_nax = False
    device_name = "unknown"

    try:
        info = {}
        if hasattr(mx, "metal") and hasattr(mx.metal, "device_info"):
            raw = mx.metal.device_info()
            if isinstance(raw, dict):
                info = raw

        device_name = str(info.get("name", info.get("device", "")))
        name_lower = device_name.lower()

        if "m5" in name_lower:
            gen = DeviceGeneration.M5
            has_nax = True
        elif "m4" in name_lower:
            gen = DeviceGeneration.M4
            has_nax = True
        elif "m3" in name_lower:
            gen = DeviceGeneration.M3
            has_nax = True
        elif "m2" in name_lower:
            gen = DeviceGeneration.M2
        elif "m1" in name_lower:
            gen = DeviceGeneration.M1

        gpu_cores = info.get("max_compute_units", 8)
        if not isinstance(gpu_cores, int):
            gpu_cores = 8
    except Exception:
        pass

    if not has_nax:
        try:
            from mlx_mfa._ext import device_has_neural_accelerators as _h

            has_nax = _h()
            if has_nax and gen == DeviceGeneration.UNKNOWN:
                gen = DeviceGeneration.M3
        except (ImportError, AttributeError):
            pass

    return DeviceInfo(
        generation=gen,
        gpu_core_count=gpu_cores,
        has_nax=has_nax,
        is_apple_silicon=is_apple_silicon,
        device_name=device_name,
    )


def get_device_info() -> DeviceInfo:
    global _DEVICE_INFO
    if _DEVICE_INFO is None:
        _DEVICE_INFO = _detect_device_info()
    return _DEVICE_INFO


def reset_device_cache() -> None:
    global _DEVICE_INFO
    _DEVICE_INFO = None


def select_backend(
    head_dim: int,
    dtype: mx.Dtype = mx.float16,
    causal: bool = False,
    window_size: int = 0,
    quantized_kv: bool = False,
    has_alibi: bool = False,
    seq_len_q: int = 0,
    seq_len_kv: int = 0,
    batch_size: int = 1,
    num_heads: int = 1,
    has_mask: bool = False,
    is_decode: bool = False,
) -> DispatchDecision:
    info = get_device_info()

    _force = os.environ.get("MFA_FORCE_BACKEND", "").upper()
    if _force:
        for be in AttentionBackend:
            if be.name == _force:
                return DispatchDecision(
                    backend=be,
                    reason=f"env override MFA_FORCE_BACKEND={_force}",
                    block_size=_tile_size(head_dim, seq_len_q, seq_len_kv, is_decode),
                )

    if quantized_kv:
        return DispatchDecision(
            backend=AttentionBackend.TURBOQUANT,
            reason="quantized K/V detected",
            supports_backward=False,
            block_size=(64, 64),
        )

    if is_decode:
        return DispatchDecision(
            backend=AttentionBackend.MLX_SDPA,
            reason="decode step: mx.fast.sdpa is optimal",
            block_size=(1, 64),
        )

    if info.has_nax and batch_size >= 2 and head_dim <= 128 and not has_alibi:
        return DispatchDecision(
            backend=AttentionBackend.NAX,
            reason="NAX available on M3+/M4+, batch >= 2, small D",
            supports_backward=False,
            block_size=(128, 64),
        )

    ext_avail = _is_mfa_ext_available()
    if ext_avail:
        if head_dim in (64, 128):
            return DispatchDecision(
                backend=AttentionBackend.STEEL,
                reason=f"STEEL v2/v3 optimal for head_dim={head_dim}",
                block_size=_tile_size(head_dim, seq_len_q, seq_len_kv, is_decode),
            )
        if head_dim in (256, 512):
            return DispatchDecision(
                backend=AttentionBackend.STEEL_DSPLIT,
                reason=f"STEEL D-split for head_dim={head_dim}",
                block_size=(64, 64),
            )

    return DispatchDecision(
        backend=AttentionBackend.MLX_SDPA,
        reason="fallback: no MFA kernel available for this config",
        block_size=(64, 64),
    )


def _tile_size(
    head_dim: int,
    seq_len_q: int,
    seq_len_kv: int,
    is_decode: bool,
) -> tuple[int, int]:
    if is_decode:
        return (1, 64)
    if head_dim == 64:
        return (128, 128)
    if head_dim == 128:
        return (64, 64)
    if head_dim == 256:
        return (32, 64)
    return (64, 64)


def _is_mfa_ext_available() -> bool:
    try:
        from mlx_mfa._ext import mfa_attention_forward  # noqa: F401

        return True
    except (ImportError, AttributeError):
        return False


def supports_backward(head_dim: int) -> bool:
    return head_dim in (64, 128) and _is_mfa_ext_available()


def get_supported_head_dims() -> list[int]:
    if _is_mfa_ext_available():
        return [64, 128, 256]
    return []


def warmup_kernels(head_dim: int, dtype: mx.Dtype = mx.float16) -> None:
    if not _is_mfa_ext_available():
        return
    try:
        from mlx_mfa import warmup_kernels as _warmup

        _warmup(head_dim=head_dim, dtype=dtype)
    except Exception as exc:
        logger.debug("MFA kernel warmup skipped (%s)", exc)
