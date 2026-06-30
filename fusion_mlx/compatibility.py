"""Model compatibility checking — adapted from whichllm."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .hardware.memory import estimate_usable_ram
from .hardware.types import GPUInfo, HardwareInfo

logger = logging.getLogger(__name__)

_FRAMEWORK_OVERHEAD_BYTES = 500_000_000
_KV_BYTES_PER_BPARAM_PER_KCTX = 3.5 * 1024 * 1024
_ACTIVATION_BASE = 400_000_000
_ACTIVATION_PER_PARAM = 0.08
_ACTIVATION_PER_4K_CTX = 150_000_000


@dataclass
class CompatibilityResult:
    model_id: str
    can_run: bool
    vram_required_bytes: int
    vram_available_bytes: int
    offload_ratio: float = 0.0
    warnings: list[str] = field(default_factory=list)
    fit_type: str = "full_gpu"
    context_fits: bool = True


def _estimate_weight_bytes(params: int, quant_type: str) -> int:
    QUANT_BYTES = {
        "F32": 4.0, "F16": 2.0, "BF16": 2.0,
        "Q8_0": 1.0625, "Q8_K": 1.0625,
        "Q6_K": 0.8125,
        "Q5_K_M": 0.6875, "Q5_K_S": 0.6875,
        "Q5_1": 0.6875, "Q5_0": 0.625,
        "Q4_K_M": 0.5625, "Q4_K_S": 0.5625,
        "Q4_1": 0.5625, "Q4_0": 0.5,
        "Q3_K_L": 0.5, "Q3_K_M": 0.4375, "Q3_K_S": 0.4375,
        "Q2_K": 0.3125,
        "IQ4_NL": 0.5, "IQ4_XS": 0.4375,
        "IQ3_M": 0.375, "IQ3_S": 0.3125,
        "IQ2_XXS": 0.25, "IQ2_XS": 0.3125, "IQ2_S": 0.3125,
        "IQ1_M": 0.1875,
        "TQ1_0": 0.1875, "TQ2_0": 0.3125,
        "4BIT": 0.5, "8BIT": 1.0,
        "4bit": 0.5, "8bit": 1.0,
    }
    bytes_per_weight = QUANT_BYTES.get(quant_type, QUANT_BYTES.get(quant_type.upper(), 2.0))
    return int(params * bytes_per_weight)


def _estimate_kv_cache(params: int, context_length: int) -> int:
    params_b = params / 1e9
    ctx_k = context_length / 1024
    return int(params_b * ctx_k * _KV_BYTES_PER_BPARAM_PER_KCTX)


def _estimate_activation(params: int, context_length: int) -> int:
    base = _ACTIVATION_BASE
    param_term = int(params * _ACTIVATION_PER_PARAM)
    ctx_term = int((context_length / 4096) * _ACTIVATION_PER_4K_CTX)
    return base + param_term + ctx_term


def _gpu_available_memory(gpu: GPUInfo, usable_ram: int) -> int:
    vram = (gpu.usable_vram_bytes if gpu.usable_vram_bytes is not None else gpu.vram_bytes)
    if gpu.shared_memory and vram < 2 * (1024**3):
        return usable_ram
    return vram


def check_compatibility(
    model_id: str,
    params: int,
    quant_type: str,
    hardware: HardwareInfo,
    context_length: int = 4096,
    max_context: int | None = None,
) -> CompatibilityResult:
    warnings: list[str] = []

    vram_required = (
        _estimate_weight_bytes(params, quant_type) +
        _estimate_kv_cache(params, context_length) +
        _estimate_activation(params, context_length) +
        _FRAMEWORK_OVERHEAD_BYTES
    )

    usable_ram = estimate_usable_ram(hardware.ram_bytes)

    best_gpu: GPUInfo | None = None
    best_gpu_available = 0
    for gpu in hardware.gpus:
        gpu_avail = _gpu_available_memory(gpu, usable_ram)
        if best_gpu is None or gpu_avail > best_gpu_available:
            best_gpu = gpu
            best_gpu_available = gpu_avail

    vram_available = best_gpu_available if best_gpu else 0
    offload_ram_available = (
        0
        if best_gpu and best_gpu.shared_memory
        else usable_ram
    )

    if vram_available >= vram_required:
        fit_type = "full_gpu"
        can_run = True
        offload_ratio = 0.0
    elif vram_available > 0 and (vram_available + offload_ram_available) >= vram_required:
        fit_type = "partial_offload"
        can_run = True
        offload_ratio = (vram_required - vram_available) / vram_required if vram_required > 0 else 0.0
        if best_gpu and best_gpu.shared_memory:
            warnings.append("Will use shared system memory")
        else:
            warnings.append(f"~{offload_ratio * 100:.0f}% of layers will be offloaded to CPU RAM")
    elif usable_ram >= vram_required:
        fit_type = "cpu_only"
        can_run = True
        offload_ratio = 0.0
        warnings.append("Will run on CPU only (much slower)")
    else:
        fit_type = "cpu_only"
        can_run = False
        offload_ratio = 0.0
        warnings.append("Insufficient memory (GPU VRAM + RAM) to run this model")

    context_fits = not (max_context and max_context < context_length)
    if not context_fits:
        warnings.append(
            f"Model max context {max_context} < requested {context_length}; "
            f"runtime will truncate or reject"
        )
    elif context_length > 8192 and max_context and max_context >= context_length:
        warnings.append(f"Large context ({context_length}) increases VRAM usage significantly")

    file_size = _estimate_weight_bytes(params, quant_type)
    if hardware.disk_free_bytes > 0 and file_size > hardware.disk_free_bytes:
        warnings.append("Insufficient disk space to download this model")
        can_run = False

    return CompatibilityResult(
        model_id=model_id,
        can_run=can_run,
        vram_required_bytes=vram_required,
        vram_available_bytes=vram_available,
        offload_ratio=offload_ratio,
        warnings=warnings,
        fit_type=fit_type,
        context_fits=context_fits,
    )
