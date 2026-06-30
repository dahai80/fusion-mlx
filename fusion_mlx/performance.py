"""Token generation speed estimation — adapted from whichllm."""

from __future__ import annotations

from .compatibility import _estimate_weight_bytes
from .hardware.types import GPUInfo

_QUANT_EFFICIENCY: dict[str, float] = {
     "F32": 0.30, "F16": 0.40, "BF16": 0.40,
      "Q8_0": 0.45, "Q8_K": 0.45,
      "Q6_K": 0.50,
      "Q5_K_M": 0.52, "Q5_K_S": 0.51,
      "Q5_1": 0.51, "Q5_0": 0.50,
      "Q4_K_M": 0.55, "Q4_K_S": 0.54,
      "Q4_1": 0.54, "Q4_0": 0.53,
      "Q3_K_L": 0.51, "Q3_K_M": 0.50, "Q3_K_S": 0.49,
      "Q2_K": 0.45,
      "IQ4_NL": 0.54, "IQ4_XS": 0.52,
      "IQ3_M": 0.48, "IQ3_S": 0.46,
      "IQ2_XXS": 0.40, "IQ2_XS": 0.42, "IQ2_S": 0.43,
      "IQ1_M": 0.35,
      "TQ1_0": 0.33, "TQ2_0": 0.43,
      "4BIT": 0.55, "8BIT": 0.45,
}
_DEFAULT_QUANT_EFFICIENCY = 0.45

_BACKEND_FACTOR: dict[str, float] = {
      "nvidia": 1.00,
      "amd": 0.78,
      "apple": 0.82,
      "intel": 0.65,
}


def _quant_efficiency(quant_type: str) -> float:
    return _QUANT_EFFICIENCY.get(quant_type.upper(), _DEFAULT_QUANT_EFFICIENCY)


def _backend_factor(vendor: str) -> float:
    return _BACKEND_FACTOR.get(vendor, 0.70)


def estimate_tok_per_sec(
    params: int,
    quant_type: str,
    gpu: GPUInfo | None,
    fit_type: str = "full_gpu",
) -> float:
    if gpu is None or fit_type == "cpu_only":
        params_b = params / 1e9
        if params_b <= 0:
            return 0.0
        quant_factor = _quant_efficiency(quant_type) / _DEFAULT_QUANT_EFFICIENCY
        return max(0.3, 18.0 / max(params_b, 0.5) * quant_factor)

    model_size = _estimate_weight_bytes(params, quant_type)
    bandwidth = gpu.memory_bandwidth_gbps * 1e9 if gpu.memory_bandwidth_gbps else 0
    if bandwidth == 0:
        return 0.0

    theoretical = bandwidth / model_size
    efficiency = _quant_efficiency(quant_type) * _backend_factor(gpu.vendor)

    if fit_type == "partial_offload":
        if gpu.shared_memory:
            efficiency *= 0.85
        else:
            efficiency *= 0.45

    return theoretical * efficiency
