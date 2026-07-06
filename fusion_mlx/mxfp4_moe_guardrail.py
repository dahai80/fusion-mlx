# SPDX-License-Identifier: Apache-2.0
"""Load-time guardrail for the MoE + MXFP4 + multi-device throughput cliff.

mlx#3402: MoE + MXFP4 + multi-device hits 60x throughput cliff
(GLM-5.1/DeepSeek-V3.2: 0.27 tok/s vs expected ~16 tok/s).

mlx#2962: MLX NVFP4 uses signed E4M3 scales instead of unsigned UE4M3,
costing ~137x dynamic range. MoE weights silently degrade.

Decision: warn only. Operators see the warning and can pick a fallback.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_mxfp4_moe_distributed_warnings_total = 0
_nvfp4_moe_warnings_total = 0

MLX_3402_URL = "https://github.com/ml-explore/mlx/issues/3402"
MLX_2962_URL = "https://github.com/ml-explore/mlx/issues/2962"


@dataclass(frozen=True)
class GuardrailSignal:
    is_moe: bool
    quant_format: str | None
    distributed_world_size: int


def _detect_quant_format(hf_path: str | None) -> str | None:
    if not hf_path:
        return None
    lowered = hf_path.lower()
    if "mxfp4" in lowered:
        return "mxfp4"
    if "nvfp4" in lowered:
        return "nvfp4"
    return None


def _detect_distributed_world_size() -> int:
    import json
    import os

    candidates: list[int] = [1]

    for env_var in ("MLX_WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE"):
        raw = os.environ.get(env_var)
        if not raw:
            continue
        try:
            size = int(raw)
        except ValueError:
            continue
        if size >= 1:
            candidates.append(size)

    hostfile_path = os.environ.get("MLX_HOSTFILE")
    if hostfile_path:
        try:
            with open(hostfile_path) as fp:
                hosts = json.load(fp)
            if isinstance(hosts, list) and len(hosts) >= 1:
                candidates.append(len(hosts))
        except Exception:
            logger.debug(
                "MLX_HOSTFILE=%s present but unreadable; treating as size 1",
                hostfile_path,
                exc_info=True,
            )

    return max(candidates)


def _emit_mxfp4_moe_distributed_warning(
    *,
    hf_path: str | None,
    alias: str | None,
    world_size: int,
) -> None:
    global _mxfp4_moe_distributed_warnings_total
    with _lock:
        _mxfp4_moe_distributed_warnings_total += 1
    logger.warning(
        "MoE + MXFP4 + multi-device throughput cliff detected "
        "(mlx#3402, ~0.27 tok/s observed on M3 Ultra distributed vs "
        "~16 tok/s expected). model=%s alias=%s world_size=%d. "
        "Consider switching to a non-MXFP4 variant (e.g. 4-bit integer) "
        "or single-device serving until upstream fix lands. "
        "Details: %s",
        hf_path or "<unknown>",
        alias or "<none>",
        world_size,
        MLX_3402_URL,
    )


def _emit_nvfp4_moe_warning(
    *,
    hf_path: str | None,
    alias: str | None,
) -> None:
    global _nvfp4_moe_warnings_total
    with _lock:
        _nvfp4_moe_warnings_total += 1
    logger.warning(
        "MoE + NVFP4 dynamic-range loss detected "
        "(mlx#2962, signed E4M3 scales instead of Blackwell unsigned UE4M3 "
        "-> ~137x dynamic-range loss). model=%s alias=%s. "
        "MoE expert routing is especially sensitive to the lost range; "
        "expect silent token-quality regressions before throughput drops. "
        "Consider a 4-bit integer or non-NVFP4 variant. Details: %s",
        hf_path or "<unknown>",
        alias or "<none>",
        MLX_2962_URL,
    )


def check_load_time_guardrails(
    signal: GuardrailSignal,
    *,
    hf_path: str | None = None,
    alias: str | None = None,
) -> list[str]:
    fired: list[str] = []

    if not signal.is_moe:
        return fired

    quant = signal.quant_format
    is_distributed = signal.distributed_world_size > 1

    if quant == "mxfp4" and is_distributed:
        _emit_mxfp4_moe_distributed_warning(
            hf_path=hf_path,
            alias=alias,
            world_size=signal.distributed_world_size,
        )
        fired.append("mxfp4_moe_distributed")
    elif quant == "nvfp4":
        _emit_nvfp4_moe_warning(hf_path=hf_path, alias=alias)
        fired.append("nvfp4_moe")

    return fired


def check_from_profile(
    *,
    model_name: str,
    profile=None,
    alias: str | None = None,
) -> list[str]:
    is_moe = bool(getattr(profile, "is_moe", False))
    hf_path = getattr(profile, "hf_path", None) if profile is not None else model_name
    quant_format = _detect_quant_format(hf_path or model_name)
    world_size = _detect_distributed_world_size()
    signal = GuardrailSignal(
        is_moe=is_moe,
        quant_format=quant_format,
        distributed_world_size=world_size,
    )
    return check_load_time_guardrails(
        signal,
        hf_path=hf_path or model_name,
        alias=alias,
    )


def snapshot() -> dict[str, int]:
    with _lock:
        return {
            "mxfp4_moe_distributed_warnings_total": (
                _mxfp4_moe_distributed_warnings_total
            ),
            "nvfp4_moe_warnings_total": _nvfp4_moe_warnings_total,
        }


_MXFP4_MOE_HELP = (
    "Load-time warnings fired for the MoE + MXFP4 + multi-device "
    "throughput cliff (upstream mlx#3402). Any non-zero value means an "
    "operator started a model matching the three-tuple; expect "
    "~0.27 tok/s vs ~16 tok/s until upstream lands a fix."
)
_NVFP4_MOE_HELP = (
    "Load-time warnings fired for the MoE + NVFP4 dynamic-range loss "
    "(upstream mlx#2962, signed E4M3 scales instead of Blackwell "
    "unsigned UE4M3 -> ~137x dynamic-range loss). Fires regardless of "
    "device count because the dynamic-range loss bites even on "
    "single-device serving."
)


def render_prometheus_lines() -> list[str]:
    stats = snapshot()
    mxfp4 = int(stats.get("mxfp4_moe_distributed_warnings_total", 0))
    nvfp4 = int(stats.get("nvfp4_moe_warnings_total", 0))
    return [
        f"# HELP fusion_mlx_mxfp4_moe_distributed_warnings_total {_MXFP4_MOE_HELP}",
        "# TYPE fusion_mlx_mxfp4_moe_distributed_warnings_total counter",
        f"fusion_mlx_mxfp4_moe_distributed_warnings_total {mxfp4}",
        f"# HELP fusion_mlx_nvfp4_moe_warnings_total {_NVFP4_MOE_HELP}",
        "# TYPE fusion_mlx_nvfp4_moe_warnings_total counter",
        f"fusion_mlx_nvfp4_moe_warnings_total {nvfp4}",
    ]


def reset_for_tests() -> None:
    global _mxfp4_moe_distributed_warnings_total
    global _nvfp4_moe_warnings_total
    with _lock:
        _mxfp4_moe_distributed_warnings_total = 0
        _nvfp4_moe_warnings_total = 0
