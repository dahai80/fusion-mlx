# SPDX-License-Identifier: Apache-2.0
# DFlash2 model-aware presets — G20 (#0911): Qwen MoE 专属窗口配比/步长优化.
#
# The flat block_size=5 / draft_bits=4 defaults were validated on Qwen3.8-27B
# dense (dflash2-perf-ceiling: ~52 tok/s). Different model families benefit
# from different window/step ratios:
#   - Dense Qwen3.8: block_size=5, draft_bits=4 (validated sweet spot)
#   - Dense Qwen3.5: block_size=4, draft_bits=4 (smaller model, tighter window)
#   - High-quant (8bit+): draft_bits=8 (more draft accuracy, memory allows)
#   - Low-quant (4bit): draft_bits=4 (match target quant for speed)
#
# MoE models are excluded from DFlash2 by the auto_router not_moe constraint
# (correctness: GroupedDynamicCausalConv assumes single-expert activation).
# This preset module does NOT relax that — it tunes the params for eligible
# (dense) models only. MoE models use MTP/ngram instead.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DFlash2Preset:
    block_size: int
    draft_bits: int
    warmup_steps: int = 3
    circuit_breaker_threshold: float = 0.20
    circuit_breaker_window: int = 10
    hybrid: bool = False
    reason: str = ""


_PRESETS: dict[str, DFlash2Preset] = {
    "qwen3_8": DFlash2Preset(
        block_size=5,
        draft_bits=4,
        warmup_steps=3,
        circuit_breaker_threshold=0.20,
        reason="Qwen3.8 dense — validated sweet spot (dflash2-perf-ceiling ~52 tok/s)",
    ),
    "qwen3_5": DFlash2Preset(
        block_size=4,
        draft_bits=4,
        warmup_steps=3,
        circuit_breaker_threshold=0.20,
        reason="Qwen3.5 dense — smaller model benefits from tighter window",
    ),
    "qwen3": DFlash2Preset(
        block_size=4,
        draft_bits=4,
        warmup_steps=2,
        circuit_breaker_threshold=0.25,
        reason="Qwen3 generic — conservative window, faster warmup",
    ),
    "default": DFlash2Preset(
        block_size=5,
        draft_bits=4,
        warmup_steps=3,
        circuit_breaker_threshold=0.20,
        reason="Default — flat block_size=5 (pre-preset behavior)",
    ),
}


def _detect_family(model_name: str, model: Any = None) -> str:
    name_lower = (model_name or "").lower()
    if "qwen3.8" in name_lower or "qwen3_8" in name_lower or "qwen3.8-" in name_lower:
        return "qwen3_8"
    if "qwen3.5" in name_lower or "qwen3_5" in name_lower or "qwen3.5-" in name_lower:
        return "qwen3_5"
    if "qwen3" in name_lower or "qwen-3" in name_lower:
        return "qwen3"
    if model is not None:
        mt = ""
        try:
            mt = getattr(getattr(model, "args", None), "model_type", "") or ""
        except Exception:
            pass
        if "qwen3" in mt:
            if "moe" in mt:
                return "qwen3"
            return "qwen3_8"
    return "default"


def _is_hybrid_model(model: Any) -> bool:
    """Detect hybrid models with non-trimmable recurrent layers (GDN/ArraysCache).

    These models mix `linear_attention` (GDN, ArraysCache — not trimmable)
    with `full_attention` (KVCache — trimmable). On draft rejection, the
    ArraysCache state cannot be trimmed, so rollback must restore a
    pre-verify snapshot AND replay accepted tokens through the full model
    to rebuild GDN state. That replay ≈ verify cost, so spec decode is
    net-negative for hybrid models unless acceptance is near 100%.
    """
    if model is None:
        return False
    try:
        args = getattr(model, "args", None)
        if args is None:
            return False
        mt = getattr(args, "model_type", "") or ""
        if mt in ("qwen3_5", "qwen3_8", "qwen3_next", "gpt_oss", "gpt_oss_text"):
            return True
        tc = getattr(args, "text_config", None)
        if isinstance(tc, dict):
            layer_types = tc.get("layer_types") or []
            if isinstance(layer_types, list) and any(
                str(t) == "linear_attention" for t in layer_types
            ):
                return True
            tc_mt = tc.get("model_type", "")
            if tc_mt in ("qwen3_5_text", "qwen3_8_text", "qwen3_next_text"):
                return True
    except Exception:
        pass
    return False


def _adjust_for_hybrid(preset: DFlash2Preset) -> DFlash2Preset:
    """Hybrid (GDN/ArraysCache) models: raise circuit breaker + shrink block.

    Replay-on-rejection cost ≈ verify cost doubles compute per rejected
    step. Break-even acceptance ≈ 100% at block_size=5. Raising the
    threshold to 0.80 + block_size 2 keeps DFlash2 active only when
    acceptance is near-perfect, minimizing net-negative steps.
    """
    return DFlash2Preset(
        block_size=2,
        draft_bits=preset.draft_bits,
        warmup_steps=preset.warmup_steps,
        circuit_breaker_threshold=0.80,
        circuit_breaker_window=6,
        hybrid=True,
        reason=(
            preset.reason
            + " | HYBRID (GDN/ArraysCache): replay-on-rejection ≈ verify cost → "
            "cb_threshold=0.80, block_size=2 (break-even ≈ 100% acceptance)"
        ),
    )


def _adjust_for_quant(preset: DFlash2Preset, quant_bits: int | None) -> DFlash2Preset:
    if quant_bits is not None and quant_bits >= 8:
        if preset.draft_bits < 8:
            return DFlash2Preset(
                block_size=preset.block_size,
                draft_bits=8,
                warmup_steps=preset.warmup_steps,
                circuit_breaker_threshold=preset.circuit_breaker_threshold,
                circuit_breaker_window=preset.circuit_breaker_window,
                reason=preset.reason + " (8bit target → draft_bits=8 for accuracy)",
            )
    return preset


def resolve_preset(
    model_name: str,
    model: Any = None,
    quant_bits: int | None = None,
) -> DFlash2Preset:
    family = _detect_family(model_name, model)
    preset = _PRESETS.get(family, _PRESETS["default"])
    adjusted = _adjust_for_quant(preset, quant_bits)
    if _is_hybrid_model(model):
        adjusted = _adjust_for_hybrid(adjusted)
        logger.warning(
            "[dflash2.preset] %s is HYBRID (GDN/ArraysCache) — DFlash2 replay-on-"
            "rejection ≈ verify cost. Break-even acceptance ≈ 100%%. For best "
            "speedup use a DENSE model (e.g. Qwen3-8B-4bit) where rollback is "
            "O(1) KVCache.trim (no replay). Adjusted: block=2, cb=0.80.",
            model_name,
        )
    logger.info(
        "[dflash2.preset] family=%s block_size=%d draft_bits=%d warmup=%d "
        "cb_threshold=%.2f quant_bits=%s reason='%s'",
        family,
        adjusted.block_size,
        adjusted.draft_bits,
        adjusted.warmup_steps,
        adjusted.circuit_breaker_threshold,
        quant_bits,
        adjusted.reason,
    )
    return adjusted
