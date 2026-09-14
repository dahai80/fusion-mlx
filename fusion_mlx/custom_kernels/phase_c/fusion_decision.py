# SPDX-License-Identifier: Apache-2.0
# O5.3 runtime fusion decision layer.
#
# Replaces the manual phase_c three-piece dispatch (gdn / glm_moe_ffn /
# w4a8) with a runtime decision function. An op sequence (list of op
# descriptors) is inspected for fusability — can adjacent ops collapse
# into a single Metal kernel graph? — and a FusionPlan is returned.
#
# The decider is pattern-based (deterministic code, not model-judged):
# each known fusion pattern registers (matcher, materializer) pairs. New
# architectures benefit automatically: register a pattern, the runtime
# dispatch picks it up without hand-wiring each model.

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OpDescriptor:
    # One op in a candidate sequence. op_type is the canonical name
    # (matmul, silu, swiglu, gdn, moe_ffn, rms_norm, ...). attrs carries
    # shape/dtype/quant hints the matcher may need.
    op_type: str
    attrs: dict = field(default_factory=dict)


@dataclass
class FusionPattern:
    # A registered fusion: name, sequence of op_types it collapses, and the
    # materializer that returns the fused callable (or None = not available,
    # fall back to unfused).
    name: str
    op_sequence: tuple[str, ...]
    matcher: Callable[[list[OpDescriptor]], bool] | None = None
    materializer: Callable[[list[OpDescriptor]], object | None] | None = None
    native_required: bool = False

    def matches(self, ops: list[OpDescriptor]) -> bool:
        if len(ops) < len(self.op_sequence):
            return False
        tail = [o.op_type for o in ops[-len(self.op_sequence) :]]
        if tuple(tail) != self.op_sequence:
            return False
        if self.matcher is not None:
            return self.matcher(ops)
        return True


@dataclass
class FusionPlan:
    # Decision result: which pattern (if any) applies, the fused callable
    # (or None to fall back to unfused execution), and a reason for logging.
    pattern_name: str | None
    fused_callable: object | None
    reason: str
    ops_consumed: int = 0


_REGISTRY: list[FusionPattern] = []


def register_pattern(pattern: FusionPattern) -> None:
    _REGISTRY.append(pattern)
    logger.debug("fusion_decision: registered pattern %s", pattern.name)


def _gdn_matcher(ops: list[OpDescriptor]) -> bool:
    # The op_sequence match already validates (square, matmul, add, rsqrt,
    # div). Skip the diagonal approximation (elementwise, no matmul fuse).
    matmul_op = next((o for o in ops if o.op_type == "matmul"), None)
    if matmul_op is None:
        return False
    return matmul_op.attrs.get("diagonal", False) is False


def _moe_ffn_matcher(ops: list[OpDescriptor]) -> bool:
    # moe_ffn: gate matmul + silu/swiglu + up matmul + down matmul, gated.
    # The three-matmul+activation collapse is the glm_moe_ffn_fused kernel.
    has_gate = any(o.op_type == "matmul" and o.attrs.get("role") == "gate" for o in ops)
    has_act = any(o.op_type in ("silu", "swiglu") for o in ops)
    has_down = any(o.op_type == "matmul" and o.attrs.get("role") == "down" for o in ops)
    return has_gate and has_act and has_down


def _w4a8_matcher(ops: list[OpDescriptor]) -> bool:
    last = ops[-1]
    return (
        last.op_type == "matmul"
        and last.attrs.get("weight_quant") in ("q4", "q6", "q8", "nvfp4")
        and last.attrs.get("act_quant") == "int8"
    )


def _gdn_materializer(ops: list[OpDescriptor]) -> object | None:
    try:
        from .fused_gdn import FusedGDN

        return FusedGDN
    except Exception as exc:
        logger.debug("fusion_decision: FusedGDN unavailable: %s", exc)
        return None


def _moe_ffn_materializer(ops: list[OpDescriptor]) -> object | None:
    try:
        from .glm_moe_ffn import is_native_available, moe_ffn_fused

        if not is_native_available():
            logger.debug("fusion_decision: moe_ffn native not built, skip fuse")
            return None
        return moe_ffn_fused
    except Exception as exc:
        logger.debug("fusion_decision: moe_ffn_fused unavailable: %s", exc)
        return None


def _w4a8_materializer(ops: list[OpDescriptor]) -> object | None:
    try:
        from . import w4a8_fused_matmul

        return w4a8_fused_matmul
    except Exception as exc:
        logger.debug("fusion_decision: w4a8 unavailable: %s", exc)
        return None


def _register_builtins() -> None:
    register_pattern(
        FusionPattern(
            name="fused_gdn",
            op_sequence=("square", "matmul", "add", "rsqrt", "div"),
            matcher=_gdn_matcher,
            materializer=_gdn_materializer,
        )
    )
    register_pattern(
        FusionPattern(
            name="glm_moe_ffn_fused",
            op_sequence=("matmul", "silu", "matmul", "matmul"),
            matcher=_moe_ffn_matcher,
            materializer=_moe_ffn_materializer,
            native_required=True,
        )
    )
    register_pattern(
        FusionPattern(
            name="w4a8_fused_matmul",
            op_sequence=("quantize_act", "matmul"),
            matcher=_w4a8_matcher,
            materializer=_w4a8_materializer,
        )
    )


_register_builtins()


def decide_fusion(ops: list[OpDescriptor]) -> FusionPlan:
    # Inspect the op sequence against registered patterns. Returns the FIRST
    # matching pattern whose materializer yields a callable; if the pattern
    # is native_required and the native kernel is absent, falls through to
    # the next candidate. No match -> unfused (fused_callable=None).
    if not ops:
        return FusionPlan(None, None, "empty op sequence")
    for pattern in _REGISTRY:
        if not pattern.matches(ops):
            continue
        if pattern.materializer is None:
            continue
        fused = pattern.materializer(ops)
        if fused is None:
            if pattern.native_required:
                logger.debug(
                    "fusion_decision: %s native unavailable, trying next",
                    pattern.name,
                )
                continue
            return FusionPlan(pattern.name, None, f"{pattern.name} materializer None")
        return FusionPlan(
            pattern.name,
            fused,
            f"matched {pattern.name}",
            ops_consumed=len(pattern.op_sequence),
        )
    return FusionPlan(None, None, "no pattern matched (unfused)")


def registered_patterns() -> list[str]:
    return [p.name for p in _REGISTRY]


__all__ = [
    "OpDescriptor",
    "FusionPattern",
    "FusionPlan",
    "register_pattern",
    "decide_fusion",
    "registered_patterns",
]
