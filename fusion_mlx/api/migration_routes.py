# SPDX-License-Identifier: Apache-2.0
"""Model adaptation level assessment API for fusion-mlx.

Provides /v1/migration-level endpoint that evaluates how well a model
adapts to the MLX runtime on Apple Silicon. Returns L0~L4 level with
migration cost and component analysis.

Issue: dahai80/fusion-mlx#230
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..middleware.auth import verify_api_key

from ..model_aliases import list_aliases, resolve_model
from ..model_auto_config import detect_model_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["migration"])


class MigrationLevel(str, Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"


class MigrationCost(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    extreme = "extreme"


class CompileStrategy(str, Enum):
    block_loop = "block+loop"
    full = "full"


_LEVEL_DESCRIPTIONS = {
    MigrationLevel.L0: "Natively supported — model is in the alias registry, no conversion needed",
    MigrationLevel.L1: "Component-level auto-migration — standard DiT/UNet/Transformer components detected",
    MigrationLevel.L2: "Minor code generation needed — some ops require custom MLX implementation",
    MigrationLevel.L3: "Manual adaptation required — critical ops missing, needs expert work",
    MigrationLevel.L4: "Not supported — architecture incompatible or core dependencies missing",
}

_COST_MAP = {
    MigrationLevel.L0: MigrationCost.low,
    MigrationLevel.L1: MigrationCost.low,
    MigrationLevel.L2: MigrationCost.medium,
    MigrationLevel.L3: MigrationCost.high,
    MigrationLevel.L4: MigrationCost.extreme,
}

_STANDARD_COMPONENTS = {
    "DiT": re.compile(r"(dit|diT|DiT)", re.I),
    "AdaLN": re.compile(r"(adaln|ada_ln|adaptive_norm)", re.I),
    "MHA": re.compile(r"(self_attn|attention|q_proj|k_proj|v_proj)", re.I),
    "GQA": re.compile(r"(gqa|grouped_query|num_key_value_heads)", re.I),
    "FFN": re.compile(r"(ffn|feed_forward|mlp|gate_proj|up_proj|down_proj)", re.I),
    "RoPE": re.compile(r"(rope|rotary_emb|rotary)", re.I),
    "PatchEmbed": re.compile(r"(patch_embed|patch_embed)", re.I),
    "UNet": re.compile(r"(unet|u_net|up_block|down_block)", re.I),
    "VAE": re.compile(r"(vae|variational_autoenc)", re.I),
    "Norm": re.compile(r"(norm|rmsnorm|layernorm|ln_)", re.I),
}

_HYBRID_OPS = re.compile(r"(mamba|ssm|gated_delta|rwkv|liquid)", re.I)
_RARE_OPS = re.compile(r"(custom_op|fused_moe|expert|moe_gate)", re.I)


class MigrationLevelRequest(BaseModel):
    model_id: str = Field(..., description="Model identifier or alias")
    hf_repo: str | None = Field(None, description="HuggingFace repo (org/name)")
    source_format: str | None = Field(
        None, description="Source format: pytorch|safetensors|gguf"
    )


class MigrationLevelResponse(BaseModel):
    model_id: str
    level: MigrationLevel
    level_desc: str
    migration_cost: MigrationCost
    components_matched: list[str]
    missing_ops: list[str]
    compile_strategy: CompileStrategy
    warnings: list[str]


def _assess_level(
    model_id: str, hf_repo: str | None
) -> tuple[MigrationLevel, list[str], list[str], list[str]]:
    aliases = list_aliases()
    if model_id in aliases:
        return MigrationLevel.L0, list(_STANDARD_COMPONENTS.keys()), [], []

    try:
        resolved = resolve_model(model_id)
        if resolved and resolved != model_id:
            logger.info(
                "model %s resolved to %s, checking alias list", model_id, resolved
            )
            if resolved in aliases:
                return MigrationLevel.L0, list(_STANDARD_COMPONENTS.keys()), [], []
    except Exception:
        pass

    try:
        cfg = detect_model_config(model_id)
    except Exception:
        cfg = None

    if cfg is not None:
        model_str = str(cfg)
        matched = [
            name for name, pat in _STANDARD_COMPONENTS.items() if pat.search(model_str)
        ]
        missing = []
        warnings = []

        if _HYBRID_OPS.search(model_str):
            missing.append("hybrid-attention ops (Mamba/SSM)")
            warnings.append(
                "Model uses hybrid attention; spec decoding and some optimizations disabled"
            )

        if _RARE_OPS.search(model_str):
            missing.append("MoE/rare ops")
            warnings.append("MoE or custom ops detected; may need manual tuning")

        if cfg.is_hybrid:
            warnings.append("Hybrid model detected; throughput may be limited")

        if not missing:
            if matched:
                return MigrationLevel.L1, matched, missing, warnings
            else:
                return MigrationLevel.L2, matched, ["unknown structure"], warnings
        elif len(missing) <= 1:
            return MigrationLevel.L2, matched, missing, warnings
        else:
            return MigrationLevel.L3, matched, missing, warnings

    return (
        MigrationLevel.L4,
        [],
        ["unrecognized architecture"],
        ["Cannot analyze model structure; manual assessment required"],
    )


def _compile_strategy(level: MigrationLevel) -> CompileStrategy:
    if level in (MigrationLevel.L0, MigrationLevel.L1, MigrationLevel.L2):
        return CompileStrategy.block_loop
    return CompileStrategy.full


@router.post("/migration-level", response_model=MigrationLevelResponse)
async def assess_migration_level(req: MigrationLevelRequest, _auth: bool = Depends(verify_api_key)) -> Any:
    level, matched, missing, warnings = _assess_level(req.model_id, req.hf_repo)
    strategy = _compile_strategy(level)

    return MigrationLevelResponse(
        model_id=req.model_id,
        level=level,
        level_desc=_LEVEL_DESCRIPTIONS[level],
        migration_cost=_COST_MAP[level],
        components_matched=matched,
        missing_ops=missing,
        compile_strategy=strategy,
        warnings=warnings,
    )
