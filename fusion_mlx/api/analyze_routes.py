# SPDX-License-Identifier: Apache-2.0
"""Model structure analysis API for fusion-mlx.

Provides /v1/analyze endpoint for parsing model structure including
architecture type, parameter counts, layer types, and special ops.
Uses existing model_auto_config.py capabilities.

Issue: dahai80/fusion-mlx#234
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..middleware.auth import verify_api_key
from ..model_auto_config import detect_model_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["analyze"])

_ARCH_PATTERNS = [
    (re.compile(r"(dit|diT|DiT)", re.I), "DiT"),
    (re.compile(r"(unet|u_net|up_block|down_block)", re.I), "UNet"),
    (re.compile(r"(vae|variational_autoenc)", re.I), "VAE"),
    (
        re.compile(
            r"(llama|qwen|mistral|gemma|phi|deepseek|yi|chatglm|solar|internlm)", re.I
        ),
        "Transformer",
    ),
]

_LAYER_PATTERNS = {
    "DiTBlock": re.compile(r"(dit_block|dit\.blocks|transformer_blocks)", re.I),
    "AdaLN": re.compile(r"(adaln|ada_ln|adaptive_norm|norm1|norm2)", re.I),
    "MHA": re.compile(r"(self_attn|attention|q_proj|k_proj|v_proj|o_proj)", re.I),
    "GQA": re.compile(r"(num_key_value_heads|grouped_query)", re.I),
    "FFN": re.compile(r"(ffn|feed_forward|mlp|gate_proj|up_proj|down_proj)", re.I),
    "RoPE": re.compile(r"(rope|rotary_emb|rotary)", re.I),
    "PatchEmbed": re.compile(r"(patch_embed|pos_embed)", re.I),
    "Norm": re.compile(r"(norm|rmsnorm|layernorm|ln_)", re.I),
    "Embedding": re.compile(r"(embed_tokens|wte|word_embedding)", re.I),
    "LMHead": re.compile(r"(lm_head|output_head|embed_out)", re.I),
}

_SPECIAL_OPS_PATTERNS = [
    (re.compile(r"(mamba|ssm|gated_delta|rwkv)", re.I), "hybrid-attention (Mamba/SSM)"),
    (
        re.compile(r"(moe|expert|gate_proj.*shared_expert)", re.I),
        "MoE (Mixture of Experts)",
    ),
    (
        re.compile(r"(cross_attention|encoder_attn)", re.I),
        "cross-attention (encoder-decoder)",
    ),
    (
        re.compile(r"(visual|vision_tower|image_encoder|patch_embedding)", re.I),
        "vision encoder (multimodal)",
    ),
]


class AnalyzeRequest(BaseModel):
    model_path: str | None = Field(None, description="Local model path on disk")
    hf_repo: str | None = Field(
        None, description="HuggingFace repo (org/name) for online models"
    )


class AnalyzeResponse(BaseModel):
    model_id: str
    architecture: str
    params_total: int
    params_by_layer: dict[str, int]
    layer_types: list[str]
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    special_ops: list[str]
    safetensors_files: list[str]
    config_json: dict[str, Any]


def _detect_architecture(model_str: str, config: dict) -> str:
    model_type = config.get("model_type", "")
    arch = config.get("architectures", [])

    if arch:
        arch_str = " ".join(arch) if isinstance(arch, list) else str(arch)
        for pat, label in _ARCH_PATTERNS:
            if pat.search(arch_str):
                return label
        if arch:
            return arch[0] if isinstance(arch, list) else str(arch)

    if model_type:
        for pat, label in _ARCH_PATTERNS:
            if pat.search(model_type):
                return label

    for pat, label in _ARCH_PATTERNS:
        if pat.search(model_str):
            return label

    return "Unknown"


def _detect_layer_types(tensor_keys: list[str]) -> list[str]:
    found = set()
    for name in _LAYER_PATTERNS:
        pat = _LAYER_PATTERNS[name]
        if any(pat.search(k) for k in tensor_keys):
            found.add(name)
    return sorted(found)


def _detect_special_ops(tensor_keys: list[str], config: dict) -> list[str]:
    found = set()
    all_str = " ".join(tensor_keys) + " " + json.dumps(config)
    for pat, label in _SPECIAL_OPS_PATTERNS:
        if pat.search(all_str):
            found.add(label)
    return sorted(found)


def _count_params_by_layer(
    tensor_keys: list[str], shapes: dict[str, list[int]]
) -> dict[str, int]:
    result: dict[str, str] = {}
    total_by_type = {
        "attention": 0,
        "ffn": 0,
        "embedding": 0,
        "norm": 0,
        "other": 0,
    }

    attn_pat = re.compile(r"(q_proj|k_proj|v_proj|o_proj|self_attn|attention)", re.I)
    ffn_pat = re.compile(r"(ffn|feed_forward|mlp|gate_proj|up_proj|down_proj)", re.I)
    embed_pat = re.compile(r"(embed|wte|word_embedding|lm_head|output)", re.I)
    norm_pat = re.compile(r"(norm|rmsnorm|layernorm|ln_)", re.I)

    for key in tensor_keys:
        size = 1
        if key in shapes:
            for dim in shapes[key]:
                size *= dim

        if attn_pat.search(key):
            total_by_type["attention"] += size
        elif ffn_pat.search(key):
            total_by_type["ffn"] += size
        elif embed_pat.search(key):
            total_by_type["embedding"] += size
        elif norm_pat.search(key):
            total_by_type["norm"] += size
        else:
            total_by_type["other"] += size

    return total_by_type


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_model(
    req: AnalyzeRequest, _auth: bool = Depends(verify_api_key)
) -> Any:
    if not req.model_path and not req.hf_repo:
        raise HTTPException(400, detail="Either model_path or hf_repo is required")

    model_id = req.model_path or req.hf_repo or "unknown"

    config: dict[str, Any] = {}
    tensor_keys: list[str] = []
    shapes: dict[str, list[int]] = {}
    safetensors_files: list[str] = []

    model_dir = None
    if req.model_path:
        model_dir = Path(req.model_path)
    elif req.hf_repo:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        repo_dir = cache_dir / f"models--{req.hf_repo.replace('/', '--')}"
        if repo_dir.exists():
            snapshot_dir = repo_dir / "snapshots"
            if snapshot_dir.exists():
                snapshots = sorted(snapshot_dir.iterdir(), reverse=True)
                if snapshots:
                    model_dir = snapshots[0]

    if model_dir and model_dir.exists():
        config_file = model_dir / "config.json"
        if config_file.exists():
            try:
                with open(config_file, encoding="utf-8") as f:
                    config = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read config.json: %s", e)

        for sf in model_dir.glob("*.safetensors"):
            safetensors_files.append(sf.name)
            try:
                from mlx.core import load_metadata

                meta = load_metadata(str(sf))
                for key, val in meta.items():
                    tensor_keys.append(key)
                    if isinstance(val, dict) and "shape" in val:
                        shapes[key] = val["shape"]
            except Exception:
                pass

    try:
        model_cfg = detect_model_config(model_id)
        if model_cfg and model_cfg.is_hybrid:
            if "hybrid-attention" not in _detect_special_ops(tensor_keys, config):
                pass
    except Exception:
        pass

    architecture = _detect_architecture(model_id, config)
    layer_types = _detect_layer_types(tensor_keys) if tensor_keys else []
    special_ops = _detect_special_ops(tensor_keys, config)

    num_layers = config.get("num_hidden_layers", config.get("n_layer", 0))
    hidden_size = config.get("hidden_size", config.get("n_embd", 0))
    num_heads = config.get("num_attention_heads", config.get("n_head", 0))

    params_by_layer = (
        _count_params_by_layer(tensor_keys, shapes)
        if tensor_keys
        else {
            "attention": 0,
            "ffn": 0,
            "embedding": 0,
            "norm": 0,
            "other": 0,
        }
    )
    params_total = sum(params_by_layer.values())

    return AnalyzeResponse(
        model_id=model_id,
        architecture=architecture,
        params_total=params_total,
        params_by_layer=params_by_layer,
        layer_types=layer_types,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        special_ops=special_ops,
        safetensors_files=safetensors_files,
        config_json=config,
    )
