# SPDX-License-Identifier: Apache-2.0
"""Known architecture templates for MLX migration.

Callers: fusion_mlx.migrate.analyzer, fusion_mlx.admin.migrate_route
API: match_template(model_type, config) -> (ArchTemplate, list[str])
Schema: ArchTemplate dataclass, KNOWN_TEMPLATES dict
User instruction: "做一个端到端的功能，做模型迁移和量化的功能"
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ArchTemplate:
    name: str
    display_name: str
    family: str
    description: str
    base_class: str
    has_bias: bool = False
    has_qkv_bias: bool = False
    has_mlp_bias: bool = False
    norm_type: str = "rmsnorm"
    activation: str = "silu"
    positional_encoding: str = "rope"
    mlp_type: str = "swiglu"
    attention_type: str = "gqa"
    tie_word_embeddings: bool = False
    adaptations: list[str] = field(default_factory=list)
    codegen_hints: dict = field(default_factory=dict)


KNOWN_TEMPLATES: dict[str, ArchTemplate] = {
    "llama": ArchTemplate(
        name="llama",
        display_name="LLaMA",
        family="llama",
        description="Standard LLaMA architecture — RMSNorm + RoPE + GQA + SwiGLU, no bias",
        base_class="LlamaModel",
        has_bias=False,
        has_qkv_bias=False,
        has_mlp_bias=False,
        norm_type="rmsnorm",
        activation="silu",
        positional_encoding="rope",
        mlp_type="swiglu",
        attention_type="gqa",
    ),
    "llama_with_bias": ArchTemplate(
        name="llama_with_bias",
        display_name="LLaMA-like with Bias",
        family="llama",
        description="LLaMA architecture with bias in all Linear layers — used by OpenPangu, Yi, etc.",
        base_class="LlamaModel",
        has_bias=True,
        has_qkv_bias=True,
        has_mlp_bias=True,
        norm_type="rmsnorm",
        activation="silu",
        positional_encoding="rope",
        mlp_type="swiglu",
        attention_type="gqa",
        adaptations=[
            "Add bias=True to all nn.Linear layers",
            "Load bias tensors alongside weights",
            "Handle custom RoPE theta values",
        ],
        codegen_hints={"bias": True, "rope_theta_var": True},
    ),
    "qwen2": ArchTemplate(
        name="qwen2",
        display_name="Qwen2",
        family="qwen2",
        description="Qwen2 architecture — similar to LLaMA but with tied embeddings option",
        base_class="Qwen2Model",
        has_bias=False,
        has_qkv_bias=False,
        has_mlp_bias=False,
        norm_type="rmsnorm",
        activation="silu",
        positional_encoding="rope",
        mlp_type="swiglu",
        attention_type="gqa",
        tie_word_embeddings=True,
    ),
    "gemma": ArchTemplate(
        name="gemma",
        display_name="Gemma",
        family="gemma",
        description="Gemma architecture — RMSNorm + RoPE + GQA + GeGLU",
        base_class="GemmaModel",
        has_bias=False,
        norm_type="rmsnorm",
        activation="gelu",
        positional_encoding="rope",
        mlp_type="geglu",
        attention_type="gqa",
    ),
    "mistral": ArchTemplate(
        name="mistral",
        display_name="Mistral",
        family="mistral",
        description="Mistral architecture — RMSNorm + RoPE + GQA + SwiGLU",
        base_class="MistralModel",
        has_bias=False,
        norm_type="rmsnorm",
        activation="silu",
        positional_encoding="rope",
        mlp_type="swiglu",
        attention_type="gqa",
    ),
    "phi3": ArchTemplate(
        name="phi3",
        display_name="Phi-3",
        family="phi3",
        description="Phi-3 architecture — RMSNorm + RoPE + GQA + SwiGLU with partial bias",
        base_class="Phi3Model",
        has_bias=False,
        has_qkv_bias=True,
        has_mlp_bias=False,
        norm_type="rmsnorm",
        activation="silu",
        positional_encoding="rope",
        mlp_type="swiglu",
        attention_type="gqa",
    ),
    "deepseek_v2": ArchTemplate(
        name="deepseek_v2",
        display_name="DeepSeek-V2 (MLA+MoE)",
        family="deepseek",
        description="DeepSeek-V2+ with MLA and MoE — complex, requires custom MLA implementation",
        base_class="DeepseekV2Model",
        has_bias=False,
        norm_type="rmsnorm",
        activation="silu",
        positional_encoding="rope",
        mlp_type="swiglu",
        attention_type="mla",
        adaptations=[
            "Implement Multi-head Latent Attention (MLA)",
            "Implement MoE routing with shared experts",
            "Handle KV compression via LoRA",
        ],
        codegen_hints={"mla": True, "moe": True},
    ),
}


HF_ARCH_TO_TEMPLATE: dict[str, str] = {
    "LlamaForCausalLM": "llama",
    "OpenPanguEmbeddedForCausalLM": "llama_with_bias",
    "PanguEmbeddedForCausalLM": "llama_with_bias",
    "Qwen2ForCausalLM": "qwen2",
    "GemmaForCausalLM": "gemma",
    "Gemma2ForCausalLM": "gemma",
    "MistralForCausalLM": "mistral",
    "Phi3ForCausalLM": "phi3",
    "DeepseekV2ForCausalLM": "deepseek_v2",
    "DeepseekV3ForCausalLM": "deepseek_v2",
}


def match_template(
    model_type: str,
    config: dict,
) -> tuple[ArchTemplate | None, list[str]]:
    template_name = HF_ARCH_TO_TEMPLATE.get(model_type)
    if template_name and template_name in KNOWN_TEMPLATES:
        template = KNOWN_TEMPLATES[template_name]
        diff = _compute_diff(template, config)
        logger.info(
            "Matched %s to template '%s' (diff=%d items)",
            model_type, template_name, len(diff),
        )
        return template, diff

    template = _infer_from_config(config)
    if template:
        diff = _compute_diff(template, config)
        logger.info(
            "Inferred template '%s' for %s (diff=%d items)",
            template.name, model_type, len(diff),
        )
        return template, diff

    logger.warning("No template match for model_type=%s", model_type)
    return None, []


def _infer_from_config(config: dict) -> ArchTemplate | None:
    model_type = config.get("model_type", "").lower()
    arch = config.get("architectures", [""])[0] if config.get("architectures") else ""

    has_bias = config.get("bias", False) or config.get("attention_bias", False)
    has_mlp_bias = config.get("mlp_bias", False)
    norm_type = "rmsnorm" if "RMSNorm" in str(config.get("norm_type", "")) else "layernorm"
    activation = config.get("hidden_act", "silu")
    rope_theta = config.get("rope_theta", 10000.0)
    n_kv_heads = config.get("num_key_value_heads", 0)
    n_heads = config.get("num_attention_heads", 0)
    is_gqa = n_kv_heads > 0 and n_kv_heads < n_heads

    if has_bias or has_mlp_bias:
        return ArchTemplate(
            name="inferred_llama_bias",
            display_name="LLaMA-like with Bias (inferred)",
            family="llama",
            description=f"Auto-detected LLaMA-like architecture with bias from {model_type or arch}",
            base_class="LlamaModel",
            has_bias=has_bias,
            has_qkv_bias=has_bias,
            has_mlp_bias=has_mlp_bias,
            norm_type=norm_type,
            activation=activation,
            positional_encoding="rope",
            mlp_type="swiglu" if activation == "silu" else "geglu",
            attention_type="gqa" if is_gqa else "mha",
            adaptations=[
                f"Add bias=True to Linear layers (detected bias={has_bias}, mlp_bias={has_mlp_bias})",
                f"Custom RoPE theta={rope_theta}" if rope_theta != 10000.0 else "",
            ],
            codegen_hints={"bias": has_bias, "rope_theta_var": rope_theta != 10000.0},
        )

    if is_gqa and activation in ("silu", "gelu"):
        return KNOWN_TEMPLATES.get("llama")

    return None


def _compute_diff(template: ArchTemplate, config: dict) -> list[str]:
    diff = []
    rope_theta = config.get("rope_theta", 10000.0)
    if rope_theta != 10000.0:
        diff.append(f"RoPE theta: {rope_theta} (template default: 10000)")

    vocab_size = config.get("vocab_size", 0)
    if vocab_size > 100000:
        diff.append(f"Large vocab: {vocab_size} (template default: ~32k)")

    if config.get("tie_word_embeddings", False) and not template.tie_word_embeddings:
        diff.append("tie_word_embeddings=True (template default: False)")

    if config.get("bias", False) and not template.has_bias:
        diff.append("bias=True in config (template default: False)")

    if config.get("mlp_bias", False) and not template.has_mlp_bias:
        diff.append("mlp_bias=True (template default: False)")

    n_heads = config.get("num_attention_heads", 0)
    n_kv = config.get("num_key_value_heads", 0)
    if n_kv > 0 and n_kv < n_heads:
        diff.append(f"GQA: {n_heads}Q / {n_kv}KV heads")

    return diff
