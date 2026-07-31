# SPDX-License-Identifier: Apache-2.0
"""Weight name mapping — HF parameter names to MLX names.

Callers: fusion_mlx.migrate.converter
API: build_weight_map(config, template) -> dict[str, str]
     find_orphan_keys(hf_keys, weight_map) -> list[str]
     find_missing_keys(weight_map, hf_keys) -> list[str]
Schema: WeightMapRule dataclass
User instruction: "做一个端到端的功能，做模型迁移和量化的功能"
"""

import logging
from dataclasses import dataclass
from typing import Optional

from .architectures import ArchTemplate

logger = logging.getLogger(__name__)


@dataclass
class WeightMapRule:
    hf_prefix: str
    mlx_prefix: str
    tensors: list[str]

    def apply(self, hf_name: str) -> Optional[str]:
        if hf_name.startswith(self.hf_prefix):
            suffix = hf_name[len(self.hf_prefix):]
            if suffix in self.tensors:
                return self.mlx_prefix + suffix
            for t in self.tensors:
                if suffix == t:
                    return self.mlx_prefix + t
        return None


def _llama_rules(template: ArchTemplate) -> list[WeightMapRule]:
    attn_tensors = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"]
    mlp_tensors = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]

    if template.has_bias:
        attn_tensors += [
            "q_proj.bias", "k_proj.bias", "v_proj.bias", "o_proj.bias",
        ]
    if template.has_mlp_bias:
        mlp_tensors += [
            "gate_proj.bias", "up_proj.bias", "down_proj.bias",
        ]

    return [
        WeightMapRule(
            hf_prefix="model.embed_tokens.",
            mlx_prefix="embed_tokens.",
            tensors=["weight"],
        ),
        WeightMapRule(
            hf_prefix="model.norm.",
            mlx_prefix="norm.",
            tensors=["weight"],
        ),
        WeightMapRule(
            hf_prefix="lm_head.",
            mlx_prefix="lm_head.",
            tensors=["weight"] + (["bias"] if template.has_bias else []),
        ),
        WeightMapRule(
            hf_prefix="model.layers.{i}.input_layernorm.",
            mlx_prefix="layers.{i}.input_layernorm.",
            tensors=["weight"],
        ),
        WeightMapRule(
            hf_prefix="model.layers.{i}.post_attention_layernorm.",
            mlx_prefix="layers.{i}.post_attention_layernorm.",
            tensors=["weight"],
        ),
        WeightMapRule(
            hf_prefix="model.layers.{i}.self_attn.",
            mlx_prefix="layers.{i}.self_attn.",
            tensors=attn_tensors,
        ),
        WeightMapRule(
            hf_prefix="model.layers.{i}.mlp.",
            mlx_prefix="layers.{i}.mlp.",
            tensors=mlp_tensors,
        ),
    ]


def _expand_layer_rules(rules: list[WeightMapRule], num_layers: int) -> list[WeightMapRule]:
    expanded = []
    for rule in rules:
        if "{i}" in rule.hf_prefix:
            for i in range(num_layers):
                expanded.append(WeightMapRule(
                    hf_prefix=rule.hf_prefix.replace("{i}", str(i)),
                    mlx_prefix=rule.mlx_prefix.replace("{i}", str(i)),
                    tensors=rule.tensors,
                ))
        else:
            expanded.append(rule)
    return expanded


def build_weight_map(
    config: dict,
    template: ArchTemplate,
) -> dict[str, str]:
    num_layers = config.get("num_hidden_layers", config.get("n_layer", 0))

    family = template.family
    if family in ("llama", "qwen2", "gemma", "mistral", "phi3", "deepseek"):
        rules = _llama_rules(template)
    else:
        logger.warning("Unknown family '%s', falling back to LLaMA rules", family)
        rules = _llama_rules(template)

    expanded = _expand_layer_rules(rules, num_layers)

    weight_map = {}
    for rule in expanded:
        for tensor in rule.tensors:
            hf_name = rule.hf_prefix + tensor
            mlx_name = rule.mlx_prefix + tensor
            weight_map[hf_name] = mlx_name

    logger.info(
        "Built weight map: %d entries for family=%s layers=%d bias=%s",
        len(weight_map), family, num_layers, template.has_bias,
    )
    return weight_map


def find_orphan_keys(
    hf_keys: list[str],
    weight_map: dict[str, str],
) -> list[str]:
    mapped = set(weight_map.keys())
    orphans = [k for k in hf_keys if k not in mapped]
    if orphans:
        logger.warning("Found %d orphan HF keys: %s", len(orphans), orphans[:10])
    return orphans


def find_missing_keys(
    weight_map: dict[str, str],
    hf_keys: list[str],
) -> list[str]:
    hf_set = set(hf_keys)
    missing = [hf for hf, mlx in weight_map.items() if hf not in hf_set]
    if missing:
        logger.warning("Found %d missing HF keys: %s", len(missing), missing[:10])
    return missing
