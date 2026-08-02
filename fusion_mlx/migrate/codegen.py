# SPDX-License-Identifier: Apache-2.0
"""MLX model class code generation — produces Python source from architecture template.

Callers: fusion_mlx.admin.migrate_route
API: generate_model_code(template, config, output_dir) -> CodegenResult
     list_model_files(template, config) -> list[str]
Schema: CodegenResult(dataclass) — output_path, files_generated, error
User instruction verbatim: "做一个端到端的功能，做模型迁移和量化的功能，以openpangu为例，把迁移的每个步骤展现在GUI上"
"""

import json
import logging
import os
from dataclasses import dataclass, field

from .architectures import ArchTemplate

logger = logging.getLogger(__name__)


@dataclass
class CodegenResult:
    output_path: str
    files_generated: list[str] = field(default_factory=list)
    error: str | None = None


def _build_imports(template: ArchTemplate) -> str:
    lines = [
        "import math",
        "",
        "import mlx.core as mx",
        "import mlx.nn as nn",
        "",
        "from mlx.utils import checkpoint",
    ]
    if template.has_qkv_bias:
        lines.append("")
        lines.append("from typing import Optional")
    return "\n".join(lines)


def _build_attention_class(template: ArchTemplate, config: dict) -> str:
    bias_lines = ""
    if template.has_bias:
        bias_lines = """
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=True)"""
    else:
        bias_lines = """
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)"""

    code = f"""

class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim ** -0.5
{bias_lines}

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.reshape(B, L, self.num_attention_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        new_cache = (k, v)
        if self.num_key_value_heads < self.num_attention_heads:
            reps = self.num_attention_heads // self.num_key_value_heads
            k = mx.repeat(k, reps, axis=1)
            v = mx.repeat(v, reps, axis=1)
        attn = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        if mask is not None:
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out), new_cache"""
    return code


def _build_mlp_class(template: ArchTemplate) -> str:
    bias = "True" if template.has_mlp_bias else "False"
    activation = (
        "nn.silu" if template.activation == "silu" else f"nn.{template.activation}"
    )

    code = f"""

class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias={bias})
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias={bias})
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias={bias})

    def __call__(self, x):
        return self.down_proj({activation}(self.gate_proj(x)) * self.up_proj(x))"""
    return code


def _build_transformer_block(template: ArchTemplate) -> str:
    code = """

class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x, mask=None, cache=None):
        h, new_cache = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_cache"""
    return code


def _build_model_class(template: ArchTemplate, config: dict) -> str:
    tie = config.get("tie_word_embeddings", False)
    tie_comment = "  # tied with embed_tokens" if tie else ""

    if tie:
        lm_head_line = "        self.lm_head = self.embed_tokens"
    elif template.has_bias:
        lm_head_line = "        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=True)"
    else:
        lm_head_line = "        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)"
    lm_head_line += tie_comment

    code = f"""

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
{lm_head_line}

    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)
        new_caches = []
        for i, layer in enumerate(self.layers):
            c = cache[i] if cache is not None else None
            h, new_cache = layer(h, mask=mask, cache=c)
            new_caches.append(new_cache)
        return self.lm_head(self.norm(h)), new_caches"""
    return code


def generate_model_code(
    template: ArchTemplate,
    config: dict,
    output_dir: str,
) -> CodegenResult:
    result = CodegenResult(output_path=output_dir)

    try:
        os.makedirs(output_dir, exist_ok=True)

        imports = _build_imports(template)
        attention = _build_attention_class(template, config)
        mlp = _build_mlp_class(template)
        block = _build_transformer_block(template)
        model = _build_model_class(template, config)

        model_py = imports + attention + mlp + block + model + "\n"
        model_path = os.path.join(output_dir, f"{template.name}.py")
        with open(model_path, "w") as f:
            f.write(model_py)
        result.files_generated.append(model_path)
        logger.info("Generated model code: %s", model_path)

        mlx_config = {
            "model_type": template.name,
            "num_hidden_layers": config.get("num_hidden_layers", 0),
            "hidden_size": config.get("hidden_size", 0),
            "intermediate_size": config.get("intermediate_size", 0),
            "num_attention_heads": config.get("num_attention_heads", 0),
            "num_key_value_heads": config.get(
                "num_key_value_heads", config.get("num_attention_heads", 0)
            ),
            "rms_norm_eps": config.get("rms_norm_eps", 1e-6),
            "vocab_size": config.get("vocab_size", 0),
            "tie_word_embeddings": config.get("tie_word_embeddings", False),
        }
        if config.get("rope_theta"):
            mlx_config["rope_theta"] = config["rope_theta"]
        if template.has_bias:
            mlx_config["bias"] = True
        if template.has_mlp_bias:
            mlx_config["mlp_bias"] = True

        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(mlx_config, f, indent=2)
        result.files_generated.append(config_path)
        logger.info("Generated config: %s", config_path)

    except Exception as e:
        logger.exception("Codegen failed: %s", e)
        result.error = str(e)

    return result


def list_model_files(template: ArchTemplate, config: dict) -> list[str]:
    files = [
        f"{template.name}.py",
        "config.json",
        "weights.npz",
    ]
    for tok in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"):
        files.append(tok)
    return files
