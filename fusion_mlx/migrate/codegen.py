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
import re
from dataclasses import dataclass, field

from .architectures import ArchTemplate

logger = logging.getLogger(__name__)


_VALID_TEMPLATE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _validate_template_name(name: str) -> str:
    if not name or not _VALID_TEMPLATE_NAME.match(name):
        raise ValueError(
            f"Invalid template name {name!r}: must be a Python identifier "
            f"(letters/digits/underscore, no quotes, newlines, or shell metachars)"
        )
    return name


@dataclass
class CodegenResult:
    output_path: str
    files_generated: list[str] = field(default_factory=list)
    error: str | None = None


def _build_model_args(template: ArchTemplate, config: dict) -> str:
    _validate_template_name(template.name)
    defaults = {
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
    rope_theta = config.get("rope_theta", 10000.0)
    max_pos = config.get("max_position_embeddings", 32768)

    lines = [
        "from dataclasses import dataclass",
        "from typing import Optional",
        "",
        "import mlx.core as mx",
        "import mlx.nn as nn",
        "from mlx_lm.models.base import BaseModelArgs",
        "from mlx_lm.models.llama import initialize_rope, scaled_dot_product_attention, create_attention_mask",
        "",
        "",
        "@dataclass",
        "class ModelArgs(BaseModelArgs):",
        f'    model_type: str = "{template.name}"',
    ]
    for k, v in defaults.items():
        if isinstance(v, bool):
            lines.append(f"    {k}: bool = {v}")
        elif isinstance(v, float):
            lines.append(f"    {k}: float = {v}")
        elif isinstance(v, int):
            lines.append(f"    {k}: int = {v}")
    lines.append(f"    rope_theta: float = {rope_theta}")
    lines.append("    rope_traditional: bool = False")
    lines.append("    rope_scaling: Optional[dict] = None")
    lines.append(f"    max_position_embeddings: int = {max_pos}")
    if template.has_bias:
        lines.append("    bias: bool = True")
    if template.has_mlp_bias:
        lines.append("    mlp_bias: bool = True")
    return "\n".join(lines)


def _build_attention_class(template: ArchTemplate) -> str:
    attn_bias = "True" if template.has_bias else "False"
    code = f"""

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(args.hidden_size, self.n_heads * self.head_dim, bias={attn_bias})
        self.k_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias={attn_bias})
        self.v_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias={attn_bias})
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias={attn_bias})
        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)"""
    return code


def _build_mlp_class(template: ArchTemplate) -> str:
    bias = "True" if template.has_mlp_bias else "False"
    code = f"""

class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias={bias})
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias={bias})
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias={bias})

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))"""
    return code


def _build_transformer_block(template: ArchTemplate) -> str:
    code = """

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x, mask=None, cache=None):
        h = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x"""
    return code


def _build_model_class(template: ArchTemplate, config: dict) -> str:
    tie = config.get("tie_word_embeddings", False)
    lm_head_bias = "True" if template.has_bias else "False"
    if tie:
        lm_head_section = ""
        call_section = """        if self.args.tie_word_embeddings:
            return self.embed_tokens.as_linear(h)
        return self.lm_head(h)"""
    else:
        lm_head_section = f"        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias={lm_head_bias})"
        call_section = "        return self.lm_head(h)"

    code = f"""

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
{lm_head_section}

    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for i, layer in enumerate(self.layers):
            h = layer(h, mask, cache=cache[i])
        h = self.norm(h)
{call_section}"""
    return code


def generate_model_code(
    template: ArchTemplate,
    config: dict,
    output_dir: str,
) -> CodegenResult:
    result = CodegenResult(output_path=output_dir)

    try:
        os.makedirs(output_dir, exist_ok=True)

        model_args = _build_model_args(template, config)
        attention = _build_attention_class(template)
        mlp = _build_mlp_class(template)
        block = _build_transformer_block(template)
        model = _build_model_class(template, config)

        model_py = model_args + attention + mlp + block + model + "\n"
        model_path = os.path.join(output_dir, f"{template.name}.py")
        with open(model_path, "w") as f:
            f.write(model_py)
        result.files_generated.append(model_path)
        logger.info("Generated model code: %s", model_path)

        mlx_config = {
            "model_type": template.name,
            "model_file": f"{template.name}.py",
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
        if config.get("max_position_embeddings"):
            mlx_config["max_position_embeddings"] = config["max_position_embeddings"]
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
    _validate_template_name(template.name)
    files = [
        f"{template.name}.py",
        "config.json",
        "model.safetensors",
    ]
    for tok in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"):
        files.append(tok)
    return files
