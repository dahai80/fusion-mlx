"""Convert released DSpark HF checkpoints into MLX-ready model directories.

The released DSpark draft checkpoints (e.g. ``deepseek-ai/dspark_qwen3_4b_block7``)
store their tensors with names that match the reference PyTorch module tree
directly (no ``model.`` prefix):

    embed_tokens.weight
    layers.{i}.self_attn.{q,k,v,o}_proj.weight
    layers.{i}.self_attn.{q,k}_norm.weight
    layers.{i}.mlp.{gate,up,down}_proj.weight
    layers.{i}.{input_layernorm,post_attention_layernorm}.weight
    norm.weight
    fc.weight                          # [hidden, len(target_layer_ids) * hidden]
    hidden_norm.weight
    lm_head.weight
    markov_head.markov_w1.weight       # nn.Embedding  [vocab, markov_rank]
    markov_head.markov_w2.weight       # nn.Linear     [vocab, markov_rank]
    confidence_head.proj.weight        # nn.Linear     [1, hidden (+ markov_rank)]
    confidence_head.proj.bias          # [1]

Both PyTorch and MLX ``nn.Linear`` store weights as ``[out_features, in_features]``
and both embeddings as ``[num_embeddings, dims]``, so the mapping into the MLX
module tree is the identity — the converter's job is validation, config schema
translation (``dspark_config`` section, flattened rope settings), and the
embed/lm_head weight audit against the runtime target checkpoint.

The MLX draft model (U3) must therefore name its modules to match the tree
above exactly (``layers``, ``fc``, ``hidden_norm``, ``norm``, ``embed_tokens``,
``lm_head``, ``markov_head.markov_w1/markov_w2``, ``confidence_head.proj``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .draft import resolve_model_path

SUPPORTED_MARKOV_HEAD_TYPES = ("vanilla",)
WEIGHTS_FILENAME = "model.safetensors"
AUDIT_FILENAME = "audit.json"
CONFIG_FILENAME = "config.json"
DSPARK_SCHEMA_VERSION = 1

# Draft-side tensors that may be shared with the runtime target model.
EMBEDDING_TENSORS = ("embed_tokens.weight", "lm_head.weight")


class ConversionError(RuntimeError):
    """Raised when a checkpoint cannot be converted safely."""


def load_config(model_path: Path) -> dict[str, Any]:
    config_file = model_path / CONFIG_FILENAME
    if not config_file.exists():
        raise ConversionError(f"No {CONFIG_FILENAME} found in {model_path}")
    return json.loads(config_file.read_text())


def resolve_rope(config: dict[str, Any]) -> tuple[float, dict | None]:
    """Resolve rope settings, handling the transformers 5.x nested layout.

    transformers 5.x nests rope config under ``rope_parameters`` (the released
    DSpark configs do this); older configs keep ``rope_theta`` at the top level.
    """
    rope_params = config.get("rope_parameters") or {}
    rope_theta = rope_params.get("rope_theta", config.get("rope_theta"))
    if rope_theta is None:
        raise ConversionError(
            "Could not resolve rope_theta: neither rope_parameters.rope_theta "
            "nor top-level rope_theta present in source config."
        )
    rope_type = rope_params.get("rope_type", "default")
    rope_scaling = config.get("rope_scaling")
    if rope_scaling is None and rope_type not in (None, "default"):
        # Non-default rope (e.g. yarn): propagate the full parameter dict so
        # mlx-lm's initialize_rope sees the scaling config.
        rope_scaling = {k: v for k, v in rope_params.items() if k != "rope_theta"}
    return float(rope_theta), rope_scaling


def validate_dspark_config(config: dict[str, Any]) -> None:
    required = (
        "block_size",
        "target_layer_ids",
        "markov_rank",
        "mask_token_id",
        "enable_confidence_head",
        "num_anchors",
        "num_target_layers",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ConversionError(
            f"Source config is missing required DSpark fields: {missing}. "
            "Is this a DSpark draft checkpoint?"
        )
    markov_rank = int(config["markov_rank"])
    if markov_rank > 0:
        markov_head_type = config.get("markov_head_type")
        if markov_head_type is None:
            raise ConversionError(
                "markov_rank > 0 but markov_head_type is missing from the "
                "source config."
            )
        if str(markov_head_type).lower() not in SUPPORTED_MARKOV_HEAD_TYPES:
            raise ConversionError(
                f"Unsupported markov_head_type {markov_head_type!r}: this "
                f"converter only supports {list(SUPPORTED_MARKOV_HEAD_TYPES)}. "
                "'gated' and 'rnn' heads carry additional weight matrices "
                "(gate_proj / joint_proj) that the MLX runtime does not "
                "implement; converting them would silently drop weights."
            )
    if bool(config["enable_confidence_head"]) and (
        "confidence_head_with_markov" not in config
    ):
        raise ConversionError(
            "enable_confidence_head is true but confidence_head_with_markov "
            "is missing from the source config."
        )


def expected_tensor_shapes(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Full expected tensor tree for a DSpark draft checkpoint, from its config."""
    hidden = int(config["hidden_size"])
    vocab = int(config["vocab_size"])
    n_heads = int(config["num_attention_heads"])
    n_kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config.get("head_dim") or hidden // n_heads)
    intermediate = int(config["intermediate_size"])
    n_layers = int(config["num_hidden_layers"])
    n_ctx_layers = len(config["target_layer_ids"])
    attention_bias = bool(config.get("attention_bias", False))

    shapes: dict[str, tuple[int, ...]] = {
        "embed_tokens.weight": (vocab, hidden),
        "lm_head.weight": (vocab, hidden),
        "fc.weight": (hidden, n_ctx_layers * hidden),
        "hidden_norm.weight": (hidden,),
        "norm.weight": (hidden,),
    }
    for i in range(n_layers):
        prefix = f"layers.{i}."
        shapes[prefix + "self_attn.q_proj.weight"] = (n_heads * head_dim, hidden)
        shapes[prefix + "self_attn.k_proj.weight"] = (n_kv_heads * head_dim, hidden)
        shapes[prefix + "self_attn.v_proj.weight"] = (n_kv_heads * head_dim, hidden)
        shapes[prefix + "self_attn.o_proj.weight"] = (hidden, n_heads * head_dim)
        if attention_bias:
            shapes[prefix + "self_attn.q_proj.bias"] = (n_heads * head_dim,)
            shapes[prefix + "self_attn.k_proj.bias"] = (n_kv_heads * head_dim,)
            shapes[prefix + "self_attn.v_proj.bias"] = (n_kv_heads * head_dim,)
            shapes[prefix + "self_attn.o_proj.bias"] = (hidden,)
        shapes[prefix + "self_attn.q_norm.weight"] = (head_dim,)
        shapes[prefix + "self_attn.k_norm.weight"] = (head_dim,)
        shapes[prefix + "mlp.gate_proj.weight"] = (intermediate, hidden)
        shapes[prefix + "mlp.up_proj.weight"] = (intermediate, hidden)
        shapes[prefix + "mlp.down_proj.weight"] = (hidden, intermediate)
        shapes[prefix + "input_layernorm.weight"] = (hidden,)
        shapes[prefix + "post_attention_layernorm.weight"] = (hidden,)

    markov_rank = int(config["markov_rank"])
    if markov_rank > 0:
        shapes["markov_head.markov_w1.weight"] = (vocab, markov_rank)
        shapes["markov_head.markov_w2.weight"] = (vocab, markov_rank)

    if bool(config["enable_confidence_head"]):
        input_dim = hidden
        if bool(config.get("confidence_head_with_markov", False)):
            input_dim += markov_rank
        shapes["confidence_head.proj.weight"] = (1, input_dim)
        shapes["confidence_head.proj.bias"] = (1,)
    return shapes


def load_source_weights(source_path: Path) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    for weight_file in sorted(source_path.glob("*.safetensors")):
        weights.update(mx.load(str(weight_file)))
    if not weights:
        raise ConversionError(f"No safetensors weights found in {source_path}")
    return weights


def validate_weights(
    weights: dict[str, mx.array], config: dict[str, Any]
) -> dict[str, tuple[int, ...]]:
    expected = expected_tensor_shapes(config)
    missing = sorted(set(expected) - set(weights))
    unexpected = sorted(set(weights) - set(expected))
    if missing or unexpected:
        raise ConversionError(
            "Checkpoint tensor set does not match the DSpark config. "
            f"Missing: {missing or 'none'}. Unexpected: {unexpected or 'none'}."
        )
    mismatched = {
        name: (tuple(weights[name].shape), shape)
        for name, shape in expected.items()
        if tuple(weights[name].shape) != shape
    }
    if mismatched:
        details = ", ".join(
            f"{name}: got {got}, expected {want}"
            for name, (got, want) in mismatched.items()
        )
        raise ConversionError(f"Tensor shape mismatches: {details}")
    return expected


def _tensor_sha256(arr: mx.array) -> str:
    data = np.array(mx.view(arr, mx.uint8), copy=False)
    return hashlib.sha256(data.tobytes()).hexdigest()


def _find_target_tensor(
    target_path: Path, candidates: list[str]
) -> tuple[str | None, mx.array | None]:
    index_file = target_path / "model.safetensors.index.json"
    if index_file.exists():
        weight_map = json.loads(index_file.read_text())["weight_map"]
        for key in candidates:
            if key in weight_map:
                return key, mx.load(str(target_path / weight_map[key]))[key]
        return None, None
    for weight_file in sorted(target_path.glob("*.safetensors")):
        shard = mx.load(str(weight_file))
        for key in candidates:
            if key in shard:
                return key, shard[key]
    return None, None


def audit_embeddings(
    source_weights: dict[str, mx.array],
    target_path: Path,
    target_config: dict[str, Any],
) -> dict[str, Any]:
    """Checksum draft embed_tokens/lm_head against the target checkpoint."""
    target_quantized = "quantization" in target_config
    tied = bool(target_config.get("tie_word_embeddings", False))
    embed_candidates = ["model.embed_tokens.weight", "embed_tokens.weight"]
    lm_head_candidates = ["lm_head.weight", "model.lm_head.weight"]
    if tied:
        # Tied targets ship only the embedding; lm_head compares against it.
        lm_head_candidates += embed_candidates

    candidates = {
        "embed_tokens.weight": embed_candidates,
        "lm_head.weight": lm_head_candidates,
    }
    tensors: dict[str, Any] = {}
    for name in EMBEDDING_TENSORS:
        source = source_weights[name]
        entry: dict[str, Any] = {
            "shape": list(source.shape),
            "dtype": str(source.dtype),
            "source_sha256": _tensor_sha256(source),
            "target_tensor": None,
            "target_sha256": None,
            "comparable": False,
            "identical": False,
            "max_abs_diff": None,
            "note": None,
        }
        if target_quantized:
            entry["note"] = (
                "target checkpoint is quantized; raw comparison skipped and "
                "bf16 draft copies are always materialized"
            )
        else:
            key, target = _find_target_tensor(target_path, candidates[name])
            if target is None:
                entry["note"] = f"no matching tensor in target ({candidates[name]})"
            else:
                entry["target_tensor"] = key
                same_layout = (
                    tuple(target.shape) == tuple(source.shape)
                    and target.dtype == source.dtype
                )
                if not same_layout:
                    entry["note"] = (
                        f"layout mismatch: target {key} has shape "
                        f"{list(target.shape)} dtype {target.dtype}"
                    )
                else:
                    entry["comparable"] = True
                    entry["target_sha256"] = _tensor_sha256(target)
                    entry["identical"] = (
                        entry["source_sha256"] == entry["target_sha256"]
                    )
                    if not entry["identical"]:
                        diff = mx.abs(
                            source.astype(mx.float32) - target.astype(mx.float32)
                        )
                        entry["max_abs_diff"] = float(mx.max(diff))
        tensors[name] = entry

    return {
        "target_quantized": target_quantized,
        "target_tie_word_embeddings": tied,
        "tensors": tensors,
        "all_identical": all(t["identical"] for t in tensors.values()),
    }


def build_output_config(
    source_config: dict[str, Any],
    *,
    rope_theta: float,
    rope_scaling: dict | None,
    reuse_target_embeddings: bool,
    conversion_meta: dict[str, Any],
) -> dict[str, Any]:
    config = dict(source_config)
    config["rope_theta"] = rope_theta
    config["rope_scaling"] = rope_scaling
    markov_rank = int(source_config["markov_rank"])
    config["dspark_config"] = {
        "block_size": int(source_config["block_size"]),
        "target_layer_ids": [int(x) for x in source_config["target_layer_ids"]],
        "markov_rank": markov_rank,
        "markov_head_type": (
            str(source_config["markov_head_type"]) if markov_rank > 0 else None
        ),
        "mask_token_id": int(source_config["mask_token_id"]),
        "enable_confidence_head": bool(source_config["enable_confidence_head"]),
        "confidence_head_with_markov": bool(
            source_config.get("confidence_head_with_markov", False)
        ),
        "num_anchors": int(source_config["num_anchors"]),
        "num_target_layers": int(source_config["num_target_layers"]),
        "reuse_target_embeddings": reuse_target_embeddings,
    }
    config["dspark_conversion"] = conversion_meta
    return config


def default_output_dir(source: str) -> Path:
    name = source.rstrip("/").split("/")[-1]
    return Path("models") / f"{name}-mlx"


def convert_checkpoint(
    source: str,
    target: str,
    output_dir: str | Path | None = None,
    *,
    reuse_target_embeddings: bool = False,
) -> dict[str, Any]:
    """Convert a DSpark HF draft checkpoint into an MLX model directory.

    Returns the audit report (also written to ``<output_dir>/audit.json``).

    Reuse policy: draft embed_tokens/lm_head are always written into the
    output unless ``reuse_target_embeddings`` was requested AND the audit
    proves they are byte-identical to the target's bf16 tensors AND the
    target checkpoint is not quantized (a quantized runtime target must not
    donate its embeddings to the bf16 draft — that would silently change
    draft logits vs the reference). The emitted config records the decision.
    """
    source_path = resolve_model_path(source)
    target_path = resolve_model_path(target)
    source_config = load_config(source_path)
    target_config = load_config(target_path)

    validate_dspark_config(source_config)
    rope_theta, rope_scaling = resolve_rope(source_config)
    weights = load_source_weights(source_path)
    validate_weights(weights, source_config)

    audit = audit_embeddings(weights, target_path, target_config)
    if not reuse_target_embeddings:
        reuse = False
        reuse_reason = "not requested (--reuse-target-embeddings not passed)"
    elif audit["target_quantized"]:
        reuse = False
        reuse_reason = "requested but target is quantized; bf16 copies materialized"
    elif not audit["all_identical"]:
        reuse = False
        reuse_reason = (
            "requested but embed/lm_head differ from target; copies materialized"
        )
    else:
        reuse = True
        reuse_reason = "requested and audit verified identical tensors"
    audit["reuse_target_embeddings"] = reuse
    audit["reuse_reason"] = reuse_reason

    conversion_meta = {
        "schema_version": DSPARK_SCHEMA_VERSION,
        "source": source,
        "target": target,
        "target_quantized": audit["target_quantized"],
        "reuse_reason": reuse_reason,
    }
    output_config = build_output_config(
        source_config,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        reuse_target_embeddings=reuse,
        conversion_meta=conversion_meta,
    )

    output_weights = dict(weights)
    if reuse:
        for name in EMBEDDING_TENSORS:
            output_weights.pop(name)

    out_dir = Path(output_dir) if output_dir is not None else default_output_dir(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(out_dir / WEIGHTS_FILENAME),
        output_weights,
        metadata={"format": "mlx"},
    )
    (out_dir / CONFIG_FILENAME).write_text(json.dumps(output_config, indent=2) + "\n")
    (out_dir / AUDIT_FILENAME).write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def format_audit_summary(audit: dict[str, Any]) -> str:
    lines = ["weight audit (draft vs target):"]
    for name, entry in audit["tensors"].items():
        if entry["comparable"]:
            verdict = (
                "identical"
                if entry["identical"]
                else (f"DIFFERS (max_abs_diff={entry['max_abs_diff']})")
            )
            lines.append(f"  {name} vs {entry['target_tensor']}: {verdict}")
        else:
            lines.append(f"  {name}: not comparable ({entry['note']})")
    lines.append(
        f"  reuse_target_embeddings={audit['reuse_target_embeddings']} "
        f"({audit['reuse_reason']})"
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dspark-metal-convert",
        description=(
            "Convert a released DSpark draft checkpoint into an MLX-ready "
            "model directory, auditing embed/lm_head against the target."
        ),
    )
    parser.add_argument(
        "source",
        help="DSpark draft checkpoint (HF repo id or local path)",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Runtime target model (HF repo id or local path), e.g. "
        "mlx-community/Qwen3-4B-bf16",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory (default: models/<source-name>-mlx)",
    )
    parser.add_argument(
        "--reuse-target-embeddings",
        action="store_true",
        help="Omit embed_tokens/lm_head from the converted weights when the "
        "audit proves they are identical to the (non-quantized) target's",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else default_output_dir(args.source)
    )
    audit = convert_checkpoint(
        args.source,
        args.target,
        output_dir,
        reuse_target_embeddings=args.reuse_target_embeddings,
    )
    print(f"Converted {args.source} -> {output_dir}")
    print(format_audit_summary(audit))
    print(f"Audit report: {output_dir / AUDIT_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
