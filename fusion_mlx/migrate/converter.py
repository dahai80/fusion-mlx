# SPDX-License-Identifier: Apache-2.0
"""Weight conversion orchestration — HF safetensors to MLX format.

Callers: fusion_mlx.admin.migrate_route
API: convert_model(hf_dir, output_dir, config, template, ...) -> ConvertResult
Schema: ConvertResult(dataclass) — output_dir, num_weights, total_params_b, orphans, missing, error
User instruction verbatim: "做一个端到端的功能，做模型迁移和量化的功能，以openpangu为例，把迁移的每个步骤展现在GUI上"
"""

import json
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from .architectures import ArchTemplate
from .weight_mapper import build_weight_map, find_missing_keys, find_orphan_keys

logger = logging.getLogger(__name__)


@dataclass
class ConvertResult:
    output_dir: str
    num_weights: int = 0
    total_params_b: float = 0.0
    orphans: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    error: str | None = None


def _load_hf_weights(hf_dir: str) -> dict[str, mx.array]:
    weights = {}
    for fname in sorted(Path(hf_dir).glob("*.safetensors")):
        logger.info("Loading %s", fname.name)
        f = safe_open(str(fname), framework="pt")
        import torch

        for key in f.keys():  # noqa: SIM118 - safe_open is not iterable
            t = f.get_tensor(key)
            if isinstance(t, torch.Tensor):
                if t.dtype == torch.bfloat16:
                    t = t.float()
                arr = t.numpy()
            else:
                arr = np.asarray(t, dtype=np.float32)
            weights[key] = mx.array(arr)
            del t
        del f
    logger.info("Loaded %d tensors from %s", len(weights), hf_dir)
    return weights


def _save_weights_safetensors(weights: dict[str, mx.array], output_dir: str):
    from safetensors.numpy import save_file

    np_weights = {}
    for name, tensor in weights.items():
        arr = np.array(tensor)
        if arr.dtype == np.float16:
            arr = arr.astype(np.float32)
        np_weights[name] = arr
    save_file(np_weights, os.path.join(output_dir, "model.safetensors"))
    del np_weights

    # fsync the weight file so a crash after save does not leave a renamed
    # but empty/partial model.safetensors that mlx_lm.load silently accepts
    # as a valid model (#811 P0).
    try:
        st_path = os.path.join(output_dir, "model.safetensors")
        with open(st_path, "rb") as f:
            os.fsync(f.fileno())
    except OSError as e:
        logger.warning("fsync of safetensors failed (non-fatal): %s", e)


def _remap_weights(
    hf_weights: dict[str, mx.array],
    weight_map: dict[str, str],
) -> dict[str, mx.array]:
    mlx_weights = {}
    for hf_name, mlx_name in weight_map.items():
        if hf_name in hf_weights:
            mlx_weights[mlx_name] = hf_weights[hf_name]
        else:
            logger.warning("Mapped key not found in HF weights: %s", hf_name)
    return mlx_weights


def _quantize_weights(
    weights: dict[str, mx.array],
    quant_bits: int = 4,
    quant_group_size: int = 64,
) -> tuple[dict[str, mx.array], bool]:
    """Quantize weights. Returns (quantized_weights, all_ok).

    #1059: if ANY weight fails to quantize, we do NOT partially quantize
    — a half-quantized model has some weights with .scales/.biases and
    others as plain .weight, which mlx_lm.load mis-handles (shape/quant
    mismatch). all_ok=False signals the caller to drop quantization
    entirely and keep all weights fp16, so config + weights stay
    consistent.
    """
    if quant_bits <= 0:
        return weights, True

    quantized = {}
    skip_patterns = ("norm.weight", "embed_tokens.weight", "lm_head.weight")
    failed: list[str] = []

    for name, tensor in weights.items():
        if any(name.endswith(p) for p in skip_patterns) or len(tensor.shape) < 2:
            quantized[name] = tensor
            continue

        base_name = name
        if base_name.endswith(".weight"):
            base_name = base_name[: -len(".weight")]

        try:
            q_weight, q_scale, q_bias = mx.quantize(
                tensor,
                group_size=quant_group_size,
                bits=quant_bits,
            )
            quantized[base_name + ".weight"] = q_weight
            quantized[base_name + ".scales"] = q_scale
            quantized[base_name + ".biases"] = q_bias
        except Exception:
            logger.warning("Quantize failed for %s", name, exc_info=True)
            failed.append(name)

    if failed:
        logger.warning(
            "Quantize failed for %d/%d weights — dropping quantization "
            "entirely to avoid a half-quantized model (config/weight "
            "mismatch on load). Affected: %s",
            len(failed),
            len(weights),
            failed[:10],
        )
        # Return original weights un-quantized so config + weights stay
        # consistent (no .scales/.biases keys → mlx_lm loads as fp16).
        return dict(weights), False

    return quantized, True


def _build_mlx_config(
    hf_config: dict,
    template: ArchTemplate,
) -> dict:
    mlx_config = {
        "model_type": template.name,
        "model_file": f"{template.name}.py",
        "num_hidden_layers": hf_config.get(
            "num_hidden_layers", hf_config.get("n_layer", 0)
        ),
        "hidden_size": hf_config.get("hidden_size", hf_config.get("d_model", 0)),
        "intermediate_size": hf_config.get("intermediate_size", 0),
        "num_attention_heads": hf_config.get("num_attention_heads", 0),
        "num_key_value_heads": hf_config.get(
            "num_key_value_heads", hf_config.get("num_attention_heads", 0)
        ),
        "rms_norm_eps": hf_config.get("rms_norm_eps", 1e-6),
        "vocab_size": hf_config.get("vocab_size", 0),
        "tie_word_embeddings": hf_config.get("tie_word_embeddings", False),
    }

    rope_theta = hf_config.get("rope_theta")
    if rope_theta:
        mlx_config["rope_theta"] = rope_theta

    rope_traditional = hf_config.get("rope_traditional", False)
    mlx_config["rope_traditional"] = rope_traditional

    if template.has_bias:
        mlx_config["bias"] = True
    if template.has_mlp_bias:
        mlx_config["mlp_bias"] = True

    max_position_embeddings = hf_config.get("max_position_embeddings")
    if max_position_embeddings:
        mlx_config["max_position_embeddings"] = max_position_embeddings

    return mlx_config


def convert_model(
    hf_dir: str,
    output_dir: str,
    config: dict,
    template: ArchTemplate,
    quant_bits: int = 0,
    quant_group_size: int = 64,
    progress_cb: Callable[[float, str], None] | None = None,
) -> ConvertResult:
    result = ConvertResult(output_dir=output_dir)

    tmp_dir: str | None = None
    try:
        if progress_cb:
            progress_cb(0.0, "Loading HF weights")
        hf_weights = _load_hf_weights(hf_dir)
        hf_keys = list(hf_weights.keys())

        if progress_cb:
            progress_cb(0.2, "Building weight map")
        weight_map = build_weight_map(config, template)
        result.orphans = find_orphan_keys(hf_keys, weight_map)
        result.missing = find_missing_keys(weight_map, hf_keys)

        # #1059: if num_hidden_layers resolved to 0, the weight map only
        # contains embed/norm/lm_head — ALL per-layer weights become
        # orphans. Writing this to disk produces a 0-layer "model" that
        # mlx_lm.load accepts as valid but generates garbage. Hard-fail
        # instead of warning + returning success.
        num_layers = config.get("num_hidden_layers", config.get("n_layer", 0))
        if num_layers == 0:
            result.error = (
                "num_hidden_layers resolved to 0 — cannot build a valid "
                "weight map (no per-layer entries). Check the HF config "
                "for 'num_hidden_layers' or 'n_layer'."
            )
            logger.error("Conversion aborted: %s", result.error)
            return result

        if result.missing:
            logger.warning(
                "Missing %d expected keys — conversion may be incomplete",
                len(result.missing),
            )

        if progress_cb:
            progress_cb(0.4, "Remapping weights")
        mlx_weights = _remap_weights(hf_weights, weight_map)

        quant_ok = True
        if quant_bits > 0:
            if progress_cb:
                progress_cb(0.6, "Quantizing weights")
            mlx_weights, quant_ok = _quantize_weights(
                mlx_weights, quant_bits, quant_group_size
            )

        if progress_cb:
            progress_cb(0.8, "Saving MLX model")
        # Atomic-ish write: build the full output tree in a sibling temp dir,
        # then swap it into output_dir on success. A crash mid-write leaves
        # the temp dir (cleaned below) rather than a half-written
        # model.safetensors + config.json that mlx_lm.load would silently
        # load as a valid but corrupt model (#811 P0).
        os.makedirs(os.path.dirname(os.path.abspath(output_dir)) or ".", exist_ok=True)
        tmp_dir = f"{output_dir}.tmp.{os.getpid()}"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        mlx_config = _build_mlx_config(config, template)
        # #1059: if quantization was requested but partially failed
        # (_quantize_weights returned all_ok=False), the weights are
        # un-quantized fp16. Strip any quantization-related fields from
        # config so mlx_lm.load doesn't expect .scales/.biases keys.
        if quant_bits > 0 and not quant_ok:
            mlx_config.pop("quantization", None)
            mlx_config.pop("quant_bits", None)
            mlx_config.pop("quant_group_size", None)
            logger.warning(
                "Quantization requested (bits=%d) but failed — writing "
                "fp16 model with no quantization config",
                quant_bits,
            )
        config_path = os.path.join(tmp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(mlx_config, f, indent=2)
        logger.info("Wrote config.json to %s", tmp_dir)

        _save_weights_safetensors(mlx_weights, tmp_dir)
        logger.info("Wrote weights (%d tensors) to %s", len(mlx_weights), tmp_dir)

        tokenizer_src = Path(hf_dir)
        for tok_name in (
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
        ):
            src = tokenizer_src / tok_name
            if src.exists():
                shutil.copy2(str(src), os.path.join(tmp_dir, tok_name))
                logger.info("Copied %s", tok_name)
        for custom_tok in sorted(tokenizer_src.glob("tokenization_*.py")):
            shutil.copy2(str(custom_tok), os.path.join(tmp_dir, custom_tok.name))
            logger.info("Copied custom tokenizer %s", custom_tok.name)

        # #1058: swap via rename-to-backup so a failed rename does NOT
        # destroy the previously-available output_dir. The old code did
        # rmtree(output_dir) then rename(tmp_dir, output_dir) — if rmtree
        # succeeded but rename failed (cross-device / permission), the
        # old model was already gone and tmp_dir was cleaned by the outer
        # finally, losing everything. Now: rename old → backup, rename
        # tmp → final, rmtree backup only after the swap succeeds.
        backup_dir = f"{output_dir}.old.{os.getpid()}"
        if os.path.exists(output_dir):
            os.rename(output_dir, backup_dir)
        try:
            os.rename(tmp_dir, output_dir)
            tmp_dir = None  # consumed
        except OSError:
            # rename failed — restore old output if we moved it.
            if os.path.exists(backup_dir) and not os.path.exists(output_dir):
                try:
                    os.rename(backup_dir, output_dir)
                except OSError:
                    logger.error(
                        "Failed to restore backup %s after rename failure",
                        backup_dir,
                        exc_info=True,
                    )
            raise
        # Swap succeeded — safe to remove the old output.
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir, ignore_errors=True)
        logger.info("Atomically installed converted model to %s", output_dir)

        result.num_weights = len(mlx_weights)
        total_elements = sum(np.prod(w.shape) for w in mlx_weights.values())
        result.total_params_b = float(total_elements) / 1e9

        if progress_cb:
            progress_cb(1.0, "Conversion complete")

    except Exception as e:
        logger.exception("Conversion failed: %s", e)
        result.error = str(e)
    finally:
        # Clean up a half-written temp dir from a failed conversion so it is
        # not mistaken for a valid model on the next run.
        if tmp_dir is not None and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return result
