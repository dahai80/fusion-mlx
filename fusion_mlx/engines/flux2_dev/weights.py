import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_PREDICATE = lambda path, module: hasattr(module, "to_quantized")


def _load_sharded(component_dir):
    component_dir = Path(component_dir)
    index_file = component_dir / "model.safetensors.index.json"
    if index_file.exists():
        idx = json.loads(index_file.read_text())
        weight_map = idx["weight_map"]
        shard_cache = {}
        merged = {}
        for tensor_name, shard_name in weight_map.items():
            if shard_name not in shard_cache:
                shard_cache[shard_name] = mx.load(
                    str(component_dir / shard_name), return_metadata=False
                )
            merged[tensor_name] = shard_cache[shard_name][tensor_name]
        q_level = idx.get("metadata", {}).get("quantization_level")
        logger.info(
            "flux2_dev weights: loaded sharded %s keys=%d quant=%s",
            component_dir.name,
            len(merged),
            q_level,
        )
        return merged, int(q_level) if q_level else None
    files = sorted(component_dir.glob("*.safetensors"))
    merged = {}
    for f in files:
        merged.update(mx.load(str(f), return_metadata=False))
    logger.info(
        "flux2_dev weights: loaded %d single-shard files keys=%d from %s",
        len(files),
        len(merged),
        component_dir.name,
    )
    return merged, None


def load_dit(transformer, transformer_dir, quantize_bits=8):
    weights, stored_q = _load_sharded(transformer_dir)
    bits = quantize_bits or stored_q
    if bits is not None:
        logger.info(
            "flux2_dev weights: quantizing transformer bits=%d group_size=64",
            bits,
        )
        nn.quantize(
            transformer,
            group_size=64,
            bits=bits,
            class_predicate=_PREDICATE,
        )
    loaded = transformer.update(weights, strict=False)
    missing = getattr(loaded, "missing", [])
    unexpected = getattr(loaded, "unexpected", [])
    logger.info(
        "flux2_dev weights: transformer applied missing=%d unexpected=%d",
        len(missing),
        len(unexpected),
    )
    if len(missing) > 0:
        logger.warning(
            "flux2_dev weights: %d transformer keys missing (first 5: %s)",
            len(missing),
            missing[:5],
        )
    return transformer


def load_vae(vae, vae_dir, quantize_bits=8):
    weights, stored_q = _load_sharded(vae_dir)
    bits = quantize_bits or stored_q
    if bits is not None:
        logger.info("flux2_dev weights: quantizing vae bits=%d group_size=64", bits)
        nn.quantize(
            vae,
            group_size=64,
            bits=bits,
            class_predicate=_PREDICATE,
        )
    loaded = vae.update(weights, strict=False)
    missing = getattr(loaded, "missing", [])
    unexpected = getattr(loaded, "unexpected", [])
    logger.info(
        "flux2_dev weights: vae applied missing=%d unexpected=%d",
        len(missing),
        len(unexpected),
    )
    if len(missing) > 0:
        logger.warning(
            "flux2_dev weights: %d vae keys missing (first 5: %s)",
            len(missing),
            missing[:5],
        )
    return vae


def load_text_encoder(text_encoder, weights_path):
    weights_path = Path(weights_path)
    logger.info(
        "flux2_dev weights: loading text encoder from %s (%.1f GB)",
        weights_path.name,
        weights_path.stat().st_size / 1e9,
    )
    raw = mx.load(str(weights_path), return_metadata=False)
    sanitized = text_encoder.sanitize(raw)
    loaded = text_encoder.update(sanitized, strict=False)
    missing = getattr(loaded, "missing", [])
    unexpected = getattr(loaded, "unexpected", [])
    logger.info(
        "flux2_dev weights: text_encoder applied missing=%d unexpected=%d",
        len(missing),
        len(unexpected),
    )
    if len(missing) > 0:
        logger.warning(
            "flux2_dev weights: %d text_encoder keys missing (first 5: %s)",
            len(missing),
            missing[:5],
        )
    return text_encoder
