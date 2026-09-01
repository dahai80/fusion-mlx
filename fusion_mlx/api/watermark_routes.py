import asyncio
import fnmatch
import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from ..admin.auth import require_admin
from ..middleware import require_model_hub_source
from ..watermark.lsb import (
    bits_to_payload,
    compute_signature,
    embed_bits,
    extract_bits,
    payload_to_bits,
)
from .convert_routes import _executor
from .watermark_models import WatermarkEmbedRequest, WatermarkVerifyRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["watermark"])

_DEFAULT_SECRET = "fusion-model-hub-default-secret"
_QUANTIZED_CONFIG_KEYS = ("quantization", "quantization_config")


def _resolve_secret(provided: str) -> str:
    env_secret = os.environ.get("FMH_WATERMARK_SECRET", "")
    secret = provided or env_secret
    if not secret or secret == _DEFAULT_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Watermark disabled: set a non-default FMH_WATERMARK_SECRET env "
            "(high entropy) before embedding/verifying watermarks",
        )
    return secret


def _seed_for(secret: str, name: str) -> int:
    return int.from_bytes(
        hashlib.sha256(secret.encode() + b":" + name.encode()).digest(), "big"
    )


def _quantized_prefixes(config: dict) -> set[str]:
    prefixes: set[str] = set()
    for key in _QUANTIZED_CONFIG_KEYS:
        qc = config.get(key)
        if isinstance(qc, dict):
            for k in ("linear_class", "modules", "keys", "weight_key_suffix"):
                v = qc.get(k)
                if isinstance(v, list):
                    prefixes.update(str(x) for x in v)
    return prefixes


def _is_quantized(name: str, arr: np.ndarray, q_prefixes: set[str]) -> bool:
    if not np.issubdtype(arr.dtype, np.floating):
        return True
    for p in q_prefixes:
        if name.startswith(p):
            return True
    return False


def _select_carriers(
    weights: list[tuple[str, np.ndarray]],
    layers: list[str] | None,
    q_prefixes: set[str],
) -> list[tuple[str, np.ndarray]]:
    carriers = []
    for name, arr in weights:
        if _is_quantized(name, arr, q_prefixes):
            continue
        if layers is not None and not any(fnmatch.fnmatch(name, pat) for pat in layers):
            continue
        carriers.append((name, arr))
    return carriers


def _run_embed(
    model_path: str,
    payload: dict,
    secret: str,
    layers: list[str] | None,
    bits_per_weight: int,
    in_place: bool,
    output_path: str | None,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.utils import load, save

    model, tokenizer, config = load(model_path, return_config=True)
    tree = tree_flatten(model.parameters())
    orig_dtypes = {n: w.dtype for n, w in tree}
    weights_np = [(n, np.array(mx.array(w))) for n, w in tree]
    q_prefixes = _quantized_prefixes(config)
    carriers = _select_carriers(weights_np, layers, q_prefixes)
    if not carriers:
        raise ValueError(
            "No eligible (non-quantized float) weight tensors to watermark"
        )
    bits = payload_to_bits(payload)
    total_capacity = sum(c[1].size for c in carriers) * bits_per_weight
    if bits.size > total_capacity:
        raise ValueError(
            f"payload {bits.size} bits exceeds capacity {total_capacity}; "
            f"narrow layers or raise bits_per_weight"
        )
    bit_cursor = 0
    watermarked: list[tuple[str, np.ndarray]] = []
    for name, arr in carriers:
        gen = np.random.default_rng(_seed_for(secret, name))
        chunk = bits[bit_cursor : bit_cursor + arr.size * bits_per_weight]
        if chunk.size == 0:
            watermarked.append((name, arr))
            continue
        out, used = embed_bits(arr, chunk, gen, bits_per_weight=bits_per_weight)
        bit_cursor += used
        watermarked.append((name, out))
    carrier_count = len(carriers)
    name_map = dict(watermarked)
    new_tree = [
        (n, mx.array(name_map.get(n, w)).astype(orig_dtypes[n])) for n, w in tree
    ]
    if not hasattr(model, "update_weights"):
        raise ValueError(
            f"model {model_path} has no update_weights; cannot apply watermark"
        )
    model.update_weights(tree_unflatten(new_tree))
    dest = model_path if in_place else output_path
    if dest is None:
        raise ValueError("output_path required when in_place is false")
    save_path = Path(dest)
    save(save_path, model_path, model, tokenizer, config, donate_model=False)
    signature = compute_signature(secret, model_path, payload)
    logger.info(
        "watermark embed: model=%s carriers=%d bits=%d dest=%s sig=%s",
        model_path,
        carrier_count,
        bits.size,
        dest,
        signature[:8],
    )
    return {
        "status": "ok",
        "model": model_path,
        "output_path": str(dest),
        "signature": signature,
        "payload_bytes": bits.size // 8,
        "layers_watermarked": [c[0] for c in carriers],
        "bits_per_weight": bits_per_weight,
        "carrier_count": carrier_count,
    }


def _run_verify(
    model_path: str,
    secret: str,
    layers: list[str] | None,
    bits_per_weight: int,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load

    _model, _tokenizer, config = load(model_path, return_config=True)
    tree = tree_flatten(_model.parameters())
    weights_np = [(n, np.array(mx.array(w))) for n, w in tree]
    q_prefixes = _quantized_prefixes(config)
    carriers = _select_carriers(weights_np, layers, q_prefixes)
    if not carriers:
        return {
            "verified": False,
            "model": model_path,
            "payload": None,
            "signature": "",
            "mismatch_rate": None,
            "carrier_count": 0,
            "reason": "no carrier tensors found",
        }
    all_bits = []
    for name, arr in carriers:
        gen = np.random.default_rng(_seed_for(secret, name))
        bits, _ = extract_bits(
            arr, arr.size * bits_per_weight, gen, bits_per_weight=bits_per_weight
        )
        all_bits.append(bits)
    stream = np.concatenate(all_bits) if all_bits else np.array([], dtype=np.uint8)
    payload = bits_to_payload(stream)
    if payload is None:
        return {
            "verified": False,
            "model": model_path,
            "payload": None,
            "signature": "",
            "mismatch_rate": None,
            "carrier_count": sum(c[1].size for c in carriers),
            "reason": "payload not recoverable (corrupted/quantized)",
        }
    signature = compute_signature(secret, model_path, payload)
    return {
        "verified": True,
        "model": model_path,
        "payload": payload,
        "signature": signature,
        "mismatch_rate": 0.0,
        "carrier_count": sum(c[1].size for c in carriers),
        "reason": "",
    }


@router.post("/watermark/embed")
async def embed_watermark(
    request: WatermarkEmbedRequest,
    _is_admin: bool = Depends(require_admin),
    _source: bool = Depends(require_model_hub_source),
) -> dict[str, Any]:
    secret = _resolve_secret(request.secret)
    if not request.in_place and not request.output_path:
        raise HTTPException(422, detail="output_path required when in_place is false")
    from ..model_aliases import resolve_model

    model_path = resolve_model(request.model)
    if _executor._broken:
        raise HTTPException(503, detail="Convert/watermark executor unavailable")
    logger.info(
        "watermark embed job: model=%s resolved=%s bits=%d",
        request.model,
        model_path,
        request.bits_per_weight,
    )
    try:
        future = _executor.submit(
            _run_embed,
            model_path,
            request.payload,
            secret,
            request.layers,
            request.bits_per_weight,
            request.in_place,
            request.output_path,
        )
        result = await asyncio.wrap_future(future)
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("watermark embed rejected: %s", exc)
        raise HTTPException(400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("watermark embed failed: model=%s", request.model)
        raise HTTPException(500, detail=f"watermark embed failed: {exc}") from exc
    return result


@router.post("/watermark/verify")
async def verify_watermark(
    request: WatermarkVerifyRequest,
    _is_admin: bool = Depends(require_admin),
    _source: bool = Depends(require_model_hub_source),
) -> dict[str, Any]:
    secret = _resolve_secret(request.secret)
    from ..model_aliases import resolve_model

    model_path = resolve_model(request.model)
    if _executor._broken:
        raise HTTPException(503, detail="Convert/watermark executor unavailable")
    try:
        future = _executor.submit(
            _run_verify,
            model_path,
            secret,
            request.layers,
            request.bits_per_weight,
        )
        result = await asyncio.wrap_future(future)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("watermark verify failed: model=%s", request.model)
        raise HTTPException(500, detail=f"watermark verify failed: {exc}") from exc
    return result
