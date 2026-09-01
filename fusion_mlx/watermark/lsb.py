import hashlib
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

_LENGTH_HEADER_BYTES = 4


def payload_to_bits(payload: dict) -> np.ndarray:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    n = len(body)
    header = n.to_bytes(_LENGTH_HEADER_BYTES, "big")
    raw = header + body
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    logger.debug("payload_to_bits: %d bytes -> %d bits", n, bits.size)
    return bits.astype(np.uint8)


def bits_to_payload(bits: np.ndarray) -> dict | None:
    if bits.size < _LENGTH_HEADER_BYTES * 8:
        logger.warning("bits_to_payload: too few bits for header: %d", bits.size)
        return None
    bytes_arr = np.packbits(bits.astype(np.uint8))
    n = int.from_bytes(bytes(bytes_arr[:_LENGTH_HEADER_BYTES]), "big")
    total_needed = (_LENGTH_HEADER_BYTES + n) * 8
    if bits.size < total_needed:
        logger.warning(
            "bits_to_payload: truncated: have %d bits, need %d", bits.size, total_needed
        )
        return None
    body = bytes(bytes_arr[_LENGTH_HEADER_BYTES : _LENGTH_HEADER_BYTES + n])
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("bits_to_payload: decode failed: %s", exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("bits_to_payload: payload is not a dict: %r", type(payload))
        return None
    return payload


def compute_signature(secret: str, model: str, payload: dict) -> str:
    raw = f"{secret}:{model}::{json.dumps(payload, sort_keys=True)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _uint_view_dtype(arr):
    return np.dtype(f"uint{arr.itemsize * 8}")


def _eligible_mask(weights, epsilon):
    if not np.issubdtype(weights.dtype, np.floating):
        raise ValueError(
            f"embed_bits: integer/quantized dtype {weights.dtype} not eligible"
        )
    return np.abs(weights) > epsilon


def embed_bits(weights, bit_array, generator, bits_per_weight=1, epsilon=1e-6):
    if bits_per_weight < 1 or bits_per_weight > 3:
        raise ValueError(
            f"embed_bits: bits_per_weight must be 1-3, got {bits_per_weight}"
        )
    eligible = _eligible_mask(weights, epsilon)
    n_eligible = int(eligible.sum())
    capacity = n_eligible * bits_per_weight
    if bit_array.size > capacity:
        raise ValueError(
            f"embed_bits: bit_array {bit_array.size} exceeds capacity {capacity} "
            f"(eligible={n_eligible}, bits_per_weight={bits_per_weight})"
        )
    eligible_indices = np.flatnonzero(eligible)
    carrier_idx = generator.choice(eligible_indices, size=bit_array.size, replace=False)
    out = weights.copy()
    uint_dt = _uint_view_dtype(out)
    mask = int("1" * bits_per_weight, 2)
    for i, idx in enumerate(carrier_idx):
        w_int = int(out[idx].view(uint_dt))
        bit_val = int(bit_array[i])
        w_int = (w_int & ~mask) | (bit_val & mask)
        out[idx] = np.frombuffer(
            np.array([w_int], dtype=uint_dt).tobytes(), dtype=out.dtype
        )[0]
    logger.info(
        "embed_bits: %d bits into %d eligible carriers (bits_per_weight=%d)",
        bit_array.size,
        n_eligible,
        bits_per_weight,
    )
    return out, bit_array.size


def extract_bits(weights, n_bits, generator, bits_per_weight=1, epsilon=1e-6):
    eligible = _eligible_mask(weights, epsilon)
    n_eligible = int(eligible.sum())
    capacity = n_eligible * bits_per_weight
    if n_bits > capacity:
        raise ValueError(f"extract_bits: n_bits {n_bits} exceeds capacity {capacity}")
    eligible_indices = np.flatnonzero(eligible)
    carrier_idx = generator.choice(eligible_indices, size=n_bits, replace=False)
    uint_dt = _uint_view_dtype(weights)
    mask = int("1" * bits_per_weight, 2)
    bits = np.empty(n_bits, dtype=np.uint8)
    for i, idx in enumerate(carrier_idx):
        w_int = int(weights[idx].view(uint_dt))
        bits[i] = w_int & mask
    logger.debug("extract_bits: read %d bits from %d carriers", n_bits, n_eligible)
    return bits, n_bits
