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


def embed_bits(weights, bit_array, generator, bits_per_weight=1, epsilon=1e-6):
    raise NotImplementedError


def extract_bits(weights, n_bits, generator, bits_per_weight=1, epsilon=1e-6):
    raise NotImplementedError
