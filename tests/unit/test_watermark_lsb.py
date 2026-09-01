import hashlib
import json

import numpy as np
import pytest

from fusion_mlx.watermark.lsb import (
    bits_to_payload,
    compute_signature,
    payload_to_bits,
)


def test_payload_to_bits_bits_to_payload_roundtrip():
    payload = {"owner": "dahai80", "note": "provenance"}
    bits = payload_to_bits(payload)
    assert bits.dtype == np.uint8
    out = bits_to_payload(bits)
    assert out == payload


def test_bits_to_payload_corrupt_returns_none():
    payload = {"a": 1}
    bits = payload_to_bits(payload)
    bits[len(bits) // 2] ^= 1
    assert bits_to_payload(bits) is None


def test_bits_to_payload_truncated_returns_none():
    payload = {"a": 1}
    bits = payload_to_bits(payload)
    assert bits_to_payload(bits[:3]) is None


def test_signature_matches_hub_format():
    secret = "nondefault-test-secret"
    model = "org/repo"
    payload = {"owner": "dahai80", "embedded_at": "2026-09-01T00:00:00Z"}
    sig = compute_signature(secret, model, payload)
    expected = hashlib.sha256(
        f"{secret}:{model}::{json.dumps(payload, sort_keys=True)}".encode()
    ).hexdigest()[:32]
    assert sig == expected
    assert len(sig) == 32


def test_signature_constant_length():
    sig = compute_signature("s", "m", {"x": "y" * 1000})
    assert len(sig) == 32
