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


from fusion_mlx.watermark.lsb import embed_bits, extract_bits


def _gen(secret, name):
    seed = hashlib.sha256(secret.encode() + b":" + name.encode()).digest()
    return np.random.default_rng(int.from_bytes(seed, "big"))


def test_embed_extract_roundtrip():
    w = np.random.default_rng(0).standard_normal(1024).astype(np.float32)
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    g = _gen("secret", "tensor.0.weight")
    out, count = embed_bits(w, bits, g, bits_per_weight=1)
    g2 = _gen("secret", "tensor.0.weight")
    recovered, _ = extract_bits(out, bits.size, g2, bits_per_weight=1)
    assert np.array_equal(recovered, bits)
    assert count >= bits.size


def test_embed_skips_tiny_weights():
    w = np.array([0.0, 1e-9, -1e-9, 5.0, -3.0], dtype=np.float32)
    bits = np.array([1, 1], dtype=np.uint8)
    g = _gen("s", "t")
    out, count = embed_bits(w, bits, g, bits_per_weight=1)
    # tiny weights unchanged
    assert out[0] == 0.0
    assert abs(out[1]) <= 1e-9
    assert abs(out[2]) <= 1e-9


def test_embed_capacity_exceeded_raises():
    w = np.array([5.0, -3.0], dtype=np.float32)
    bits = np.array([1, 1, 1], dtype=np.uint8)
    g = _gen("s", "t")
    with pytest.raises(ValueError, match="capacity"):
        embed_bits(w, bits, g, bits_per_weight=1)


def test_prng_determinism_same_seed():
    w = np.random.default_rng(1).standard_normal(512).astype(np.float32)
    bits = np.array([1, 0, 1, 0], dtype=np.uint8)
    g1 = _gen("secret", "tensor.0.weight")
    g2 = _gen("secret", "tensor.0.weight")
    out1, _ = embed_bits(w.copy(), bits, g1)
    out2, _ = embed_bits(w.copy(), bits, g2)
    assert np.array_equal(out1, out2)


def test_quantized_tensor_skipped():
    w = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
    bits = np.array([1, 0], dtype=np.uint8)
    g = _gen("s", "t")
    with pytest.raises(ValueError, match="capacity|integer"):
        embed_bits(w, bits, g, bits_per_weight=1)


def test_magnitude_safety_no_nan():
    w = np.random.default_rng(2).standard_normal(2048).astype(np.float32)
    bits = np.random.default_rng(3).integers(0, 2, 64).astype(np.uint8)
    g = _gen("s", "t")
    out, _ = embed_bits(w, bits, g, bits_per_weight=1)
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))
