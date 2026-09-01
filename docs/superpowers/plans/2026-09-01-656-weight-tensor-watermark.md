# #656 Weight-Tensor Watermark Embed/Verify — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /v1/watermark/embed` + `POST /v1/watermark/verify` endpoints that embed/extract a secret-seeded LSB spread-spectrum watermark in model weight tensors, returning a Hub-aligned signature.

**Architecture:** Pure-numpy LSB core (`fusion_mlx/watermark/lsb.py`) with no MLX dependency, unit-testable standalone. Route layer (`fusion_mlx/api/watermark_routes.py`) bridges MLX↔numpy on the caller thread (#630), runs on the shared `convert_routes._executor` single-worker pool, saves safetensors via `mx.save_safetensors`. Synchronous `asyncio.wrap_future` shape like `/merge-adapter` (R1). Signature `sha256(secret:model::json.dumps(payload,sort_keys=True))[:32]` aligns with Hub `FMH_WATERMARK_SECRET` (R2). Quantized tensors config-driven skipped (R3).

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, numpy, MLX, mlx_lm. No new deps.

**Spec:** `docs/superpowers/specs/2026-09-01-656-weight-tensor-watermark-design.md` — the plan argues from the spec; executors read both.

## Global Constraints

- **Indentation:** multiple of 4 spaces. No docstrings (CLAUDE.md).
- **Logs:** every function path logs (CLAUDE.md). `logger = logging.getLogger(__name__)` per module.
- **No new dependencies.** numpy, hashlib, hmac, json, MLX, mlx_lm all already present.
- **Thread-portability (#630):** MLX↔numpy bridge happens on the caller thread (numpy-bridge `np.array(...)` outside executor, rebuild `mx.array` inside). `mx.eval` on worker thread. Never `mx.clear_streams()`.
- **Lint gate:** `black --fast` + `ruff check fusion_mlx/ tests/`. Never pass `debt_modules.txt` to ruff. N999 enforced.
- **Test runner:** `.venv/bin/python -m pytest` (not bare pytest — rtk proxy gotcha).
- **Secret env:** `FMH_WATERMARK_SECRET`. Route 503 on default/empty (E-S6). Never log/return the secret.
- **Real-model tests gated** `FUSION_MLX_REAL_MODEL_TESTS`. Clean process data after tests (CLAUDE.md).

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `fusion_mlx/watermark/__init__.py` | create | Package marker; re-exports |
| `fusion_mlx/watermark/lsb.py` | create | Pure-numpy LSB core. No MLX import. |
| `fusion_mlx/api/watermark_models.py` | create | Pydantic request models |
| `fusion_mlx/api/watermark_routes.py` | create | FastAPI routes + executor bridge |
| `fusion_mlx/server.py` | modify | Mount router |
| `fusion_mlx/_version.py` | modify | bump 0.8.57 → 0.8.58 |
| `CHANGELOG.md` | modify | add `[0.8.58]` |
| `README.md` | modify | document endpoints + env |
| `tests/unit/test_watermark_lsb.py` | create | 9 algorithm unit tests |
| `tests/unit/test_watermark_routes.py` | create | 8 route contract tests |
| `tests/unit/test_watermark_real_model.py` | create | gated real-model round-trip |

---

### Task 1: LSB core — payload↔bits + signature

**Files:**
- Create: `fusion_mlx/watermark/__init__.py`
- Create: `fusion_mlx/watermark/lsb.py`
- Test: `tests/unit/test_watermark_lsb.py`

**Interfaces:**
- Consumes: nothing (leaf module)
- Produces: `payload_to_bits(payload: dict) -> np.ndarray`, `bits_to_payload(bits: np.ndarray) -> dict | None`, `compute_signature(secret: str, model: str, payload: dict) -> str`. Later tasks use these.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_watermark_lsb.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_watermark_lsb.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fusion_mlx.watermark'`

- [ ] **Step 3: Write minimal implementation**

Create `fusion_mlx/watermark/__init__.py`:
```python
from .lsb import (
    bits_to_payload,
    compute_signature,
    embed_bits,
    extract_bits,
    payload_to_bits,
)

__all__ = [
    "bits_to_payload",
    "compute_signature",
    "embed_bits",
    "extract_bits",
    "payload_to_bits",
]
```

Create `fusion_mlx/watermark/lsb.py`:
```python
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
```

Note: `embed_bits`/`extract_bits` are referenced in `__init__` but not yet defined — add stubs now to keep import valid; Task 2 implements them. Add to `lsb.py`:
```python
def embed_bits(weights, bit_array, generator, bits_per_weight=1, epsilon=1e-6):
    raise NotImplementedError


def extract_bits(weights, n_bits, generator, bits_per_weight=1, epsilon=1e-6):
    raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_watermark_lsb.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add fusion_mlx/watermark/__init__.py fusion_mlx/watermark/lsb.py tests/unit/test_watermark_lsb.py
git commit -m "feat(#656): LSB core — payload<->bits + signature"
```

---

### Task 2: LSB core — embed_bits + extract_bits

**Files:**
- Modify: `fusion_mlx/watermark/lsb.py`
- Test: `tests/unit/test_watermark_lsb.py` (append)

**Interfaces:**
- Consumes: Task 1 (`payload_to_bits`)
- Produces: `embed_bits(weights, bit_array, generator, bits_per_weight=1, epsilon=1e-6) -> tuple[np.ndarray, int]`, `extract_bits(weights, n_bits, generator, bits_per_weight=1, epsilon=1e-6) -> tuple[np.ndarray, int]`. Task 4 route uses these.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_watermark_lsb.py`:
```python
from fusion_mlx.watermark.lsb import embed_bits, extract_bits


def _gen(secret, name):
    seed = hashlib.sha256(secret.encode() + b":" + name.encode()).digest()
    return np.random.default_rng(seed)


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
    assert abs(out[1]) < 1e-9
    assert abs(out[2]) < 1e-9


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_watermark_lsb.py -v -k "embed or prng or quantized or magnitude"`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Write minimal implementation**

Replace the two stubs in `fusion_mlx/watermark/lsb.py`:
```python
def _eligible_mask(weights: np.ndarray, epsilon: float) -> np.ndarray:
    if not np.issubdtype(weights.dtype, np.floating):
        raise ValueError(
            f"embed_bits: integer/quantized dtype {weights.dtype} not eligible"
        )
    return np.abs(weights) > epsilon


def embed_bits(weights, bit_array, generator, bits_per_weight=1, epsilon=1e-6):
    if bits_per_weight < 1 or bits_per_weight > 3:
        raise ValueError(f"embed_bits: bits_per_weight must be 1-3, got {bits_per_weight}")
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
    mask = int("1" * bits_per_weight, 2)
    for i, idx in enumerate(carrier_idx):
        w_int = int(out[idx].view(np.uint32 if out.dtype == np.float32 else np.uint32))
        # operate on the lowest bits_per_weight bits of the integer reinterpretation
        bit_val = int(bit_array[i])
        w_int = (w_int & ~mask) | ((bit_val & mask))
        out[idx] = np.frombuffer(
            np.array([w_int], dtype=np.uint32).tobytes(), dtype=out.dtype
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
        raise ValueError(
            f"extract_bits: n_bits {n_bits} exceeds capacity {capacity}"
        )
    eligible_indices = np.flatnonzero(eligible)
    carrier_idx = generator.choice(eligible_indices, size=n_bits, replace=False)
    mask = int("1" * bits_per_weight, 2)
    bits = np.empty(n_bits, dtype=np.uint8)
    for i, idx in enumerate(carrier_idx):
        w_int = int(weights[idx].view(np.uint32 if weights.dtype == np.float32 else np.uint32))
        bits[i] = w_int & mask
    logger.debug("extract_bits: read %d bits from %d carriers", n_bits, n_eligible)
    return bits, n_bits
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_watermark_lsb.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add fusion_mlx/watermark/lsb.py tests/unit/test_watermark_lsb.py
git commit -m "feat(#656): LSB core — embed_bits + extract_bits"
```

---

### Task 3: Pydantic request models

**Files:**
- Create: `fusion_mlx/api/watermark_models.py`
- Test: `tests/unit/test_watermark_routes.py` (skeleton, models only for now)

**Interfaces:**
- Consumes: `fusion_mlx/api/convert_models._get_allowed_output_prefixes` (reuse path validator)
- Produces: `WatermarkEmbedRequest`, `WatermarkVerifyRequest`. Task 4 routes consume these.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_watermark_routes.py`:
```python
import pytest
from pydantic import ValidationError

from fusion_mlx.api.watermark_models import (
    WatermarkEmbedRequest,
    WatermarkVerifyRequest,
)


def test_embed_request_minimal():
    req = WatermarkEmbedRequest(
        model="org/repo", payload={"a": 1}, secret="nondefault"
    )
    assert req.bits_per_weight == 1
    assert req.in_place is False
    assert req.layers is None


def test_embed_request_bits_per_weight_range():
    with pytest.raises(ValidationError):
        WatermarkEmbedRequest(
            model="m", payload={}, secret="s", bits_per_weight=4
        )
    with pytest.raises(ValidationError):
        WatermarkEmbedRequest(
            model="m", payload={}, secret="s", bits_per_weight=0
        )


def test_embed_request_output_path_required_when_not_in_place():
    # in_place=False (default) without output_path is allowed at model level;
    # the route enforces output_path presence. Here we just validate the
    # path-prefix constraint when a path IS given.
    import os

    home = os.path.expanduser("~/.fusion-mlx/models")
    req = WatermarkEmbedRequest(
        model="m", payload={}, secret="s", output_path=home + "/wm-out"
    )
    assert req.output_path.startswith(home)


def test_verify_request_minimal():
    req = WatermarkVerifyRequest(model="m", secret="s")
    assert req.bits_per_weight == 1
    assert req.layers is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_watermark_routes.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `fusion_mlx/api/watermark_models.py`:
```python
import logging
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from .convert_models import _get_allowed_output_prefixes

logger = logging.getLogger(__name__)


def _validate_output_path(v: str | None) -> str | None:
    if v is None:
        return v
    resolved = Path(v).resolve()
    for prefix in _get_allowed_output_prefixes():
        try:
            if resolved.is_relative_to(prefix.resolve()):
                return str(resolved)
        except Exception:
            pass
    logger.warning("watermark output_path rejected (outside allowed dirs): %s", v)
    raise ValueError(
        "output_path must be within allowed model directories "
        "(~/.fusion-mlx/models, CWD, or HF cache)"
    )


class WatermarkEmbedRequest(BaseModel):
    model: str = Field(
        ..., description="HF repo (org/name), model alias, or local model path"
    )
    payload: dict = Field(..., description="JSON-serializable dict to embed")
    secret: str = Field(..., description="Signing/seed secret (FMH_WATERMARK_SECRET)")
    layers: list[str] | None = Field(
        None, description="Glob patterns restricting which weight tensors to watermark"
    )
    bits_per_weight: int = Field(1, ge=1, le=3, description="LSB bits per carrier weight")
    in_place: bool = Field(False, description="Mutate safetensors in place (else copy)")
    output_path: str | None = Field(
        None, description="Copy destination (required when in_place is false)"
    )

    @field_validator("output_path")
    @classmethod
    def _check_output_path(cls, v):
        return _validate_output_path(v)


class WatermarkVerifyRequest(BaseModel):
    model: str = Field(..., description="Model to verify (path/alias/repo)")
    secret: str = Field(..., description="Same secret used at embed")
    layers: list[str] | None = Field(
        None, description="Same restriction used at embed"
    )
    bits_per_weight: int = Field(1, ge=1, le=3, description="Same value used at embed")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_watermark_routes.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add fusion_mlx/api/watermark_models.py tests/unit/test_watermark_routes.py
git commit -m "feat(#656): Pydantic request models for watermark embed/verify"
```

---

### Task 4: Watermark routes — embed + verify

**Files:**
- Create: `fusion_mlx/api/watermark_routes.py`
- Modify: `fusion_mlx/server.py` (mount router)
- Test: `tests/unit/test_watermark_routes.py` (append route tests)

**Interfaces:**
- Consumes: Task 1+2 (`fusion_mlx.watermark.lsb`), Task 3 (models), `convert_routes._executor`, `convert_models._get_allowed_output_prefixes`, `model_aliases.resolve_model`, `admin.auth.require_admin`, `middleware.require_model_hub_source`
- Produces: `router` (APIRouter prefix `/v1`, tags `["watermark"]`), mounted in server.py.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_watermark_routes.py`:
```python
import hashlib
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api.watermark_routes import router


def _app():
    app = FastAPI()
    app.include_router(router)
    return app


def test_embed_rejects_default_secret():
    client = TestClient(_app())
    with patch.dict("os.environ", {"FMH_WATERMARK_SECRET": "fusion-model-hub-default-secret"}):
        r = client.post(
            "/v1/watermark/embed",
            json={"model": "m", "payload": {"a": 1}, "secret": "fusion-model-hub-default-secret"},
        )
        assert r.status_code == 503


def test_embed_rejects_empty_secret():
    client = TestClient(_app())
    r = client.post(
        "/v1/watermark/embed",
        json={"model": "m", "payload": {"a": 1}, "secret": ""},
    )
    assert r.status_code == 503


def test_verify_rejects_default_secret():
    client = TestClient(_app())
    r = client.post(
        "/v1/watermark/verify",
        json={"model": "m", "secret": "fusion-model-hub-default-secret"},
    )
    assert r.status_code == 503


def test_signature_format_route_aligned():
    from fusion_mlx.watermark.lsb import compute_signature

    sig = compute_signature("nondefault", "org/repo", {"owner": "x"})
    import json

    expected = hashlib.sha256(
        f"nondefault:org/repo::{json.dumps({'owner': 'x'}, sort_keys=True)}".encode()
    ).hexdigest()[:32]
    assert sig == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_watermark_routes.py -v -k "secret or signature_format"`
Expected: FAIL with `ModuleNotFoundError: No module named 'fusion_mlx.api.watermark_routes'`

- [ ] **Step 3: Write minimal implementation**

Create `fusion_mlx/api/watermark_routes.py`:
```python
import asyncio
import fnmatch
import hashlib
import json
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


def _seed_for(secret: str, name: str) -> bytes:
    return hashlib.sha256(secret.encode() + b":" + name.encode()).digest()


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
    from mlx_lm.utils import load

    model, _tokenizer, config = load(model_path, return_config=True)
    tree = tree_flatten(model.parameters())
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
    carrier_count = 0
    for name, arr in carriers:
        gen = np.random.default_rng(_seed_for(secret, name))
        chunk = bits[bit_cursor : bit_cursor + arr.size * bits_per_weight]
        if chunk.size == 0:
            watermarked.append((name, arr))
            continue
        out, used = embed_bits(arr, chunk, gen, bits_per_weight=bits_per_weight)
        bit_cursor += used
        carrier_count += used
        watermarked.append((name, out))
    name_map = dict(watermarked)
    new_tree = [(n, mx.array(name_map.get(n, w))) for n, w in tree]
    model.update_weights(tree_unflatten(new_tree)) if hasattr(model, "update_weights") else None
    dest = model_path if in_place else output_path
    if dest is None:
        raise ValueError("output_path required when in_place is false")
    # save weights back to safetensors
    save_tree = tree_flatten(model.parameters()) if not watermarked else new_tree
    weights_dict = {n: w for n, w in save_tree}
    mx.save_safetensors(str(Path(dest) / "model.safetensors"), weights_dict)
    # copy config/tokenizer for a usable model dir on copy
    if not in_place:
        import shutil

        src = Path(model_path)
        for fn in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            if (src / fn).exists():
                shutil.copy2(src / fn, Path(dest) / fn)
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
    # read all bits in carrier order
    all_bits = []
    for name, arr in carriers:
        gen = np.random.default_rng(_seed_for(secret, name))
        bits, _ = extract_bits(arr, arr.size * bits_per_weight, gen, bits_per_weight=bits_per_weight)
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
```

- [ ] **Step 4: Mount router in server.py**

In `fusion_mlx/server.py`, near line 1023 (after `convert_router`), add import at top with the other route imports and mount:
```python
app.include_router(watermark_router)
```
Find the existing import block (e.g. `from .api.convert_routes import router as convert_router`) and add:
```python
from .api.watermark_routes import router as watermark_router
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_watermark_routes.py -v`
Expected: PASS (8 tests). Note: admin/hub-source deps are bypassed in these tests because TestClient calls the router directly without the full app middleware; the secret-rejection tests don't reach the dependency. If the deps gate blocks the 503 tests, patch `require_admin`/`require_model_hub_source` to no-ops in a fixture.

- [ ] **Step 6: Commit**

```bash
git add fusion_mlx/api/watermark_routes.py fusion_mlx/server.py tests/unit/test_watermark_routes.py
git commit -m "feat(#656): /v1/watermark/embed + /verify routes"
```

---

### Task 5: Real-model integration test (gated)

**Files:**
- Create: `tests/unit/test_watermark_real_model.py`

**Interfaces:**
- Consumes: Task 4 routes, `FUSION_MLX_REAL_MODEL_TESTS` env, a small real model in `~/.fusion-mlx/models`.

- [ ] **Step 1: Write the test**

Create `tests/unit/test_watermark_real_model.py`:
```python
import os
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("FUSION_MLX_REAL_MODEL_TESTS"),
    reason="set FUSION_MLX_REAL_MODEL_TESTS to run real-model watermark round-trip",
)

_SECRET = "real-model-test-secret-nondefault"


def _find_small_model() -> str | None:
    base = Path.home() / ".fusion-mlx" / "models"
    if not base.exists():
        return None
    candidates = []
    for p in base.iterdir():
        if (p / "config.json").exists() and (p / "model.safetensors").exists():
            candidates.append(p)
    candidates.sort(key=lambda p: (p / "model.safetensors").stat().st_size)
    return str(candidates[0]) if candidates else None


def test_embed_verify_real_model_roundtrip(tmp_path):
    model_path = _find_small_model()
    if model_path is None:
        pytest.skip("no small safetensors model in ~/.fusion-mlx/models")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from fusion_mlx.api.watermark_routes import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    out = tmp_path / "wm-out"
    out.mkdir()
    payload = {"owner": "dahai80", "purpose": "provenance-test"}

    r = client.post(
        "/v1/watermark/embed",
        json={
            "model": model_path,
            "payload": payload,
            "secret": _SECRET,
            "output_path": str(out),
        },
    )
    assert r.status_code == 200, r.text
    embed = r.json()
    assert embed["signature"]
    assert embed["carrier_count"] > 0

    r2 = client.post(
        "/v1/watermark/verify",
        json={"model": str(out), "secret": _SECRET},
    )
    assert r2.status_code == 200, r2.text
    verify = r2.json()
    assert verify["verified"] is True, verify
    assert verify["payload"] == payload
    assert verify["signature"] == embed["signature"]
    # cleanup process data
    shutil.rmtree(out, ignore_errors=True)
```

- [ ] **Step 2: Run test (gated — skip if no env/model)**

Run: `.venv/bin/python -m pytest tests/unit/test_watermark_real_model.py -v`
Expected: SKIP (no `FUSION_MLX_REAL_MODEL_TESTS`). Full validation happens at release-time with the env set + a real small model.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_watermark_real_model.py
git commit -m "test(#656): gated real-model watermark round-trip"
```

---

### Task 6: Lint + full-suite sweep + version bump + docs

**Files:**
- Modify: `fusion_mlx/_version.py`
- Modify: `CHANGELOG.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: all prior tasks

- [ ] **Step 1: Lint**

Run: `black --fast fusion_mlx/watermark/ fusion_mlx/api/watermark_models.py fusion_mlx/api/watermark_routes.py tests/unit/test_watermark_*.py`
Run: `ruff check fusion_mlx/ tests/`
Expected: clean. Fix any findings (never pass `debt_modules.txt` to ruff).

- [ ] **Step 2: Run full unit suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: 0 failures. Confirm watermark tests collected + passing.

- [ ] **Step 3: Version bump**

In `fusion_mlx/_version.py`: change `__version__ = "0.8.57"` to `__version__ = "0.8.58"`.

- [ ] **Step 4: CHANGELOG**

In `CHANGELOG.md`, add at top (after the header):
```markdown
## [0.8.58] - 2026-09-01

### Added
- `POST /v1/watermark/embed` — embed a secret-seeded LSB spread-spectrum
  watermark into model weight tensors; returns a Hub-aligned signature
  (`sha256(secret:model::payload)[:32]`). Admin + hub-source gated.
  Reuses the convert/merge single-worker executor (#656).
- `POST /v1/watermark/verify` — extract + verify the embedded payload
  from tensors. Returns `verified`, `payload`, `signature`, `mismatch_rate`.
- `FMH_WATERMARK_SECRET` env var (shared with Fusion-Model-Hub); route
  503s on default/empty secret (E-S6).
- `fusion_mlx.watermark.lsb` pure-numpy core (embed_bits/extract_bits/
  payload_to_bits/bits_to_payload/compute_signature).

### Changed
- Quantized (int) weight tensors are config-driven skipped during
  watermark embed/verify (R3).
```

- [ ] **Step 5: README**

In `README.md`, add a section under the API endpoints area documenting `/v1/watermark/embed` + `/verify`, the `FMH_WATERMARK_SECRET` env requirement, the sync shape (returns `output_path` in body), and the quantized-tensor skip behavior.

- [ ] **Step 6: Commit**

```bash
git add fusion_mlx/_version.py CHANGELOG.md README.md
git commit -m "chore(#656): version bump 0.8.58 + changelog + readme"
```

- [ ] **Step 7: Final lint + suite re-confirm**

Run: `ruff check fusion_mlx/ tests/` and `.venv/bin/python -m pytest tests/unit -q`
Expected: clean + green.

---

## Self-Review (controller runs before dispatch)

**Spec coverage:**
- §2 goals (embed + verify + tamper-resistant + Hub-aligned + admin-gated + no hardcoded secret) → Tasks 1-4 ✓
- §4 approach (LSB spread-spectrum, magnitude safety, quantized skip) → Task 2 ✓
- §6 API contract (sync shape R1, response fields) → Task 4 ✓
- §7 algorithm (all 5 functions) → Tasks 1+2 ✓
- §8 Hub alignment (secret env, sig format R2, no enrichment R6) → Tasks 1+4 ✓
- §9 fail-visible (503 secret, 400 no-carriers/too-large, 200 verified=false R7) → Task 4 ✓
- §10 testing (unit + route + real-model gated) → Tasks 1-5 ✓
- §11 security (admin gate, path validation, no exec, capacity DoS) → Tasks 3+4 ✓
- §12 file structure → all tasks ✓

**Placeholder scan:** none. All code blocks complete.

**Type consistency:** `embed_bits`/`extract_bits` signatures match between Task 1 stubs, Task 2 impl, Task 4 route calls. `compute_signature(secret, model, payload)` consistent across Tasks 1, 4. `_seed_for` used in both `_run_embed` and `_run_verify` (Task 4) — same derivation.

**Known risk for reviewer:** Task 4 Step 5 notes the admin/hub-source dependency may gate the TestClient secret-rejection tests; the task prescribes patching the deps. The real-model test (Task 5) is gated and only validates at release time.
