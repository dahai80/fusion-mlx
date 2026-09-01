# #656 Weight-Tensor Watermark Embed/Verify — Design Spec

**Issue:** [#656](https://github.com/dahai80/fusion-mlx/issues/656)
**Date:** 2026-09-01
**Status:** Design
**Author:** dahai80

## 1. Problem

Fusion-Model-Hub's watermark (`POST /api/v1/watermark/embed`) stores a DB row + HMAC signature + a signed `watermark.json` sidecar. None of these travel **inside** the weights: copy just the safetensors elsewhere and the watermark is gone. The Hub cannot fix this — its hard contract forbids importing `mlx`/`mlx-lm`/`torch`/`transformers`. All weight-tensor manipulation is delegated to fusion-mlx, which already owns `/convert`, `/quantize`, `/merge-adapter`.

This issue tracks the tensor-level embed that only fusion-mlx can provide.

## 2. Goals

- `POST /v1/watermark/embed` — embed a low-magnitude watermark into model weight tensors. Returns a signature over `(model, payload)`.
- `POST /v1/watermark/verify` — extract the embedded payload from tensors, verify signature.
- Tamper-resistant against a user who redistributes just the weights (strips sidecar/DB).
- Aligns with Hub's signing scheme so a payload signed in both systems verifies identically.
- Admin-gated; secret never hardcoded (E-S6 lesson).

## 3. Non-Goals

- Cryptographic robustness against a determined adversary who knows the algorithm and targets the watermark bits (open research; out of scope). Goal is **tamper-evidence against naive redistribution**, not proof-of-ownership in court.
- Watermarking quantized/packed weight formats in-place beyond what LSB naturally survives (we test and report survival; we do not guarantee it).
- Hub DB integration — fusion-mlx is stateless re: watermark; it embeds in tensors and verifies tensors. Hub owns the DB row + sidecar.

## 4. Approach — Secret-Seeded LSB Spread-Spectrum

Three approaches were weighed:

| Approach | Tamper-resistance | Survives quantize? | Complexity | Verdict |
|----------|-------------------|--------------------|------------|---------|
| **Reserved tensor** (inject `watermark.payload` key) | Low — delete the key, gone | No — quantize may merge/drop | Low | Rejected: too easy to strip |
| **Plain LSB** (flip lowest bit of every Nth weight) | Medium — local edits break it | Partial | Low | Rejected: not spread, fragile |
| **Secret-seeded LSB spread-spectrum** (PRNG picks which weights+bits) | Highest of the three | Partial (test + report mismatch_rate) | Medium | **Chosen** |

### Chosen: secret-seeded LSB spread-spectrum

A deterministic PRNG seeded from `(secret, tensor_name)` selects **which** weight elements carry watermark bits and **which** bit position. Encoding writes the payload bits into those positions; decoding reads the same positions via the same PRNG. Because positions are secret-dependent, an attacker who doesn't have the secret cannot locate (and thus cannot selectively strip) the watermark bits without destroying the model.

**Bit budget:** payload is a JSON dict → UTF-8 bytes → bit array. Each payload bit occupies one selected weight element's lowest `bits_per_weight` (default 1) bit(s). Spread across many weights → redundancy not required for correctness (deterministic read-back), but the spread itself is the robustness.

**Magnitude safety:** only weights with `abs(w) > epsilon` (epsilon = 1e-6) are eligible carriers. Flipping the LSB of a 1e-9 weight can flip its sign or push it toward denormal; skipping tiny weights avoids NaN/denormal injection. This is a hard rule, not a heuristic.

**Dtype handling:** MLX safetensors store float weights in bf16/fp16/fp32. LSB on float bits is **not** the mantissa LSB (float bits have sign/exponent/mantissa layout). Flipping the true LSB of a bf16 reinterpreted as uint16 changes the mantissa's lowest bit — a tiny relative perturbation (~2^-7 of mantissa for bf16). This is the intended low-magnitude perturbation. We operate on the **integer reinterpretation** of the float buffer (`view(uint_bits)`) so the bit flip is well-defined regardless of float format. **Quantized** (int4/int8 affine-packed) tensors are skipped entirely — their "weights" are packed codes, not floats; watermarking them corrupts the scale/zero-point contract.

### Why not a reserved tensor

A reserved tensor (`watermark.payload`) is trivially stripped by `safetensors` key deletion — `save_safetensors` with the key removed, or any HF loader that filters unknown keys. The issue explicitly calls out that a sidecar "can be stripped independently of the weights — only tensor-level embedding is tamper-resistant." A reserved tensor is just a named sidecar inside the safetensors; same strip vector. Spread-spectrum in real weight tensors has no such single key to delete.

## 5. Data Flow

### Embed

```
client ──POST /v1/watermark/embed──▶ route
  │  {model, payload, secret, layers?, bits?, in_place?, output_path?}
  ▼
route: resolve_model(model) ──▶ local path
  │  validate secret non-default
  ▼
_executor.submit(embed_job):
  1. load weights via mlx (tree_flatten → [(name, mx.array)])
  2. select carrier tensors:
     - skip quantized tensors (detect: key in quantization config OR dtype is int)
     - if layers specified, restrict to those names (glob match)
  3. payload → JSON bytes → bit array
  4. for each carrier tensor: PRNG(seed=sha256(secret:name)) picks positions
     flip lowest `bits` bit of each selected element (abs(w)>eps only)
  5. tree_unflatten → save_safetensors (copy unless in_place)
  6. signature = sha256(secret:model::json.dumps(payload,sort_keys=True))[:32]
  7. return {signature, layers_watermarked, output_path, bits_per_weight, payload_bytes}
```

### Verify

```
client ──POST /v1/watermark/verify──▶ route
  │  {model, secret, layers?, bits?}
  ▼
route: resolve_model(model)
  │
  ▼
_executor.submit(verify_job):
  1. load weights (read-only; load_weights only, no full model build)
  2. select carrier tensors (same skip logic)
  3. for each carrier: PRNG(seed=sha256(secret:name)) → same positions
     read lowest `bits` bit of each → bit array
  4. reconstruct payload bytes → JSON → payload dict
     (deterministic: read order = write order, so bits reassemble exactly
      IF no bit was perturbed; if perturbed, JSON parse fails → verified=False)
  5. signature = sha256(secret:model::json.dumps(payload,sort_keys=True))[:32]
  6. return {verified, payload (if parse ok), signature, mismatch_rate}
```

**mismatch_rate:** fraction of carrier weights whose current LSB ≠ the PRNG-expected bit (computed against a re-embedded reference, not the original). Detects partial tampering / quantize damage even when payload still parses. If payload JSON fails to parse, `verified=False`, `mismatch_rate` reported over the raw bit stream vs the last-known-good bit pattern is **not** available (no reference) → report `mismatch_rate=None`.

## 5. Data Flow

PLACEHOLDER: data flow

## 6. API Contract

### `POST /v1/watermark/embed`

**Request** (`WatermarkEmbedRequest`):
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | `str` | yes | — | HF repo, alias, or local path (same resolution as `/convert`) |
| `payload` | `dict` | yes | — | Arbitrary JSON-serializable dict to embed |
| `secret` | `str` | yes | — | Signing/seed secret. Must be non-default; route 503s on default/empty (E-S6) |
| `layers` | `list[str]` \| `null` | no | `null` | Glob patterns restricting which weight tensors to watermark. `null` = all eligible (non-quantized float) tensors |
| `bits_per_weight` | `int` | no | `1` | Bits of LSB to use per carrier weight (1-3). Higher = more payload capacity, more perturbation |
| `in_place` | `bool` | no | `false` | Mutate the model's safetensors in place. `false` (default) writes a copy to `output_path` |
| `output_path` | `str` \| `null` | no | `null` | Copy destination (required when `in_place=false`). Same allowed-prefix validation as `/merge-adapter` |

**Response** (200):
```json
{
  "status": "ok",
  "model": "org/repo",
  "output_path": "/abs/path",
  "signature": "abcd1234...",
  "payload_bytes": 42,
  "layers_watermarked": ["model.layers.0.self_attn.q_proj.weight", "..."],
  "bits_per_weight": 1,
  "carrier_count": 336
}
```

**Errors:** 400 (bad payload/bits/empty model), 403 (not admin / not hub source), 503 (default/empty secret; executor unavailable), 404 (model unresolvable), 500 (embed failed).

**Shape:** long-running like `/convert` (loads full model, writes safetensors). BUT: the Hub calls `/merge-adapter` **synchronously** and expects `output_path` in the body — watermark embed is the same class (Hub calls it synchronously, bounded by its 300s client timeout). So `embed`/`verify` use the **synchronous `asyncio.wrap_future` shape** (like `/merge-adapter`), NOT the async job-poll shape (like `/convert`). This is ruling R1.

### `POST /v1/watermark/verify`

**Request** (`WatermarkVerifyRequest`):
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | `str` | yes | — | Model to verify (path/alias/repo) |
| `secret` | `str` | yes | — | Same secret used at embed |
| `layers` | `list[str]` \| `null` | no | `null` | Same restriction used at embed (must match for correct read-back) |
| `bits_per_weight` | `int` | no | `1` | Same value used at embed |

**Response** (200):
```json
{
  "verified": true,
  "model": "org/repo",
  "payload": {"owner": "dahai80", "embedded_at": "..."},
  "signature": "abcd1234...",
  "mismatch_rate": 0.0,
  "carrier_count": 336
}
```
On failure: `verified=false`, `payload=null`, `signature=""`, `mismatch_rate` = float or `null` (if no reference reconstructable), `reason` field explains (e.g. "payload bytes did not parse as JSON", "signature mismatch", "no carrier tensors found").

**Note:** `verify` does NOT return the secret. It returns the embedded payload + a re-computed signature; the caller (Hub) compares the signature to the one it stored at embed time (defense-in-depth, mirroring Hub's sidecar+DB dual-source verify).

## 7. Core Algorithm (lsb.py)

Pure-Python + numpy, no MLX import. Operates on numpy arrays so it's unit-testable without loading a model. The route layer bridges MLX↔numpy (numpy-bridge on caller thread, per #630 thread-portability invariant).

### Functions

```
PRNG_SEED(seed_bytes: bytes) -> numpy.random.Generator
    # Deterministic from (secret, tensor_name). seed = sha256(secret + b":" + tensor_name.encode())
    # numpy default_rng(seed_bytes) accepts a seed of arbitrary length (hashes internally).

def embed_bits(weights: np.ndarray, bit_array: np.ndarray, generator: np.random.Generator,
               bits_per_weight: int = 1, epsilon: float = 1e-6) -> tuple[np.ndarray, int]:
    """Embed bit_array into weights' LSBs at PRNG-chosen positions.
    Returns (watermarked_weights, carrier_count). Skips |w|<=epsilon.
    Raises ValueError if bit_array longer than capacity (carrier_count * bits_per_weight)."""

def extract_bits(weights: np.ndarray, n_bits: int, generator: np.random.Generator,
                 bits_per_weight: int = 1, epsilon: float = 1e-6) -> tuple[np.ndarray, int]:
    """Read n_bits from the same PRNG positions. Returns (bit_array, carrier_count_read)."""

def payload_to_bits(payload: dict) -> np.ndarray:
    """JSON-dumps(sort_keys=True) → UTF-8 bytes → uint8 bit array (MSB-first per byte).
    Prepends a 4-byte big-endian length header so extract knows how many bytes to read."""

def bits_to_payload(bits: np.ndarray) -> dict | None:
    """Inverse. Reads length header, slices, UTF-8 decode, json.loads.
    Returns None on any failure (truncated, invalid UTF-8, invalid JSON)."""

def compute_signature(secret: str, model: str, payload: dict) -> str:
    """sha256(f'{secret}:{model}::' + json.dumps(payload, sort_keys=True)).hexdigest()[:32].
    Double-colon separates model from payload-json because model_id may contain ':' in edge cases.
    NOTE: Hub uses single ':' + version_id; fusion-mlx has no version concept → empty version.
    Ruling R2: signature format = sha256(secret:model::json.dumps(payload,sort_keys=True))[:32]."""
```

### Capacity check

`capacity = carrier_count * bits_per_weight`. Payload bits (with 4-byte length header) must fit. Route returns 400 with the capacity number if payload too large for the selected tensors, so the caller can narrow `layers` or raise `bits_per_weight`.

### Quantized-tensor detection

A tensor is **ineligible** (skipped) if:
- its dtype is integer (`np.integer`), OR
- its name appears in the model's quantization config (`config.json` `quantization`/`quantization_config` keys map to weight-name prefixes), OR
- MLX exposes it as a packed `QuantizedLinear` weight (route detects via `hasattr(module, 'quantization')` when a full model load is available; for weights-only loads, the name-prefix check against config is the signal).

The route reads `config.json` alongside the weights to get the quantized-prefix list. Ruling R3: quantized-tensor detection = config-driven prefix match + integer-dtype check; never touch packed codes.

## 8. Hub Alignment

The Hub (`fusion_model_hub/server/routers/watermark.py`) signs with:
```python
raw = f"{secret}:{model_id}:{version_id}:{json.dumps(payload, sort_keys=True)}"
signature = hashlib.sha256(raw.encode()).hexdigest()[:32]
```
Secret from `FMH_WATERMARK_SECRET` env; refuses to sign on default/empty (E-S6).

fusion-mlx aligns:
- **Secret env var:** `FMH_WATERMARK_SECRET` (same name → one secret configures both systems). Route reuses the same non-default check; 503 on default/empty.
- **Signature scheme:** `sha256(f'{secret}:{model}::{json.dumps(payload, sort_keys=True)}')[:32]`. The Hub includes a `version_id`; fusion-mlx has no version concept → empty version, but the delimiter is kept as `::` so a Hub caller can map `version_id=""` and recompute identically. This lets the Hub embed the **same payload** via both paths (DB+sidecar AND tensor) and have one signature verify against both. Ruling R2 locks the format.
- **payload enrichment:** the Hub adds `embedded_at` + `owner` to the payload before signing. fusion-mlx does NOT enrich — it embeds exactly what the caller sends. If the Hub wants those fields, it sends them in the `payload` dict. Keeps fusion-mlx stateless and avoids a clock dependency (no `datetime.now` in the engine path; aligns with #630/no-`Date.now` discipline).
- **No DB, no sidecar:** fusion-mlx touches only tensors. The Hub continues to own the DB row + `watermark.json` sidecar. The tensor watermark is a **third** source of truth that survives even when DB and sidecar are stripped.

## 9. Error Handling & Fail-Visible

(Rule 12: fail visibly, not silently. Rule 7: surface conflicts, don't average.)

| Condition | Behavior |
|-----------|----------|
| Default/empty secret | 503 "Watermark disabled: set non-default FMH_WATERMARK_SECRET" (E-S6) |
| Model unresolvable | 404 with resolved-attempt detail |
| All tensors quantized / no eligible carriers | 400 "No eligible (non-quantized float) weight tensors to watermark" — do NOT silently no-op and return a fake signature |
| Payload too large for capacity | 400 with capacity + payload_bytes numbers |
| `bits_per_weight` not in 1-3 | 422 (Pydantic validation) |
| `in_place=false` and no `output_path` | 422 (Pydantic validation — `output_path` required when `in_place` is false) |
| Embed write fails (disk full, perms) | 500 with the underlying OSError message, executor job stays failed |
| Verify: no carrier tensors found | 200 with `verified=false, reason="no carrier tensors found"` (verify is a query, not a mutation — 200 not 4xx) |
| Verify: payload bits don't parse as JSON | 200 `verified=false, reason="payload not recoverable (corrupted/quantized)"` |
| Verify: signature mismatch | 200 `verified=false, reason="signature mismatch"` |
| Verify: mismatch_rate > 0 but payload parses | 200 `verified=true, mismatch_rate=<float>` (bits survived enough to reconstruct; report damage) |

**Never** swallow an exception into a `verified=false` 200 without logging it at WARNING/ERROR. Every verify failure path logs the reason at INFO (expected) or WARNING (unexpected). Embed failures log at ERROR + `logger.exception`.

## 10. Testing Strategy

### Unit tests (`tests/unit/test_watermark_lsb.py`) — no model load

Pure algorithm tests against random numpy arrays:
1. `test_embed_extract_roundtrip` — embed bits, extract bits, bit-exact match (deterministic PRNG).
2. `test_payload_to_bits_bits_to_payload_roundtrip` — dict → bits → dict, equal.
3. `test_bits_to_payload_corrupt_returns_none` — flip a payload bit → `bits_to_payload` returns None.
4. `test_embed_skips_tiny_weights` — weights below epsilon are never carriers (assert no bit flipped there).
5. `test_embed_capacity_exceeded_raises` — payload bits > capacity → ValueError.
6. `test_prng_determinism_same_seed` — same `(secret, name)` → same positions (two generators, same seed, identical position sequence).
7. `test_quantized_tensor_skipped` — integer-dtype array → skipped (capacity 0).
8. `test_signature_matches_hub_format` — signature string equals `sha256(f'{secret}:{model}::{json.dumps(payload,sort_keys=True)}')[:32]`.
9. `test_magnitude_safety_no_nan` — watermarked weights contain no NaN/inf (flip didn't hit denormal sign bit).

### Route tests (`tests/unit/test_watermark_routes.py`) — TestClient, no real model

1. `test_embed_rejects_default_secret` → 503.
2. `test_embed_rejects_empty_secret` → 503.
3. `test_embed_requires_admin` → 403 without admin.
4. `test_embed_requires_hub_source` → 403 without hub source (mirrors `/merge-adapter`).
5. `test_embed_payload_too_large` → 400 with capacity (monkeypatched small weights).
6. `test_verify_no_carriers` → 200 `verified=false`.
7. `test_embed_then_verify_roundtrip` — monkeypatch the executor + load to use an in-memory random-weight fixture; embed → verify → `verified=true`, payload matches, `mismatch_rate=0.0`.
8. `test_verify_corrupted_returns_false` — embed, flip a payload bit in the fixture, verify → `verified=false`.

Route tests monkeypatch `mlx_lm.utils.load`/weight loading + `save_safetensors` to an in-memory fixture so no real model loads (unit-test speed). The real-model test is gated separately.

### Real-model integration test (gated `FUSION_MLX_REAL_MODEL_TESTS`) — `tests/unit/test_watermark_real_model.py`

Per CLAUDE.md ("涉及到大模型测试，须真实加载模型"), the genuine round-trip loads a small real model from `~/.fusion-mlx/models`:
1. Skip if env unset.
2. Load a small LLM (e.g. `Qwen2.5-0.5B` if present; skip-with-reason if no small model available — do NOT download in-test).
3. Embed a payload → write copy to temp dir.
4. Reload copy → verify → assert `verified=true`, payload matches, `mismatch_rate=0.0`.
5. Quantize the copy (4-bit) → reload → verify → assert `mismatch_rate` reported (bits may partially survive; test asserts the route returns a structured result, NOT that bits survive quantize — that's a known partial-survival case, documented).
6. Clean up temp dir (CLAUDE.md: clean process data, keep only final outputs + logs).

This test is the load-bearing correctness gate (Rule 9: meaningful test). Unit tests prove the algorithm; this proves the MLX↔numpy bridge + safetensors round-trip on a real format.

## 11. Security Considerations

- **Secret handling:** `FMH_WATERMARK_SECRET` read from env at call time, never logged, never returned in any response. Route 503s on default/empty (E-S6 — prior Hub default was a source-public constant, forgeable by anyone with the source). The secret must be high-entropy; the route does not enforce entropy (out of scope) but does refuse the documented default sentinel.
- **Constant-time compare:** verify's signature comparison uses `hmac.compare_digest` (mirrors Hub E-S6) so a mismatch does not leak how many leading bytes matched (timing oracle for forgery). The internal bit-match for `mismatch_rate` is not timing-sensitive (it's a count, not a gate).
- **Admin gate:** both routes `Depends(require_admin)` + `Depends(require_model_hub_source)`, exactly like `/merge-adapter`. Watermark embed mutates weights — same trust level as quantize/convert.
- **Path validation:** `output_path` reuses `_get_allowed_output_prefixes()` from `convert_models.py` (within `~/.fusion-mlx/models`, HF cache, or CWD). No arbitrary filesystem write.
- **No code execution from payload:** payload is JSON-serialized to bytes then to bits; never `eval`'d or `exec`'d. `json.loads` only.
- **DoS via huge payload:** capacity check (§7) rejects payload larger than `carrier_count * bits_per_weight` bits. A payload that fits but is large just takes proportionally longer — bounded by model size. No unbounded loop.
- **Quantized weights untouched:** §7 quantized-tensor detection. Watermarking packed int4 codes would corrupt the scale/zero contract and produce garbage inference — hard skip, not a graceful degrade.

## 12. File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `fusion_mlx/watermark/__init__.py` | create | Package marker; re-exports public API |
| `fusion_mlx/watermark/lsb.py` | create | Pure-numpy LSB spread-spectrum core: `embed_bits`, `extract_bits`, `payload_to_bits`, `bits_to_payload`, `compute_signature`, `PRNG_SEED`. No MLX import. Unit-testable standalone. |
| `fusion_mlx/api/watermark_models.py` | create | Pydantic: `WatermarkEmbedRequest`, `WatermarkVerifyRequest`. Mirrors `convert_models.py` pattern + `output_path` validator reuse. |
| `fusion_mlx/api/watermark_routes.py` | create | APIRouter `/v1/watermark/embed` + `/verify`. Secret env check, model resolution, executor submit, MLX↔numpy bridge, safetensors save/load. Reuses `convert_routes._executor` (same single-worker pool — watermark is OOM-class alongside convert/quantize/merge). |
| `fusion_mlx/server.py` | modify | `include_router(watermark_router)` next to `convert_router` (line ~1023). Import + mount. |
| `fusion_mlx/_version.py` | modify | bump 0.8.57 → 0.8.58 |
| `CHANGELOG.md` | modify | add `[0.8.58]` entry |
| `README.md` | modify | document `/v1/watermark/*` endpoints + `FMH_WATERMARK_SECRET` |
| `tests/unit/test_watermark_lsb.py` | create | 9 algorithm unit tests (§10) |
| `tests/unit/test_watermark_routes.py` | create | 8 route contract tests, monkeypatched load (§10) |
| `tests/unit/test_watermark_real_model.py` | create | gated real-model round-trip (§10) |

**No new deps.** numpy already a dependency. `hashlib`/`hmac`/`json` stdlib. MLX already present.

## 13. Open Questions / Rulings

- **R1 (sync shape, not job-poll):** `/v1/watermark/embed` + `/verify` use the synchronous `asyncio.wrap_future` shape (like `/merge-adapter`), NOT the async job-poll shape (like `/convert`). Rationale: the Hub calls weight-mutating endpoints synchronously and expects `output_path` in the body, bounded by its 300s client timeout. Watermark embed is the same class. Cost if wrong: Hub integration breaks if we return a `job_id` to poll — but this is detectable in integration and reversible.
- **R2 (signature format):** `sha256(f'{secret}:{model}::{json.dumps(payload, sort_keys=True)}')[:32]`. Double-colon delimits model from payload-json (model_id may contain `:` in edge cases; version_id is empty for fusion-mlx). Aligns with Hub's single-`:`+version format when version=`""`. Cost if wrong: cross-system signature mismatch → Hub's dual-source verify fails; detectable, reversible.
- **R3 (quantized-tensor detection):** config-driven prefix match (read `config.json` quantization keys → weight-name prefixes) + integer-dtype check. Never touch packed int codes. Cost if wrong: watermarking packed codes corrupts inference (garbage outputs) — severe, but the detection is conservative (skip-on-doubt).
- **R4 (secret env name):** `FMH_WATERMARK_SECRET`, same as Hub. One env configures both. Cost if wrong: operator configures two secrets, signatures diverge — detectable.
- **R5 (executor reuse):** watermark jobs run on `convert_routes._executor` (single-worker pool). Watermark is OOM-class (loads full model). Serializing alongside convert/quantize/merge avoids OOM. Cost if wrong: a concurrent convert + watermark could OOM on one machine — but the single-worker pool prevents it.
- **R6 (no payload enrichment by fusion-mlx):** fusion-mlx embeds exactly the payload dict sent. Hub owns `embedded_at`/`owner` enrichment. Keeps fusion-mlx stateless, no clock dependency. Cost if wrong: none — caller controls payload.
- **R7 (verify returns 200 on `verified=false`):** verify is a query, not a mutation. A negative result is a successful query result, not an error. Error codes (4xx/5xx) reserved for request-shape/server failures. Cost if wrong: if caller treats non-200 as "not watermarked," a 4xx would be indistinguishable from "no watermark" — but we return 200 with structured `verified=false`, so the caller gates on the `verified` field.
