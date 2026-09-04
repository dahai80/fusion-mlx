# Telemetry & Observability Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Align fusion-mlx telemetry/observability with Rapid-MLX (parity on shared surface) and lead beyond it (multimodal/Paged-KV/distributed/lifespan dimensions Rapid structurally lacks).

**Architecture:** Two PRs split at prod-behavior boundary. PR1 = telemetry package + activation funnel (dark, consent-gated, zero prod call sites). PR2 = per-request/error wiring + Prometheus metric expansion (prod-visible). Section 4 lead capabilities span both PRs: activation milestones in PR1, multimodal/Paged-KV/distributed/lifespan metrics in PR2.

**Tech Stack:** Python (pure). MLX 0.32.0, mlx_lm 0.31.3, Python 3.12, FastAPI. No Rust — telemetry is fire-and-forget bounded queue, not perf-critical; target is Python; Rust = toolchain overhead with no benefit.

**Spec:** `docs/superpowers/specs/2026-09-04-telemetry-observability-alignment-design.md` (approved). Feature gaps filed: #787 (suffix-decode), #788 (MTP), #789 (mxfp4 guardrail) — not telemetry responsibility; their metric families wire when features land.

## Global Constraints

- Only modify `/Users/dahai/claude-home/fusion-mlx` code
- 4-multiple indentation, NO docstrings, code must log (binding)
- No `mx.clear_streams()` in tests (#630 GOTCHA); `mx.clear_cache()` allowed
- Lint: `black --check --target-version py313` + `ruff`; never touch `debt_modules.txt`
- Real model loads gated `FUSION_*_REAL_MODEL=on`; default small model; no OOM (binding)
- README.md English-only; README_CN.md Chinese; GitHub ops English (binding)
- PyPI SKIPPED (no token); squash-merge PRs to main; push via SSH `git@github.com:dahai80/fusion-mlx.git`
- Clean process data after tests, keep only final artifacts + logs (binding)
- Tests must assert something meaningful (Rule 9)
- Telemetry consent-gated: all emit calls short-circuit when `is_enabled()==False`; wiring tests assert no-fire when disabled

## File Structure

**PR1 (dark library) — modified/new files:**
- Create: `fusion_mlx/telemetry/activation_spec.py` — activation funnel spec (constants + predicates)
- Create: `fusion_mlx/telemetry/coherence.py` — deterministic output-degenerate heuristic (used by PR2 wiring but ships in PR1 as pure lib)
- Modify: `fusion_mlx/telemetry/schema.py` — +ActivationPayload, RequestPayload fields, sample_request_preview_payload, TelemetryPayload.activation slot
- Modify: `fusion_mlx/telemetry/redact.py` — +normalize_caller_agent + _CALLER_AGENT_MARKERS
- Modify: `fusion_mlx/telemetry/state.py` — +activation marker functions, reset_state extension, CURRENT_CONSENT_SCHEMA_VERSION 1→3
- Modify: `fusion_mlx/telemetry/emit.py` — +activation(), +server_surface(), request() field widening + sampling gate
- Modify: `fusion_mlx/cli.py` — telemetry preview shows request sample
- Create: `docs/telemetry-activation.md` — human-facing activation funnel spec (English)
- Create: `tests/unit/test_telemetry_activation_spec.py`
- Create: `tests/unit/test_telemetry_activation_marker.py`
- Create: `tests/unit/test_telemetry_request_preview_payload.py`
- Create: `tests/unit/test_telemetry_caller_agent_redact.py`
- Create: `tests/unit/test_telemetry_coherence.py`

**PR2 (prod-visible) — modified/new files:**
- Modify: `fusion_mlx/routes_internal/chat.py` — emit.request + emit.activation call sites (stream + non-stream)
- Modify: `fusion_mlx/server.py` — emit.error call sites (lifespan startup/shutdown)
- Modify: `fusion_mlx/cli.py` — emit.activation(model_pull) + emit.error(model_load_failure)
- Modify: `fusion_mlx/server_metrics.py` — per-model histogram collection, uptime, modality counters
- Modify: `fusion_mlx/routes_internal/metrics.py` — histogram rendering, prefix/radix/KV-checkpoint/spec-decode/UBC/turboquant/pflash/embedding/queue/metal/lifespan/distributed families, fix rapid_mlx_ prefix, Section 4 lead families
- Create: `tests/unit/test_telemetry_request_wiring.py`
- Create: `tests/unit/test_telemetry_streaming_request_wiring.py`
- Create: `tests/unit/test_telemetry_error_wiring.py`
- Create: `tests/unit/test_telemetry_route_attribution.py`
- Create: `tests/unit/test_prometheus_metric_expansion.py`
- Create: `tests/unit/test_prometheus_histogram_render.py`
- Create: `tests/unit/test_prometheus_multimodal_metrics.py`
- Create: `tests/unit/test_telemetry_activation_multimodal.py`
- Create: `tests/unit/test_prometheus_kv_cache_advanced.py`
- Create: `tests/unit/test_prometheus_lifespan_metrics.py`

---

## PR1 — Telemetry package + activation funnel (dark)

**Ruling — spec v2→v3 in PR1:** Spec Section 1 ports Rapid's activation_spec at v2 (7 milestones). Spec Section 4 extends to v3 (9 milestones: +`first_image_generation`, +`first_video_generation`). Since `activation_spec.py` is a single PR1 file and Section 4's multimodal milestones are part of the approved lead capability, PR1 ships the FULL v3 spec (9 milestones) in one file rather than port v2 then re-edit to v3 in PR2. One file, one version, no churn. `CURRENT_CONSENT_SCHEMA_VERSION` likewise jumps 1→3 directly (skips 2). This supersedes the spec's Section 1 "v2" wording — Section 4's v3 is the binding final state, and the plan lands the final state.

### Task 1: activation_spec.py module

**Files:**
- Create: `fusion_mlx/telemetry/activation_spec.py`
- Create: `tests/unit/test_telemetry_activation_spec.py`

**Interfaces:**
- Produces: `ACTIVATION_SPEC_VERSION=3`, 9 milestone kinds (7 Rapid + `first_image_generation` + `first_video_generation`), 3 surfaces, `ACTIVATION_KIND_SURFACE_PAIRS`, `is_allowed_activation(kind, surface)`, `CHAT_SPAWN_ENV="FUSION_MLX_CHAT_SPAWN"`, `INFERENCE_ENDPOINTS`, `is_successful_inference(status, completion_tokens)`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_telemetry_activation_spec.py
# SPDX-License-Identifier: Apache-2.0
from fusion_mlx.telemetry import activation_spec as spec


def test_spec_version_is_3():
    assert spec.ACTIVATION_SPEC_VERSION == 3


def test_nine_milestone_kinds():
    expected = {
        "first_inference", "model_pull", "agent_setup",
        "first_chat_reply", "first_vision_reply", "first_dictation",
        "first_image", "first_image_generation", "first_video_generation",
    }
    assert spec.ACTIVATION_KINDS == expected


def test_multimodal_pairs_on_api_surface():
    assert ("first_image_generation", "api") in spec.ACTIVATION_KIND_SURFACE_PAIRS
    assert ("first_video_generation", "api") in spec.ACTIVATION_KIND_SURFACE_PAIRS


def test_is_allowed_activation_rejects_unknown():
    assert spec.is_allowed_activation("first_inference", "api") is True
    assert spec.is_allowed_activation("bogus", "api") is False
    assert spec.is_allowed_activation("first_inference", "bogus") is False


def test_chat_spawn_env_renamed():
    assert spec.CHAT_SPAWN_ENV == "FUSION_MLX_CHAT_SPAWN"


def test_inference_endpoints_chat_only():
    assert spec.INFERENCE_ENDPOINTS == frozenset({"/v1/chat/completions"})


def test_is_successful_inference_2xx_nonempty():
    assert spec.is_successful_inference(200, 5) is True
    assert spec.is_successful_inference(200, 0) is False
    assert spec.is_successful_inference(500, 5) is False
    assert spec.is_successful_inference(299, 1) is True
    assert spec.is_successful_inference(300, 5) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_telemetry_activation_spec.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# fusion_mlx/telemetry/activation_spec.py
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

ACTIVATION_SPEC_VERSION = 3

ACTIVATION_FIRST_INFERENCE = "first_inference"
ACTIVATION_MODEL_PULL = "model_pull"
ACTIVATION_AGENT_SETUP = "agent_setup"
ACTIVATION_FIRST_CHAT_REPLY = "first_chat_reply"
ACTIVATION_FIRST_VISION_REPLY = "first_vision_reply"
ACTIVATION_FIRST_DICTATION = "first_dictation"
ACTIVATION_FIRST_IMAGE = "first_image"
ACTIVATION_FIRST_IMAGE_GENERATION = "first_image_generation"
ACTIVATION_FIRST_VIDEO_GENERATION = "first_video_generation"

SURFACE_CLI = "cli"
SURFACE_API = "api"
SURFACE_DESKTOP = "desktop"

ACTIVATION_KIND_SURFACE_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        (ACTIVATION_FIRST_INFERENCE, SURFACE_CLI),
        (ACTIVATION_FIRST_INFERENCE, SURFACE_API),
        (ACTIVATION_MODEL_PULL, SURFACE_CLI),
        (ACTIVATION_AGENT_SETUP, SURFACE_CLI),
        (ACTIVATION_FIRST_CHAT_REPLY, SURFACE_DESKTOP),
        (ACTIVATION_FIRST_VISION_REPLY, SURFACE_DESKTOP),
        (ACTIVATION_FIRST_DICTATION, SURFACE_DESKTOP),
        (ACTIVATION_FIRST_IMAGE, SURFACE_DESKTOP),
        (ACTIVATION_FIRST_IMAGE_GENERATION, SURFACE_API),
        (ACTIVATION_FIRST_VIDEO_GENERATION, SURFACE_API),
    }
)
ACTIVATION_KINDS: frozenset[str] = frozenset(kind for kind, _ in ACTIVATION_KIND_SURFACE_PAIRS)
ACTIVATION_SURFACES: frozenset[str] = frozenset(surface for _, surface in ACTIVATION_KIND_SURFACE_PAIRS)
DESKTOP_ACTIVATION_KINDS: frozenset[str] = frozenset(
    kind for kind, surface in ACTIVATION_KIND_SURFACE_PAIRS if surface == SURFACE_DESKTOP
)


def is_allowed_activation(activation_kind: str, surface: str) -> bool:
    return (activation_kind, surface) in ACTIVATION_KIND_SURFACE_PAIRS


CHAT_SPAWN_ENV = "FUSION_MLX_CHAT_SPAWN"

INFERENCE_ENDPOINTS: frozenset[str] = frozenset({"/v1/chat/completions"})


def is_successful_inference(status: int, completion_tokens: int) -> bool:
    try:
        status_ok = 200 <= int(status) < 300
        nonempty = int(completion_tokens) > 0
    except (TypeError, ValueError):
        logger.warning("activation_spec: bad inference args status=%r tokens=%r", status, completion_tokens)
        return False
    return status_ok and nonempty
```

NOTE: fix `ACTIVATION_FIRST_VISION_REPLY = "first_vision_reply"` typo — must be `"first_vision_reply"` (lowercase) to match Rapid + test expectation. Verify all kind constants lowercase.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_telemetry_activation_spec.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Lint + commit**

Run: `black --check --target-version py313 fusion_mlx/telemetry/activation_spec.py tests/unit/test_telemetry_activation_spec.py && ruff check fusion_mlx/telemetry/activation_spec.py tests/unit/test_telemetry_activation_spec.py`
Commit: `feat(telemetry): add activation_spec.py funnel module (9 milestones, spec v3)`

### Task 2: schema.py — ActivationPayload + request fields + request preview

**Files:**
- Modify: `fusion_mlx/telemetry/schema.py`
- Create: `tests/unit/test_telemetry_request_preview_payload.py`

**Interfaces:**
- Consumes: `SCHEMA_VERSION`, `PlatformInfo`, `platform_info()`, `_utc_now_iso()`
- Produces: `ActivationPayload(activation_kind, surface, client_id, spec_version, occurred_at_epoch)`, `RequestPayload` +`caller_agent`/`output_degenerate`/`completion_empty`/`completion_abnormally_short`, `TelemetryPayload.activation` slot, `sample_request_preview_payload(client_id, fusion_mlx_version)`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_telemetry_request_preview_payload.py
# SPDX-License-Identifier: Apache-2.0
from fusion_mlx.telemetry.schema import (
    ActivationPayload, RequestPayload, TelemetryPayload,
    sample_request_preview_payload,
)


def test_activation_payload_fields():
    ap = ActivationPayload(
        activation_kind="first_inference", surface="api",
        client_id="abc", spec_version=3, occurred_at_epoch=1700000000,
    )
    assert ap.activation_kind == "first_inference"
    assert ap.surface == "api"


def test_request_payload_new_fields_default():
    rp = RequestPayload(
        endpoint="/v1/chat/completions", model_alias="m", stream=False,
        tool_call_used=False, prompt_tokens_bucket="0", completion_tokens_bucket="1-100",
        ttft_ms_bucket="100-500", tps_bucket="10-50", status=200,
        caller_agent="claude-code", output_degenerate=False,
        completion_empty=False, completion_abnormally_short=False,
    )
    assert rp.caller_agent == "claude-code"


def test_sample_request_preview_payload_event():
    p = sample_request_preview_payload(client_id="x", fusion_mlx_version="0.9.0")
    assert p.event == "request"
    assert p.request is not None
    assert p.request.endpoint == "/v1/chat/completions"


def test_telemetry_payload_has_activation_slot():
    import dataclasses
    fields = {f.name for f in dataclasses.fields(TelemetryPayload)}
    assert "activation" in fields
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_telemetry_request_preview_payload.py -v`
Expected: FAIL — ActivationPayload missing, fields missing

- [ ] **Step 3: Implement — add ActivationPayload dataclass**

After `ErrorPayload` class, add:

```python
@dataclass(frozen=True)
class ActivationPayload:
    activation_kind: str
    surface: str
    client_id: str
    spec_version: int
    occurred_at_epoch: int
```

- [ ] **Step 4: Implement — extend RequestPayload**

Add 4 fields to `RequestPayload` (after `status: int`):

```python
    caller_agent: str = "other"
    output_degenerate: bool = False
    completion_empty: bool = False
    completion_abnormally_short: bool = False
```

- [ ] **Step 5: Implement — extend TelemetryPayload**

Add `event` value `"activation"` to docstring; add field after `error`:

```python
    activation: ActivationPayload | None = None
```

Update `to_dict()` to drop `activation` when None — add `"activation"` to the drop loop:

```python
        for key in ("session", "request", "error", "activation"):
```

- [ ] **Step 6: Implement — add sample_request_preview_payload**

After `sample_preview_payload`, add:

```python
def sample_request_preview_payload(
    *,
    client_id: str,
    fusion_mlx_version: str,
) -> TelemetryPayload:
    info = platform_info()
    return TelemetryPayload(
        schema_version=SCHEMA_VERSION,
        client_id=client_id,
        session_id="preview-0000000000000000",
        fusion_mlx_version=fusion_mlx_version,
        platform=PlatformInfo(
            os=info["os"], os_version=info["os_version"], arch=info["arch"],
            chip=info["chip"], memory_gb=info["memory_gb"], python_version=info["python_version"],
        ),
        event="request",
        timestamp=_utc_now_iso(),
        request=RequestPayload(
            endpoint="/v1/chat/completions",
            model_alias="mlx-community/Qwen3.5-9B-4bit",
            stream=False,
            tool_call_used=False,
            prompt_tokens_bucket="100-500",
            completion_tokens_bucket="100-500",
            ttft_ms_bucket="100-500",
            tps_bucket="10-50",
            status=200,
            caller_agent="claude-code",
        ),
    )
```

- [ ] **Step 7: Add to `__all__`**

Add `"ActivationPayload"`, `"sample_request_preview_payload"` to `__all__`.

- [ ] **Step 8: Run tests + lint + commit**

Run: `python -m pytest tests/unit/test_telemetry_request_preview_payload.py tests/unit/test_telemetry_activation_spec.py -v`
Run: `black --check --target-version py313 fusion_mlx/telemetry/schema.py && ruff check fusion_mlx/telemetry/schema.py`
Commit: `feat(telemetry): add ActivationPayload + request fields + request preview to schema`

### Task 3: redact.py — normalize_caller_agent

**Files:**
- Modify: `fusion_mlx/telemetry/redact.py`
- Create: `tests/unit/test_telemetry_caller_agent_redact.py`

**Interfaces:**
- Produces: `normalize_caller_agent(user_agent: str) -> str`, `_CALLER_AGENT_MARKERS: list[tuple[str, str]]`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_telemetry_caller_agent_redact.py
# SPDX-License-Identifier: Apache-2.0
from fusion_mlx.telemetry.redact import normalize_caller_agent


def test_known_agents_bucketed():
    assert normalize_caller_agent("claude-cli/1.0 something") == "claude-code"
    assert normalize_caller_agent("Cursor/0.42") == "cursor"
    assert normalize_caller_agent("aider 0.5") == "aider"


def test_unknown_agent_other():
    assert normalize_caller_agent("Mozilla/5.0") == "other"
    assert normalize_caller_agent("") == "other"
    assert normalize_caller_agent(None) == "other"


def test_no_raw_ua_leaks():
    out = normalize_caller_agent("claude-cli/1.0 (secret-token-here)")
    assert "secret-token-here" not in out
    assert out == "claude-code"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_telemetry_caller_agent_redact.py -v`
Expected: FAIL — function missing

- [ ] **Step 3: Implement**

Add after `normalize_model_path` in redact.py:

```python
_CALLER_AGENT_MARKERS: list[tuple[str, str]] = [
    ("claude-cli", "claude-code"),
    ("claude-code", "claude-code"),
    ("cursor", "cursor"),
    ("aider", "aider"),
    ("continue", "continue"),
    ("cline", "cline"),
    ("roo", "roo"),
]


def normalize_caller_agent(user_agent: str | None) -> str:
    if not user_agent:
        return "other"
    ua_lower = user_agent.lower()
    for marker, label in _CALLER_AGENT_MARKERS:
        if marker in ua_lower:
            return label
    logger.debug("redact: unrecognized caller agent, bucketed as other")
    return "other"
```

Ensure `logger` is defined at module top (check existing redact.py — if not, add `logger = logging.getLogger(__name__)`).

- [ ] **Step 4: Run tests + lint + commit**

Run: `python -m pytest tests/unit/test_telemetry_caller_agent_redact.py -v`
Run: `black --check --target-version py313 fusion_mlx/telemetry/redact.py && ruff check fusion_mlx/telemetry/redact.py`
Commit: `feat(telemetry): add normalize_caller_agent UA bucketing`

### Task 4: state.py — activation markers + consent schema v3

**Files:**
- Modify: `fusion_mlx/telemetry/state.py`
- Create: `tests/unit/test_telemetry_activation_marker.py`

**Interfaces:**
- Produces: `activation_marker_path(kind) -> Path`, `claim_activation_marker(kind) -> bool`, `_validate_activation_kind(kind) -> bool`; `reset_state()` extended to wipe markers; `CURRENT_CONSENT_SCHEMA_VERSION = 3`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_telemetry_activation_marker.py
# SPDX-License-Identifier: Apache-2.0
import pytest
from fusion_mlx.telemetry import state


@pytest.fixture
def tmp_telemetry_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "_default_telemetry_dir", lambda: tmp_path)
    return tmp_path


def test_claim_once_per_install(tmp_telemetry_dir):
    assert state.claim_activation_marker("first_inference") is True
    assert state.claim_activation_marker("first_inference") is False


def test_claim_rejects_unknown_kind(tmp_telemetry_dir):
    with pytest.raises(ValueError):
        state.claim_activation_marker("bogus_kind")


def test_reset_wipes_markers(tmp_telemetry_dir):
    state.claim_activation_marker("first_inference")
    state.reset_state()
    assert state.claim_activation_marker("first_inference") is True


def test_consent_schema_version_is_3():
    assert state.CURRENT_CONSENT_SCHEMA_VERSION == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_telemetry_activation_marker.py -v`
Expected: FAIL — functions missing

- [ ] **Step 3: Implement — bump schema version**

Change `CURRENT_CONSENT_SCHEMA_VERSION = 1` → `CURRENT_CONSENT_SCHEMA_VERSION = 3`.

NOTE: bump to 3 (not 2) because fusion's Section 4 extends the activation spec beyond Rapid's v2 — schema gains activation slot + multimodal milestones. Forces re-consent.

- [ ] **Step 4: Implement — marker functions**

Add after `client_id_path`/`consent_path` helpers, before `ConsentState`:

```python
_activation_latch: set[str] = set()


def _validate_activation_kind(kind: str) -> bool:
    from .activation_spec import ACTIVATION_KINDS
    return kind in ACTIVATION_KINDS


def activation_marker_path(kind: str) -> Path:
    if not _validate_activation_kind(kind):
        raise ValueError(f"unknown activation kind: {kind}")
    return _default_telemetry_dir() / f"activation_seen_{kind}"


def claim_activation_marker(kind: str) -> bool:
    if kind in _activation_latch:
        return False
    path = activation_marker_path(kind)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        _activation_latch.add(kind)
        logger.info("activation marker already present: %s", kind)
        return False
    except OSError:
        logger.warning("activation marker write failed: %s", kind, exc_info=True)
        return False
    _activation_latch.add(kind)
    logger.info("activation milestone claimed: %s", kind)
    return True
```

Ensure `import os` at top (check existing imports).

- [ ] **Step 5: Implement — extend reset_state**

In existing `reset_state()`, after clearing consent/client-id, add marker cleanup:

```python
    import glob
    for marker in glob.glob(str(_default_telemetry_dir() / "activation_seen_*")):
        try:
            os.remove(marker)
        except OSError:
            logger.warning("failed to remove activation marker %s", marker, exc_info=True)
    _activation_latch.clear()
```

- [ ] **Step 6: Run tests + lint + commit**

Run: `python -m pytest tests/unit/test_telemetry_activation_marker.py -v`
Run: `black --check --target-version py313 fusion_mlx/telemetry/state.py && ruff check fusion_mlx/telemetry/state.py`
Commit: `feat(telemetry): add activation markers + bump consent schema to v3`

### Task 5: emit.py — activation() + server_surface() + request() widening + sampling

**Files:**
- Modify: `fusion_mlx/telemetry/emit.py`

**Interfaces:**
- Produces: `activation(activation_kind, surface, **extra)`, `server_surface() -> str`, `request()` widened with `caller_agent`/`output_degenerate`/`completion_empty`/`completion_abnormally_short` + sampling gate (`_request_sample_rate()`, `_should_sample_request()`)

- [ ] **Step 1: Implement — server_surface()**

Add near top of emit.py (after imports):

```python
def server_surface() -> str:
    import os
    from .activation_spec import CHAT_SPAWN_ENV, SURFACE_CLI, SURFACE_API
    return SURFACE_CLI if os.environ.get(CHAT_SPAWN_ENV) else SURFACE_API
```

- [ ] **Step 2: Implement — activation()**

Add after `request()`:

```python
def activation(activation_kind: str, surface: str, **extra: object) -> None:
    if not is_enabled():
        return
    from .activation_spec import is_allowed_activation, ACTIVATION_SPEC_VERSION
    if not is_allowed_activation(activation_kind, surface):
        logger.warning("emit.activation: rejected pair %s/%s", activation_kind, surface)
        return
    if not claim_activation_marker(activation_kind):
        return
    import time
    payload = _envelope("activation")
    payload["activation"] = {
        "activation_kind": activation_kind,
        "surface": surface,
        "client_id": get_or_create_client_id(),
        "spec_version": ACTIVATION_SPEC_VERSION,
        "occurred_at_epoch": int(time.time()),
    }
    get_queue().enqueue(payload)
```

Ensure `claim_activation_marker`, `get_or_create_client_id` imported from state (check existing imports — `get_or_create_client_id` already imported per `__init__` exports).

- [ ] **Step 3: Implement — request sampling gate**

Add before `request()`:

```python
def _request_sample_rate() -> float:
    import os
    raw = os.environ.get("FUSION_MLX_TELEMETRY_REQUEST_SAMPLE", "0.1")
    try:
        rate = float(raw)
    except ValueError:
        logger.warning("bad FUSION_MLX_TELEMETRY_REQUEST_SAMPLE=%r, default 0.1", raw)
        return 0.1
    return max(0.0, min(1.0, rate))


def _should_sample_request() -> bool:
    import random
    return random.random() < _request_sample_rate()
```

- [ ] **Step 4: Implement — widen request() signature + fields + sampling**

Modify `request()` signature — add params after `status`:

```python
def request(
    *,
    endpoint: str,
    model_alias: str,
    stream: bool,
    tool_call_used: bool,
    prompt_tokens: int,
    completion_tokens: int,
    ttft_ms: float,
    tps: float,
    status: int,
    caller_agent: str | None = None,
    output_degenerate: bool = False,
    completion_empty: bool = False,
    completion_abnormally_short: bool = False,
) -> None:
```

Inside, after `is_enabled()` guard, add sampling:

```python
    if not _should_sample_request():
        return
```

In the `payload["request"]` dict, add the new fields:

```python
        "caller_agent": normalize_caller_agent(caller_agent),
        "output_degenerate": bool(output_degenerate),
        "completion_empty": bool(completion_empty),
        "completion_abnormally_short": bool(completion_abnormally_short),
```

Ensure `normalize_caller_agent` imported from redact (add to existing redact import).

- [ ] **Step 5: Run all PR1 tests + lint + commit**

Run: `python -m pytest tests/unit/test_telemetry_activation_spec.py tests/unit/test_telemetry_activation_marker.py tests/unit/test_telemetry_request_preview_payload.py tests/unit/test_telemetry_caller_agent_redact.py -v`
Run: `black --check --target-version py313 fusion_mlx/telemetry/emit.py && ruff check fusion_mlx/telemetry/emit.py`
Commit: `feat(telemetry): add activation()+server_surface()+request sampling+field widening (dark)`

### Task 6: coherence.py — deterministic output-degenerate heuristic

**Files:**
- Create: `fusion_mlx/telemetry/coherence.py`
- Create: `tests/unit/test_telemetry_coherence.py`

**Interfaces:**
- Produces: `looks_like_garbage(text: str, completion_tokens: int) -> bool`, `is_empty(completion_tokens: int) -> bool`, `is_abnormally_short(text: str, completion_tokens: int, threshold: int = 3) -> bool`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_telemetry_coherence.py
# SPDX-License-Identifier: Apache-2.0
from fusion_mlx.telemetry.coherence import looks_like_garbage, is_empty, is_abnormally_short


def test_empty_when_zero_tokens():
    assert is_empty(0) is True
    assert is_empty(5) is False


def test_garbage_repetition():
    assert looks_like_garbage("aaaaaaaaaaaaaaaa", 16) is True
    assert looks_like_garbage("hello world this is fine", 10) is False


def test_garbage_when_tokens_but_empty_text():
    assert looks_like_garbage("", 10) is True


def test_abnormally_short():
    assert is_abnormally_short("hi", 2) is True
    assert is_abnormally_short("hello world paragraph", 20) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_telemetry_coherence.py -v`
Expected: FAIL — module missing

- [ ] **Step 3: Implement**

```python
# fusion_mlx/telemetry/coherence.py
# SPDX-License-Identifier: Apache-2.0
import logging

logger = logging.getLogger(__name__)

_GARBLE_REPEAT_RATIO = 0.7


def is_empty(completion_tokens: int) -> bool:
    return int(completion_tokens) <= 0


def is_abnormally_short(text: str, completion_tokens: int, threshold: int = 3) -> bool:
    if is_empty(completion_tokens):
        return False
    return int(completion_tokens) <= threshold


def looks_like_garbage(text: str, completion_tokens: int) -> bool:
    if is_empty(completion_tokens):
        return False
    if not text or not text.strip():
        logger.debug("coherence: empty text with tokens=%d -> garbage", completion_tokens)
        return True
    if len(text) < 4:
        return True
    most_common_char = max(set(text), key=text.count)
    ratio = text.count(most_common_char) / len(text)
    if ratio >= _GARBLE_REPEAT_RATIO:
        logger.debug("coherence: repetition ratio %.2f >= %.2f -> garbage", ratio, _GARBLE_REPEAT_RATIO)
        return True
    return False
```

- [ ] **Step 4: Run tests + lint + commit**

Run: `python -m pytest tests/unit/test_telemetry_coherence.py -v`
Run: `black --check --target-version py313 fusion_mlx/telemetry/coherence.py && ruff check fusion_mlx/telemetry/coherence.py`
Commit: `feat(telemetry): add deterministic coherence heuristic (no model routing)`

### Task 7: CLI telemetry preview — show request sample

**Files:**
- Modify: `fusion_mlx/cli.py` (telemetry preview action)

**Interfaces:**
- Consumes: `sample_preview_payload`, `sample_request_preview_payload`

- [ ] **Step 1: Locate telemetry preview action**

Run: `grep -n "sample_preview_payload\|telemetry.*preview\|def.*telemetry" fusion_mlx/cli.py`
Find the preview branch that calls `sample_preview_payload`.

- [ ] **Step 2: Add request sample display**

In the preview branch, after the session sample is printed, add:

```python
        req_preview = sample_request_preview_payload(
            client_id=client_id, fusion_mlx_version=__version__,
        )
        print(json.dumps(req_preview.to_dict(), indent=2, sort_keys=True))
```

Ensure `sample_request_preview_payload` imported from `fusion_mlx.telemetry.schema`.

- [ ] **Step 3: Verify + lint + commit**

Run: `python -c "from fusion_mlx.cli import *"` (import sanity)
Run: `black --check --target-version py313 fusion_mlx/cli.py && ruff check fusion_mlx/cli.py`
Commit: `feat(telemetry): CLI preview shows request sample alongside session`

### Task 8: docs/telemetry-activation.md

**Files:**
- Create: `docs/telemetry-activation.md`

- [ ] **Step 1: Write doc (English)**

Document: purpose (growth/engagement funnel), 9 milestones, 3 surfaces, once-per-install marker semantics, `FUSION_MLX_CHAT_SPAWN` surface attribution, `INFERENCE_ENDPOINTS` scope, `is_successful_inference` predicate, spec version 3 vs Rapid v2 (multimodal additions). Reference `activation_spec.py` as the code contract.

- [ ] **Step 2: Commit**

Commit: `docs: add telemetry-activation.md human-facing funnel spec`

### Task 9: PR1 release — version bump, CHANGELOG, PR, merge

- [ ] **Step 1: Bump `fusion_mlx/_version.py`** 0.8.76 → 0.8.77
- [ ] **Step 2: CHANGELOG `## [0.8.77] - 2026-09-04`** — Added: activation funnel (9 milestones, spec v3), ActivationPayload, per-request telemetry fields (caller_agent, output_degenerate, completion_empty, completion_abnormally_short), request sampling gate, activation markers + consent schema v3, normalize_caller_agent, coherence heuristic, CLI request preview. Dark — no prod call sites. English.
- [ ] **Step 3: Full PR1 test suite green + lint clean**
- [ ] **Step 4: Commit, push branch `feat/telemetry-activation-funnel`, create PR (English), merge squash to main, tag v0.8.77**
- [ ] **Step 5: Update memory (release file + MEMORY.md)**

---

## PR2 — Wiring + Prometheus metrics (prod-visible)

### Task 10: per-request wiring (non-streaming) in chat.py

**Files:**
- Modify: `fusion_mlx/routes_internal/chat.py`
- Create: `tests/unit/test_telemetry_request_wiring.py`

**Interfaces:**
- Consumes: `emit.request()`, `emit.activation()`, `emit.server_surface()`, `is_successful_inference`, `coherence.looks_like_garbage`/`is_empty`/`is_abnormally_short`
- Produces: `emit.request(...)` call at non-streaming chat completion

- [ ] **Step 1: Locate non-streaming completion path**

Run: `grep -n "def.*chat\|completion_tokens\|generation_duration\|prefill_duration\|return.*response" fusion_mlx/routes_internal/chat.py`
Find the point where final token counts + status are known, before response return.

- [ ] **Step 2: Write failing test**

```python
# tests/unit/test_telemetry_request_wiring.py
# SPDX-License-Identifier: Apache-2.0
# Asserts emit.request fires with correct fields at non-streaming completion.
# Consent-gated: no fire when is_enabled()==False.
```

Test structure: monkeypatch `is_enabled` → True, patch `get_queue().enqueue`, drive a minimal chat completion (mocked engine), assert enqueued payload has `event=="request"`, correct bucket fields, `caller_agent` bucketed. Then `is_enabled` → False, assert no enqueue.

(Full test body written by implementer after reading chat.py completion path — the exact call-site shape determines assertions.)

- [ ] **Step 3: Implement — add emit.request call**

At completion point, after token counts known:

```python
        try:
            from ..telemetry import emit
            from ..telemetry.coherence import looks_like_garbage, is_empty, is_abnormally_short
            from ..telemetry.activation_spec import is_successful_inference
            emit.request(
                endpoint="/v1/chat/completions",
                model_alias=model_alias,
                stream=False,
                tool_call_used=bool(tool_call_used),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_ms=ttft_ms,
                tps=tps,
                status=status,
                caller_agent=request.headers.get("user-agent"),
                output_degenerate=looks_like_garbage(output_text, completion_tokens),
                completion_empty=is_empty(completion_tokens),
                completion_abnormally_short=is_abnormally_short(output_text, completion_tokens),
            )
            if is_successful_inference(status, completion_tokens):
                emit.activation(
                    activation_kind="first_inference",
                    surface=emit.server_surface(),
                )
        except Exception:
            logger.debug("telemetry request emit failed", exc_info=True)
```

Wrap in try/except — telemetry must never break a request (fail-visible in log, not crash).

- [ ] **Step 4: Run test + lint + commit**

Commit: `feat(telemetry): wire emit.request+activation into non-streaming chat completion`

### Task 11: per-request wiring (streaming) in chat.py

**Files:**
- Modify: `fusion_mlx/routes_internal/chat.py`
- Create: `tests/unit/test_telemetry_streaming_request_wiring.py`

- [ ] **Step 1: Locate streaming generator finalize**

Run: `grep -n "StreamingResponse\|async def.*stream\|yield\|finally" fusion_mlx/routes_internal/chat.py`
Find the generator's finalize/finally block where final token count is known.

- [ ] **Step 2: Write failing test** — asserts `emit.request` fires once at stream finalize, `stream=True`. Consent-gated.

- [ ] **Step 3: Implement** — add same `emit.request(...)` + `emit.activation(...)` call in the generator's finally/finalize, with `stream=True`.

- [ ] **Step 4: Run test + lint + commit**

Commit: `feat(telemetry): wire emit.request+activation into streaming chat finalize`

### Task 12: error wiring in server.py + cli.py

**Files:**
- Modify: `fusion_mlx/server.py`
- Modify: `fusion_mlx/cli.py`
- Create: `tests/unit/test_telemetry_error_wiring.py`

**Interfaces:**
- Consumes: `emit.error(category, phase, fingerprint)`, `redact.fingerprint_traceback`

- [ ] **Step 1: Locate lifespan startup/shutdown + cli model-load**

Run: `grep -n "lifespan\|model.*load\|except.*Exception\|shutdown" fusion_mlx/server.py | head -30`
Run: `grep -n "model.*load\|except.*Exception\|pull" fusion_mlx/cli.py | head -30`

- [ ] **Step 2: Write failing test** — asserts `emit.error` fires on lifespan load failure with `category="model_load_failure"`, `phase="startup"`. Consent-gated.

- [ ] **Step 3: Implement — server.py startup**

In lifespan startup except block (model load failure):

```python
            try:
                from .telemetry import emit
                from .telemetry.redact import fingerprint_traceback
                emit.error(
                    category="model_load_failure",
                    phase="startup",
                    fingerprint=fingerprint_traceback(exc),
                )
            except Exception:
                logger.debug("telemetry error emit failed", exc_info=True)
```

- [ ] **Step 4: Implement — server.py shutdown**

In lifespan shutdown except block (unhandled traceback):

```python
            try:
                from .telemetry import emit
                from .telemetry.redact import fingerprint_traceback
                emit.error(
                    category="shutdown_traceback",
                    phase="shutdown",
                    fingerprint=fingerprint_traceback(exc),
                )
            except Exception:
                pass
```

- [ ] **Step 5: Implement — cli.py model-load + model_pull activation**

In cli.py model-load failure except: `emit.error(category="model_load_failure", phase="chat", fingerprint=...)`.
In cli.py pull success path: `emit.activation(activation_kind="model_pull", surface="cli")`.

- [ ] **Step 6: Run test + lint + commit**

Commit: `feat(telemetry): wire emit.error into lifespan + cli, emit.activation model_pull`

### Task 13: route attribution test

**Files:**
- Create: `tests/unit/test_telemetry_route_attribution.py`

- [ ] **Step 1: Write test** — asserts `server_surface()` returns `"cli"` when `FUSION_MLX_CHAT_SPAWN` set, `"api"` when unset. Asserts activation surface matches.

- [ ] **Step 2: Run + commit**

Commit: `test(telemetry): route attribution surface=cli when CHAT_SPAWN set`

### Task 14: server_metrics.py — per-model histograms + uptime + modality counters

**Files:**
- Modify: `fusion_mlx/server_metrics.py`
- Create: `tests/unit/test_prometheus_histogram_render.py`

**Interfaces:**
- Produces: `Histogram` helper class (observe + render), per-model TTFT + decode-TPS histograms in `ServerMetrics`, `record_request_complete` extended to feed histograms, modality counters (`vision_requests`, `audio_requests`, `video_requests`, `image_generation_requests`), `record_modality_request(modality)`, startup/shutdown timestamps.

- [ ] **Step 1: Write failing test** — `Histogram` observes values, renders bucket/count/sum/max correctly.

```python
# tests/unit/test_prometheus_histogram_render.py
# SPDX-License-Identifier: Apache-2.0
from fusion_mlx.server_metrics import Histogram


def test_histogram_observe_and_render():
    h = Histogram(buckets=[0.1, 0.5, 1.0, float("inf")])
    h.observe(0.05)
    h.observe(0.3)
    h.observe(2.0)
    lines = h.render("fusion_mlx_model_ttft_seconds", labels={"model": "m"})
    text = "\n".join(lines)
    assert "fusion_mlx_model_ttft_seconds_bucket" in text
    assert "fusion_mlx_model_ttft_seconds_count" in text
    assert "fusion_mlx_model_ttft_seconds_sum" in text
    assert "fusion_mlx_model_ttft_seconds_max" in text
    # bucket le="0.1" has 1 (0.05), le="0.5" has 2, le="+Inf" has 3
    assert 'le="0.1"' in text
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement Histogram**

```python
class Histogram:
    def __init__(self, buckets: list[float]):
        self.buckets = sorted(buckets)
        self.counts = [0] * len(self.buckets)
        self.sum = 0.0
        self.count = 0
        self.max = 0.0

    def observe(self, value: float) -> None:
        self.count += 1
        self.sum += float(value)
        if value > self.max:
            self.max = float(value)
        for i, b in enumerate(self.buckets):
            if value <= b:
                self.counts[i] += 1

    def render(self, name: str, labels: dict | None = None) -> list[str]:
        label_pairs = [f'{k}="{v}"' for k, v in (labels or {}).items()]
        base_labels = ",".join(label_pairs)
        lines = []
        for i, b in enumerate(self.buckets):
            le = "+Inf" if b == float("inf") else str(b)
            pairs = [f'le="{le}"'] + label_pairs
            lines.append(f'{name}_bucket{{{",".join(pairs)}}} {self.counts[i]}')
        suffix = (f"{{{base_labels}}}" if base_labels else "")
        lines.append(f"{name}_count{suffix} {self.count}")
        lines.append(f"{name}_sum{suffix} {self.sum}")
        lines.append(f"{name}_max{suffix} {self.max}")
        return lines
```

- [ ] **Step 4: Add per-model histograms to ServerMetrics**

Add `_ttft_hist: dict[str, Histogram]`, `_tps_hist: dict[str, Histogram]` with TTFT buckets `[0.01,0.05,0.1,0.25,0.5,1,2.5,5,10,30,float("inf")]` and TPS buckets `[1,5,10,20,40,80,160,float("inf")]`. In `record_request_complete`, observe ttft into `_ttft_hist[model_id]` and tps into `_tps_hist[model_id]`. Add `get_ttft_histograms()` / `get_tps_histograms()` accessors.

- [ ] **Step 5: Add modality counters + lifespan timestamps**

Add `vision_requests`, `audio_requests`, `video_requests`, `image_generation_requests` counters + `record_modality_request(modality)`. Add `_startup_epoch`, `_shutdown_epoch` + `record_startup()` / `record_shutdown()`.

- [ ] **Step 6: Run tests + lint + commit**

Commit: `feat(metrics): per-model TTFT/TPS histograms + modality counters + lifespan timestamps`

### Task 15: metrics.py — fix rapid_mlx_ prefix + queue/uptime/metal gauges

**Files:**
- Modify: `fusion_mlx/routes_internal/metrics.py`
- Create: `tests/unit/test_prometheus_metric_expansion.py`

- [ ] **Step 1: Write failing test** — asserts `rapid_mlx_` prefix gone, `fusion_mlx_requests_running`/`_waiting`/`_uptime_seconds`/`_metal_*_bytes` render.

- [ ] **Step 2: Rename rapid_mlx_ prefix**

Find the 5 `rapid_mlx_response_format_strict_*` occurrences → rename to `fusion_mlx_response_format_strict_*`.

- [ ] **Step 3: Add queue gauges**

```python
def _render_queue_metrics() -> list[str]:
    from ..server import _server_state
    pool = _server_state.get("engine_pool")
    running = waiting = 0
    if pool:
        # read scheduler running/waiting counts
        for _mid, entry in getattr(pool, "_entries", {}).items():
            sched = getattr(getattr(entry, "engine", None), "scheduler", None)
            if sched:
                running += len(getattr(sched, "running", []))
                waiting += len(getattr(sched, "waiting", []))
    return [
        *_fmt_metric("fusion_mlx_requests_running", "gauge", "Running requests", running),
        *_fmt_metric("fusion_mlx_requests_waiting", "gauge", "Waiting requests", waiting),
    ]
```

(Verify actual scheduler field names via codegraph before finalizing.)

- [ ] **Step 4: Add uptime + metal memory**

```python
def _render_uptime_metal() -> list[str]:
    lines = []
    from ..server_metrics import get_server_metrics
    m = get_server_metrics()
    lines.extend(_fmt_metric("fusion_mlx_uptime_seconds", "gauge", "Process uptime", m.to_dict().get("uptime_seconds", 0)))
    try:
        import mlx.core as mx
        if mx.metal.is_available():
            lines.extend(_fmt_metric("fusion_mlx_metal_active_bytes", "gauge", "MLX active memory", mx.get_active_memory() or 0))
            lines.extend(_fmt_metric("fusion_mlx_metal_cache_bytes", "gauge", "MLX cache memory", mx.get_cache_memory() or 0))
            lines.extend(_fmt_metric("fusion_mlx_metal_peak_bytes", "gauge", "MLX peak memory", mx.get_peak_memory() or 0))
    except Exception:
        logger.debug("metrics: mlx memory unavailable", exc_info=True)
    return lines
```

- [ ] **Step 5: Wire into render_prometheus_metrics dispatch**

Add `_render_queue_metrics()` + `_render_uptime_metal()` calls.

- [ ] **Step 6: Run tests + lint + commit**

Commit: `fix(metrics): rename rapid_mlx_ prefix + add queue/uptime/metal gauges`

### Task 16: metrics.py — prefix/radix/KV-checkpoint/spec-decode/UBC/turboquant/pflash/embedding families

**Files:**
- Modify: `fusion_mlx/routes_internal/metrics.py`
- Modify: `tests/unit/test_prometheus_metric_expansion.py`

- [ ] **Step 1: Write failing test** — asserts each new family renders at zero (no engine) OR reflects subsystem state; skipped families (suffix_decode, mtp, mxfp4) absent.

- [ ] **Step 2: Implement _render_prefix_cache_metrics()**

Wire to `radix_index.to_dict()` (verify field names via codegraph). Wrap in try/except → empty list if absent.

- [ ] **Step 3: Implement _render_kv_checkpoint_metrics()** — wire to `disk_kv_checkpoint` stats.
- [ ] **Step 4: Implement _render_spec_decode_metrics()** — wire to `spec_decode.total_draft_accepted`.
- [ ] **Step 5: Implement _render_ubc_metrics()** — wire to `ubc_evict`.
- [ ] **Step 6: Implement _render_turboquant_metrics()** — wire to `turboquant`.
- [ ] **Step 7: Implement _render_pflash_metrics()** — wire to `pflash`.
- [ ] **Step 8: Implement _render_embedding_metrics()** — `fusion_mlx_embedding_truncations_total`.
- [ ] **Step 9: Wire all into render dispatch**
- [ ] **Step 10: Run tests + lint + commit**

Commit: `feat(metrics): prefix/radix/KV-checkpoint/spec-decode/UBC/turboquant/pflash/embedding families`

### Task 17: Section 4 lead metrics — multimodal + KV-cache advanced + distributed + lifespan

**Files:**
- Modify: `fusion_mlx/routes_internal/metrics.py`
- Modify: `fusion_mlx/server_metrics.py`
- Create: `tests/unit/test_prometheus_multimodal_metrics.py`
- Create: `tests/unit/test_prometheus_kv_cache_advanced.py`
- Create: `tests/unit/test_prometheus_lifespan_metrics.py`

- [ ] **Step 1: Write failing tests** — multimodal counters + VAE/video histograms render; paged-KV gauges render from pool; lifespan histograms render.

- [ ] **Step 2: Implement _render_multimodal_metrics()** — `vision_requests_total`, `audio_requests_total`, `video_requests_total`, `image_generation_requests_total` from `ServerMetrics` modality counters; VAE/video histograms.

- [ ] **Step 3: Implement _render_kv_cache_advanced()** — `fusion_mlx_kv_cache_pages_total`, `_evictions_total`, `_block_utilization` from Paged-KV pool (verify via codegraph).

- [ ] **Step 4: Implement _render_distributed_decode()** — `fusion_mlx_distributed_decode_tokens_total{node}`, `_rtt_seconds` (wire to #630 distributed decode if loaded; empty if absent).

- [ ] **Step 5: Implement _render_lifespan_metrics()** — `fusion_mlx_lifespan_startup_seconds`, `_shutdown_seconds` histograms from `ServerMetrics` timestamps.

- [ ] **Step 6: Wire into dispatch**
- [ ] **Step 7: Run tests + lint + commit**

Commit: `feat(metrics): Section 4 lead — multimodal + paged-KV + distributed + lifespan metrics`

### Task 18: activation multimodal milestones wiring

**Files:**
- Modify: `fusion_mlx/routes_internal/chat.py` (or vision/image/video route)
- Create: `tests/unit/test_telemetry_activation_multimodal.py`

- [ ] **Step 1: Write failing test** — `first_image_generation` + `first_video_generation` fire once-per-install on API surface, spec version 3.

- [ ] **Step 2: Wire** — in image-generation route completion: `emit.activation(activation_kind="first_image_generation", surface="api")`. In video-generation route completion: `emit.activation(activation_kind="first_video_generation", surface="api")`. In audio transcription route: `emit.activation(activation_kind="first_dictation", surface="api")`.

- [ ] **Step 3: Run test + lint + commit**

Commit: `feat(telemetry): wire multimodal activation milestones (image/video gen, dictation)`

### Task 19: PR2 release — version bump, CHANGELOG, PR, merge

- [ ] **Step 1: Bump `fusion_mlx/_version.py`** 0.8.77 → 0.8.78
- [ ] **Step 2: CHANGELOG `## [0.8.78] - 2026-09-04`** — Added: per-request telemetry wiring (stream+non-stream), error wiring (lifespan+cli), activation wiring (first_inference, model_pull, multimodal milestones), Prometheus metric expansion (TTFT/decode-TPS histograms, prefix/radix/KV-checkpoint/spec-decode/UBC/turboquant/pflash/embedding/queue/uptime/metal), Section 4 lead (multimodal counters+VAE/video histograms, paged-KV gauges, distributed decode, lifespan), rapid_mlx_ prefix fix. English.
- [ ] **Step 3: Full PR2 test suite green + lint clean**
- [ ] **Step 4: Commit, push branch `feat/telemetry-wiring-metrics`, create PR (English), merge squash to main, tag v0.8.78**
- [ ] **Step 5: Update memory (release file + MEMORY.md)**
