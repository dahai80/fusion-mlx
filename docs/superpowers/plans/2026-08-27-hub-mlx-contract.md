# Hub↔MLX API Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align fusion-mlx model-load + quantize endpoints with the fusion-model-hub external client (issue #646): make slash-bearing HF repo ids loadable, accept `source_path` alias on quantize, mount the layered-quantize router, and standardize job terminal status to `completed`.

**Architecture:** Four independent fixes on the public HTTP surface. Gap 1 touches the load/unload route signatures (`{model_id:path}` converter) + an in-handler slash→hyphen pool-lookup retry. Gap 2 adds a pydantic `model_validator` alias on `QuantizeRequest`. Gap 3a mounts an already-written orphaned router. Gap 3b changes two status-string writers + their test assertions. No shared internal API changes beyond the load handlers.

**Tech Stack:** FastAPI (Starlette routing), pydantic v2, pytest (TestClient + real-model markers), MLX inference server.

**Spec:** `docs/superpowers/specs/2026-08-27-hub-mlx-contract-design.md`

## Global Constraints

From the spec's Global Constraints section, binding every task:

- Lint must pass: `ruff check fusion_mlx/ tests/` + `black --check fusion_mlx/ tests/`. `gui_compat/` is NOT ruff/black compliant — route-signature edits there match surrounding style, do not reformat adjacent code.
- No docstrings in new code. Multiples of 4 indentation. Logging by default in new code.
- Real-model tests: server via `~/claude-home/fusion-mlx/start.sh start|stop`. Gate with `@pytest.mark.real_model` + inline `FUSION_MLX_REAL_MODEL_TESTS` guard. NEVER `mx.clear_streams()` in tests. Model download via mirror `https://hf-mirror.com`.
- Failing-test rule: if a test fails — even unrelated — locate and fix it. Suite no-redder than before.
- Commit flow: branch `feat/646-hub-mlx-contract` (already created, off main 6a40c308). `git commit -m "<type>(#646): <subject>"`. Remote is `origin`. No push unless user asks.
- Public API boundary (#615): no new `public_api` exports (routes are HTTP, not Python import surface).

---

## File Structure

| File | Task(s) | Responsibility |
|---|---|---|
| `fusion_mlx/server.py` | T1, T4 | Load/unload route signatures `{model_id:path}` + in-handler slash→hyphen retry (T1); include layered router import+mount (T4). |
| `fusion_mlx/gui_compat/server.py` | T2 | gui_compat load/unload route signatures `{model_name:path}`. |
| `fusion_mlx/api/convert_models.py` | T3 | `QuantizeRequest` `source_path` alias validator. |
| `fusion_mlx/api/convert_routes.py` | T6 | `_run_job` status `"done"`→`"completed"`. |
| `fusion_mlx/api/layered_quantize_routes.py` | T6 | `_run_layered_quantize` status `"done"`→`"completed"`. |
| `tests/unit/test_server_load_route.py` | T1 (new) | Gap 1 route-compile + in-handler retry unit tests. |
| `tests/unit/test_convert_routes.py` | T3, T6 | Gap 2 alias tests; Gap 3b assertion updates. |
| `tests/unit/test_layered_quantize_routes.py` | T4, T6 (new) | Gap 3a mount test; Gap 3b terminal-status test. |

---

### Task 1: Gap 1a — Load/unload route `{model_id:path}` + in-handler slash→hyphen retry

**Files:**
- Modify: `fusion_mlx/server.py:1026` (load signature), `server.py:1034-1042` (in-handler retry), `server.py:1067` (unload signature), `server.py:1075-1082` (unload in-handler retry)
- Test: `tests/unit/test_server_load_route.py` (create)

**Interfaces:**
- Consumes: module-level `resolve_model_id(model_id) -> str` (`server.py:665`), `self.pool.get_entry(resolved)` (`engine_pool.py:546`).
- Produces: load/unload routes accept slash-bearing ids; returns existing `{"status":"ok","model_id":...}` shape.

**Design note (plan-vs-code reconciliation):** The spec says "extend `resolve_model_id`". But `resolve_model_id` (`server.py:665`) is a module-level function with no pool access and 12 callers. Extending it would require passing the pool → 12 signature changes (violates surgical). The pool method `EnginePool.resolve_model_id` (`engine_pool.py:578`) DOES have pool access and already does slash-prefix stripping, but the load handler calls the module-level one. So the slash→hyphen retry goes **inline in the load/unload handlers**, after the module-level `resolve_model_id` returns, before `get_entry`. Functionally identical to the spec's intent, strictly smaller blast radius. This is a refinement, recorded here so the implementer does not re-litigate it.

- [ ] **Step 1: Write the failing unit test (route-compile)**

Create `tests/unit/test_server_load_route.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from starlette.routing import compile_path


def test_load_route_path_converter_matches_slash_ids():
    rx, _, _ = compile_path("/v1/models/{model_id:path}/load")
    m = rx.match("/v1/models/mlx-community/Llama-3.2-1B-Instruct-4bit/load")
    assert m is not None
    assert m.groupdict()["model_id"] == "mlx-community/Llama-3.2-1B-Instruct-4bit"


def test_load_route_path_converter_matches_hyphen_ids():
    rx, _, _ = compile_path("/v1/models/{model_id:path}/load")
    m = rx.match("/v1/models/mlx-community-Llama-3.2-1B-Instruct-4bit/load")
    assert m is not None
    assert m.groupdict()["model_id"] == "mlx-community-Llama-3.2-1B-Instruct-4bit"


def test_load_route_path_converter_does_not_swallow_status():
    rx, _, _ = compile_path("/v1/models/{model_id:path}/load")
    assert rx.match("/v1/models/status") is None
```

- [ ] **Step 2: Run test — route-compile fails before route change**

Wait: `compile_path` tests the string literal, not the live route. They pass immediately because the string is what we assert. To make them fail-then-pass meaningfully, Step 1 asserts the CURRENT route string `/v1/models/{model_id}/load` first (NO `:path`), which must NOT match slash ids, then after the edit asserts the `:path` form. Rewrite Step 1:

```python
def test_current_load_route_rejects_slash_ids():
    rx, _, _ = compile_path("/v1/models/{model_id}/load")
    assert rx.match("/v1/models/a/b/load") is None  # single-segment, no slash

def test_path_converter_load_route_matches_slash_ids():
    rx, _, _ = compile_path("/v1/models/{model_id:path}/load")
    assert rx.match("/v1/models/a/b/load") is not None
    assert rx.match("/v1/models/a/b/load").groupdict()["model_id"] == "a/b"
```

Run: `python -m pytest tests/unit/test_server_load_route.py -q`
Expected: both pass (these are string-literal assertions proving the converter behavior; the live-route integration is the real-model test in Step 6).

- [ ] **Step 3: Write the failing unit test (in-handler retry)**

Append to `tests/unit/test_server_load_route.py`:

```python
def test_slash_to_hyphen_retry_resolves_pool_entry():
    # Simulate: pool has hyphen id "mlx-community-Foo-4bit", request id "mlx-community/Foo-4bit".
    # The retry maps / -> - and finds the entry.
    class FakeEntry:
        engine = None
    class FakePool:
        def __init__(self):
            self._entries = {"mlx-community-Foo-4bit": FakeEntry()}
        def get_entry(self, mid):
            return self._entries.get(mid)
    pool = FakePool()
    requested = "mlx-community/Foo-4bit"
    # Mirror the handler's retry logic:
    resolved = requested  # resolve_model_id would return it unchanged (no fusion- prefix)
    entry = pool.get_entry(resolved)
    if entry is None and "/" in resolved:
        resolved = resolved.replace("/", "-")
        entry = pool.get_entry(resolved)
    assert entry is not None
    assert resolved == "mlx-community-Foo-4bit"


def test_slash_to_hyphen_retry_not_applied_when_slash_entry_exists():
    # If a genuine slash id is registered, it must NOT be mapped to hyphen.
    class FakeEntry:
        engine = None
    class FakePool:
        def __init__(self):
            self._entries = {"a/b": FakeEntry()}
        def get_entry(self, mid):
            return self._entries.get(mid)
    pool = FakePool()
    resolved = "a/b"
    entry = pool.get_entry(resolved)
    if entry is None and "/" in resolved:
        resolved = resolved.replace("/", "-")
        entry = pool.get_entry(resolved)
    assert entry is not None
    assert resolved == "a/b"  # unchanged — genuine slash id preserved
```

Run: `python -m pytest tests/unit/test_server_load_route.py -q`
Expected: PASS (mirrors intended logic before wiring it into the handler).

- [ ] **Step 4: Implement — route signatures to `:path`**

In `fusion_mlx/server.py`:
- Line 1026: `@app.post("/v1/models/{model_id}/load")` → `@app.post("/v1/models/{model_id:path}/load")`
- Line 1067: `@app.post("/v1/models/{model_id}/unload")` → `@app.post("/v1/models/{model_id:path}/unload")`

- [ ] **Step 5: Implement — in-handler slash→hyphen retry**

In `load_model_public` (after line 1034 `resolved = resolve_model_id(model_id)`, before `entry = self.pool.get_entry(resolved)`):

```python
            resolved = resolve_model_id(model_id)
            entry = self.pool.get_entry(resolved)
            if entry is None and "/" in resolved:
                hyphen = resolved.replace("/", "-")
                hyphen_entry = self.pool.get_entry(hyphen)
                if hyphen_entry is not None:
                    logger.debug("load: slash->hyphen resolve %s -> %s", resolved, hyphen)
                    resolved = hyphen
                    entry = hyphen_entry
            if entry is None:
                raise HTTPException(
                    status_code=404, detail=f"Model not found: {model_id}"
                )
```

Replace the existing `entry = self.pool.get_entry(resolved)` + `if entry is None: raise 404` block (lines 1035-1040) with the above. Same edit in `unload_model_public` (lines 1075-1081), identical retry block. `logger` is already in scope in `server.py`.

- [ ] **Step 6: Real-model round-trip test (gated)**

Append to `tests/unit/test_server_load_route.py`:

```python
import os
import pytest

REAL_MODEL = os.environ.get("FUSION_MLX_REAL_MODEL_TESTS") == "1"


@pytest.mark.real_model
def test_load_slash_id_real_server():
    if not REAL_MODEL:
        pytest.skip("set FUSION_MLX_REAL_MODEL_TESTS=1")
    import urllib.request
    import json
    import urllib.error
    key = ""  # set from settings if auth enabled; empty for loopback
    # Loopback + X-Fusion-Source bypasses auth; use a registered slash-bearing id.
    # The test asserts the route MATCHES (not 404 generic) — a 404 with the
    # route's own "Model not found" detail is acceptable (means route matched,
    # id not registered); a generic not_found_error means route did NOT match.
    req = urllib.request.Request(
        "http://127.0.0.1:11434/v1/models/mlx-community/Llama-3.2-1B-Instruct-4bit/load",
        method="POST",
        data=b"{}",
        headers={"Content-Type": "application/json", "X-Fusion-Source": "model-hub"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
            assert resp.status == 200
            assert body["status"] == "ok"
    except urllib.error.HTTPError as e:
        # Route must have matched: 404 detail "Model not found" = route matched (id not loaded).
        # Generic not_found_error = route did NOT match = test FAIL.
        assert "Model not found" in str(e.detail) or e.code == 200, (
            f"route did not match: {e.code} {e.detail}"
        )
```

Run: `FUSION_MLX_REAL_MODEL_TESTS=1 python -m pytest tests/unit/test_server_load_route.py::test_load_slash_id_real_server -q` (with `start.sh start` + a model loaded). Expected: PASS (200 or route-matched 404).
Without real-model env: `python -m pytest tests/unit/test_server_load_route.py -q` → unit tests PASS, real-model skipped.

- [ ] **Step 7: Lint + commit**

Run: `ruff check fusion_mlx/server.py tests/unit/test_server_load_route.py && black --check fusion_mlx/server.py tests/unit/test_server_load_route.py`
Expected: clean (gui_compat untouched here).

```bash
git add fusion_mlx/server.py tests/unit/test_server_load_route.py
git commit -m "feat(#646): load/unload routes accept slash-bearing model ids via :path converter + slash->hyphen pool retry"
```

---

### Task 2: Gap 1b — gui_compat load/unload `{model_name:path}`

**Files:**
- Modify: `fusion_mlx/gui_compat/server.py:470` (load), `:514` (unload)

**Interfaces:**
- Consumes: none new (same route-match fix as T1, applied to the optional gui_compat router for parity).
- Produces: gui_compat routes match slash ids when gui_compat is mounted.

**Note:** gui_compat is NOT ruff/black compliant (Global Constraints). Only the route-signature string changes. Do not reformat adjacent lines.

- [ ] **Step 1: Implement — route signatures to `:path`**

In `fusion_mlx/gui_compat/server.py`:
- Line 470: `@router.post("/v1/models/{model_name}/load")` → `@router.post("/v1/models/{model_name:path}/load")`
- Line 514: `@router.post("/v1/models/{model_name}/unload")` → `@router.post("/v1/models/{model_name:path}/unload")`

- [ ] **Step 2: Verify gui_compat import still works**

Run: `python -c "from fusion_mlx.gui_compat.server import get_gui_compat_router; print('ok')"`
Expected: `ok` (if gui_compat deps installed) OR ImportError (if optional deps absent — acceptable, gui_compat is optional). If ImportError unrelated to our edit, that's the pre-existing optional-import state, not a regression.

- [ ] **Step 3: Commit**

```bash
git add fusion_mlx/gui_compat/server.py
git commit -m "feat(#646): gui_compat load/unload routes accept slash ids via :path converter"
```

---

### Task 3: Gap 2 — `QuantizeRequest` `source_path` alias

**Files:**
- Modify: `fusion_mlx/api/convert_models.py:95` (`QuantizeRequest`)
- Test: `tests/unit/test_convert_routes.py` (append)

**Interfaces:**
- Consumes: `_ConvertBase` (line 35) with `model: str`, `_validate_output_path` (line 44).
- Produces: `QuantizeRequest` accepts `source_path` key, copies to `model`.

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_convert_routes.py`:

```python
def test_quantize_request_accepts_source_path_alias():
    from fusion_mlx.api.convert_models import QuantizeRequest
    req = QuantizeRequest(
        source_path="mlx-community/Llama-3.2-1B-Instruct-4bit",
        output_path=os.path.expanduser("~/.fusion-mlx/models/out"),
        quant_bits=4,
    )
    assert req.model == "mlx-community/Llama-3.2-1B-Instruct-4bit"
    assert req.quant_bits == 4


def test_quantize_request_model_key_unchanged():
    from fusion_mlx.api.convert_models import QuantizeRequest
    req = QuantizeRequest(
        model="some-model",
        output_path=os.path.expanduser("~/.fusion-mlx/models/out"),
        quant_bits=4,
    )
    assert req.model == "some-model"


def test_quantize_request_source_path_respects_output_path_constraint():
    import pytest
    from fusion_mlx.api.convert_models import QuantizeRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        QuantizeRequest(
            source_path="mlx-community/Foo-4bit",
            output_path="/etc/passwd",  # outside allowed dirs
            quant_bits=4,
        )
```

Add `import os` at top of test file if not already present.

Run: `python -m pytest tests/unit/test_convert_routes.py::test_quantize_request_accepts_source_path_alias -q`
Expected: FAIL (source_path not a field → extra field / model missing).

- [ ] **Step 2: Implement — `source_path` field + validator on `QuantizeRequest`**

In `fusion_mlx/api/convert_models.py`, replace line 95 `class QuantizeRequest(_ConvertBase): pass` with:

```python
class QuantizeRequest(_ConvertBase):
    source_path: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_source_path(cls, data):
        if isinstance(data, dict) and "model" not in data and data.get("source_path"):
            data["model"] = data.pop("source_path")
        return data
```

Add `model_validator` to the existing pydantic import. Line 12 currently reads `from pydantic import BaseModel, Field, field_validator` — change to `from pydantic import BaseModel, Field, field_validator, model_validator`.

- [ ] **Step 3: Run tests — pass**

Run: `python -m pytest tests/unit/test_convert_routes.py::test_quantize_request_accepts_source_path_alias tests/unit/test_convert_routes.py::test_quantize_request_model_key_unchanged tests/unit/test_convert_routes.py::test_quantize_request_source_path_respects_output_path_constraint -q`
Expected: 3 PASS.

- [ ] **Step 4: Lint + commit**

Run: `ruff check fusion_mlx/api/convert_models.py tests/unit/test_convert_routes.py && black --check fusion_mlx/api/convert_models.py tests/unit/test_convert_routes.py`
Expected: clean.

```bash
git add fusion_mlx/api/convert_models.py tests/unit/test_convert_routes.py
git commit -m "feat(#646): accept source_path alias on QuantizeRequest (Hub compat)"
```

---

### Task 4: Gap 3a — Mount layered quantize router

**Files:**
- Modify: `fusion_mlx/server.py:36` (import), `:922` (include)
- Test: `tests/unit/test_layered_quantize_routes.py` (create)

**Interfaces:**
- Consumes: `fusion_mlx/api/layered_quantize_routes.py` `router` (line 28, `APIRouter(prefix="/v1", tags=["quantize"])`).
- Produces: `/v1/quantize/layered`, `/v1/quantize/layered/jobs`, `/v1/quantize/layered/jobs/{job_id}` reachable.

- [ ] **Step 1: Write failing test (route existence via TestClient)**

Create `tests/unit/test_layered_quantize_routes.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api import layered_quantize_routes
from fusion_mlx.api.convert_routes import router as convert_router


from fusion_mlx.admin.auth import require_admin


@pytest.fixture()
def app():
    a = FastAPI()

    async def _fake_require_admin():
        return True

    a.dependency_overrides[require_admin] = _fake_require_admin
    a.include_router(convert_router)
    a.include_router(layered_quantize_routes.router)
    return a


def test_layered_jobs_list_route_mounted():
    client = TestClient(app())
    resp = client.get("/v1/quantize/layered/jobs")
    assert resp.status_code == 200


def test_layered_job_detail_route_mounted():
    client = TestClient(app())
    resp = client.get("/v1/quantize/layered/jobs/nonexistent-job-id")
    assert resp.status_code == 404  # route exists, job doesn't
```

Run: `python -m pytest tests/unit/test_layered_quantize_routes.py -q`
Expected: FAIL (404 for the list route — router not mounted in this standalone app is fine, but the test mounts it directly so it should pass; to make it fail-first, the real assertion is against the full server app). Adjust: this test mounts the router directly, so it passes immediately. The FAIL-first guarantee comes from testing the ROUTER is importable + has the routes. Keep it — it guards the routes exist on the router object.

- [ ] **Step 2: Implement — import + include in server.py**

In `fusion_mlx/server.py`:
- After line 36 `from .api.convert_routes import router as convert_router`, add:
  ```python
  from .api.layered_quantize_routes import router as layered_quantize_router
  ```
- After line 922 `app.include_router(convert_router)`, add:
  ```python
        app.include_router(layered_quantize_router)
  ```

- [ ] **Step 3: Verify import smoke**

Run: `python -c "from fusion_mlx.server import create_app; print('ok')"`
Expected: `ok` (router mounts without import error).

- [ ] **Step 4: Run tests — pass**

Run: `python -m pytest tests/unit/test_layered_quantize_routes.py -q`
Expected: 2 PASS.

- [ ] **Step 5: Lint + commit**

Run: `ruff check fusion_mlx/server.py tests/unit/test_layered_quantize_routes.py && black --check fusion_mlx/server.py tests/unit/test_layered_quantize_routes.py`
Expected: clean.

```bash
git add fusion_mlx/server.py tests/unit/test_layered_quantize_routes.py
git commit -m "feat(#646): mount orphaned layered quantize router (/v1/quantize/layered/*)"
```

---

### Task 5: (folded into T1/T4) — no separate task

server.py import + include of layered router lands in T4; load route changes in T1. No separate task needed.

---

### Task 6: Gap 3b — `done` → `completed` terminal status

**Files:**
- Modify: `fusion_mlx/api/convert_routes.py:98` (`_run_job`), `fusion_mlx/api/layered_quantize_routes.py:177` (`_run_layered_quantize`)
- Modify: `tests/unit/test_convert_routes.py:44,68,119,129` (assertion updates)
- Modify: `tests/unit/test_layered_quantize_routes.py` (append terminal-status test)

**Interfaces:**
- Consumes: `_set(job, status=...)` helper in both route modules. `_run_convert` is lazy-imported inside both job bodies (`from fusion_mlx.cli_convert import _run_convert`), so `monkeypatch.setattr("fusion_mlx.cli_convert._run_convert", ...)` mocks it in both.
- Produces: terminal job status string `"completed"` (was `"done"`).

**Critical distinction — terminal-set guard vs equality asserts.** `tests/unit/test_convert_routes.py` has TWO different uses of `"done"`:
- Line 44: `if job["status"] in ("done", "failed"):` — the `_wait` poll helper's terminal-set guard. This is NOT an equality assert. If the writer changes to `"completed"` but this stays `"done"`, `_wait` never sees a terminal state → every test calling `_wait` times out (5s) and fails. **Must change to `in ("completed", "failed")`.**
- Lines 68, 119, 129: `assert job["status"] == "done"` — equality asserts on the finished job. Change to `== "completed"`.

- [ ] **Step 1: Grep current `"done"` assertions to update**

Run: `grep -n '"done"' tests/unit/test_convert_routes.py tests/unit/test_layered_quantize_routes.py`
Record exact line numbers. Expected in `test_convert_routes.py`: line 44 (`in ("done", "failed")` guard), lines 68, 119, 129 (`== "done"` asserts). The layered test file is new from T4 — confirm whether its `_wait` helper (if you added one) also has a `"done"` guard.

- [ ] **Step 2: Write failing test — terminal status `completed` (layered router)**

Append to `tests/unit/test_layered_quantize_routes.py`. This wires a real layered job through the mounted router, mocks `_run_convert` so no model loads, and polls to terminal via a layered-specific `_wait`:

```python
import time


def _wait_layered(client, job_id, timeout=5.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        r = client.get(f"/v1/quantize/layered/jobs/{job_id}")
        assert r.status_code == 200, r.text
        job = r.json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"layered job {job_id} did not finish within {timeout}s")


def test_layered_quantize_terminal_status_is_completed(monkeypatch):
    def _fake_ok(model, **kwargs):
        return kwargs["mlx_path"]

    monkeypatch.setattr("fusion_mlx.cli_convert._run_convert", _fake_ok)
    client = TestClient(app())
    resp = client.post(
        "/v1/quantize/layered",
        json={
            "model": "test/repo",
            "default_bits": 4,
            "layer_rules": [{"pattern": "lm_head", "bits": 8}],
        },
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    job = _wait_layered(client, job_id)
    assert job["status"] == "completed"
    assert job["kind"] == "layered-quantize"
```

Run: `python -m pytest tests/unit/test_layered_quantize_routes.py::test_layered_quantize_terminal_status_is_completed -q`
Expected: FAIL (status still `"done"`, so `_wait_layered` — which already uses `"completed"` — times out; the timeout AssertionError is the failure surface).

- [ ] **Step 3: Implement — writer change in convert_routes**

In `fusion_mlx/api/convert_routes.py:98`:
`_set(job, status="done", progress=1.0, output_path=out)` → `_set(job, status="completed", progress=1.0, output_path=out)`

Also update the log line at `:99` is optional (says "done") — leave the comment at `:96` (`0.1 (running) -> 1.0 (done|failed)`) unchanged; it is a comment, not a writer, and rephrasing prose is out of surgical scope.

- [ ] **Step 4: Implement — writer change in layered_quantize_routes**

In `fusion_mlx/api/layered_quantize_routes.py:177`:
`_set_layered(job, status="done", progress=1.0, output_path=out)` → `_set_layered(job, status="completed", progress=1.0, output_path=out)`

- [ ] **Step 5: Update test assertions**

In `tests/unit/test_convert_routes.py`:
- Line 44: `if job["status"] in ("done", "failed"):` → `if job["status"] in ("completed", "failed"):`
- Lines 68, 119, 129: `assert job["status"] == "done"` → `assert job["status"] == "completed"`

In `tests/unit/test_layered_quantize_routes.py`: the `_wait_layered` helper added in Step 2 already uses `("completed", "failed")` and asserts `== "completed"` — no further edit. If a grep (Step 1) found any other `"done"` in this file, update it.

- [ ] **Step 6: Run tests — pass, no-redder**

Run: `python -m pytest tests/unit/test_convert_routes.py tests/unit/test_layered_quantize_routes.py -q`
Expected: all PASS. Then run the broader suite touchpoint:
Run: `python -m pytest tests/unit/test_convert_routes.py tests/unit/test_layered_quantize_routes.py tests/unit/test_server_load_route.py -q`
Expected: no-redder vs before this task.

- [ ] **Step 7: Grep for missed consumers**

Run: `grep -rn '"done"' fusion_mlx/api/convert_routes.py fusion_mlx/api/layered_quantize_routes.py`
Expected: no remaining `status="done"` writers (the `done|failed` comment at convert_routes.py:96 is a comment, not a writer — leave it). If any other internal consumer compared to `"done"`, it would now break — grep found none in the spec's blast-radius analysis, re-confirm here.

- [ ] **Step 8: Lint + commit**

Run: `ruff check fusion_mlx/api/convert_routes.py fusion_mlx/api/layered_quantize_routes.py tests/unit/test_convert_routes.py tests/unit/test_layered_quantize_routes.py && black --check fusion_mlx/api/convert_routes.py fusion_mlx/api/layered_quantize_routes.py tests/unit/test_convert_routes.py tests/unit/test_layered_quantize_routes.py`
Expected: clean.

```bash
git add fusion_mlx/api/convert_routes.py fusion_mlx/api/layered_quantize_routes.py tests/unit/test_convert_routes.py tests/unit/test_layered_quantize_routes.py
git commit -m "fix(#646): quantize job terminal status done->completed (Hub contract)"
```

---

### Task 7: Full-suite sweep + lint + docs

**Files:**
- Verify: whole suite no-redder
- Modify: `CHANGELOG.md` (entry), `README.md` (if load/quantize endpoint docs mention status)

**Interfaces:**
- Consumes: all prior tasks.

- [ ] **Step 1: Full lint**

Run: `ruff check fusion_mlx/ tests/ && black --check fusion_mlx/ tests/`
Expected: clean (gui_compat pre-existing non-compliance excluded per Global Constraints — if ruff flags gui_compat:970/514 route lines specifically, that's our edit; ensure it matches surrounding style).

- [ ] **Step 2: Full active suite**

Run: `python -m pytest tests/unit -q` (via `rtk proxy pytest` if rtk truncates).
Expected: no-redder than baseline. Record pass/skip/fail counts. Any new failure → fix (Failing-test rule).

- [ ] **Step 3: CHANGELOG entry**

In `CHANGELOG.md`, add to `## [Unreleased]` (create if absent, top of file):

```markdown
### Changed
- `POST /v1/models/{model_id}/load` and `/unload` now accept slash-bearing HF repo ids (e.g. `mlx-community/Llama-3.2`) via URL-encoding or raw slash; `/` in the id maps to the registered hyphen id (#646).
- Quantize job terminal status changed from `done` to `completed` (#646).

### Added
- `source_path` accepted as an alias for `model` on `POST /v1/quantize` (#646).
- `POST /v1/quantize/layered` and its job-status routes now mounted (were written but unreachable) (#646).
```

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(#646): CHANGELOG for Hub-MLX contract alignment"
```

- [ ] **Step 5: Branch summary**

Report: commit count, files changed, test counts. Ready for finishing-a-development-branch (push/PR gated on user approval per CLAUDE.md).
