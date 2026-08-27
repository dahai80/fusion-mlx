# Disconnect Metrics (#645) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `fusion_mlx_requests_cancelled_total` Prometheus counter ticked from the live disconnect handlers, delete the dead `_disconnect_guard` streaming wrapper and recorder stubs, and retire the 4 test files pinning never-shipped C-01/M-01 architecture.

**Architecture:** Single counter on `ServerMetrics`, rendered in `/metrics`, ticked at the two LIVE disconnect sites (streaming `CancelledError` handler + `_wait_with_disconnect`). The dead `_disconnect_guard`/`_force_abort_request`/recorder symbols are removed; `_wait_with_disconnect` is kept (live). Tests: delete 4 files, add 1 new (5 unit + 2 integration).

**Tech Stack:** Python 3.12 (venv), FastAPI, pytest + httpx TestClient, ruff + black (line-length 88), mypy. No MLX / real model needed.

**Spec:** `docs/superpowers/specs/2026-08-26-disconnect-metrics-design.md`

## Global Constraints

- Indentation is multiples of 4. **No docstrings in production code** (`fusion_mlx/`). Test files (`tests/`) DO use docstrings (codebase convention).
- New production code logs by default.
- Metric recording never breaks a request (fail-visible-but-quiet, matches `record_llm_metrics` convention).
- Metric name: `fusion_mlx_requests_cancelled_total` (NOT `rapid_mlx_`). Counter type. `# HELP` text: "Client-disconnected requests (streaming + non-stream)".
- `_wait_with_disconnect` behavior is unchanged except the added counter tick on disconnect.
- Streaming inline `CancelledError` handler (`openai_routes.py`) behavior unchanged except the added tick.
- Do NOT add: scheduler `num_requests_cancelled` keys, lifetime ledgers, 2-hop resolver, `recorder(engine)` seam, `_run_with_disconnect_guard`, `via_disconnect` sub-counter. Out of scope (spec §Out of Scope).
- Lint scope: `fusion_mlx/` + `tests/` via `ruff check` + `black --check`. Do not lint `fusion_mlx/patches/`.
- Suite must be no redder: `pytest tests/unit -q` 0 new failures.
- Branch: `feat/645-disconnect-metrics` (already created, spec committed `b13c5ea`).
- Commit/push only when user asks. GitHub operations in English.

---

<!-- TASKS BELOW -->

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `fusion_mlx/server_metrics.py` | `ServerMetrics` dataclass + global singleton + `record_llm_*` wrappers | modify: +field, +method, +to_dict, +clear, +alltime, +wrapper |
| `fusion_mlx/routes_internal/metrics.py` | Prometheus `/metrics` render | modify: +`_render_disconnect_metrics()`, +extend aggregator |
| `fusion_mlx/api/openai_routes.py` | OpenAI streaming route | modify: +tick in `except CancelledError` (`:1383`), +import |
| `fusion_mlx/service/disconnect_guard.py` | disconnect wait/abort | modify: delete 7 symbols + 3 globals, keep `_wait_with_disconnect`, +tick |
| `fusion_mlx/service/__init__.py` | service package re-exports | modify: drop 2 re-exports |
| `fusion_mlx/service/helpers.py` | service helpers re-exports | modify: drop 2 re-exports (`:1733-1735`) |
| `tests/unit/test_disconnect_cancel_metric.py` | new test module | create: 5 unit + 2 integration |
| `tests/unit/test_cancelled_requests_metric.py` | retired | delete |
| `tests/unit/test_disconnect_counter_prod_shape.py` | retired | delete |
| `tests/unit/test_disconnect_guard.py` | retired | delete |
| `tests/unit/test_disconnect_guard_aborts_scheduler.py` | retired (tests dead symbol) | delete |
| `tests/unit/debt_modules.txt` | quarantine registry | modify: remove 4 verdict blocks |
| `README.md` | project docs | modify: +metric to listing |
| `CHANGELOG.md` | release log | modify: +Unreleased entry |

Tasks ordered by dependency: counter field first (foundation), render next, tick sites next, dead-code removal, re-export cleanup, test retire + new, docs last.

---

### Task 1: Add `cancelled_requests` counter field + method to `ServerMetrics`

**Files:**
- Modify: `fusion_mlx/server_metrics.py:77-200`
- Test: `tests/unit/test_disconnect_cancel_metric.py` (create)

**Interfaces:**
- Consumes: nothing
- Produces: `ServerMetrics.cancelled_requests: int` field; `ServerMetrics.record_disconnect_cancel() -> None`; `to_dict()["cancelled_requests"]`; `clear_metrics()` zeroes it; alltime `total_cancelled_requests`; module-level `record_llm_disconnect_cancel() -> None` wrapper

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_disconnect_cancel_metric.py`:
```python
# SPDX-License-Identifier: Apache-2.0
"""Tests for the client-disconnect cancel counter (#645)."""

from __future__ import annotations

import threading

import pytest

from fusion_mlx.server_metrics import (
    ServerMetrics,
    get_server_metrics,
    record_llm_disconnect_cancel,
)


def _fresh() -> ServerMetrics:
    return ServerMetrics()


def test_record_disconnect_cancel_bumps_counter():
    sm = _fresh()
    assert sm.cancelled_requests == 0
    sm.record_disconnect_cancel()
    sm.record_disconnect_cancel()
    assert sm.cancelled_requests == 2


def test_cancel_counter_in_to_dict():
    sm = _fresh()
    sm.record_disconnect_cancel()
    d = sm.to_dict()
    assert d["cancelled_requests"] == 1


def test_cancel_counter_resets_on_clear_metrics():
    sm = _fresh()
    sm.record_disconnect_cancel()
    sm.record_disconnect_cancel()
    sm.clear_metrics()
    assert sm.cancelled_requests == 0


def test_cancel_counter_thread_safe():
    sm = _fresh()
    n_threads = 20
    n_each = 50

    def _bump():
        for _ in range(n_each):
            sm.record_disconnect_cancel()

    threads = [threading.Thread(target=_bump) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sm.cancelled_requests == n_threads * n_each


def test_record_llm_disconnect_cancel_swallows_errors(monkeypatch):
    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "fusion_mlx.server_metrics.get_server_metrics", lambda: _boom()
    )
    # Must not raise even though get_server_metrics blows up.
    record_llm_disconnect_cancel()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_disconnect_cancel_metric.py::test_record_disconnect_cancel_bumps_counter tests/unit/test_disconnect_cancel_metric.py::test_record_llm_disconnect_cancel_swallows_errors -v`
Expected: FAIL with `AttributeError: 'ServerMetrics' object has no attribute 'cancelled_requests'` / `ImportError: cannot import name 'record_llm_disconnect_cancel'`

- [ ] **Step 3: Add the field + method + dict + clear + alltime + wrapper**

In `fusion_mlx/server_metrics.py`:

Add field after `active_requests` (L87):
```python
    cancelled_requests: int = 0
```

Add method after `update_active_requests` (after L106):
```python
    def record_disconnect_cancel(self) -> None:
        with self._lock:
            self.cancelled_requests += 1
            at = self._alltime
            at["total_cancelled_requests"] = (
                at.get("total_cancelled_requests", 0) + 1
            )
            self._alltime_dirty = True
            now = time.monotonic()
            if now - self._alltime_last_save >= _ALLTIME_SAVE_INTERVAL:
                _save_alltime_to_disk(self._alltime)
                self._alltime_dirty = False
                self._alltime_last_save = now
```

Add to `to_dict()` return dict (after `"active_requests"` key, L274):
```python
            "cancelled_requests": self.cancelled_requests,
```

Add to `clear_metrics()` (after `self.active_requests = 0`, L192):
```python
            self.cancelled_requests = 0
```

Add to `to_alltime_dict()` return dict (after `"total_requests"`, L235):
```python
            "total_cancelled_requests": at.get("total_cancelled_requests", 0),
```

Add module-level wrapper after `record_llm_metrics` (after L309):
```python
def record_llm_disconnect_cancel() -> None:
    try:
        get_server_metrics().record_disconnect_cancel()
    except Exception as exc:
        logger.debug("Failed to record disconnect cancel: %s", exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_disconnect_cancel_metric.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add fusion_mlx/server_metrics.py tests/unit/test_disconnect_cancel_metric.py
git commit -m "feat(#645): add cancelled_requests counter to ServerMetrics"
```

---

### Task 2: Render `fusion_mlx_requests_cancelled_total` in `/metrics`

**Files:**
- Modify: `fusion_mlx/routes_internal/metrics.py:281-289`
- Test: `tests/unit/test_disconnect_cancel_metric.py` (append)

**Interfaces:**
- Consumes: `get_server_metrics().to_dict()["cancelled_requests"]` (Task 1)
- Produces: `_render_disconnect_metrics() -> list[str]`; appended in `render_prometheus_metrics()`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_disconnect_cancel_metric.py`:
```python
def test_metrics_renders_cancelled_total():
    from fusion_mlx.routes_internal.metrics import render_prometheus_metrics

    sm = get_server_metrics()
    before = sm.cancelled_requests
    sm.record_disconnect_cancel()
    body = render_prometheus_metrics()
    assert "# TYPE fusion_mlx_requests_cancelled_total counter" in body
    assert "# HELP fusion_mlx_requests_cancelled_total" in body
    # Global singleton — assert the delta lands in the rendered value, not an
    # absolute number (other tests may have bumped the counter).
    assert f"fusion_mlx_requests_cancelled_total {before + 1}" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_disconnect_cancel_metric.py::test_metrics_renders_cancelled_total -v`
Expected: FAIL — series absent from `/metrics` body

- [ ] **Step 3: Add the render function + aggregator append**

In `fusion_mlx/routes_internal/metrics.py`, add before `render_prometheus_metrics` (before L281):
```python
def _render_disconnect_metrics() -> list[str]:
    sm = get_server_metrics().to_dict()
    return _fmt_metric(
        "fusion_mlx_requests_cancelled_total",
        "counter",
        "Client-disconnected requests (streaming + non-stream)",
        sm["cancelled_requests"],
        None,
    )
```

`get_server_metrics` is imported locally inside `_render_engine_metrics` at `:95` (`from ..server_metrics import get_server_metrics`), NOT at module top. The new module-level `_render_disconnect_metrics()` needs it at top-level. Add to the top-level import block (after `from ..api import response_format_metrics`, `:10`):
```python
from ..server_metrics import get_server_metrics
```
The `:95` local import inside `_render_engine_metrics` stays as-is (surgical — do not touch it; Rule 3). The top-level import is a separate, new line for the new function.

Append in `render_prometheus_metrics` (after `lines.extend(_render_engine_metrics())`):
```python
    lines.extend(_render_disconnect_metrics())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_disconnect_cancel_metric.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Lint the touched file**

Run: `ruff check fusion_mlx/routes_internal/metrics.py && black --check fusion_mlx/routes_internal/metrics.py`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add fusion_mlx/routes_internal/metrics.py tests/unit/test_disconnect_cancel_metric.py
git commit -m "feat(#645): render fusion_mlx_requests_cancelled_total in /metrics"
```

---

### Task 3: Tick counter from `_wait_with_disconnect` (non-stream, LIVE)

**Files:**
- Modify: `fusion_mlx/service/disconnect_guard.py:345-356`

**Interfaces:**
- Consumes: `record_llm_disconnect_cancel()` (Task 1)
- Produces: counter tick on non-stream disconnect

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_disconnect_cancel_metric.py`:
```python
@pytest.mark.asyncio
async def test_wait_with_disconnect_ticks_counter_on_disconnect():
    import time

    from fusion_mlx.service.disconnect_guard import _wait_with_disconnect

    class _Disconnects:
        async def is_disconnected(self) -> bool:
            return True

    async def _noop():
        await asyncio.sleep(10)

    sm = get_server_metrics()
    before = sm.cancelled_requests
    result = await _wait_with_disconnect(_noop(), _Disconnects(), timeout=5.0)
    assert result is None
    assert sm.cancelled_requests == before + 1
```

Add `import asyncio` to the test module imports if not present.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_disconnect_cancel_metric.py::test_wait_with_disconnect_ticks_counter_on_disconnect -v`
Expected: FAIL — counter not incremented (before == after)

- [ ] **Step 3: Add the tick**

In `fusion_mlx/service/disconnect_guard.py`, in `_wait_with_disconnect`, the `if disconnect_task in done:` branch (L345). Add the tick right after the `logger.info(...)` call, before `task.cancel()`:
```python
        if disconnect_task in done:
            logger.info(
                f"[disconnect_guard] CLIENT DISCONNECTED (non-stream) "
                f"elapsed={_time.monotonic() - _t0:.1f}s"
            )
            try:
                from ..server_metrics import record_llm_disconnect_cancel

                record_llm_disconnect_cancel()
            except Exception:
                logger.debug("disconnect cancel metric tick failed", exc_info=True)
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            return None
```
(Local import mirrors the file's existing `from ..scheduler import BackpressureError` local-import pattern — keeps the module's top-level import surface minimal. The try/except is belt-and-suspenders since the wrapper already swallows, but the guard docstring says "never break a request over a metric tick".)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_disconnect_cancel_metric.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add fusion_mlx/service/disconnect_guard.py tests/unit/test_disconnect_cancel_metric.py
git commit -m "feat(#645): tick cancel counter from _wait_with_disconnect"
```

---

### Task 4: Tick counter from streaming `CancelledError` handler (LIVE)

**Files:**
- Modify: `fusion_mlx/api/openai_routes.py:51,1383`

**Interfaces:**
- Consumes: `record_llm_disconnect_cancel()` (Task 1)
- Produces: counter tick on stream cancel

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_disconnect_cancel_metric.py`:
```python
def test_streaming_cancel_tick_is_wired_in_cancellederror_handler():
    """AST-verify the streaming CancelledError handler calls the cancel-counter
    tick. A real mid-stream TestClient cancel is fragile across Starlette/uvicorn
    versions (the CancelledError path depends on live ASGI disconnect), so this
    pins the wiring deterministically instead of asserting behavior through a
    brittle harness. If the tick call is removed or moved out of the handler,
    this test fails — which is exactly the regression that matters."""
    import ast
    from pathlib import Path

    src = Path("fusion_mlx/api/openai_routes.py").read_text()
    tree = ast.parse(src)
    wired = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        exc = node.type
        if exc is None:
            continue
        # Match `asyncio.CancelledError` or bare `CancelledError`.
        name = exc.id if isinstance(exc, ast.Name) else getattr(exc, "attr", None)
        if name != "CancelledError":
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "record_llm_disconnect_cancel"
            ):
                wired = True
    assert wired, "record_llm_disconnect_cancel() must be called inside the streaming CancelledError handler"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_disconnect_cancel_metric.py::test_streaming_cancel_tick_is_wired_in_cancellederror_handler -v`
Expected: FAIL — `record_llm_disconnect_cancel` not yet present in openai_routes.py (assert wired is False).

- [ ] **Step 3: Add the tick + import**

In `fusion_mlx/api/openai_routes.py`, extend the import at L51:
```python
from ..server_metrics import record_llm_disconnect_cancel, record_llm_metrics
```

In the `except asyncio.CancelledError:` block (L1382), add the tick right after the `logger.info(...)` call, before `if engine:`:
```python
    except asyncio.CancelledError:
        logger.info("Client disconnected during streaming: %s", request_id)
        record_llm_disconnect_cancel()
        if engine:
            ...
```

- [ ] **Step 4: Verify the tick is wired (static)**

Run: `grep -n "record_llm_disconnect_cancel" fusion_mlx/api/openai_routes.py`
Expected: 2 matches — the import (L51) and the call (~L1384).

- [ ] **Step 5: Lint + run full new test file**

Run: `ruff check fusion_mlx/api/openai_routes.py && black --check fusion_mlx/api/openai_routes.py && pytest tests/unit/test_disconnect_cancel_metric.py -v`
Expected: lint clean, tests pass (8 tests)

- [ ] **Step 6: Commit**

```bash
git add fusion_mlx/api/openai_routes.py tests/unit/test_disconnect_cancel_metric.py
git commit -m "feat(#645): tick cancel counter from streaming CancelledError handler"
```

---

### Task 5: Delete dead `_disconnect_guard` symbols from `disconnect_guard.py`

**Files:**
- Modify: `fusion_mlx/service/disconnect_guard.py` (delete L16-21, 24-136, 139-294; keep L297-377 + tick)

**Interfaces:**
- Consumes: nothing
- Produces: `disconnect_guard.py` holding only `_wait_with_disconnect` (+ module docstring, imports, tick from Task 3)

- [ ] **Step 1: Snapshot current prod callers of the doomed symbols**

Run: `grep -rn "_disconnect_guard\b\|_force_abort_request\|_record_disconnect_abort_on_scheduler\|_resolve_disconnect_abort_recorder\|_resolve_sync_scheduler_for_abort\|_unresolved_engine_dedupe_key\|_pending_force_abort_tasks\|_disconnect_abort_recorder\|_disconnect_abort_lock" fusion_mlx/ --include="*.py" | grep -v __pycache__ | grep -v "disconnect_guard.py:"`
Expected: matches ONLY in `service/__init__.py` and `service/helpers.py` (re-exports removed in Task 6). Zero route/engine callers. If any route/engine caller appears, STOP — the symbol is not dead; re-scope.

- [ ] **Step 2: Delete the dead symbols**

In `fusion_mlx/service/disconnect_guard.py`, delete:
- `_disconnect_abort_recorder` global (L16)
- `_disconnect_abort_lock` global (L17)
- `_pending_force_abort_tasks` set (L19-21)
- `_resolve_sync_scheduler_for_abort` (L24-30)
- `_resolve_disconnect_abort_recorder` (L33-43)
- `_unresolved_engine_dedupe_key` (L46-52)
- `_record_disconnect_abort_on_scheduler` (L55-63)
- `_force_abort_request` (L66-136)
- `_disconnect_guard` (L139-294)

Keep: module docstring (L1-2), imports (L4-14), `_wait_with_disconnect` (L297-377, with Task 3 tick), and the logger.

After deletion, prune now-unused imports: `threading` (only the lock used it) — drop. `collections.abc.AsyncIterator` — check if `_wait_with_disconnect` references it; it does not (params are untyped `coro`, `raw_request`), so drop. Keep `asyncio`, `logging`, `fastapi.HTTPException`, `starlette.requests.Request`.

- [ ] **Step 3: Verify the file holds only `_wait_with_disconnect`**

Run: `grep -nE "^def |^async def |^class " fusion_mlx/service/disconnect_guard.py`
Expected: exactly one match — `async def _wait_with_disconnect` (plus any top-level constants). Zero matches for the deleted symbol names.

Run: `grep -cE "_disconnect_guard|_force_abort_request|_record_disconnect_abort|_resolve_disconnect_abort_recorder|_resolve_sync_scheduler_for_abort|_unresolved_engine_dedupe_key|_pending_force_abort_tasks" fusion_mlx/service/disconnect_guard.py`
Expected: 0

- [ ] **Step 4: Lint + import smoke test**

Run: `ruff check fusion_mlx/service/disconnect_guard.py && black --check fusion_mlx/service/disconnect_guard.py && python -c "from fusion_mlx.service.disconnect_guard import _wait_with_disconnect; print('ok')"`
Expected: lint clean, import succeeds

- [ ] **Step 5: Run the cancel-metric tests (Task 3's tick must survive)**

Run: `pytest tests/unit/test_disconnect_cancel_metric.py -v`
Expected: PASS (8 tests) — the `_wait_with_disconnect` tick still works

- [ ] **Step 6: Commit**

```bash
git add fusion_mlx/service/disconnect_guard.py
git commit -m "refactor(#645): delete dead _disconnect_guard streaming wrapper + recorder stubs"
```

---

### Task 6: Clean up re-exports in `service/__init__.py` + `helpers.py`

**Files:**
- Modify: `fusion_mlx/service/__init__.py:11,32,55,76`
- Modify: `fusion_mlx/service/helpers.py:1733-1735`

**Interfaces:**
- Consumes: Task 5 deletions
- Produces: no dangling re-exports of deleted symbols

- [ ] **Step 1: Find exact re-export lines**

Run: `grep -n "_disconnect_guard\|_force_abort_request\|_wait_with_disconnect" fusion_mlx/service/__init__.py fusion_mlx/service/helpers.py`
Expected: shows the lines to edit. `_wait_with_disconnect` STAYS; `_disconnect_guard` + `_force_abort_request` go.

- [ ] **Step 2: Remove the dead re-exports**

In `fusion_mlx/service/__init__.py`: remove `_disconnect_guard` from the import block (L11) and from `__all__` (L55). Keep `_wait_with_disconnect` (L32, L76). (Also remove `_force_abort_request` if present in `__all__`.)

In `fusion_mlx/service/helpers.py`: at L1733-1735, remove the `_disconnect_guard` and `_force_abort_request` re-export lines; keep the `_wait_with_disconnect` line.

- [ ] **Step 3: Verify zero orphan refs**

Run: `grep -rn "_disconnect_guard\|_force_abort_request" fusion_mlx/ --include="*.py" | grep -v __pycache__`
Expected: 0 matches (all deleted from `disconnect_guard.py` in Task 5, re-exports removed here).

- [ ] **Step 4: Lint + import smoke test**

Run: `ruff check fusion_mlx/service/__init__.py fusion_mlx/service/helpers.py && python -c "from fusion_mlx.service import _wait_with_disconnect; from fusion_mlx.service.helpers import _wait_with_disconnect; print('ok')"`
Expected: lint clean, both imports succeed

- [ ] **Step 5: Run the suite smoke (no import errors)**

Run: `pytest tests/unit -q -x 2>&1 | tail -15`
Expected: no `ImportError`/`AttributeError` from the removed re-exports; suite collects and runs (failures unrelated to #645 are pre-existing — note them but do not fix here unless they are #645-caused).

- [ ] **Step 6: Commit**

```bash
git add fusion_mlx/service/__init__.py fusion_mlx/service/helpers.py
git commit -m "refactor(#645): drop dead _disconnect_guard/_force_abort_request re-exports"
```

---

### Task 7: Retire the 4 test files + clean `debt_modules.txt`

**Files:**
- Delete: `tests/unit/test_cancelled_requests_metric.py`
- Delete: `tests/unit/test_disconnect_counter_prod_shape.py`
- Delete: `tests/unit/test_disconnect_guard.py`
- Delete: `tests/unit/test_disconnect_guard_aborts_scheduler.py`
- Modify: `tests/unit/debt_modules.txt`

**Interfaces:**
- Consumes: Task 5+6 (symbols gone; tests would now ImportError anyway)
- Produces: 4 fewer quarantined files; `debt_modules.txt` minus 4 verdict blocks

- [ ] **Step 1: Delete the 4 test files**

```bash
git rm tests/unit/test_cancelled_requests_metric.py tests/unit/test_disconnect_counter_prod_shape.py tests/unit/test_disconnect_guard.py tests/unit/test_disconnect_guard_aborts_scheduler.py
```

- [ ] **Step 2: Remove the 4 verdict blocks from `debt_modules.txt`**

Open `tests/unit/debt_modules.txt`. Remove the comment-block + filename entry for each of:
- `test_cancelled_requests_metric.py` (cluster verdict, references C-01/M-01)
- `test_disconnect_counter_prod_shape.py` (verdict, references 2-hop resolver)
- `test_disconnect_guard.py` (verdict, references `_run_with_disconnect_guard`)
- `test_disconnect_guard_aborts_scheduler.py` (verdict, references 5 xfail + rescued)

Each block is a `# ...` comment paragraph immediately followed by the bare filename line. Delete both the comment and the filename line for each.

- [ ] **Step 3: Verify the quarantine list is clean**

Run: `grep -ciE "disconnect|cancel" tests/unit/debt_modules.txt`
Expected: 0 (no remaining disconnect/cancel verdict blocks). If >0, inspect — may be an incidental match in an unrelated verdict; leave unrelated, only the 4 #645 blocks go.

- [ ] **Step 4: Verify the 4 files are gone from the active quarantine**

Run: `pytest --collect-only -q 2>&1 | grep -cE "test_cancelled_requests_metric|test_disconnect_counter_prod_shape|test_disconnect_guard\.py|test_disconnect_guard_aborts_scheduler"`
Expected: 0

- [ ] **Step 5: Run the full suite (no redder)**

Run: `pytest tests/unit -q 2>&1 | tail -5`
Expected: 0 failures attributable to #645. Compare pass/skip/xfail counts to baseline (12003 passed / 391 skipped / 314 xfailed). Deleted quarantined files do not change the active pass count (they were quarantined = not collected). The new `test_disconnect_cancel_metric.py` ADDS ~7 passing + 1 skip.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/debt_modules.txt
git commit -m "test(#645): retire 4 disconnect/cancel quarantine files pinning dead C-01/M-01"
```

---

### Task 8: Update README + CHANGELOG

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: all prior tasks
- Produces: user-facing docs

- [ ] **Step 1: Find the README metrics listing**

Run: `grep -n "fusion_mlx_requests_total\|prometheus\|/metrics" README.md | head`
Expected: the metrics/Prometheus section line numbers.

- [ ] **Step 2: Add the new series to README**

In the README metrics listing, after `fusion_mlx_requests_total`, add (match the surrounding format exactly):
```
- `fusion_mlx_requests_cancelled_total` — client-disconnected requests (streaming + non-stream)
```

- [ ] **Step 3: Add CHANGELOG entry**

In `CHANGELOG.md`, at the TOP (before the most recent entry; do NOT backfill the 0.8.15–0.8.39 gap), add:
```markdown
## [Unreleased]
### Added
- `fusion_mlx_requests_cancelled_total` Prometheus counter for client-disconnected requests, ticked from the live streaming and `/v1/responses` non-stream disconnect handlers (#645).

### Removed
- Dead `_disconnect_guard` streaming wrapper, `_force_abort_request`, and always-no-op telemetry recorder stubs — 0 production callers; streaming routes handle disconnect inline (#645).
```

- [ ] **Step 4: Commit**

```bash
git add README.md CHANGELOG.md
git commit -m "docs(#645): README metrics listing + CHANGELOG entry for disconnect counter"
```

---

## Verification (final sweep)

- [ ] `ruff check fusion_mlx/ tests/` clean
- [ ] `black --check fusion_mlx/ tests/` clean
- [ ] `pytest tests/unit -q` — 0 new failures; net +~7 tests, -4 quarantined files
- [ ] `grep -rn "_disconnect_guard\|_force_abort_request\|_record_disconnect_abort\|_resolve_disconnect_abort_recorder\|_resolve_sync_scheduler_for_abort\|_unresolved_engine_dedupe_key\|_pending_force_abort_tasks" fusion_mlx/ --include="*.py" | grep -v __pycache__` → 0
- [ ] `grep -rn "_disconnect_guard\|_force_abort_request" tests/ --include="*.py" | grep -v __pycache__` → 0
- [ ] `python -c "from fusion_mlx.routes_internal.metrics import render_prometheus_metrics; print(render_prometheus_metrics())"` → contains `fusion_mlx_requests_cancelled_total`
- [ ] Public API boundary guard: `python scripts/check_public_api_boundary.py --root tests/` → no new violations (no `public_api` symbol touched)

