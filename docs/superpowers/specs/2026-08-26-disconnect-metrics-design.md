# Disconnect Metrics (#645) — Design

> **Status:** Approved (Hybrid: real counter + delete dead) 2026-08-26
> **Issue:** [dahai80/fusion-mlx#645](https://github.com/dahai80/fusion-mlx/issues/645)
> **Path:** Hybrid — fix the real observability gap, delete dead code, retire architecture-pinning tests

## Decision

**Hybrid: real counter + delete dead.**

Issue #645 frames a binary: wire the full C-01 disconnect-attribution feature (scheduler counters, lifetime ledgers, 2-hop inner-engine resolver, recorder(engine) seam, `_run_with_disconnect_guard` wrapper, Prometheus series) OR remove the dead code and retire the tests. The issue author notes removal is "likely the correct path given 0 callers."

This spec takes a third path that matches the verified production reality: the observability gap is REAL (live disconnect handlers emit nothing), but the architecture the quarantined tests pin was NEVER shipped and is not the right design to resurrect. So:

- **Fix the real gap:** add a `fusion_mlx_requests_cancelled_total` counter, ticked from the LIVE disconnect handlers.
- **Delete dead code:** the `_disconnect_guard` streaming wrapper + `_force_abort_request` + recorder stubs have 0 production callers; remove them.
- **Retire architecture-pinning tests:** the 3 quarantined tests + 5 xfails pin never-shipped C-01/M-01 architecture (rapid_mlx_ series, ledgers, 2-hop resolver, `_run_with_disconnect_guard`). Delete them; write one new test pinning the live counter.

This aligns with the standing FINAL SCOPE DECISION: fix the real production gap; do NOT rollback or resurrect shipped architecture to satisfy tests pinning removed/redesign-drift symbols.

## Current State (verified this session)

Three disconnect mechanisms exist in `fusion_mlx/service/disconnect_guard.py` (377 lines). Not one — prior memory conflated them.

### 1. `_disconnect_guard` (streaming wrapper, `:139-294`) — DEAD
- 0 production callers. No route passes `request_id_holder`.
- Streaming routes handle disconnect inline: `openai_routes.py:1382` `except asyncio.CancelledError → engine.abort_request(request_id)`.
- The 7 "passing" tests in `test_disconnect_guard_aborts_scheduler.py` call `_disconnect_guard` directly — they test dead code.

### 2. `_wait_with_disconnect` (non-stream, `:297-377`) — LIVE
- Called by `/v1/responses` (`routes_internal/responses.py:440,453`).
- Polls `raw_request.is_disconnected()`; on disconnect cancels the task and returns `None`. Raises `HTTPException(504)` on timeout, `HTTPException(503)` on `BackpressureError`.
- Calls NO abort, NO recorder, emits NO metric.

### 3. Telemetry recorder — DOES NOT EXIST
- `get_disconnect_abort_recorder` is absent from `fusion_mlx/telemetry/` (dir holds only consent/emit/queue/redact/schema/state/transport).
- `_resolve_disconnect_abort_recorder()` imports it in a try/except → always returns `None` → `_record_disconnect_abort_on_scheduler` is a silent no-op. Aspirational stub.

### Supporting findings
- `ServerMetrics` (`server_metrics.py:77`) has NO cancel fields. `to_dict()` exposes total/successful/failed_requests, token counts, active_requests, model_stats, kv_cache_dtype.
- `/metrics` (`routes_internal/metrics.py`, 296 lines) renders 6 families: build_info, requests_total, prompt/completion tokens, models_discovered/loaded, model_memory_bytes, kv_cache_dtype, response_format_strict_*, response_cache_*. ZERO cancel/disconnect references.
- Anthropic routes (`anthropic_routes.py`): 0 disconnect/abort/cancel refs.
- The real production observability gap: live inline handler + live `_wait_with_disconnect` both handle disconnect but emit nothing.

## Design — New Metric Surface

### Counter field

Add `cancelled_requests: int = 0` field to `ServerMetrics` (`server_metrics.py:77`), alongside `total_requests`/`successful_requests`/`failed_requests`.

Add method `record_disconnect_cancel(self) -> None`:
```python
def record_disconnect_cancel(self) -> None:
    with self._lock:
        self.cancelled_requests += 1
```

Add to `to_dict()` (`:264`): `"cancelled_requests": self.cancelled_requests,`.
Add to `clear_metrics()` (`:184`): `self.cancelled_requests = 0`.

Alltime accumulator: add `at["cancelled_requests"] = at.get("cancelled_requests", 0) + 1` inside `record_disconnect_cancel` (NOT inside `record_request_complete` — a cancel is not a complete request). Mirror the dirty-flag + interval-save pattern. Add `total_cancelled_requests` to `to_alltime_dict()`.

### Metric series

Add `_render_disconnect_metrics()` to `routes_internal/metrics.py`:
```python
def _render_disconnect_metrics() -> str:
    sm = get_server_metrics().to_dict()
    return _fmt_metric(
        "fusion_mlx_requests_cancelled_total",
        "counter",
        "Client-disconnected requests (streaming + non-stream)",
        sm["cancelled_requests"],
        {},
    )
```
Append its output in `render_prometheus_metrics()` aggregator (`:281-289`).

**Naming:** `fusion_mlx_` prefix matches the existing `fusion_mlx_requests_total` series. NOT the never-shipped `rapid_mlx_requests_cancelled_total` the quarantined tests want — those tests retire.

### Tick sites (LIVE handlers only)

**1. Streaming inline handler (`openai_routes.py:1382`):** inside `except asyncio.CancelledError`, after the `engine.abort_request` task is created, call `record_llm_disconnect_cancel()`:
```python
except asyncio.CancelledError:
    logger.info("Client disconnected during streaming: %s", request_id)
    record_llm_disconnect_cancel()
    if engine:
        ...  # existing abort_request task unchanged
```

**2. `_wait_with_disconnect` (`disconnect_guard.py:345`):** tick when `disconnect_task in done`, before cancelling/returning None:
```python
if disconnect_task in done:
    logger.info(...)
    record_llm_disconnect_cancel()
    task.cancel()
    ...
    return None
```
Single site covers both `/v1/responses` call sites automatically. Co-located with disconnect detection.

### Thin wrapper

Add `record_llm_disconnect_cancel()` to `server_metrics.py`, mirroring `record_llm_metrics` (`:291`):
```python
def record_llm_disconnect_cancel() -> None:
    try:
        get_server_metrics().record_disconnect_cancel()
    except Exception as exc:
        logger.debug("Failed to record disconnect cancel: %s", exc)
```
Fail-visible-but-quiet (matches existing metric-recording convention — never break a request over a metric tick).

## Design — Dead-Code Removal

### DELETE from `disconnect_guard.py`

| Symbol | Lines | Why dead |
|---|---|---|
| `_disconnect_guard` | `:139-294` | 0 prod callers; streaming routes handle inline |
| `_force_abort_request` | `:66-136` | only called by dead `_disconnect_guard` |
| `_record_disconnect_abort_on_scheduler` | `:55-63` | only called by `_force_abort_request` |
| `_resolve_disconnect_abort_recorder` | `:33-43` | recorder never existed; always None |
| `_unresolved_engine_dedupe_key` | `:46-52` | only called by dead recorder path |
| `_resolve_sync_scheduler_for_abort` | `:24-30` | only called by dead `_force_abort_request` |
| `_disconnect_abort_recorder` global | `:16` | recorder never existed |
| `_disconnect_abort_lock` global | `:17` | guarded the absent recorder |
| `_pending_force_abort_tasks` set | `:21` | only dead wrapper uses |

Also drop now-unused imports after deletion: `FastAPI.HTTPException` stays (used by `_wait_with_disconnect`), `starlette.requests.Request` stays (param type), `collections.abc.AsyncIterator` stays (param type). `threading` — check if still needed; if only the lock used it, drop.

### KEEP in `disconnect_guard.py`

- `_wait_with_disconnect` (`:297-377`) — LIVE, 2 callers in `responses.py`.
- Its dependencies: `asyncio`, `logging`, `time as _time` (local import), `from ..scheduler import BackpressureError` (local import), `FastAPI.HTTPException`, `starlette.requests.Request`.

### Post-state

File shrinks ~377 → ~80 lines: module docstring + imports + `_wait_with_disconnect` + the new counter tick. Module name `disconnect_guard` still accurate (non-stream disconnect wait). No rename — renaming = scope creep (Rule 3).

### Re-export cleanup

- `service/__init__.py:11` — drop `_disconnect_guard` from imports; `:55` drop from `__all__`. Keep `_wait_with_disconnect` (`:32`, `:76`).
- `service/helpers.py:1733-1735` — drop `_disconnect_guard` and `_force_abort_request` re-exports; keep `_wait_with_disconnect`.

### Verify no orphan callers

After removal, run: `grep -rn "_disconnect_guard\|_force_abort_request\|_record_disconnect_abort\|_resolve_disconnect_abort_recorder\|_resolve_sync_scheduler_for_abort\|_pending_force_abort_tasks" fusion_mlx/ tests/` excluding `__pycache__` — must return only the deleted-test files (removed in Test Disposition) and zero prod refs.

## Design — Test Disposition

### Retire (delete file + remove verdict block from `debt_modules.txt`)

| File | Lines removed from debt_modules.txt | What it pinned (never-shipped) |
|---|---|---|
| `test_cancelled_requests_metric.py` | cluster verdict (~debt_modules.txt:575) | `rapid_mlx_requests_cancelled_total`/`_via_disconnect_total` series, scheduler `num_requests_cancelled` keys, `record_disconnect_abort` ledger |
| `test_disconnect_counter_prod_shape.py` | verdict (~:909) | 2-hop inner-engine resolver, recorder(engine) arg-form, lifetime ledgers |
| `test_disconnect_guard.py` | verdict (~:913) | `_run_with_disconnect_guard` (exists in NEITHER fusion-mlx nor rapid-mlx) |
| `test_disconnect_guard_aborts_scheduler.py` | verdict (~:921) | ALL 12 tests: 7 "passing" call dead `_disconnect_guard` directly; 5 xfail pin dead C-01 path. Deleting the symbol = deleting all its tests. |

### New test: `tests/unit/test_disconnect_cancel_metric.py`

No docstring in prod code per CLAUDE.md — but test files DO use docstrings per codebase convention.

**Unit tests:**
- `test_record_disconnect_cancel_bumps_counter` — `ServerMetrics.record_disconnect_cancel()` increments `cancelled_requests`; `to_dict()` exposes the key.
- `test_cancel_counter_resets_on_clear_metrics` — `clear_metrics()` zeroes it.
- `test_cancel_counter_thread_safe` — concurrent `record_disconnect_cancel()` calls lose no increments (mirror existing lock tests).
- `test_metrics_renders_cancelled_total` — `render_prometheus_metrics()` output contains `fusion_mlx_requests_cancelled_total` with `# TYPE counter`, value matches `cancelled_requests`.
- `test_record_llm_disconnect_cancel_swallows_errors` — wrapper never raises even if `get_server_metrics` blows up.

**Integration tests (TestClient, mock engine, no real model):**
- `test_streaming_cancel_ticks_counter` — TestClient streaming `/v1/chat/completions`, cancel mid-stream (httpx disconnect or explicit `aclose`), assert `cancelled_requests` incremented. Mock engine `abort_request` (no real MLX load).
- `test_wait_with_disconnect_ticks_counter_on_disconnect` — call `_wait_with_disconnect` with a fake `raw_request` whose `is_disconnected()` returns True; assert counter incremented AND return value is None.

**No real model needed** — pure async + TestClient + fakes. Gated to default suite (NOT `@pytest.mark.real_model`).

### debt_modules.txt cleanup

Remove the 4 verdict blocks. Re-verify with `grep -ciE "disconnect|cancel" tests/unit/debt_modules.txt` after — expect 0 (or only unrelated incidental matches, which there are none per the grep this session).

## Design — Docs + CHANGELOG

### README.md

Metrics/Prometheus section: add `fusion_mlx_requests_cancelled_total` to the listed series, with one-line description: "Client-disconnected requests (streaming + non-stream)." Locate the existing metrics listing (search `fusion_mlx_requests_total` in README) and append in the same format.

### CHANGELOG.md

Append new entry at TOP (do NOT backfill the 0.8.15–0.8.39 gap — pre-existing, tracked in `fusion-mlx-v0838-release` memory). Entry:

```markdown
## [Unreleased]
### Added
- `fusion_mlx_requests_cancelled_total` Prometheus counter for client-disconnected requests, ticked from the live streaming and `/v1/responses` non-stream disconnect handlers (#645).

### Removed
- Dead `_disconnect_guard` streaming wrapper, `_force_abort_request`, and always-no-op telemetry recorder stubs — 0 production callers; streaming routes handle disconnect inline (#645).
```

### No new docs/ file

The metric is self-documenting via `# HELP` / `# TYPE` in the `/metrics` output. A standalone doc page for a single counter is over-documentation (Rule 2).

## Files Touched

| File | Action | Lines |
|---|---|---|
| `fusion_mlx/server_metrics.py` | modify | +field, +method, +to_dict key, +clear_metrics, +alltime, +wrapper |
| `fusion_mlx/routes_internal/metrics.py` | modify | +`_render_disconnect_metrics()`, +append in aggregator |
| `fusion_mlx/api/openai_routes.py` | modify | +counter tick in `except CancelledError` (`:1382`) |
| `fusion_mlx/service/disconnect_guard.py` | modify | delete 7 symbols + 3 globals, keep `_wait_with_disconnect`, +counter tick (`:345`) |
| `fusion_mlx/service/__init__.py` | modify | drop 2 re-exports (`_disconnect_guard`, `_force_abort_request`) |
| `fusion_mlx/service/helpers.py` | modify | drop 2 re-exports (`:1733-1735`) |
| `tests/unit/test_disconnect_cancel_metric.py` | create | new: 5 unit + 2 integration |
| `tests/unit/test_cancelled_requests_metric.py` | delete | retire |
| `tests/unit/test_disconnect_counter_prod_shape.py` | delete | retire |
| `tests/unit/test_disconnect_guard.py` | delete | retire |
| `tests/unit/test_disconnect_guard_aborts_scheduler.py` | delete | retire (all 12 test dead symbol) |
| `tests/unit/debt_modules.txt` | modify | remove 4 verdict blocks |
| `README.md` | modify | +metric to Prometheus listing |
| `CHANGELOG.md` | modify | +Unreleased entry |

14 files. Bounded.

## Out of Scope

- **The full C-01/M-01 architecture.** No scheduler `num_requests_cancelled`/`num_requests_cancelled_via_disconnect` keys, no lifetime ledgers (`_cancelled_request_ids`/`_disconnect_abort_ids`), no 2-hop inner-engine scheduler resolver, no `recorder(engine)` arg-form seam, no `_run_with_disconnect_guard` wrapper. These were never shipped; resurrecting them to satisfy retired tests is over-engineering.
- **Distinguishing via_disconnect from other cancels.** The counter is a single `fusion_mlx_requests_cancelled_total`. A `via_disconnect` sub-counter (the M-01 `rapid_mlx_requests_cancelled_via_disconnect_total`) is NOT added — the live handlers only detect client disconnect, so every tick IS a disconnect. Splitting adds a label with one value; no information gained.
- **Wiring `_disconnect_guard` into streaming routes.** The working inline `CancelledError → engine.abort_request` design stays. Replacing it with a guard wrapper to resurrect dead architecture is out of scope.
- **Anthropic route disconnect metrics.** `anthropic_routes.py` has 0 disconnect handling (verified). Adding it = new feature, separate issue. The counter ticks only where disconnect is actually detected today.
- **Renaming `disconnect_guard.py`.** Module shrinks but keeps its name. Rename = churn, no behavior change.
- **Backfilling CHANGELOG 0.8.15–0.8.39.** Pre-existing gap, tracked separately.
- **Releasing to PyPI/homebrew.** This lands code; release is a separate step the user triggers.

## Success Criteria

1. **Counter ticks on real disconnect.** A TestClient streaming request cancelled mid-stream increments `ServerMetrics.cancelled_requests`; a `_wait_with_disconnect` call whose `is_disconnected()` returns True increments it. Verified by the 2 integration tests.
2. **Metric renders.** `GET /metrics` response contains `# TYPE fusion_mlx_requests_cancelled_total counter` and a value line matching `cancelled_requests`. Verified by `test_metrics_renders_cancelled_total`.
3. **Dead code gone.** `grep -rn "_disconnect_guard\|_force_abort_request\|_record_disconnect_abort\|_resolve_disconnect_abort_recorder\|_resolve_sync_scheduler_for_abort\|_pending_force_abort_tasks\|_disconnect_abort_recorder\|_disconnect_abort_lock" fusion_mlx/` returns 0 matches. `disconnect_guard.py` holds only `_wait_with_disconnect`.
4. **No orphan callers.** `grep -rn "_disconnect_guard\|_force_abort_request" tests/` returns 0 matches after the 4 test files are deleted.
5. **Suite no redder.** `pytest tests/unit -q` passes with 0 new failures; the 4 deleted test files removed from `debt_modules.txt`; net test count change = -4 retired files + 1 new file (7 tests). `ruff check` + `black --check` clean on all touched files.
6. **Docs updated.** README lists the new series; CHANGELOG has the Unreleased entry.
7. **`_wait_with_disconnect` still works.** `/v1/responses` non-stream path unchanged in behavior (only adds a metric tick on disconnect).
