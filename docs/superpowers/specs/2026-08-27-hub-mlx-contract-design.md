# Hub↔MLX API Contract Design (#646)

> **Status:** Design approved 2026-08-27. Pre-implementation spec.
> **Issue:** dahai80/fusion-mlx#646 — "Expose model-load REST endpoint + align quantize request schema for external clients (fusion-model-hub)"
> **Branch (pending):** `feat/646-hub-mlx-contract`

## Context

Issue #646 was filed by the fusion-model-hub integration: three gaps between what Hub sends and what fusion-mlx exposes. Two of the three gap premises are **stale** — the endpoints exist in code but fail at runtime:

- **Gap 1** (premise "no model-load REST endpoint"): STALE. `POST /v1/models/{model_id}/load` exists (`server.py:1026`, added 2026-06-28 commit 3d58e137). It fails at runtime for slash-bearing HF repo ids (`mlx-community/Llama-3.2-1B-Instruct-4bit`) — route-match 404 + pool-lookup miss. This is the real gap.
- **Gap 2** (premise "quantize schema mismatch"): VALID. Hub sends `{"source_path", "output_path", "quant_bits"}`; `QuantizeRequest` requires `model` → 422.
- **Gap 3** (premise "layered job-status path mismatch"): PARTIALLY STALE. The layered router (`layered_quantize_routes.py`) is written but **never mounted** → all `/v1/quantize/layered/*` 404. Also `status:"done"` vs Hub's expected `"completed"`.

Empirical confirmation (live server repro, model loaded, `X-Fusion-Source: model-hub` + API key):
```
A: %2F encoded slash → 404 not_found_error   (route did not match)
B: raw slash (4 segments)  → 404              (route did not match)
C: hyphenated id mlx-community-Llama-3.2-1B-Instruct-4bit → 200 {"status":"ok","model_id":...,"message":"Already loaded"}
```

## Decisions

Four decisions locked via clarifying questions (user, 2026-08-27):

1. **Gap 1 — Accept encoded slash** (Approach A). Keep path-param `{model_id}`; switch to `{model_id:path}` converter so slash-bearing ids match; extend `resolve_model_id` to map `/`→`-` on pool-lookup miss. Hub works unchanged. Minimal Hub change.
2. **Gap 2 — Accept `source_path` alias.** `model_validator` copies `source_path`→`model` when `model` absent. Hub works unchanged.
3. **Gap 3a — Mount layered router.** Include the existing orphaned `layered_quantize_routes` router. Hub's `/v1/quantize/layered/jobs/{id}` works as-is. Keep two job stores.
4. **Gap 3b — `done` → `completed`.** Change MLX terminal status in both job stores. Aligns with Hub's expectation.

## Gap 1 — Model-load route + id resolution

### Root cause (two distinct failures)

1. **Route-match failure.** Both load routes use single-segment `{model_id}`:
   - `server.py:1026` `@app.post("/v1/models/{model_id}/load")` (`load_model_public`)
   - `server.py:1067` `@app.post("/v1/models/{model_id}/unload")` (`unload_model_public`)
   - `gui_compat/server.py:470` `@router.post("/v1/models/{model_name}/load")` (mounted first via `server.py:957`, when gui_compat import succeeds)

   uvicorn decodes `%2F`→`/` into `scope["path"]` before Starlette routes on it → `mlx-community/Llama-3.2` is 2 path segments → no single-segment route matches → generic `not_found_error` (NOT the route's own "Model not found" detail). Confirmed empirically.

2. **Pool-lookup failure.** Even with a matching route, `resolve_model_id` (`server.py:665`) only strips `fusion-mlx/`/`fusion/` prefixes — it does NOT map `mlx-community/X`→`mlx-community-X`. `pool.get_entry` (`engine_pool.py:546`) is a raw `self._entries.get(model_id)`. Registered ids are hyphenated (HF repo `/`→`-` at download; `model_id` = directory name on disk, `engine_pool.py:63`). So a slash id fails the pool lookup too.

### Approach A — path converter + resolve_model_id slash→hyphen

**1.1 Route signature — public load/unload routes use `{model_id:path}`**

Change `server.py:1026` → `@app.post("/v1/models/{model_id:path}/load")`, and `server.py:1067` unload likewise. Also change `gui_compat/server.py:470` load and `:514` unload to `{model_name:path}` for consistency (same route-match bug regardless of which is mounted).

Verified via `compile_path` probe: `{model_id:path}/load` compiles to regex `^/v1/models/(?P<model_id>.*)/load$`; greedy `.*` backtracks, captures `mlx-community/Llama-3.2` while leaving `/load` as the suffix. `gui_compat:794/931` already use `{model_id:path}` — no new convention.

Route-order safety: `/v1/models/status` (GET, `server.py:946`, registered before load) is NOT swallowed by `{model_id:path}/load` — the `:path` regex requires the `/load` suffix, so a bare `/v1/models/status` (no `/load`) does not match the load route (probe confirms NO MATCH). Existing hyphen-id clients also unaffected: `{model_id:path}/load` still matches `/v1/models/a-b/load` (probe confirms MATCH `a-b`). The `:path` converter is a superset of single-segment matching — strictly additive, no regression.

**1.2 `resolve_model_id` — slash→hyphen map on pool-lookup miss**

Current `server.py:665`: strips `fusion-mlx/`/`fusion/` only. Add: if id contains `/` and (after prefix-strip) `pool.get_entry(resolved)` is None, map `/`→`-` and retry the pool lookup once.

Scoping rule (critical — prevents shadowing a genuine slash id): the slash→hyphen retry fires only when `pool.get_entry` lookup fails. If a genuine `a/b` entry exists, it returns before the retry and is never mapped. Existing client behavior unchanged.

Why `resolve_model_id` not `pool.get_entry`: `resolve_model_id` is the single choke point called before `get_entry` in the route handler (`server.py:1029` resolved, `:1038` `get_engine(resolved)`). Mapping here means `get_engine` also receives the hyphen id. Centralized.

### Test strategy — real server, not TestClient

Known constraint: Starlette TestClient (httpx) does NOT pre-decode `%2F` like uvicorn — it lies about route match. So route-match tests run against real uvicorn (`start.sh` + `FUSION_MLX_REAL_MODEL_TESTS=1`, codebase convention).

- **Unit route-compile test:** `compile_path` regex assertion (like the probe) — covers converter logic, no server.
- **Unit pool-lookup test:** `resolve_model_id` slash→hyphen map with a mock pool (entry exists under hyphen id, lookup with slash id resolves).
- **Real-model round-trip:** encoded-slash `/load` → 200, under the real-model marker.

### Blast radius

- `server.py`: 2 route-signature lines (`:path`), `resolve_model_id` ~5 lines added.
- `gui_compat/server.py`: 2 route-signature lines (`:path`) — co-located, same change.
- `pool/engine_pool.py`: unchanged (mapping lives in `resolve_model_id`).
- Tests: 1 route-compile unit, 1 `resolve_model_id` unit, 1 real-model round-trip.
- Existing clients: unchanged — hyphen-id path resolves exactly as before; slash path is additive.

## Gap 2 — Quantize request `source_path` alias

### Problem

Hub sends `{"source_path": ..., "output_path": ..., "quant_bits": ...}`. `QuantizeRequest` (`convert_models.py:95`, `class QuantizeRequest(_ConvertBase): pass`) inherits `_ConvertBase` which requires `model: str` (line 36). `source_path` key rejected → 422.

### Fix — `model_validator` on `QuantizeRequest` only (surgical)

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

- Hub sends `source_path` → validator copies to `model` → `_ConvertBase` validation proceeds.
- Existing callers sending `model` → validator no-op. No regression.
- `output_path` constraint (`_validate_output_path` line 44, allows only `~/.fusion-mlx/models`, `~/.cache/huggingface`, CWD) untouched — Hub's `output_path` must fall under these. Per decision ("Hub works unchanged"), Hub writes to the models dir → in-bounds. If Hub's path is outside, that is a separate gap to raise, not a silent constraint relaxation (CLAUDE.md: fail visibly).

### Why validator not `Field(alias=...)`

pydantic v2 alias changes the serialization key + OpenAPI doc field. Validator keeps `model` as the canonical documented field, treats `source_path` as a tolerated alias — smaller change to existing API docs. Matches the decision phrasing ("validator that copies `source_path`→`model`").

### Why `QuantizeRequest` only, not shared `_ConvertBase`

`MergeAdapterRequest` (line 99) already has its own slash-safe body approach. Adding the alias to the base class would inherit it to all subclasses — harmless but unnecessary, widens blast radius. Keep the validator on `QuantizeRequest` subclass only = one class, one validator, base + merge-adapter untouched.

### Tests

- Unit: build `QuantizeRequest` with `source_path` key + valid `output_path` + `quant_bits` → assert `.model == source_path`, `.quant_bits == 4`.
- Unit: build with `model` key → assert `.model` preserved, `source_path` no-op (regression guard).
- Unit: `source_path` only but `output_path` outside allowed dirs → still raises validation error (constraint not bypassed).

### Blast radius

- `convert_models.py`: ~6 lines on `QuantizeRequest`. `_ConvertBase`, `MergeAdapterRequest` untouched.
- Tests: 3 unit tests in `tests/unit/test_convert_routes.py`.
- Existing clients: unchanged (`model` path intact).

## Gap 3a — Mount orphaned layered quantize router

### Root cause

`layered_quantize_routes.py` (223 lines) defines `router = APIRouter(prefix="/v1", tags=["quantize"])` (line 28) with routes `POST /quantize/layered` (184), `GET /quantize/layered/jobs/{job_id}` (203), `GET /quantize/layered/jobs` (215). Separate store `_layered_jobs` (line 30) + `_layered_executor` (line 35). Router never imported/included in `server.py` → all `/v1/quantize/layered/*` 404.

### Fix

Include the router in `server.py`, next to the `convert_router` include (`server.py:922` `app.include_router(convert_router)`). Add import + `app.include_router(layered_quantize_router)`. Job-store isolation maintained (two stores, two executors) — decision explicitly "keep two job stores". Hub's `/v1/quantize/layered/jobs/{job_id}` poll now works.

### Pre-requisite: route conflict check

Layered router paths: `prefix="/v1"` + `"/quantize/layered"` family. `convert_router` (prefix `/v1`, tag `convert`) has `/quantize`, `/quantize/jobs`, `/quantize/jobs/{job_id}` — distinct paths. No conflict. `LayeredQuantizeRequest` (line 54: `model`, `output_path`, `default_bits`, `layer_rules`, `quant_group_size`, `quant_mode`, `trust_remote_code`) is already consumed by its own handler — mounting only exposes already-written code.

### Tests

- Mount test: `GET /v1/quantize/layered/jobs` against live server → 200 (was 404). Route-existence test.
- Real-model: run a layered quantize job, poll to terminal → assert response.

### Blast radius

- `server.py`: +1 import, +1 `include_router`.
- Tests: new mount-existence test + layered-router terminal test.
- No prod logic change — only mounting already-written code.

## Gap 3b — `done` → `completed` terminal status

### Fix — both job stores

Two terminal-status writers:
- `convert_routes.py:98` `_run_job` → `_set(job, status="done", ...)`. Change to `status="completed"`.
- `layered_quantize_routes.py:177` `_run_layered_quantize` → `_set(job, status="done", ...)`. Change to `status="completed"`.

### Test assertion updates (blast radius from grep)

- `tests/unit/test_convert_routes.py` lines 44, 68, 119, 129: `assert status == "done"` → `== "completed"`.
- `test_layered_quantize_routes.py` (if present): grep for any `"done"` assertion and update. This grep is a task step, not an assumption — implementer runs it.

### Consumer safety

Status-string readers: `get_quantize_job` (`convert_routes.py:173`) and layered `get` (`:203`) return the full job dict including `status` — Hub reads `status == "done"`. Other internal readers: grep `status.*done` / `"done"` for non-test consumers. Prior grep found none beyond the 2 writers + 4 test assertions — blast radius confined to files being changed. Hub explicitly expects `completed` (decision basis), so this aligns the contract.

### Known limitation — in-memory job stores

Both stores are process-local dicts (`_jobs`, `_layered_jobs`), no persistence across restart. Hub polling a job across a restart → 404. Pre-existing (even the orphaned layered store is just a dict; convert store same). Out of scope for this fix (decision: "keep two job stores"). Documented here as a known limitation; raised as a separate follow-up issue if Hub depends on cross-restart polling. No silent persistence added.

### Tests

- `completed` test: run a quantize job, poll to terminal → assert `status == "completed"` (both routers).
- Regression: existing `test_convert_routes.py` 4 assertions updated to new string; suite must be no-redder.

### Blast radius

- `convert_routes.py`: 1 line (`"done"`→`"completed"`).
- `layered_quantize_routes.py`: 1 line (`"done"`→`"completed"`).
- Tests: 4 assertion updates + new terminal-status test + layered-router grep+update step.

## File map

| File | Change | Responsibility |
|---|---|---|
| `fusion_mlx/server.py` | Gap 1: 2 route signatures `{model_id:path}`; `resolve_model_id` slash→hyphen map. Gap 3a: import + `include_router(layered_quantize_router)`. | Public load/unload routes + resolver + router wiring. |
| `fusion_mlx/gui_compat/server.py` | Gap 1: 2 route signatures `{model_name:path}` (load `:470`, unload `:514`). | gui_compat load/unload route-match parity. |
| `fusion_mlx/api/convert_models.py` | Gap 2: `QuantizeRequest` `source_path` field + `model_validator`. | Quantize schema alias. |
| `fusion_mlx/api/convert_routes.py` | Gap 3b: `_run_job` `status="done"`→`"completed"`. | Plain quantize terminal status. |
| `fusion_mlx/api/layered_quantize_routes.py` | Gap 3b: `_run_layered_quantize` `status="done"`→`"completed"`. | Layered quantize terminal status (router already written, now mounted). |
| `tests/unit/test_convert_routes.py` | Gap 2: 3 alias unit tests. Gap 3b: 4 assertion updates (lines 44, 68, 119, 129). | Schema + terminal-status coverage. |
| `tests/unit/test_layered_quantize_routes.py` | Gap 3a: mount-existence test. Gap 3b: grep+update `"done"` assertions, terminal-status test. | Layered router coverage. |
| `tests/unit/test_server_load_route.py` (new) | Gap 1: route-compile unit test + `resolve_model_id` unit test (mock pool). Gap 1 real-model round-trip gated by `@pytest.mark.real_model`. | Load route-match + id-resolution. |

## Global constraints

From CLAUDE.md / CONTRIBUTING.md, binding every task:

- Lint: `ruff check fusion_mlx/ tests/` + `black --check fusion_mlx/ tests/` must pass. `gui_compat/` is NOT yet ruff/black compliant — changes there are co-located route-signature edits only, match surrounding style, do not reformat adjacent code.
- No docstrings in new code. Multiples of 4 indentation. Logging by default in new code.
- Real-model tests: load real MLX weights, server via `~/claude-home/fusion-mlx/start.sh start|stop`. Gate with `@pytest.mark.real_model` + inline `FUSION_MLX_REAL_MODEL_TESTS` guard (codebase convention, test_distributed_decode_step_e2e.py). NEVER `mx.clear_streams()` in tests (poisons default Stream handle — see kv-cache-0-root-cause). Model download via mirror `https://hf-mirror.com`.
- Public API boundary (#615): downstream imports via `fusion_mlx.public_api`, not internals. This work touches internal route handlers — no new public_api exports required (routes are HTTP, not Python import surface).
- Failing-test rule: if a test fails — even unrelated — locate and fix it. Suite must be no-redder than before.
- Commit/PR flow: branch `feat/646-hub-mlx-contract` off `main` (current default). `git commit -m "<type>(#646): <subject>"`. Remote is `origin` (`git@github.com:dahai80/fusion-mlx.git`) — NOT `fusion-mlx` (CLAUDE.md `push -u fusion-mlx` instruction is stale). GitHub ops in English. Upstream issue (#646) already filed → this work lands the code per the issue-first flow.
- Token budget: per-task cap, stop at cap (Rule 6).

## Out of scope

- **Job-store persistence across restart.** Both `_jobs` and `_layered_jobs` are in-memory dicts. Cross-restart poll 404 is pre-existing; persistence is a separate architectural concern, not part of this contract alignment.
- **`output_path` constraint relaxation.** `_validate_output_path` (allowed dirs only) stays. If Hub needs a path outside, raise a separate issue.
- **`MergeAdapterRequest` `source_path` alias.** Decision scoped the alias to `QuantizeRequest` only. Merge adapter already has its own slash-safe body pattern.
- **`done`→`completed` for non-quantize job kinds.** The `_jobs` store (`convert_routes.py`) holds both `kind="quantize"` and `kind="convert"` jobs; the `convert_routes.py:98` writer sets terminal status for both kinds in one line, so the change covers convert jobs too. No separate writer exists for convert. Blast radius confined to the 2 writer lines (one per store).
- **#630 multi-token decode release.** Pushed to origin/main @ 6a40c308 but not tagged/released. Separate task; not part of #646.
- **Slash→hyphen transform source pinning.** The transform producing registered hyphen ids lives in the HF-snapshot→dir download step (`replace("/", "-")` not found repo-wide via grep; repro empirically confirms hyphen ids). Implementer pins the exact source during implementation; the design's load-bearing claim (hyphen id works, slash id fails at route + pool) is empirically proven, not assumed.

## Open questions / follow-up issues

- **Cross-restart job poll persistence** — if Hub polls a quantize job across a fusion-mlx restart, it 404s. File a separate issue if Hub depends on this; document as known limitation in the PR body regardless.
- **Slash→hyphen transform source** — implementer must confirm the exact code site that transforms HF repo `/`→`-` at download, to document `resolve_model_id`'s new mapping against the real transform (not a guessed one). If the transform is in an external tool (HF snapshot script), the `resolve_model_id` map is the correct in-server normalization regardless.
