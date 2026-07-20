# #431 Step 2 — Per-request spec routing: router-driven dispatch

## Goal
Replace the **implicit priority chain** in `_try_spec_decode`
(`ngram → dflash → dspark → draft`, with mtp implicitly winning the
forward pass via `last_step_was_mtp` bail) with an **explicit per-request
router decision** via `select_active_method` (Slice 1, already committed).
The router decides ONCE per request (at first pure-decode step) which spec
method runs; the decision is fixed for the request's lifetime. mtp's
forward-pass stepping is brought under router control via a per-step
suppress hook so a non-mtp choice actually runs.

Plan-doc definition (L208-211): *"wire `select_active_method` (Slice 1)
into `_try_spec_decode` + mtp eligibility so the router, not implicit
mtp-priority, decides which method runs."* Step 1 guard (commit 69e00040)
is the prerequisite and is complete.

## Key facts established by exploration
- `Request` is a plain `@dataclass` (no `__slots__`); per-request state is
  flat fields (SpecPrefill convention). Add a flat field — no side-dict.
- `_try_spec_decode` (sched_step.py:479) is called ONLY from
  `_step_pure_decode` (sched_step.py:456), the single-running-request fast
  path. The full step path never calls it.
- No single registry of loaded spec methods. Assemble from scheduler attrs:
  - suffix: `self._ngram_spec_state` (ALWAYS loaded — engine_core.py:261
    creates `NGramSpecState()` unconditionally → router's NGRAM default is
    always available)
  - dflash: `self._dflash_runtime`
  - dspark: `self._dspark_runtime`
  - mtp: `getattr(self.model, "_fusion_mlx_mtp_decode_enabled", False)`
    (per-model load flag; `self.model` set at sched_init.py:86)
- mtp stepping is decided **inside `bg._next()`** by `_is_mtp_eligible`
  (no stride — eligible ⇒ steps every step). `_mtp_common_eligible`
  (batch_generator.py:301) is the lowest-level gate; it reads
  `gen_batch.model` (= `self.model`). Existing tests mock `_is_mtp_eligible`
  directly ⇒ a check added inside `_mtp_common_eligible` is invisible to them.
- `request.num_prompt_tokens` set at admission (sched_admission.py:61),
  available before decode.
- `select_active_method(prompt_token_count, loaded, *, has_mtp=False,
  recent_accept_rate=None, current_method=None)` returns a METHOD_* string
  or None. Fresh path: long-doc(≥4096)→DFLASH, has_mtp→MTP, else NGRAM.
- draft-model (`spec_decode_step`, via `_spec_decode_state.draft_model`) is
  NOT in the router's 4 methods. Out of Step 2 scope — keep as fallback.

## Design

### 1. Per-request decision cache — `fusion_mlx/request.py`
Add field to `Request` dataclass (alongside SpecPrefill fields):
```python
_active_spec_method: str | None = None
```
- `None` = not yet decided; `""` = decided "no method"; else METHOD_*.
- Auto-resets per new Request (lifecycle) — no manual reset.

### 2. Loaded-methods + decision helpers — `fusion_mlx/scheduler/sched_step.py`
Two scheduler mixin methods (module-level defs, near `_try_spec_decode`):
```python
def _loaded_spec_methods(self) -> dict[str, bool]:
    from ..speculative.per_request_route import loaded_methods
    return loaded_methods(
        suffix=self._ngram_spec_state is not None,
        dflash=self._dflash_runtime is not None,
        dspark=self._dspark_runtime is not None,
        mtp=bool(getattr(self.model, "_fusion_mlx_mtp_decode_enabled", False)),
    )

def _decide_spec_method(self, request) -> str:
    from ..speculative.per_request_route import select_active_method
    loaded = self._loaded_spec_methods()
    method = select_active_method(
        request.num_prompt_tokens, loaded, has_mtp=loaded["mtp"],
    )
    return method or ""
```
- `recent_accept_rate=None`, `current_method=None` (fresh per request;
  abandon/hysteresis deferred — decision is fixed per request lifetime).

### 3. Decision + mtp suppress BEFORE `bg._next()` — `_step_pure_decode`
Before the forward pass (sched_step.py:438), decide once + gate mtp:
```python
# Router-driven per-request spec decision (Step 2). Decided once at first
# pure-decode step; fixed for the request lifetime. mtp runs inside
# bg._next() on its own eligibility, so a non-mtp choice must suppress mtp
# this step or the forward pass steals the step and the choice is bailed.
request = next(iter(self.running.values()))  # fast path: len(running)==1
if request._active_spec_method is None:
    request._active_spec_method = self._decide_spec_method(request)
from ..speculative.auto_router import METHOD_MTP
suppress = request._active_spec_method not in (METHOD_MTP, "")
self.model._fusion_mlx_mtp_suppressed = suppress
try:
    with mx.stream(self._stream):
        _, responses = bg._next()
finally:
    self.model._fusion_mlx_mtp_suppressed = False
```
- Suppress when router chose a post-forward heuristic (ngram/dflash/dspark);
  allow mtp when MTP chosen or no method (let mtp own forward pass).

### 4. mtp suppress hook — `patches/mlx_lm_mtp/batch_generator.py:_mtp_common_eligible`
Add right after the `hasattr(gen_batch, "model")` check:
```python
if getattr(gen_batch.model, "_fusion_mlx_mtp_suppressed", False):
    return False
```
- Invisible to `test_mtp_suffix_coexistence.py` (mocks `_is_mtp_eligible`).

### 5. Router-driven dispatch — `_try_spec_decode`
Keep ALL existing bail guards (single running, vlm_mtp, `last_step_was_mtp`,
output_parser, pending_abort). After guards, replace the
`ngram→dflash→dspark→draft` chain with single-method dispatch:
```python
request = self.running.get(request_id)
method = request._active_spec_method or ""
if method == METHOD_NGRAM and self._ngram_spec_state is not None:
    result = ngram_spec_step(...);  if result: return result
elif method == METHOD_DFLASH and self._dflash_runtime is not None:
    result = dflash_spec_step(...);  if result: return result
elif method == METHOD_DSPARK and self._dspark_runtime is not None:
    result = dspark_spec_step(...);  if result: return result
# METHOD_MTP / "": no post-forward heuristic spec; fall through.
# Draft-model (NOT router-controlled): fallback when chosen heuristic
# produced no output this step. Preserves today's ngram-then-draft behavior.
if self._spec_decode_state is not None and self._spec_decode_state.draft_model is not None:
    return spec_decode_step(self, output, current_token, request_id)
return []
```
- NO fall-through among heuristic methods (router picked one → no double-spec).
- draft-model fallback preserved (runs only when heuristic returned [] —
  same as today, since `*_spec_step` returns [] when no spec fires).

## Files to touch
1. `fusion_mlx/request.py` — add `_active_spec_method` field.
2. `fusion_mlx/scheduler/sched_step.py` — `_step_pure_decode` (decide +
   suppress before `bg._next()`, finally-reset), `_try_spec_decode`
   (router-driven dispatch), add `_loaded_spec_methods` + `_decide_spec_method`.
3. `fusion_mlx/patches/mlx_lm_mtp/batch_generator.py` — `_mtp_common_eligible`
   suppress check.
4. `tests/unit/test_per_request_route.py` — extend: loaded-assembly +
   decision-cache unit tests.
5. `tests/unit/test_mtp_suffix_coexistence.py` — add: suppress flag respected
   by `_mtp_common_eligible` (model attr → ineligible).
6. NEW `tests/unit/test_spec_routing_dispatch.py` — `_try_spec_decode`
   dispatch integration (mock scheduler, assert ONLY chosen method runs;
   assert mtp suppress set before bg._next()).

## Behavior changes (must be covered by tests)
- mtp loaded + long-doc + dflash loaded → router picks DFLASH, suppresses
  mtp, DFLASH runs. **Today:** mtp wins forward pass, DFLASH bailed.
  Intended improvement (router decides).
- mtp loaded + short-doc → router picks MTP, mtp runs. **Today:** mtp runs.
  No change.
- ngram/dflash/dspark no longer fall through to each other. In practice
  low-impact: each `*_spec_step` returns [] when no spec fires, so the
  draft-model fallback (preserved) covers the "no heuristic fired" case.
- draft-model: unchanged (fallback after chosen heuristic returns []).

## Risks / checkpoints
- **suppress flag lifecycle**: `finally`-reset after `bg._next()` so full
  step path / vlm mtp / exceptions leave it False. Verify with a test that
  raises inside bg._next().
- **decision before bg._next()**: `num_prompt_tokens` is set at admission
  (before running) ✓. `next(iter(self.running.values()))` is the decoded
  request (fast path guarantees len==1) ✓.
- **cached decision across preemption**: based on prompt length (fixed) +
  loaded methods (fixed for scheduler lifetime) ⇒ safe to persist. Request
  objects aren't reused.
- **single-thread MLX executor**: no race on `self.model._fusion_mlx_mtp_suppressed`.

## Verification
- `rtk proxy python -m pytest tests/unit/test_per_request_route.py
  tests/unit/test_spec_auto_router.py tests/unit/test_mtp_suffix_coexistence.py
  tests/unit/test_spec_routing_dispatch.py -x`
- `rtk proxy python -m pytest tests/unit/ -k "spec or mtp or ngram or dflash
  or dspark or route" -x`
- Collection: `rtk proxy python -m pytest --co -q tests/unit/ | tail -3`
- Lint: `black` + `ruff check` on touched files.
- Import: `python -c "import fusion_mlx.scheduler.sched_step,
  fusion_mlx.request, fusion_mlx.patches.mlx_lm_mtp.batch_generator"`
- Commit on `feat/per-request-spec-routing` (HEAD 69e00040). Do NOT merge to
  main / repackage .dmg until #451 (commit 5fc1d015 on
  fix/mllm-reexport-deadcode-cleanup) merge is confirmed with user.

## Out of scope (deferred to Step 3+)
- `recent_accept_rate` cross-request feedback (router abandon/hysteresis).
- Within-request method switching (decision fixed per request).
- draft-model under router control.
- dflash/dspark into BatchedEngine (break early-fork).
- CLI `--spec-decode auto` per-request mode.
