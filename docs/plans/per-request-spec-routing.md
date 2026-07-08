# Per-Request Spec Routing — Engine Refactor Plan

## Status

- **Slice 1 DONE** (commit c93407a5, branch `feat/per-request-spec-routing`):
  `fusion_mlx/speculative/per_request_route.py` (`select_active_method` +
  `loaded_methods`) + 16 unit tests. Pure decision function, zero wiring,
  zero regression risk. Reusable foundation.
- **Step 1 (break the suffix/mtp mutex) DONE** (Changes 1-6, branch
  `feat/per-request-spec-routing`). The strictly-additive per-step guard
  is in place: mtp sets `_fusion_mlx_mtp_stepped` on the verify+accept
  path; `last_step_was_mtp` reads it; `_try_spec_decode` bails when mtp
  owned the step; the CLI `sys.exit` mutex is lifted to an info print.
  Verified by 12 new coexistence unit tests + 60 existing spec tests
  (84 hw/mlx-gated skips, 0 regressions); full unit suite collects
  cleanly with quarantine active. **E2E coherence (Change 6) is
  environment-gated**: no local MTP-head model loads with `mtp_enabled`
  today (Qwen3.5-9B-4bit has a `language_model.` weight-prefix mismatch
  that fails even without mtp; Qwen3.6-27B-mxfp8's mtp patch breaks
  strict load with 160 extra params; DeepSeek-V4-Flash's model type is
  absent from mlx_lm). All three are pre-existing load issues unrelated
  to this change. The E2E test (`tests/integration/test_mtp_suffix_
  coherence.py`) skips gracefully on load failure and will run once an
  mtp-loadable model is available; the guard's correctness rests on the
  unit tests + the strictly-additive safety argument (the guard only
  makes `_try_spec_decode` bail more, never less - the double-spec
  corruption path is precisely what it blocks).
- **Slice 2/3 REPLANNED** below. The original 3-slice plan assumed
  "multi-method resident per-request routing" was possible at boot. Code
  investigation proved that assumption WRONG. This document records the
  corrected architecture and the real first step.

## Architecture reality (corrected)

The original plan claimed suffix and mtp "both monkey-patch the
BatchGenerator step" and are therefore mutually exclusive. That is
INACCURATE. The actual mechanisms:

| Method | Integration | Patch site | When it runs |
|---|---|---|---|
| fusion throughput opt | monkey-patch `GenerationBatch._step` | `scheduler/monkeypatches.py` L343 | every decode step (inner forward+sampling) |
| **mtp** | monkey-patch `GenerationBatch.__init__/next/filter/extend` | `patches/mlx_lm_mtp/batch_generator.py` `apply()` | inside `bg._next()`, per-batch gated by `_is_mtp_eligible`/`_is_mtp_batch_eligible`; does its own 2-token verify + MTP-head forward, bypassing standard `_step` |
| **suffix (ngram)** | NONE on `GenerationBatch` — engine-agnostic drafter | driven by scheduler `_try_spec_decode` -> `ngram_spec_step` | AFTER the decode step, single-seq only |
| dflash / dspark | scheduler `_try_spec_decode` (dflash/dspark `*_spec_step`); runtime injected into `scheduler._dflash_runtime`/`_dspark_runtime` | AFTER the decode step | single-seq only |
| draft-model (SpecPrefill) | scheduler `_try_spec_decode` -> `spec_decode_step` | AFTER the decode step | single-seq only |

### The real mutex reason

`_try_spec_decode` (`scheduler/sched_step.py` L479) runs AFTER the regular
decode step (called from `_step_pure_decode` L456). It is a static
priority chain: ngram -> dflash -> dspark -> draft-model, single-seq only.

mtp runs INSIDE the decode step (`bg._next()` -> `GenerationBatch.next` ->
`patched_next` -> `_mtp_next`), advancing the cache and emitting
accepted/rejected tokens. When mtp handles a step, `_try_spec_decode` then
runs on top of the mtp-produced response and tries a SECOND independent
spec loop (suffix draft-verify) on the same step = **double-spec**. Two
independent cache-advancing speculative loops on one step is exactly the
corruption mode documented in `fusion-mlx-spec-corruption-fix.md`.

`_try_spec_decode` already guards VLM-MTP (`if self._vlm_mtp_active:
return []`, L508) but has NO guard for standard mtp. The CLI mutex at
`cli_serve.py` L1859 (`--suffix-decoding` XOR `--enable-mtp`) prevents
the double-spec by forbidding coexistence — but its stated reason
("both monkey-patch the BatchGenerator step") is wrong, and forbidding
coexistence blocks the most valuable per-request routing combination
(mtp for MTP-eligible models, suffix fallback for the rest).

### dflash / dspark stay excluded from Step 1

dflash and dspark **early-fork** to separate servers
(`cli_serve.py` L1090-1105 `run_dspark_server` + `return`, bypassing
BatchedEngine entirely). Integrating them into BatchedEngine is a
separate, larger refactor explicitly deferred to 0.10 (L1555 comment).
Step 1 does NOT touch them. The dflash-vs-{suffix,mtp} mutex at L1563
stays.

## Step 1 — Break the suffix/mtp mutex safely (THIS task)

**Goal**: allow `--suffix-decoding` and `--enable-mtp` to coexist, with
mtp taking priority for MTP-eligible steps and suffix running only when
mtp did NOT handle the step. This unlocks mtp<->suffix per-request
routing (the highest-value combination) with a strictly-additive guard.

**Behavior change is one-directional and safe**: the new guard only
makes `_try_spec_decode` bail MORE often (when mtp ran). It never causes
suffix to run when it should not. Status quo (mutex forbids coexistence)
is strictly more restrictive than the post-change behavior.

### Change 1 — mtp per-step "handled" signal

File: `fusion_mlx/patches/mlx_lm_mtp/batch_generator.py`, in `patched_next`
(L120-149).

- Reset `self._fusion_mlx_mtp_stepped = False` at the top of `patched_next`.
- Set `self._fusion_mlx_mtp_stepped = True` ONLY after `_mtp_batch_next`
  (L129) or `_mtp_next` (L142) returns successfully — i.e. capture the
  result, set the flag, then return. Do NOT set it before the call (a
  `_MtpStepFallback` exception must leave the flag False so suffix may
  run on the fallback standard step).
- Restructure the two `try`/`except _MtpStepFallback` blocks so the flag
  is set on the success path only:
  ```python
  result = _mtp_batch_next(self, batch_state)
  self._fusion_mlx_mtp_stepped = True
  return result
  ```
  inside the `try`, leaving the `except` (fallback) path with the flag
  still False.

### Change 2 — helper to read the signal

File: `fusion_mlx/patches/mlx_lm_mtp/__init__.py` (or `batch_generator.py`
re-exported).

Add:
```python
def last_step_was_mtp(batch_generator) -> bool:
    """True iff the most recent ``GenerationBatch.next`` ran the mtp path."""
    try:
        gb = getattr(batch_generator, "_generation_batch", None)
        return bool(getattr(gb, "_fusion_mlx_mtp_stepped", False))
    except Exception:
        return False
```
The scheduler reaches the gen batch via `self.batch_generator._generation_batch`
(this attribute is already used by the mtp patch at L193).

### Change 3 — guard in `_try_spec_decode`

File: `fusion_mlx/scheduler/sched_step.py`, `_try_spec_decode` (L479),
right after the existing `_vlm_mtp_active` guard (L508):
```python
if self._vlm_mtp_active:
    return []
# Standard mtp owns this decode step (verify+accept inside bg._next());
# running suffix/dflash/dspark/draft on top would double-spec.
if _last_step_was_mtp(self.batch_generator):
    return []
```
with a lazy import of the helper to keep the scheduler import-light.

### Change 4 — lift the CLI mutex

File: `fusion_mlx/cli_serve.py` L1857-1865. Replace the hard
`sys.exit(1)` with a warning + allow coexistence:
```python
if args.suffix_decoding and args.enable_mtp:
    logger.info(
        "--suffix-decoding + --enable-mtp: mtp takes priority for "
        "MTP-eligible steps; suffix runs only when mtp did not handle "
        "the step (per-request routing)."
    )
```
Also check `speculative/auto_resolve.py` boot-time resolution: ensure
`auto` no longer forces suffix XOR mtp when both are eligible — both
should be loaded and the per-step guard decides dispatch. (Verify the
exact gate in auto_resolve; adjust to allow both loaded.)

### Change 5 — tests

File: `tests/unit/test_mtp_suffix_coexistence.py` (new) + extend
existing mtp/suffix tests.

1. **mtp eligible + suffix loaded**: mtp `patched_next` sets
   `_fusion_mlx_mtp_stepped=True`; `_try_spec_decode` returns `[]`
   (suffix not double-run). Assert via a fake gen_batch with the flag.
2. **non-mtp model + suffix loaded**: flag stays False; suffix
   `ngram_spec_step` runs normally. (Existing suffix tests cover this;
   add an assertion that the guard does not fire.)
3. **mtp fallback** (`_MtpStepFallback`): flag stays False; suffix may
   run. Simulate by making `_mtp_next` raise `_MtpStepFallback`.
4. **flag reset each step**: two consecutive `patched_next` calls, first
   mtp-eligible then not — flag is True then False.
5. **existing mtp identity tests** (`tests/test_mlx_lm_mtp_patch.py`)
   still pass — the flag is additive, does not alter mtp token output.
6. **existing suffix tests** still pass.

### Change 6 — E2E coherence + CI

- Run the spec-corruption coherence smoke test (the historical pain
  point: greedy identity across mtp+suffix coexistence) on an
  MTP-eligible model (e.g. Qwen3.5) with both `--suffix-decoding` and
  `--enable-mtp`.
- Assert output is byte-identical to mtp-only and suffix-only baselines
  for a fixed prompt (greedy, temp 0).
- CI: black + ruff + the new + existing unit tests.

## Risk analysis

- **Spec corruption (historical pain point)**: mitigated by the guard
  being strictly additive — it only suppresses suffix when mtp ran, never
  the reverse. The double-spec path that would cause corruption is
  precisely the path now blocked. Coherence E2E (Change 6) is the
  verification gate.
- **mtp fallback correctness**: the flag must NOT be set on
  `_MtpStepFallback`. Test 3 covers this. If missed, suffix would be
  suppressed on a step where mtp fell back to standard decode — a
  missed-speedup bug, not a corruption bug (suffix simply doesn't run).
- **flag staleness across steps**: reset at top of `patched_next` (Change
  1). Test 4 covers this.
- **`_generation_batch` attribute access**: already used by the mtp patch
  itself (L193), so the path is known-good. Helper swallows exceptions
  (Change 2) so a missing attr degrades to "suffix may run" (safe).

## Explicitly deferred (later steps / sessions)

- **Explicit router-driven dispatch**: wire `select_active_method`
  (Slice 1) into `_try_spec_decode` + mtp eligibility so the router, not
  implicit mtp-priority, decides which method runs. Larger; needs the
  guard from Step 1 as a prerequisite.
- **dflash / dspark into BatchedEngine**: break the early-fork
  (`run_dspark_server` + `return`). Explicitly deferred to 0.10 per the
  L1555 comment. Multi-session.
- **CLI `--spec-decode auto` per-request mode** (original Slice 3):
  depends on both deferrals above.

## Success criteria for Step 1

1. DONE - `--suffix-decoding --enable-mtp` boots without `sys.exit`
   (CLI mutex lifted to an info print; `ModelSettings` has no
   mtp+ngram_spec validation, so per-load settings coexist too).
2. DONE (unit) - On an MTP-eligible step, mtp sets
   `_fusion_mlx_mtp_stepped`; `last_step_was_mtp` returns True;
   `_try_spec_decode` returns `[]` (suffix not double-run). Covered by
   `tests/unit/test_mtp_suffix_coexistence.py`.
3. DONE (unit) - On a non-mtp step the flag is False (reset each call,
   left False on `_MtpStepFallback`), so suffix may run as before.
4. ENVIRONMENT-GATED - Greedy byte-identity across mtp-only / suffix-only
   / both. No local MTP-head model loads with `mtp_enabled` today
   (pre-existing checkpoint/patch issues, see Status). The E2E test
   skips on load failure; correctness rests on (2)+(3) + the
   strictly-additive safety argument.
5. DONE - 12 new coexistence tests pass; 60 existing spec tests pass
   (84 skips, 0 regressions); black + ruff clean on all touched files;
   full unit suite collects cleanly with quarantine active.
