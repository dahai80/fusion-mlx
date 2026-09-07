# DEBT_FATES.md — Rapid-MLX migration test-debt triage

This file is the per-file fate registry referenced by `debt_modules.txt`.
Every module excluded from CI collection has a deliberate fate here:
KEEP_QUARANTINED is **documented** (pollution / deep drift / ambiguous prod
contract), not silent. The quarantine shrinks over time as prod APIs stabilize.

## Fate categories

| Fate | Meaning | Action |
|------|---------|--------|
| **RESCUED** | Re-verified green against current prod; real coverage. | Comment out line in `debt_modules.txt` → collected in CI. |
| **KEEP_QUARANTINED** | Collects but fails on stale assertions / removed attrs / deep drift. Migration debt, not prod defect. | Stay listed; prod NOT modified to satisfy. |
| **EMPTY** | Module has no collected tests (helpers / param-gated / all-skipped). | Stay listed; harmless. |
| **TIMEOUT** | Hangs or exceeds the unit-gate budget (integration-style). | Stay listed. |

## Rule: a passing test is not always rescuable

A quarantined module that "passes" is only un-quarantined when the passing
tests exercise **real prod code**. A test that passes against a stub/shim
installer or a degenerate no-op is **false coverage** (Rule 9: a test that
passes for the wrong reason is worse than no test). Such modules stay
KEEP_QUARANTINED even with a green run.

## Current state (audit #0907, 2026-09-07)

Total excluded modules: **116** (after 2 RESCUED this audit).

Fresh strict re-classification of all 118 previously-quarantined modules
(run each in isolation, strict `N failed` regex so `xfailed` is not mistaken
for failure):

- **RESCUED (un-quarantined this audit): 2**
- **KEEP_QUARANTINED (real fail): 106**
- **EMPTY (no tests collected): 6**
- **TIMEOUT: 2**
- **stub/false-coverage pass (stay quarantined): 2**

<!-- RESCUED sections are filled per-file. KEEP_QUARANTINED/EMPTY/TIMEOUT
     are summarized by count + representative failure modes; a full per-file
     breakdown lives in git history (commit that introduced this file). -->

## RESCUED — un-quarantined this audit (#0907)

### test_suffix_decoding.py — RESCUED (partial: 14 pass / 6 xfail)
- 14 drafter unit tests (`TestDrafterBasics` / `HistoryTrimming` / `Stats` /
  `Validation` / `RealisticAgentWorkload`) import from
  `fusion_mlx.speculative.suffix_decoding` (correct prod path) — all pass.
- 6 `TestInstallSuffixDecoding` tests xfailed: import
  `_install_suffix_decoding` from `fusion_mlx.scheduler` (REMOVED —
  ImportError). Suffix-decoding moved to `speculative/suffix_decoding.py` with
  a different architecture (no `GenerationBatch` monkey-patch installer, no
  profile allowlist gate, no `_step`/`.next` replacement, no `_suffix_stats`
  telemetry). Class-level `xfail` documents the removed install-hook contract.
- Un-quarantined: the 14 real-coverage tests belong in the active suite; the
  6 xfails are legit removed-feature markers, not false fails.

### test_responses_sse_event_order.py — RESCUED (3 pass)
- `TestResponsesStreamEventOrder` pins `response.created` →
  `response.in_progress` → `output_item.added` SSE event order + payload
  shape against the real `routes_internal/responses` router (lightweight
  engine modules installed via `monkeypatch`).
- Re-verified green: 3 passed. Original round13 quarantine reason
  (`output_text.done` event + `object` key drift) no longer reproduces
  against current prod.

## Stub / false-coverage — stay quarantined despite green run

### test_dense_sampler_fastpath.py — KEEP_QUARANTINED (false coverage)
- 7 tests pass, but they exercise a **no-op shim** installer
  (`_install_dense_sampler_fastpath` conftest stub), not prod. The real
  dense-sampler fast-path was NOT ported (replaced by a simpler
  module-level `get_or_create_fused_sampler` singleton, deliberately no
  bounded-LRU). Un-quarantining = false coverage. Perf feature gap
  (homogeneous-batch fast path) remains a future feature decision. Issue #674.

### test_request_time_alias_resolution.py — KEEP_QUARANTINED (deep drift)
- 5 real STT-resolver tests pass, but 15 tests xfail (embeddings/chat route
  alias resolution: `_resolve_request_alias_or_default` /
  `_aliases_match` signature changed to 1-arg degenerate stubs with zero
  production callers — #598/#515 deep drift). File uses a heavy stub-server
  harness. Audio-only un-quarantine would need test-file refactor (wide touch).
  Stay quarantined until the alias-resolution route contract is redesigned.

## KEEP_QUARANTINED — 106 modules (real fail, migration debt)

Representative failure modes across the 106 (not every file enumerated; full
list = the still-uncommented lines in `debt_modules.txt`):

- **fixture/registry-shape drift** (most common): `is_hybrid` / `count` /
  `context_window` registry KeyError; `_resolve_request_alias_or_default`
  signature drift; vllm_mlx.routes fixture cluster pins removed attrs.
  Files: `test_routes.py`, `test_responses_route.py`, `test_responses_bundle.py`,
  `test_responses_input_default_type.py`, `test_engine_preflight.py`.
- **removed prod symbol import** (ImportError at runtime): `DEFAULT_BURST_DECODE_MODE`
  removed from `fusion_mlx.settings` → `test_settings.py` errors.
- **scheduler/sampler contract drift**: `test_batching.py`, `test_seed_reproducibility.py`,
  `test_scheduler_chunked_prefill.py`, `test_thinking_budget.py`.
- **tool-calling parser dispatch drift** (57 fail / 173 pass): `test_tool_calling.py`
  — large file, partial real coverage but enough stale assertions to stay quarantined.
- **speculative-decode engine contract**: `test_dflash_*.py` (4 files),
  `test_mtp_*.py` (3 files), `test_pflash_*.py` (3 files) — engine/scheduler
  internal API changed since tests written.
- **VLM engine shape drift**: `test_vlm_engine.py` (58 fail / 30 pass).
- **release/CI-meta tests** (error at collection): `test_check_gha_pinning.py`,
  `test_check_mlx_upstream_calls.py`, `test_validate_release_subject.py`,
  `test_release_check_random.py`, `test_ready_banner_timing.py` — assert repo
  invariants from an older contract.

## EMPTY — 6 modules (no tests collected; harmless, stay listed)

`test_langchain.py`, `test_librechat_docker.py`, `test_openwebui.py`,
`test_paged_cache_real_inference.py`, `test_paged_cache_real_model.py`,
`test_request_cancellation.py` — contain only helpers / skipped /
param-gated cases that collect zero. Listed so the quarantine roster stays
complete; no CI noise.

## TIMEOUT — 2 modules (integration-style, exceed unit-gate budget)

`test_cli.py`, `test_hf_downloader.py` — hang or exceed the unit-gate
timeout. Need real subprocess / network. Belong in an integration suite, not
the unit gate. Stay quarantined.

## Re-activation process

1. Run the candidate in isolation: `pytest tests/unit/test_X.py -q --tb=short`.
2. Confirm the **passing** tests exercise real prod (not a stub/shim installer).
   If green is shim-driven → KEEP_QUARANTINED (false coverage).
3. Confirm `xfailed` markers are legit removed-feature documentation, not
   stale `xfail` hiding a now-passing test (a passing `xfail` = `XPASS` =
   pytest exit nonzero unless `strict=False`; check intent).
4. Comment out the line in `debt_modules.txt` (commented = collected = active).
5. Update this file: move the module from KEEP_QUARANTINED to RESCUED with
   the date + pass/xfail counts + why it's real coverage.
6. Verify `pytest tests/unit --collect-only -q | tail -1` count increases by
   the module's test count and the full active gate stays no-redder.
