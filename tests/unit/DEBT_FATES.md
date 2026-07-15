# Debt Modules - Fate Manifest (#80)

Purpose: record a **deliberate fate** for every file in
`tests/unit/debt_modules.txt`. Each file gets exactly one fate:

| Fate | Meaning |
|------|---------|
| **FIX** | Un-quarantine after repairing stale test-only assertions. |
| **DELETE** | Dead test (contract removed); remove the test method. |
| **GUARD** | Optional-dep file; guard via `importorskip` / `_OPT_DEP_SUITES` (NOT debt). |
| **KEEP_QUARANTINED** | Documented permanent/temporary block (pollution, deep drift, ambiguous prod contract). |

This manifest is the companion to `debt_modules.txt`. The list says *what* is
excluded; this file says *why* and *what would un-block it*.

---

## Categorization (isolated per-file runs, order-independent parser)

Run with `debt_modules.txt` emptied so quarantined files actually collect.
**Isolated per-file runs are the only trustworthy signal** - running quarantined
files together produces massive false failures from monkeypatch leakage.

| Category | Count | Signal | Fate |
|----------|-------|--------|------|
| PARTIAL | 161 | some pass + some fail | KEEP_QUARANTINED until per-file verified (see §Rescue Process) |
| BROKEN | 78 | 0 pass, all fail | KEEP_QUARANTINED (deep runtime-drift / removed API) |
| TIMEOUT/ZERO | 24 | rc=1, unparseable or integration-timeout | KEEP_QUARANTINED (over unit-gate budget) |
| DEEP | 12 | collection error (module-level import break) | KEEP_QUARANTINED (migration breakage) |
| GUARD (opt-dep) | 3 | all-skipped; guarded by `_OPT_DEP_SUITES` | debt entry is REDUNDANT - safe to remove (done for 3+1) |

Total: 278 categorized. 4 rescued this session (3 GUARD + 1 opt-dep double-list)
-> 274 remain quarantined.

---

## Rescued this session (commit 25e57a3)

Four files were **double-listed**: present in both `debt_modules.txt` AND
`_OPT_DEP_SUITES` in `conftest.py`. The opt-dep guard is the legitimate
quarantine mechanism (dflash / torch / mlx_audio absent); the `debt_modules.txt`
entry was redundant noise. Removing the redundant entry does NOT un-collect them
(opt-dep guard still applies) but cleans the debt list.

- `test_dflash_integration.py` - guarded by dflash suite (dflash absent locally).
- `test_batching_deterministic.py` - guarded by torch suite (torch absent).
- `test_prompt_lookup_bench.py` - guarded by torch suite (torch absent).
- `test_audio_r11_b_pure.py` - guarded by mlx_audio suite (mlx_audio absent).

Full unit suite green: 6695 passed, 0 failed.

---

## Case studies (investigated this session, NOT rescued - blockers documented)

### test_oq.py (254p/8f) - BLOCKED by cross-test pollution

All 8 failures are **clear test-only stale fixes** (no prod change needed):
- 5x `get_system_memory` patch path: tests patch `fusion_mlx.settings` but the
  symbol moved to `fusion_mlx.pool.settings` (prod `oq.py:3242` imports it there).
- 2x `omlx_oq_proxy_` prefix: tests assert old prefix; prod `oq.py:4407` uses
  `fmlx_oq_proxy_` (omlx->fmlx migration rename).
- 1x `for k in f:` iteration: safetensors `safe_open` needs `f.keys()`.

**Fixes applied + verified in isolation (262p/0f).** But un-quarantining test_oq
**regresses `test_vlm_sanitize_patch.py`** (2 failures) via cross-test pollution:
running test_oq before `test_vlm_sanitize_patch` inverts the
`should_shift_norm_weights` decision (sanitize shifts when it shouldn't and
vice-versa). Importing `fusion_mlx.oq` alone does NOT trigger the vlm patch
(verified: `_VLM_SANITIZE_PATCHED=False`), so the leak is a runtime side-effect
of some test_oq test that bisection has not yet isolated.

**Fate: KEEP_QUARANTINED.** The 8 stale fixes are correct and ready (re-apply
when the polluter is found). Blocker = isolate the test_oq test that mutates
global vlm sanitize state, then add teardown. Reverted to pristine in this
session to keep the suite green.

### test_eval.py (66p/1f) - unmigrated VALID_BENCHMARKS

`test_parity` asserts `set(BENCHMARKS.keys()) == set(VALID_BENCHMARKS)`.
- `BENCHMARKS` is real, in `fusion_mlx/eval/__init__.py:27` (16 keys: arc_challenge, bbq, ...).
- `VALID_BENCHMARKS` is imported from `fusion_mlx.admin.accuracy_benchmark` -
  but that module is a literal migration stub
  (`"""Accuracy benchmark admin module (stub for test migration)."""`),
  committed in `31ce9b6` (the omlx->fusion-mlx migration). `VALID_BENCHMARKS`
  was **never migrated** to fusion-mlx prod (git log -S finds no occurrence).

**Fate: KEEP_QUARANTINED** (migration debt, not a regression). To rescue: either
DELETE `test_parity` (unmigrated contract) or GUARD it (skip when
`VALID_BENCHMARKS` absent). Per migration rule, do NOT add `VALID_BENCHMARKS` to
prod for a debt test. If the parity contract is wanted, file an issue first.

### test_share_cli.py (65p/1f) - branding ambiguity

`test_banner_has_cheetah_brand_line` asserts `🐆` in the share command output.
- The share banner (`fusion_mlx/share/warning.py:43`) uses `🔥 Fusion-MLX share`,
  NOT `🐆`.
- `🐆` appears only in the **serve** banner (`cli_commands.py:1489`, `cli_serve.py:1635`).

**Fate: KEEP_QUARANTINED** (needs prod decision). Is the missing `🐆` in the
share banner a branding-consistency gap (prod should add it) or an over-specified
test (should expect `🔥`)? Either way it is a prod-contract question, not a
test-only fix. File an issue; do not modify prod for a debt test.

### test_tool_parsers.py (153p/3f) - bracket-format contract change

3 tests in `TestGenericToolCallParsing` use bare `Calling tool: write(...)` (no
brackets). Prod `tool_calling.py:681` handles only the **bracketed** form
(`[Calling tool: name(args)]`; see also `_bracket_prefixes` at line 875). The
bare form is not handled by the current fallback -> `tool_calls=None` -> tests
fail. The tests also pass `parse_tool_calls(text)` missing the now-required
`tokenizer` arg (defaults safely to None via `getattr`).

**Fate: KEEP_QUARANTINED** (contract change, not pure staleness). Whether bare
`Calling tool:` should still be supported is a prod-contract decision. If yes,
file an issue + add a bare-format handler; if no, update the tests to the
bracketed form. Not a safe mechanical fix.

---

## Rescue Process (how to safely rescue a PARTIAL file)

The test_oq case proved: **a file that passes in isolation can still regress the
full suite via cross-test pollution.** Mass un-quarantine is therefore unsafe.
Per-file rescue requires:

1. **Empty `debt_modules.txt`** temporarily (the `conftest` `collect_ignore`
   suppresses quarantined files even by explicit path). Back up + restore.
2. **Run the file isolated** to get exact failures:
   `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m pytest tests/unit/<f>.py --timeout=25 --timeout-method=signal --override-ini=addopts= --tb=short`
3. **Classify each failure**: clear-stale (mechanical) / ambiguous / pollution /
   prod-bug. Use `git log -S <symbol>` to distinguish stale-test from regression.
4. **Fix ONLY clear-stale, test-only** failures. File an issue for prod-bugs
   (migration rule: prod NOT modified for debt tests).
5. **Full-suite verify** (mandatory - pollution check):
   `python3 -m pytest tests/unit/ --timeout=30 --timeout-method=signal --override-ini=addopts=`
6. **If green**: remove the file line from `debt_modules.txt`, update the header
   count, commit. **If pollution appears**: revert, KEEP_QUARANTINED, document
   the blocker here.
7. Use the **order-independent parser** for per-file results: separate regexes
   `(\d+) passed`, `(\d+) failed`, `(\d+) skipped`, `(\d+) errors`. The naive
   `(\d+) passed(?:, (\d+) failed)?` misreports files where failures >= passes
   (pytest prints "failed" before "passed").

---

## DEEP (12) - collection errors (module-level import/migration break)

These fail at collection (module-level import error or conftest stub mismatch).
Fate: **KEEP_QUARANTINED** - need module-level migration repair first.

```
test_anthropic_streaming_reasoning.py
test_audio_api.py
test_body_receive_timeout.py
test_check_gha_pinning.py
test_check_mlx_upstream_calls.py
test_internal_route_auth.py
test_microbench_parsers.py
test_mtp_spec_decode.py
test_ready_banner_timing.py
test_release_check_random.py
test_request_body_size_limit.py
test_validate_release_subject.py
```

---

## TIMEOUT/ZERO (24) - rc=1, unparseable or integration-style over budget

These exit rc=1 with 0 pass/0 fail/0 skip (collection blocked by opt-dep guard,
or genuine integration timeout). Several are opt-dep double-listed (see GUARD).
Fate: **KEEP_QUARANTINED** - integration-style, exceed the unit-gate budget.

```
test_config.py
test_integrations.py
test_markitdown_integration.py
test_per_request_thinking_budget.py
test_server.py
test_settings.py
test_audio_extras_lockin.py
test_cli.py
test_dflash_integration.py        # opt-dep guarded (dflash) - debt entry removed
test_hermes.py
test_hf_downloader.py
test_langchain.py
test_librechat_docker.py
test_openwebui.py
test_paged_cache_real_inference.py
test_paged_cache_real_model.py
test_platform.py
test_prefill_oom_graceful.py
test_request_cancellation.py
test_routes_models_effective_parsers.py
test_schema_v2_accepts_v1.py
test_server_api_key_env_fallback.py
test_vlm_engine.py
test_boundary_snapshot_store.py
```

---

## GUARD (opt-dep, 3) - redundant debt entries removed this session

These are guarded by `_OPT_DEP_SUITES` in `conftest.py` (the legitimate
mechanism for optional deps). They were double-listed in `debt_modules.txt`;
the redundant debt entries were removed (commit 25e57a3). The opt-dep guard
still applies, so they remain correctly excluded from CI.

```
test_audio_r11_b_pure.py          # mlx_audio suite
test_batching_deterministic.py    # torch suite
test_prompt_lookup_bench.py       # torch suite
```

---

## BROKEN (78) - 0 pass, all fail (deep runtime-drift / removed API)

These collect but every test fails on removed/renamed prod symbols or changed
behavior. Fate: **KEEP_QUARANTINED** - bulk runtime-drift, needs per-file
migration to current API. Not safe to rescue without per-file investigation.

```
test_active_models_visibility.py
test_admin_dashboard_draft_filters.py
test_admin_update_check.py
test_alias_recommended_sampling.py
test_app_bundle_cli_wrapper.py
test_audio_memory.py
test_audio_path_shaped_model.py
test_audio_probe_consistency.py
test_audio_r11_b_bundle.py
test_audio_r7_c_bundle.py
test_audio_stt.py
test_audio_tts.py
test_audio_upload_size_limit.py
test_audio_utils.py
test_batched_engine_chat_template.py
test_bench_vs_ollama.py
test_cancelled_requests_metric.py
test_capabilities_field.py
test_cli_config_fidelity.py
test_cli_embeddings_extra.py
test_codex_profile.py
test_context_window.py
test_cors_env_configurable.py
test_cors_lockdown_response_shape.py
test_deepseek_v4_vendored.py
test_disconnect_counter_prod_shape.py
test_disconnect_guard.py
test_download_gate.py
test_embeddings_route.py
test_engine_keepalive.py
... (48 more - see tests/unit/debt_modules.txt for the full list)
```

---

## PARTIAL (161) - rescue candidates (pass+fail mix)

Top rescue candidates by passing-test yield (highest value to rescue). Each
needs the §Rescue Process: investigate failures, fix clear-stale only, **full-
suite pollution verify** (test_oq proved this mandatory).

| File | pass/fail | Notes |
|------|-----------|-------|
| test_oq.py | 254/8 | BLOCKED - pollution into test_vlm_sanitize_patch (see case study) |
| test_model_auto_config.py | 196/10 | uninvestigated |
| test_tool_calling.py | 173/57 | high fail count - likely deep drift |
| test_tool_parsers.py | 153/3 | bracket-contract change (see case study) |
| test_reasoning_content_null_rescue.py | 100/4 | tests sanitizer strip `</tool_call>` - possible prod bug |
| test_ui_tars_parser.py | 90/10 | uninvestigated |
| test_api_models.py | 89/19 | uninvestigated |
| test_audio_alias_registry.py | 81/45 | high fail count |
| test_diffusion_engine.py | 74/9 | uninvestigated |
| test_tool_choice_enforcement.py | 73/7 | uninvestigated |
| test_ui_tars_fixes.py | 68/3 | AST inspection of routes/responses.py - possible prod regression |
| test_eval.py | 66/1 | unmigrated VALID_BENCHMARKS (see case study) |
| test_share_cli.py | 65/1 | 🐆 branding ambiguity (see case study) |
| test_ui_tars_lane_parity.py | 64/6 | uninvestigated |
| test_embedding.py | 59/29 | high fail count |
| test_sampling_param_finite_range.py | 59/81 | fail > pass - likely deep drift |
| test_engine_pool.py | 57/36 | uninvestigated |
| test_mirror_pull.py | 56/7 | uninvestigated |
| test_memory_cache.py | 47/2 | uninvestigated |
| test_sampling_validation.py | 46/63 | fail > pass - likely deep drift |
| test_generation_config_loader.py | 44/3 | uninvestigated |
| test_r10_scrub_validation_bundle.py | 44/13 | uninvestigated |
| test_finalize_harmony_raw_text.py | 43/3 | uninvestigated |
| test_benchmark.py | 41/20 | uninvestigated |
| test_gemma4_messages.py | 40/7 | uninvestigated |
| ... (135 more PARTIAL files) | | see debt_modules.txt |

**Fate: KEEP_QUARANTINED** until each is run through the §Rescue Process. The
1-failure files (test_eval, test_share_cli) are highest-confidence targets but
both turned out to be contract/branding questions, not mechanical fixes - a
reminder that low failure count does not guarantee easy rescue.

---

## Open issues to file (per "遇到上游问题，先提issue")

- **test_oq pollution**: find the test_oq test that mutates global vlm sanitize
  state and leaks into `test_vlm_sanitize_patch`. Once isolated, test_oq's 8
  stale fixes can be re-applied and the file rescued (+262 tests).
- **VALID_BENCHMARKS parity contract** (test_eval): decide whether the
  BENCHMARKS/VALID_BENCHMARKS parity check should be restored (migrate
  VALID_BENCHMARKS) or dropped (delete test_parity).
- **Share banner `🐆` branding** (test_share_cli): decide whether the share
  banner should match the serve banner's `🐆` branding or keep `🔥`.
- **Bare `Calling tool:` format** (test_tool_parsers): decide whether bare
  (un-bracketed) tool-call format should be re-supported.

---

## Maintenance

- When you rescue a file: remove its line from `debt_modules.txt`, decrement the
  header `Total: N` count, add a one-line entry to §Rescued above, commit.
- When you investigate a file: add/update its case study here so the next person
  doesn't re-derive it.
- The categorization above is a snapshot from isolated per-file runs
  (`/tmp/perf2.json` in the session that produced this). Re-run to refresh.
