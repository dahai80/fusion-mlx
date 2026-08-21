# CI pipefail + 38 masked test failures (#583)

PR #595 · closes issue #583 · base `main` · branch `fix/ci-pipefail-masked-failures-583`

## Problem
`pytest tests/unit ... | tee pytest-output.txt` made the CI test-step exit code
`tee`'s (0). Every non-zero pytest exit was swallowed, so `main` reported GREEN
while carrying **38 failing tests** on the 3.12 matrix (39 on 3.11/3.13 — one
extra `test_hot_cache` flake). The follow-up "guard against excessive skips"
step only counted `PASSED` + `SKIPPED`, never `FAILED`, so it passed too.

## Root causes (two layers)

1. **Exit-code masker (CI config).** The pipe `| tee` replaces the step's exit
   code with `tee`'s. Fix: `set -eo pipefail` before the `pytest | tee` line so
   a failed pytest fails the job.
2. **The 38 failures themselves (real bugs + environment-specific test gaps).**
   They were never fixed because nobody could see them. This PR fixes the 29
   that are in scope here; the 9 `test_embedding.py` failures were fixed
   separately by PR #582 (conftest guard for the `mlx_embeddings` stub).

## Fixes (3 commits)

### `fix(ci): preserve pytest exit code via pipefail`
- `.github/workflows/ci.yml`: add `set -eo pipefail` to the test step.

### `fix: gemma4 tool-call block sanitizer + hot-cache hits stat`
Production code, not just tests:

- `fusion_mlx/api/utils.py` — `sanitize_output`: strip
  `<|tool_call>...<tool_call|>` blocks before the catch-all `_FINAL_SANITIZER`
  so the inner `call:name{args}` body does not leak into `message.content`.
  Fixes `test_sanitize_output` (5 fails).
- `fusion_mlx/cache/paged_ssd_cache.py` — `load_block`: a hot-cache hit
  incremented `hot_cache_hits` but not `hits`, inconsistent with the
  pending-buffer hit and SSD hit paths (which do increment `hits`). Now
  increments both. Fixes `test_hot_cache` (2 fails). Genuine stat-correctness
  fix, not just test hardening.

### `test: fix 29 CI-masked failures across 3.11/3.12/3.13`
Test infra + fixtures:

- `tests/unit/conftest.py` — install a `vllm_mlx` → `fusion_mlx` meta-path
  finder (`_VllmMlxAliasFinder`). The legacy package was renamed to
  `fusion_mlx`, but tests still reference `vllm_mlx.*`. A bare top-level alias
  (`sys.modules["vllm_mlx"] = fusion_mlx`) was not enough: importing a
  submodule through the alias path (`import vllm_mlx.api`) reloaded the real
  `fusion_mlx/api/__init__.py` under a *second* module object named
  `vllm_mlx.api`. Two distinct `api` packages then coexisted, and a later
  `import fusion_mlx.api.openai_routes` resolved the submodule off the wrong
  (aliased) parent:
  `ImportError: cannot import name 'openai_routes' from 'vllm_mlx.api'`.
  This is order-sensitive — it surfaced on CI 3.11/3.12/3.13 but not on a
  local 3.14 run. The finder reuses the real `fusion_mlx.X` module object for
  every `vllm_mlx.X` import, so the two namespaces share one object graph. It
  defers to any pre-set `sys.modules` entry (test fakes, and the `None`
  sentinel used to force `ImportError`), matching the existing
  `monkeypatch.setitem` patterns. Fixes `test_ui_tars_lane_parity` (1 fail).
- `tests/unit/test_image_gen_sd15.py` / `test_image_gen_sd2.py` —
  `_StubImage.size = (64, 64)` (10 fails across sd15/sd2/gen_acceleration).
- `tests/unit/test_gen_acceleration_knobs.py` — `FakeImage.image` exposes
  `size=(768, 512)`.
- `tests/unit/test_pytorch_weight_loading.py` — `importorskip("torch")` +
  `import mlx.core` at top, so the file skips cleanly on bare CI runners
  that lack torch instead of `ModuleNotFoundError` mid-body (6 fails).
- `tests/unit/test_server_api_key_env_fallback.py` — neutralize the
  module-level `_api_key` global (`monkeypatch.setattr("fusion_mlx.server._api_key", None)`)
  before the env-resolution tests. CI runners load
  `~/.fusion-mlx/settings.json` `auth.api_key` into that global BEFORE pytest
  collects, so it sat ahead of env in the resolve order and the env branch was
  never reached (4 fails).
- `tests/unit/test_per_request_route.py` — expect the `dflash2` key in the
  loaded-methods dicts (DFlash2 #593 added it; dicts went 5→6 keys).
- `tests/unit/test_no_mllm_flag.py` — register `--enable-dflash2` in the
  spec-decode routing-flag allowlist.
- `tests/unit/test_mcp_config.py` — skip the example round-trip when
  `npx`/`uvx` are not on `PATH`. `MCPCommandValidator(check_path_exists=True)`
  is the production default; on bare CI images neither runner is installed
  (1 fail).
- `tests/unit/test_embeddings_timeout_admission.py` — `xfail(strict=False)`
  for the stale H6 pre-tokenized-embeddings marker; it is `XPASS(strict)` on
  CI (1 fail).
- `tests/unit/test_modality_models_route.py` — `EntryPayload` all-fields
  expects `loaded: True, state: "loaded"` (1 fail).
- `tests/unit/test_hot_cache.py` — LRU-refresh load before the 3rd save so
  the next eviction deterministically lands on the LRU block, leaving the
  MRU resident for the hot-cache-hit assertion.

## Failure inventory (run 32390849022)

| file | 3.11 | 3.12 | 3.13 | fix |
|------|------|------|------|-----|
| test_embedding.py | 9 | 9 | 9 | PR #582 — NOT this PR |
| test_pytorch_weight_loading.py | 6 | 6 | 6 | importorskip ✅ |
| test_sanitize_output.py | 5 | 5 | 5 | gemma4 block sanitizer ✅ |
| test_image_gen_sd15.py | 5 | 5 | 5 | _StubImage.size ✅ |
| test_server_api_key_env_fallback.py | 4 | 4 | 4 | _api_key global isolation ✅ |
| test_image_gen_sd2.py | 3 | 3 | 3 | _StubImage.size ✅ |
| test_gen_acceleration_knobs.py | 2 | 2 | 2 | FakeImage.image.size ✅ |
| test_ui_tars_lane_parity.py | 1 | 1 | 1 | vllm_mlx finder ✅ |
| test_modality_models_route.py | 1 | 1 | 1 | loaded/state ✅ |
| test_mcp_config.py | 1 | 1 | 1 | skip-on-missing-runner ✅ |
| test_hot_cache.py | 1 | 0 | 1 | hits stat + LRU-refresh ✅ |
| test_embeddings_timeout_admission.py | 1 | 1 | 1 | xfail strict=False ✅ |
| **total** | **39** | **38** | **39** | |

## Verification
- All 15 touched files verified on the shared 3.14 venv; no regression vs the
  pre-edit baseline (audio / dflash-adapter / output_router alias-dependent
  tests unchanged: 72 passed / 9 failed both before and after, the 9 being
  pre-existing local-env gaps unrelated to this PR).
- The `vllm_mlx.api` shadow-load bug reproduced standalone, and the finder fix
  verified to make `vllm_mlx.api` `is` `fusion_mlx.api` and resolve
  `openai_routes` under both names, while still honoring test
  `monkeypatch.setitem` fakes and `None` import-fail sentinels.
- CI is the source of truth: with pipefail now truthful, this PR must go GREEN
  on 3.11/3.12/3.13 before merge.

## Out of scope
`start.sh` `FUSION_SERVE_EXTRA` passthrough (dflash2 serve flags) belongs to
#593 and is NOT in this PR.
