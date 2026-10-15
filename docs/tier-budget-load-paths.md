# Tier Budget Enforcement on Model-Load Paths (#1194)

## Single choke point

Every pool model load funnels through `EnginePool._load_engine`
(`fusion_mlx/pool/engine_pool.py` ~line 2873; the budget check sits at
~2895-2913). It reads the entry's `priority_tier`
(`P0_NEVER_EVICT` / `P0_DYNAMIC` / `P1_TDM`), computes
`incoming = actual_size or estimated_size`, calls `check_tier_budget(tier,
incoming)`, and raises `ModelTooLargeError` on rejection — before any weights
are allocated. Everything that reaches this function is checked.

- Budget `0` = unbounded (back-compat default; `_tier_budgets_bytes` starts at
  `{"P0": 0, "P1": 0}`).
- `P0_NEVER_EVICT` (laya pinned model) bypasses the check by design — it is
  pinned and reserved; documented exemption, not a gap.

## Checked paths (route / CLI / startup triggers)

All of the following reach `_load_engine` (or the `_load_single_model` boot
path, which now enforces the same budget) and are therefore checked:

| Trigger | Entry point |
| --- | --- |
| `/v1/models/{id}/load`, load_model_public | `server.py` `load_model_public` (~1534) |
| Admin load | `server.py` admin load (~1644) |
| `Server.load_model` | `server.py` ~2636 (via `pool.get_engine`) |
| Startup preload | `server.py` `_preload_models` (~2775) |
| Pinned preload | `server.py` `preload_pinned_models` (~3538) |
| Admin refresh helpers | `admin/helpers.py` (~359) |
| Admin model manager | `admin/model_manager_route.py` (~70), `admin/models_route.py` (~259, ~781) |
| Admin benchmark | `admin/benchmark.py` (~800) |
| Engine resolve helpers | `api/_engine_helpers.py` `resolve_engine` (~15), `api/_typed_engine.py` (~27) |
| OpenClaw | `api/openclaw_routes.py` (~434) |
| OCR | `api/ocr_routes.py` (~76) |
| Images | `api/images.py` (~158) |
| Audio | `api/audio_routes.py` (~526, ~721, ~897, ~1013, ~1066, ~1126, ~1243) |
| Video | `api/videos_routes.py` (~580) |
| Embeddings | `api/embeddings_routes.py` (~67) |
| Responses | `routes_internal/responses.py` (~101) |
| Runtime cache | `runtime/cache.py` (~195) |
| GUI compat | `gui_compat/server.py` (~88), `gui_compat/model_manager.py` (~444, ~668, ~679) |
| Training | `training/service.py` (~1176) |

## Formerly-bypassing paths and their #1194 fix

- `Server._load_single_model` (`fusion_mlx/server.py` ~2680) — created
  BatchedEngine/DiffusionEngine directly and registered with
  `estimated_size=0`, so the model was invisible to `tier_resident_bytes` and
  no budget check ran. Fixed: it now estimates on-disk weights via
  `estimate_model_size`, enforces the P0 budget **before** `engine.start()`,
  and registers the real `estimated_size` under both the served name and the
  original name (applied only at entry creation, so `_current_model_memory`
  is counted once). Rejection is a loud boot failure (`ModelTooLargeError`)
  plus a `record_tier_budget_rejection("P0", ...)` metric.
- Laya decisions route direct entry — `P0_NEVER_EVICT`, exempt by design
  (pinned, reserved at boot); documented, not a gap.
- Standalone audio CLI (`tts.py` / `stt.py` / `probe.py`) — outside the pool
  entirely; out of scope for pool budget enforcement.
- `_acquire_inplace_adapter` (LoRA swap) — adapter-only, loads no model
  weights; out of scope.

## How to set budgets

```bash
# MemoryConfig (settings/config):
#   memory.tier_budget_p0_gb / memory.tier_budget_p1_gb   (GB; 0 = unbounded)
# Env overrides (ops convenience, pure memory-capacity knobs):
export FUSION_MLX_TIER_BUDGET_P0_GB=50
export FUSION_MLX_TIER_BUDGET_P1_GB=20
```

Values are clamped: negatives are treated as config errors and clamped to 0
(unbounded) in `MemoryConfig.__post_init__`; invalid env values fall back to
the config value with a WARNING (`_resolve_tier_budget_gb` in server.py).
Wiring happens once at boot in `Server._startup`, right after the #1187
heavy-admission gate configure, and logs
`Tier budgets wired: P0=…GB P1=…GB (0=unbounded)`.

A rejection raises `ModelTooLargeError` (surfaces at routes as 503/413) and
increments `fusion_mlx_tier_budget_rejections_total` (plus the per-tier series
`fusion_mlx_tier_budget_rejections_by_tier{tier="P0|P1"}`) on `/metrics`.
