# Speculative Denoise (#177)

Diffusion-model analog of LLM speculative decoding for the SkyReels-V3 DiT family.
Draft DiT predicts K future denoise steps cheaply; the full DiT verifies all K in
one batched forward; the longest agreeing prefix is accepted and the full model
corrects at the divergence point. Target: 2-3x speedup on 14B.

Status: **Phase 1 + Phase 2 + Phase 3 landed.** Phase 1 = algorithm + draft co-load API +
synthetic-DiT unit tests (env-gated, zero production risk). Phase 2 = real 14B
R2V wiring (`forward_partial` + spec denoise loop) + 14B acceptance sweep.
**Phase-2 result: layer-pruned draft gives 0% acceptance at safe epsilon and NO
speedup (0.2-0.4x, i.e. slower than baseline).** See "Phase 2 results" below.
Phase 3 = additive stats surface (`VideoBackend.last_denoise_stats` +
`GET /v1/videos/denoise-stats`) so clients can read the last run's acceptance
stats; feature surface for when a real distilled draft arrives.

## Algorithm

Flow-matching ODE step (1st-order Euler): `x_{t+dt} = x_t + dt * v(x_t, t)`,
`dt = t_next - t_cur` (timesteps decrease 1 -> 0).

One macro-step starting at index `i` with committed latent `x_i`:

1. **Draft predict** (sequential, cheap): for `j = 0..K_eff-1` run the layer-pruned
   draft at `(x_{i+j}^d, t_{i+j})` to get `v_d[j]`, then
   `x_{i+j+1}^d = x_{i+j}^d + dt[i+j] * v_d[j]`. `x_i^d = x_i`.
2. **Batched verify** (one full forward): stack the K departure latents
   `[x_i, x_{i+1}^d, ..., x_{i+K_eff-1}^d]` and their timesteps into a single
   batch; the full DiT returns `v_f[0..K_eff-1]` in one forward. Per-element
   timesteps are natively supported on the Wan2/SkyReels DiT
   (`t.ndim == 1` -> per-batch modulation, see `wan_2.py`).
3. **Accept prefix**: largest `a` such that
   `||v_d[j] - v_f[j]|| / ||v_f[j]|| < epsilon` for all `j < a`.
4. **Commit**: if `a < K_eff` (diverged) the full velocity `v_f[a]` was already
   computed at the divergence latent `x_{i+a}^d`, so take one bonus Euler step
   `x = x_{i+a}^d + dt[i+a] * v_f[a]`, `i += a + 1` (always advances >= 1); if
   `a == K_eff` (all accepted) `x = x_{i+K_eff}^d`, `i += K_eff`.

Correctness invariant (verified by tests): whenever each accepted draft velocity
equals the full velocity, the committed trajectory is bit-identical to baseline
1st-order Euler with the full model. The bonus step always uses the full velocity
at the divergence latent, so divergence only changes *where* the draft prefix
stops, not the final trajectory.

`K_eff = min(K, N - 1 - i)` clamps the last macro-step so the index never
overruns the schedule.

Cost per macro-step: 1 full forward (batched K) + K draft forwards. Baseline for
the same K steps is K full forwards. Speedup ratio =
`(accepted + bonus) / 1 full forward`; approaching K when acceptance is high and
the draft is much cheaper than the full model.

## Draft co-loading API

No fast 1B/3B SkyReels draft checkpoint exists, and MLX weight quantization is
not a speed path (#166). The only MLX-feasible fast draft is a **layer-pruned**
variant of the same DiT: load the 14B once, run only the first `M` of `N`
transformer blocks plus the shared patch/time/cross-KV and output head. This is
`M/N` of the compute and co-loads from the same weights - no separate
checkpoint.

- `DraftDiTMixin`: contract a block-list DiT implements. Adds
  `forward_partial(latent_input, t, n_blocks, **kw)` running the first `n_blocks`
  blocks + shared head.
- `LayerPrunedDraft(dit, n_blocks, **call_kw)`: adapts a `forward_partial` DiT
  into the batched `VelocityFn` the loop expects.
- `VelocityFn = (x_batch, t_batch) -> v_batch`: model-agnostic velocity callable.
  The full model and the draft both satisfy it.

## Module API (`fusion_mlx/video/skyreels_v3/speculative_denoise.py`)

- `speculative_enabled() -> bool`: env gate.
- `SpeculativeConfig(K, epsilon, relative)` + `.from_env()`.
- `speculative_denoise(full_velocity, draft_velocity, latents, timesteps, config)
  -> (latents, SpecStats)`: the loop above.
- `baseline_euler(full_velocity, latents, timesteps) -> latents`: reference for
  tests/validation.
- `SpecStats`: `macro_steps`, `accepted[]`, `full_forwards`, `draft_forwards`,
  `baseline_steps`, `avg_accept`, `speedup`.

## Env knobs (all default off / no-op in Phase 1)

| env | default | meaning |
| --- | --- | --- |
| `FUSION_SPECULATIVE_DENOISE` | `0` | master switch (Phase 2 wires it into the T2V loop) |
| `FUSION_SPEC_K` | `4` | draft lookahead depth |
| `FUSION_SPEC_EPSILON` | `0.1` | relative velocity error threshold |
| `FUSION_SPEC_DRAFT_BLOCKS` | - | layer-pruned block count (Phase 2) |
| `FUSION_SPEC_EVAL_STEPS` | `1` | eval draft velocities each step for stats (`SpecStats`) |

## Feasibility evidence

- Per-batch timesteps: `WanModel.__call__` (`wan_2.py`) handles `t.ndim == 1` as
  "scalar timestep per batch element" -> `[B,1,6,dim]` modulation. Batched verify
  of K latents at K timesteps (x2 for CFG = 2K) is mechanically supported.
- `FlowUniPCMultistepScheduler.add_noise` already accepts per-batch `[B]`
  timesteps.
- `get_timestep_embedding` (`ltx_video_legacy/transformer.py`,
  `ltx2/transformer.py`) reshapes timesteps to `(-1)` and broadcasts per element.
- `self.blocks` is a plain list on the Wan2/SkyReels DiT -> layer pruning is
  `blocks[:M]` + `head(x, e)`.

## Phase 1 scope (this PR)

- `speculative_denoise.py`: algorithm + draft API + env gate + `SpecStats`.
- `tests/unit/test_speculative_denoise.py`: 10 tests with a synthetic velocity
  field and a tiny `nn.Module` DiT exposing `forward_partial`. Covers:
  accept-all (draft == full), divergence (prefix accept + bonus == baseline),
  zero-accept (always advances 1), batched-verify == sequential, K clamp at end,
  too-few-timesteps no-op, env flag, `SpecStats.speedup`, layer-pruned draft runs
  fewer blocks, draft==full on a real (tiny) `nn.Module` matches baseline.
- `tests/unit/conftest.py`: `test_speculative_denoise.py` added to the `_OPT_DEP_SUITES`
  "mlx" skip list (Linux CI has mlx absent/mocked; runs on macOS).
- No production path changed. Default-off. Zero regression risk.

## Phase 2 results (landed)

- Implemented `forward_partial` on `SkyReelsR2VDiT` (`blocks[:M]` + shared head,
  reusing patch/time/cross-KV + unpatchify). Bit-identical to `__call__` at
  `n_blocks == num_layers` for B=1 and B=2 (unit-tested).
- Fixed a modulation broadcast bug in `WanAttentionBlock.__call__` that crashed
  for batch B>1 (`mod[:, k, :]` [B,dim] vs `x_norm` [B,L,dim]); reshaped
  modulation to `[B,6,1,dim]`. Broadcast-equivalent (byte-identical) at B=1, so
  production (single-video B=1) is unchanged; only enables the B=2K CFG verify.
- Wired `speculative_denoise` into the R2V denoise loop behind
  `FUSION_SPECULATIVE_DENOISE` (1st-order Euler; UniPC 2nd-order corrector
  bypassed in spec mode). CFG batched 2K (cond+uncond) with per-element
  timesteps; `perform_guidance` on the verified velocities.
- Real 14B R2V acceptance sweep (128x128, 5 frames, N=6, K=3, bf16, no compile):

  | draft blocks | % kept | epsilon | avg accept | full fw | draft fw | wall vs baseline | maxdiff vs baseline |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | 10/40 | 25% | 0.10 | 0.00 | 5 | 12 | 0.34x (slower) | 0.00073 |
  | 20/40 | 50% | 0.10 | 0.00 | 5 | 12 | 0.25x (slower) | 0.00073 |
  | 30/40 | 75% | 0.10 | 0.00 | 5 | 12 | 0.20x (slower) | 0.00073 |
  | 38/40 | 95% | 0.10 | 2.50 | 2 | 5  | 0.42x (slower) | 0.09687 |
  | 20/40 | 50% | 0.30 | 0.00 | 5 | 12 | 0.23x (slower) | 0.00073 |
  | 20/40 | 50% | 0.50 | 0.00 | 5 | 12 | 0.23x (slower) | 0.00073 |
  | 30/40 | 75% | 0.30 | 0.00 | 5 | 12 | 0.19x (slower) | 0.00073 |

- **Conclusion**: at safe epsilon (0.1), acceptance is 0% for any layer-prune
  down to 25% blocks kept (75% pruned). Acceptance only appears at 95% blocks
  kept (near-full draft) where the draft costs ~= full -> no speedup (0.42x,
  still slower) AND quality degrades (maxdiff 0.097 vs 0.00073). Loosening
  epsilon to 0.5 does not help at moderate depth -> the pruned-draft velocity
  error is far above 0.5, i.e. a fundamentally different velocity field, not a
  marginally-off one. The #177 hypothesis (layer-pruned draft + batched verify
  -> 2-3x) is **falsified on MLX SkyReels-V3 R2V 14B**: DiT velocity fields
  require full depth; unlike LLM token prediction, denoise steps are not
  predictable from a sub-network. The machinery is correct (all-rejected spec
  matches baseline Euler to 7e-4) and stays landed (env-gated, default off,
  zero prod risk) as infrastructure for a future separately-distilled small
  draft (the path classic speculative-diffusion papers take for high
  acceptance).
- Done (#186): `forward_partial` + spec path on V2V/A2V DiT. V2V shares R2V's
  single-context forward convention, so it reuses the base spec path (no
  production change). A2V's DiT forward signature differs (audio + text
  embeds), so it gets its own `_denoise_sample_speculative` override
  (`cross_kv_cache=None` since spec batches K steps at varying batch sizes)
  plus a base branch guard (`branch != "a2v"`) keeping the single-context
  base path R2V/V2V-only. Items 1 (modulation B>1 fix) + 2 (V2V/A2V
  `forward_partial`) landed in PR #187; item 3 (pipeline wiring) in PR #189.
- Deferred: UniPC 2nd-order corrector in spec mode; production `mx.compile`
  of the spec path.

## Phase 3 (landed) - denoise-stats surface

The released Stage API (`denoise() -> mx.array`, PR #170) and step callback
(`on_step: Callable[[int, int], Awaitable[None]]`, PR #171) signatures cannot
carry a stats dict without breaking the released contracts, and `SpecStats` is
computed once at the end of `_denoise_sample_speculative` (a single batched
verify call), not per-step. So the Phase-3 surface is an **additive accessor +
poll route**, not a per-step callback argument:

- `SpecStats.to_dict() -> dict`: plain-dict serialization
  (`macro_steps`, `accepted`, `avg_accept`, `full_forwards`, `draft_forwards`,
  `baseline_steps`, `speedup`).
- `VideoBackend.last_denoise_stats() -> dict[str, Any]`: base default `{}` (every
  backend without a spec-denoise path returns empty - no break to the released
  stage API).
- `SkyReelsBackend.last_denoise_stats()`: serializes the pipeline's
  `_last_spec_stats` (set by `_denoise_sample_speculative`) plus the current env
  config. When spec is off or no run happened, returns `available=False` with
  zeroed counters - honest "feature surface", no speedup claimed today.
- `VideoGenEngine.last_denoise_stats()`: delegates to the backend.
- `GET /v1/videos/denoise-stats?model=<name>`: HTTP surface. Default model
  `ltx-2`; 404 if the model is not loaded / not a video model; 503 if the engine
  pool is not initialized.

Response schema:

```json
{
  "model": "skyreels-r2v-14b",
  "stats": {
    "macro_steps": 0,
    "accepted": [],
    "avg_accept": 0.0,
    "full_forwards": 0,
    "draft_forwards": 0,
    "baseline_steps": 0,
    "speedup": 0.0,
    "available": false,
    "enabled": false,
    "config": { "K": 4, "epsilon": 0.1, "relative": true, "eval_steps": true }
  }
}
```

`available=false` + zeroed counters is the honest state today (Phase-2 falsified
the layer-pruned draft). When a real distilled draft lands and
`FUSION_SPECULATIVE_DENOISE=1` produces accepted runs, the same endpoint reports
`available=true` with non-zero `accepted` / `avg_accept` / `speedup` - no further
code change needed.

Tests: `tests/unit/test_spec_denoise_stats.py` (18 tests, macOS-only via the
`_OPT_DEP_SUITES` "mlx" skip list). Covers `SpecStats.to_dict` roundtrip, the
backend default + SkyReels override (available reflects a run, enabled/config
reflect env), engine delegation, and the route (ok / default model / 404 / 503 /
500).

## Draft-quality caveat (resolved by Phase 2)

Layer-pruned drafts are shallow-feature predictors (DeepCache / Delta-DiT
style). Phase 2 measured their acceptance on the real 14B R2V flow-matching
DiT: **0% at safe epsilon for any prune down to 25% blocks kept**; acceptance
only at 95% kept, where the draft is near-full-cost and quality breaks. So a
layer-pruned draft is NOT good enough on MLX SkyReels-V3. A separately
distilled small draft (the classic speculative-diffusion approach for high
acceptance) remains the path to a real speedup; the landed machinery
(`forward_partial`, `LayerPrunedDraft`, `speculative_denoise`, env knobs) is
the infrastructure that draft would plug into.
