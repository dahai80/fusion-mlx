# Speculative Denoise (#177)

Diffusion-model analog of LLM speculative decoding for the SkyReels-V3 DiT family.
Draft DiT predicts K future denoise steps cheaply; the full DiT verifies all K in
one batched forward; the longest agreeing prefix is accepted and the full model
corrects at the divergence point. Target: 2-3x speedup on 14B.

Status: **Phase 1 + Phase 2 landed.** Phase 1 = algorithm + draft co-load API +
synthetic-DiT unit tests (env-gated, zero production risk). Phase 2 = real 14B
R2V wiring (`forward_partial` + spec denoise loop) + 14B acceptance sweep.
**Phase-2 result: layer-pruned draft gives 0% acceptance at safe epsilon and NO
speedup (0.2-0.4x, i.e. slower than baseline).** See "Phase 2 results" below.
Phase 3 (fusion-comfyUI Stage API) is a follow-up.

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
- Deferred: `forward_partial` + spec path on V2V/A2V DiT (same modulation fix
  needed there, tracked as a follow-up issue); UniPC 2nd-order corrector in
  spec mode; production `mx.compile` of the spec path.

## Phase 3 (follow-up)

- fusion-comfyUI Stage API exposure (`on_step` reports `SpecStats`).

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
