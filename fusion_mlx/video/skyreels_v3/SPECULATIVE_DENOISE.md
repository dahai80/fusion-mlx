# Speculative Denoise (#177)

Diffusion-model analog of LLM speculative decoding for the SkyReels-V3 DiT family.
Draft DiT predicts K future denoise steps cheaply; the full DiT verifies all K in
one batched forward; the longest agreeing prefix is accepted and the full model
corrects at the divergence point. Target: 2-3x speedup on 14B.

Status: **Phase 1 landed** (algorithm + draft co-load API + synthetic-DiT unit
tests, env-gated, zero production risk). Phase 2 (real 14B wiring + benchmark)
and Phase 3 (fusion-comfyUI Stage API) are follow-ups.

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

## Phase 2 (follow-up)

- Implement `forward_partial` on the SkyReels-V3 T2V DiT (`blocks[:M]` + head +
  unpatchify, reusing patch/time/cross-KV).
- Wire `speculative_denoise` into the T2V `generate()` denoise loop behind
  `FUSION_SPECULATIVE_DENOISE`. Handle CFG by batching 2K (cond + uncond) with
  per-element timesteps; `perform_guidance` on the verified velocities.
- Real 14B E2E benchmark; tune `K` / `epsilon` / `M`. The achievable speedup
  depends on layer-pruned draft acceptance - measure before claiming 2-3x.
- Decide 1st-order Euler speculative loop vs UniPC 2nd-order corrector (the
  corrector needs the previous full output; speculative mode currently bypasses
  it).

## Phase 3 (follow-up)

- fusion-comfyUI Stage API exposure (`on_step` reports `SpecStats`).

## Draft-quality caveat

Layer-pruned drafts are shallow-feature predictors (DeepCache / Delta-DiT style).
Their acceptance rate on a 14B flow-matching DiT is empirically unknown in MLX;
classic speculative-diffusion papers train a separate small draft for high
acceptance. Phase 1 deliberately lands the machinery without a speedup claim;
Phase 2 measures whether the layer-pruned draft is good enough, or whether a
separately distilled small draft is needed.
