# Phase 2: Speculative Denoise - wire into SkyReels-V3 R2V (T2V) denoise loop

## Goal
Wire the Phase-1 `speculative_denoise` module into the real SkyReels-V3 R2V DiT + denoise loop behind `FUSION_SPECULATIVE_DENOISE`, validate on the real 14B weights already on disk, and measure SpecStats. Default-off; zero change to the production UniPC path when off.

## Scope decision (conflict surfaced, not averaged - Rule 7)
- Production loop uses **UniPC 2nd-order** (`scheduler.step`). Phase-1 module uses **1st-order Euler** internally. Mixing is incoherent.
- Decision: speculative path is a **parallel 1st-order Euler path** (full + draft), gated by env. When off, the existing UniPC loop runs byte-for-byte unchanged (zero prod risk preserved). Spec mode ≈ production with solver_order=1 + draft skipping. The 2nd-order corrector is the only quality delta (documented).
- CFG: spec mode always runs b=2 (cond+uncond batched) per latent - ignores dynamic-CFG for coherence. Production base `_encode_context` is a stub (zeros) so CFG is effectively no-op anyway; this lets us measure draft-vs-full acceptance cleanly.

## Files to change (surgical)

### 1. `fusion_mlx/video/skyreels_v3/transformer_r2v.py` - add `forward_partial`
Mirror `__call__` exactly but run `self.blocks[:n_blocks]`:
```
def forward_partial(self, x, t, context, seq_lens, grid_sizes, n_blocks,
                    context_lens=None, rope_cos_sin=None, attn_mask=None):
    # identical embed/time-projection as __call__; loop self.blocks[:n_blocks]; head; unpatchify
```
When `n_blocks == num_layers` -> bit-identical to `__call__` (correctness invariant). ~15 lines, no `__call__` change.

### 2. `fusion_mlx/video/skyreels_v3/speculative_denoise.py` - add `eval_steps` flag
Add `eval_steps: bool = False` to `SpeculativeConfig` (+ `.from_env()` reads `FUSION_SPEC_EVAL_STEPS` default "1"). When True, `mx.eval(x)` after each macro-step commit + `mx.eval(vs_f)` after the batched full verify. Bounds memory for 14B (#146 OOM lesson). Pure-for-Phase-1-tests: eval doesn't change math, so existing 10 tests unaffected (flag defaults don't change their behavior - they construct config without eval_steps or with default False; verify tests still pass).

### 3. `fusion_mlx/video/skyreels_v3/pipelines/__init__.py` - wire the path
- Add `_denoise_sample_speculative(self, latents, context, *, seq_lens, grid_sizes)`:
  - Read `SpeculativeConfig.from_env()`, `FUSION_SPEC_DRAFT_BLOCKS` (M, default `num_layers//4`).
  - Build `FlowUniPCMultistepScheduler` only for its `timesteps` schedule (not the solver).
  - `full_velocity(x_batch[K,...], t_batch[K])`: CFG-expand to 2K (`concat([x_batch]*2)`, `t_2k=concat([t_batch]*2)`, `ctx_2k=concat([context]*2K)`, `seq/grid *2K`), `self.dit(...)`, `perform_guidance(noise, guidance_scale)` -> `[K,...]`. `mx.eval` if eval_steps.
  - `draft_velocity(x_batch[K,...], t_batch[K])`: same CFG-expand, `self.dit.forward_partial(..., n_blocks=M)`, `perform_guidance` -> `[K,...]`. `mx.eval` if eval_steps.
  - `latents, stats = speculative_denoise(full_velocity, draft_velocity, latents, timesteps, config)`.
  - Log SpecStats summary (macro_steps, full_forwards, draft_forwards, avg_accept, speedup).
- In `_denoise_sample`, add at top:
  ```
  if speculative_enabled() and hasattr(self.dit, "forward_partial"):
      return self._denoise_sample_speculative(latents, context, seq_lens=seq_lens, grid_sizes=grid_sizes)
  ```
  Existing loop untouched otherwise. Scoped to R2V (only R2V has forward_partial in Phase 2).

### 4. `tests/unit/test_speculative_denoise_phase2.py` - CI-safe tests
- Mock-DiT wiring test: a fake `dit` with `__call__` + `forward_partial` (returns `-x + 0.3*t` style) -> exercise `_denoise_sample_speculative` velocity closures (CFG expand, perform_guidance), assert output shape == input, SpecStats populated, env gate routes correctly (off -> baseline path).
- Tiny real `SkyReelsR2VDiT` (config dict: dim=16,ffn_dim=32,num_heads=4,num_layers=4,patch_size=(1,2,2),in_dim=16,out_dim=16,text_dim=32,text_len=16,freq_dim=16): assert `forward_partial(n_blocks=4) == __call__` (bit-identical) on non-trivial random input; assert `forward_partial(n_blocks=2)` runs + differs.
- Add to conftest `_OPT_DEP_SUITES` "mlx" skip list (hard-imports mlx).
- black + ruff clean before push (CI gotcha #2).

### 5. `bench_speculative_denoise.py` (repo root) - real 14B benchmark
- Load `SkyReelsR2VPipeline` with real R2V-14B-MLX weights (no download - already on disk).
- Reduced scale: 256x256, 9 frames, N=8 steps (warmup-compatible shape), prepared latents/context (skip generate/VAE).
- Run baseline (`_denoise_sample`, env off) vs speculative (env on, K=3, M=10, epsilon=0.1): wall-clock + full-forward count + SpecStats.
- Print comparison table. Real model load per governance.

### 6. Docs
- `README.md` #177 section: Phase-2 status update (R2V wired, env knobs active, benchmark numbers).
- `SPECULATIVE_DENOISE.md`: Phase-2 done, memory-scaling caveat (2K batched verify), UniPC-vs-Euler trade-off, deferred (V2V/A2V forward_partial, 720p chunked verify, ComfyUI Stage API = Phase 3).

## Execution checkpoints (Rule 10)
1. Implement forward_partial + eval_steps flag + _denoise_sample_speculative + env gate. **Checkpoint: unit tests pass (mock + tiny real).**
2. black + ruff clean. **Checkpoint: lint green.**
3. Run real 14B benchmark (reduced scale). **Checkpoint: no crash, SpecStats populated, numbers recorded.**
4. Update docs with real numbers (honest: speedup = measured; acceptance = measured).
5. Commit, push branch, open PR, comment on #177 (Phase-2 landed, real numbers), update memory, compact.

## Known risks / honest caveats
- **2K batched-verify memory**: full verify batches 2K (K latents × CFG) full-DiT forwards. At 720p this is 4× production b=2 activation - may OOM. Benchmark at 256x256 first; 720p may need chunked verify (deferred). Documented.
- **1st-order Euler vs UniPC**: spec path loses the 2nd-order corrector. Quality delta documented; speedup comes from fewer full forwards, not solver order.
- **Stub context**: base `_encode_context` returns zeros -> CFG no-op, text not conditioned. Valid for measuring draft-vs-full mechanism + acceptance; real-prompt acceptance may differ. Documented.
- **Draft = layer-pruned same weights** (M=10 of 40 blocks). Acceptance empirically unknown until benchmark - claim nothing until measured.

## Out of scope (deferred)
- V2V/A2V `forward_partial` (R2V/T2V only in Phase 2).
- 720p chunked verify (memory).
- flicker_fix / boundary-align in spec mode (R2V has no boundary align; flicker deferred).
- ComfyUI Stage API (Phase 3).
