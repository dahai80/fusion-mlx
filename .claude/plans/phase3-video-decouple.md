# Video Decoupling Plan: Remove mlx-video, Self-Contained MLX Video Backends

> Supersedes the Phase 3–4 sections of `video-generation-multibackend.md` (whose
> Phase 0–2 are SHIPPED: registry + LTX2Backend + Wan2Backend + features 2/3).
> User decision (2026-07-09): **full mlx-video removal** + **pure-MLX T5 port**.

## Why decouple (two proven breakages)

The inherited Phase 3 plan assumed "reuse mlx-video `ltx_2/` blocks, adapt config
loader". Verified FALSE this session — two pillars of that reuse are broken:

1. **VAE topology mismatch.** mlx-video `VideoDecoder` IS `LTX2VideoDecoder` — the
   LTX-2 redesigned VAE (1024/512/256/128, 5 layers/block, timestep conditioning,
   d2s upsamplers). 0.9.x VAE is `block_out_channels=[128,256,512,512]`,
   `layers_per_block=[4,3,3,3,4]`, NO timestep conditioning. Weights won't load into
   the new VAE. The old-format `video_vae.VideoDecoder` matches 0.9.x but has no
   `from_pretrained`/`sanitize`/`decode_tiled` — those must be written regardless.
2. **Text encoder is a different family.** mlx-video `LanguageModel` is a Gemma-style
   causal LM (Gemma scaling + sliding window + causal mask) = LTX-2's encoder. 0.9.x
   uses **T5-xxl** (`caption_channels=4096`, bidirectional encoder). Not
   interchangeable; no MLX T5 exists → pure-MLX port required (user-confirmed).

**Reusable & proven equivalent** (so cheap to write our own, ~50 lines total):
- `ltx2_scheduler` ≡ 0.9.x `RectifiedFlowScheduler` (SD3 token-dependent shift +
  stretch to terminal 0.1). Math identical.
- `denoise_dev` at default flags (`cfg_rescale=0, use_apg=False, stg=0`) ≡ standard
  CFG + Euler flow-matching step. ~30 lines of real logic.

**Verified inputs:** mlx-video = MIT (vendoring legal w/ attribution); ltx_2=14.5k
LOC/37 files (incl. `audio_vae` subpkg), wan_2=5.9k/17 — ~20k total. HF mirror
(`HF_ENDPOINT=hf-mirror.com`) reachable (HTTP 200) → weights downloadable, so
**parity + real-E2E are now POSSIBLE** (slow, multi-GB), not blocked. No local
weights currently. `transformers 5.0.0` available → usable as a correctness oracle
for the T5 port. src root = `fusion_mlx/` (also a stale copy in
`apps/fusion-mac/build/Stage` → repackage memo applies).

## End-state architecture

One unified, mlx-video-free package:

```
fusion_mlx/video/
  __init__.py
  primitives/         # shared: rope, attention, rms_norm, conv3d, scheduler, denoise
  t5_encoder.py       # pure-MLX T5-xxl encoder (0.9.x + CogVideo reuse)
  ltx_video_legacy/   # 0.9.x: transformer3d, vae, backend  (Phase 3, fresh)
  ltx2/               # LTX-2: vendored-then-evolved from mlx_video ltx_2 (Phase 4)
  wan2/               # Wan2: vendored-then-evolved from mlx_video wan_2 (Phase 5)
```

`engines/video_backends/{ltx2,wan2,ltx_video_legacy}.py` become thin wrappers over
`fusion_mlx.video.*` (keep the VideoBackend ABC + registry — unchanged, shipped).

**Approach by family** (vendor-evolve for the two working backends = safe removal;
fresh reimpl for 0.9.x = it's greenfield, mlx-video lacks it anyway):
- **0.9.x**: fresh reference-impl referencing diffusers `transformer_ltx.py` +
  mlx-video `ltx_2/`. Zero mlx-video import.
- **LTX-2 / Wan2**: **vendor-then-evolve** — copy mlx-video modules into
  `fusion_mlx/video/{ltx2,wan2}/` with MIT attribution header (removes the EXTERNAL
  pip dep immediately; code becomes ours to fix/extend — directly solves the
  "4-month-stale, upstream-blocked" concern), then refactor to share Phase-3
  primitives, drop unused subsystems (e.g. `audio_vae` if LTX2Backend is video-only),
  each change guarded by a parity-oracle test. Chosen over blind from-scratch reimpl
  of 20k unverifiable-blind lines (Rule 1/12). Full reimpl remains an option if you
  prefer purity over safety — flag at approval.

## Phases (checkpoint + verify after each; "now" = start now, 0.9.x zero-dep first)

### Phase 3 — 0.9.x self-contained (greenfield, the real new capability)
Sub-steps, each with mock-weight unit tests before next (Rule 9/10):
- **3a T5-xxl encoder** (`t5_encoder.py`): embed + N encoder layers (self-attn + FF,
  RMSNorm pre-norm, relative position bias) + final norm. ~300 LOC. **Oracle: compare
  our output to `transformers.T5EncoderModel` on identical input** (download
  `t5-v1_1-xxl` via mirror, or a small t5 for fast CI). allclose gate.
- **3b Transformer3DModel** (`ltx_video_legacy/transformer.py`): 28 layers, 32 heads,
  head_dim 64, inner_dim 2048, cross_attn_dim 2048, qk_norm=RMSNorm(2048),
  caption_projection (PixArtAlphaTextProjection, caption_channels=4096),
  scale_shift_table(6,2048), AdaLayerNormSingle, patchify_proj(128→2048). ~600 LOC.
- **3c VAE decoder** (`ltx_video_legacy/vae.py`): 0.9.x topology
  (`OURS_VAE_CONFIG.blocks`: res_x/compress_all/res_x_y), write
  `from_pretrained`+`sanitize` (5D conv weight transpose `(0,2,3,4,1)`,
  per_channel_statistics mean/std rename) + `decode_tiled` (reference
  `video_vae/tiling.py`). ~400 LOC.
- **3d scheduler + denoise + backend** (`primitives/scheduler.py`,
  `primitives/denoise.py`, `ltx_video_legacy/backend.py`): SD3 shift+stretch (~20
  LOC), CFG+Euler loop (~30 LOC), `LegacyLTXBackend` ABC impl (detect/start/stop/
  generate/constraints — match ltx2.py/wan2.py patterns). Swap the stub import in
  `engines/video_backends/__init__.py` from `.unimplemented` → new package; remove
  `LegacyLTXBackend` from `unimplemented.py`.
- **3e real E2E** (mirror): download LTX-Video-0.9.x weights, T2V smoke → coherent
  mp4. Deferred-but-possible (mirror works).
- Gate: mock tests green + T5 oracle green + (3e) smoke.

### Phase 4 — LTX-2 self-contained (vendor-then-evolve)
- **4a Vendor**: copy `mlx_video/models/ltx_2/*` → `fusion_mlx/video/ltx2/` with MIT
  attribution; rewrite the single `from mlx_video...` import surface to local; keep
  `generate_video` callable. LTX2Backend imports from `fusion_mlx.video.ltx2`.
  Parity gate: identical prompt+seed → mlx-video vs vendored → output allclose.
- **4b Evolve**: drop unused subsystems (confirm LTX2Backend audio usage first;
  drop `audio_vae`/`denoise_dev_av` if video-only), refactor to share Phase-3
  primitives (rope/attention/rms_norm/scheduler/denoise where identical). Each
  refactor re-runs parity gate.
- Gate: parity green on a fixed prompt set before 4b destructive drops.

### Phase 5 — Wan2 self-contained (vendor-then-evolve)
- Same as Phase 4 for `mlx_video/models/wan_2/*` → `fusion_mlx/video/wan2/`.
  Wan2Backend imports locally. Parity gate (T2V + I2V).

### Phase 6 — Cut mlx-video dependency
- Remove `mlx-video` from `pyproject.toml` `[video]` group; delete any remaining
  `import mlx_video` (grep-verify zero). Update `tests/unit/test_video_backends.py`
  (currently stubs mlx_video — replace wan2/ltx2 stubs with real local imports).
- macOS app: repackage (`apps/fusion-mac`) — standing constraint, Python lands on main.
- Gate: `pytest tests/unit/test_video*.py` green with mlx-video UNINSTALLED (true
  zero-dep proof); `black`+`ruff` clean.

## Verification strategy
- **Mock-weight unit tests** per module (shape/dtype/numerical sanity) — fast, no
  download. Gate every sub-step.
- **Parity oracle** (Phase 4/5): same prompt+seed through old mlx-video backend and
  new self-contained → compare (allclose on latents / frame PSIM). The correctness
  gate before any destructive drop or dep removal.
- **T5 oracle** (Phase 3a): our MLX T5 vs `transformers.T5EncoderModel` allclose.
- **Real E2E** (mirror): 0.9.x T2V smoke; LTX-2/Wan2 parity runs. Multi-GB, gated,
  parallelizable, non-blocking for unit tests.

## Touchpoints
| File | Change |
|---|---|
| `fusion_mlx/video/*` | NEW unified package (primitives, t5_encoder, ltx_video_legacy, ltx2, wan2) |
| `fusion_mlx/engines/video_backends/__init__.py` | swap `LegacyLTXBackend` import: `.unimplemented` → `.ltx_video_legacy` |
| `fusion_mlx/engines/video_backends/unimplemented.py` | remove `LegacyLTXBackend` (keep `CogVideoBackend`) |
| `fusion_mlx/engines/video_backends/ltx2.py` | import surface → `fusion_mlx.video.ltx2` (Phase 4) |
| `fusion_mlx/engines/video_backends/wan2.py` | import surface → `fusion_mlx.video.wan2` (Phase 5) |
| `tests/unit/test_video_backends.py` | replace mlx_video stubs with real local imports (Phase 6) |
| `tests/unit/test_ltx_video_legacy.py` | NEW — per-module mock tests + T5 oracle |
| `pyproject.toml` | drop `mlx-video` (Phase 6) |
| `README.md` | video section: self-contained backends, no mlx-video |

## Prerequisites / pre-flight
- **Confirm LTX2Backend audio usage** (read `ltx2.py` generate path) — decides
  whether `audio_vae`/`denoise_dev_av` survive Phase 4b. Do before 4b.
- **PYTHONSAFEPATH crash-fix** (build.sh:482 +8L, PythonRuntime.swift:141 +7L) is
  UNCOMMITTED on main (memory). Needed for the macOS app to run Python at all.
  Commit it as a prerequisite (separate commit) — pending your go-ahead.
- **Memory correction**: update `fusion-mlx-phase3-ltx-video-legacy-port.md` —
  scheduler math is identical (not LTX-2-specific); adapter-only plan superseded by
  this decoupling plan. Non-blocking doc task.

## Risks / effort honesty
- **Scope is large** (~20k LOC across ltx_2+wan_2; 0.9.x adds ~2–2.5k fresh). Phased
  with checkpoints (Rule 10). Phase 3 lands independently and is the real unlock.
- **T5 port correctness**: no local MLX-T5 reference → mitigated by transformers
  oracle (3a). Highest single risk in Phase 3.
- **Vendor-then-evolve attribution**: MIT header required on each vendored file;
  track upstream provenance in a `NOTICE`.
- **Mirror slowness**: multi-GB weight downloads (T5-xxl ~9.8G) gate only real-E2E
  and parity runs, not unit tests. Parallelizable.
- **"现在就全量移除"** is sequenced, not one commit: 0.9.x zero-dep immediately
  (Phase 3); the `mlx-video` pip dep is deleted only at Phase 6 after LTX-2/Wan2 are
  self-contained + parity-verified. Deleting it earlier would break working backends
  (Rule 3/12).
