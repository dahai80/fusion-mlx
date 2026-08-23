# MiniMax-H3 (33B Omni-Transformer)

MiniMax-H3 is a single-flow packed-sequence DiT that jointly generates video and
audio from text. Unlike cross-attention DiTs, H3 scatters text/video/audio rows
into **one** packed sequence and runs full self-attention over all of it
(`is_causal=False`, no mask). This is a pure-MLX port in `fusion_mlx/video/minimax_h3/`.

> **Status: P0–P10 landed, P10 native-audio E2E VERIFIED on real 33B weights.**
> Config / VAE / DiT / scheduler / text-encoder / backend+registry are complete,
> the **t2va video-only** path is verified, and the **joint audio+video** path
> (#588) is wired and verified: `audio=true` produces an MP4 with a real `aac`
> audio stream (32 kHz mono, non-silent, content varies by seed) muxed alongside
> the h264 video. The text_encoder (Qwen3-VL, 67 GB) loads real weights for E2E;
> the audio branch uses a decode-only MLX port of upstream DAC+BigVGAN.

## Phase map

| Phase | Scope | Status |
|---|---|---|
| P0 | H3Config / H3VAEConfig / H3AudioVAEConfig / H3Partition | ✅ |
| P1 | Video VAE (spatial 16×, temporal 4×, z=24, ViT3D decoder) | ✅ |
| P2 | DiT transformer (packed-scatter forward, AdaLN 3-modality table) | ✅ |
| P3 | Rectified-flow scheduler (data-ward velocity, two shifts) | ✅ |
| P4 | Text encoder (Qwen3-VL, layer 49 hidden states) | ✅ |
| P5 | Backend + BACKENDS registry + constraints | ✅ |
| **P6** | **t2va video-only packed-sequence assembly + denoise loop** | **✅ verified** |
| P7 | Prompt skill (h3-prompt-writing Context-IR) | deferred |
| **P8** | **Real-model E2E** | **✅ verified** |
| **P9** | **Tests (118 pass) + real-model E2E validation** | **✅ verified** |
| **P10** | **Native audio (#588): joint t2va audio+video denoise + AudioVAE decode + MP4 mux** | **✅ verified** |

## Packed-sequence contract

The DiT `forward` signature (verified against diffusers `transformer_minimax_h3.py`):

```python
dit(
    hidden_states,          # (B, N_video, patch_dim)   video patch tokens
    audio_hidden_states,    # (B, N_audio, audio_dim)   audio patch tokens
    encoder_hidden_states,  # (B, N_text, 5120)         text embeddings
    timestep,               # (K,)  DISTINCT noise levels in [0,1]
    timestep_indices,       # (seq,)  per-row index into timestep
    token_tags,             # (seq,)  0=video, 1=text, 2=audio
    position_ids,           # (seq, 3)  (t,h,w) rotary coords per row
    video_indices,          # (N_video,) row positions of video
    audio_indices,          # (N_audio,) row positions of audio
    text_indices,           # (N_text,) row positions of text
)
```

The transformer docstring states: *"The caller is responsible for building the
packed layout: patchifying the video latents, ordering the rows, and producing
the (t, h, w) position grid, the per-row modality tags and the per-row timestep
indices."*

### t2va video-only layout (P6, inferred)

For text→video without audio or conditioning frames, the layout is:

- **text rows** (tag=1): `timestep=1.0` (clean), `timestep_index=0`,
  `position=(0,0,0)`.
- **video rows** (tag=0): `timestep=1-sigma` (current noise level),
  `timestep_index=1`, `position=(t,h,w)` patch grid.
- `timestep = [1.0, t_video]` — two distinct levels; `audio_indices` empty.

Scatter order is text→video (order is cosmetic; `*_indices` decide placement).
No padding, one attention document, no mask.

See `condition.py::build_t2va_packed` for the assembly and `generate.py` for the
denoise loop.

## Patchify

Video latents `(B, 24, T', H', W')` → tokens `(B, N_v, 96)` with
`patch_size=(1,2,2)`: `N_v = T'·(H'/2)·(W'/2)`, each token = `24·1·2·2 = 96`.
Row order is t-outer, h-middle, w-inner (matches the VAE ViT decoder
`_pack_tensors_3d`). Latents are normalized `(z - mean)/std` before patchify and
denormalized before VAE decode; the 24-dim `mean`/`std` are hardcoded from the
VAE `config.json` (see `condition.py`).

## Scheduler

`MiniMaxH3Scheduler` — rectified-flow Euler, `eta=0`. Three properties
verified against the diffusers source **and the real model**:

1. **Velocity sign is data-ward (PLUS)**: `x0 = x_t + sigma·v`. H3's
   transformer predicts a *data-ward* velocity, the opposite of diffusers'
   default `x0 = x_t - sigma·v`. An early port wrongly used the standard
   minus sign, which made denoising move in the wrong direction and oscillate
   near the fixed point — the signature was heavy motion jitter (frame-to-frame
   motion 49 vs the official ~12). Corrected to the official PLUS sign;
   real-model E2E then gave motion 9.8 / std 83 (from 49 / 112), matching the
   official range. See the **P8 corrections** note below.
2. **`t = 1 - sigma`**, t=1 is clean; `timesteps = 1 - sigmas[:-1]`.
3. **sigma grid** `linspace(1,0,N)` + exponential shift
   `s·σ/(1+(s-1)σ)` + `_unique_consecutive` fold.

Euler step: `denoised = sample + sigma·output; prev = ratio·sample + (1-ratio)·denoised`
where `ratio = sigma_next/sigma`. Two instances: video `shift=12.0`,
audio `shift=3.0`. Default `num_inference_steps=20` (matches the ComfyUI
base-quality profile).

## P8 corrections (real-model bugs found & fixed)

Loading the real 33B FL2VA weights surfaced six bugs in the inferred P0–P6 code.
All fixed and verified end-to-end (DiT 534/536 params, VAE 559/560, non-trivial
frames):

1. **Weight path resolution** (`generate.py::_resolve_subdir`). Loaders globbed
   `<dir>/*.safetensors` but real weights nest under `source/`
   (`video_vae/source/model.safetensors`). Now resolves the nested subdir.
2. **VAE FeedForward inner_dim** (`vae.py`). ViT3D decoder FF used
   `round(dim*8/3)`; real ckpt is `dim*4` (dim=2048 → inner_dim=8192, gated `w1`
   outputs 2×inner_dim=16384). Fixed to `inner_dim = dim * mult`, `mult=4`.
3. **DiT param tree** (`transformer.py::load_dit_from_pretrained`). The old
   `_flatten_params`/`_update_module` did not recurse `ModuleList`, so only
   18/20 top-level params loaded — the 50-block 33B body was random init.
   Rewritten with `mlx.utils.tree_flatten`/`tree_unflatten` + `model.update`;
   now 534/536 matched (2 unmatched: `rope.inv_freq` non-learned,
   `final_layer.norm.weight` dead-norm, both benign).
4. **DiT final_layer.norm remap** (`transformer.py::_remap_transformer_weights`).
   ckpt key `final_layer.norm.weight` is the AdaLN norm used in `__call__`, but
   was loaded into the unused dead `self.norm`. Redirected to
   `final_layer.adaln_proj.norm.weight`.
5. **Velocity sign** (`scheduler.py`). Minus → plus (see Scheduler above).
   The port initially used the standard `x0 = x_t - sigma·v`, but the official
   diffusers source uses the *data-ward* `x0 = x_t + sigma·v`. The wrong sign
   made denoising move against the flow and oscillate near the fixed point —
   the visible symptom was heavy frame-to-frame motion jitter (motion 49,
   std 112 vs the official ~12 / ~68). Corrected to PLUS; real-model E2E
   then gave motion 9.8 / std 83. (The earlier "plus collapses frames"
   conclusion came from a 2-step 256×256 smoke test with confounding factors
   and was a misread; the official source is authoritative.)
6. **Multi-step OOM** (`generate.py`). MLX lazy graph accumulates across denoise
   steps → EXIT=137. Added per-step `mx.eval(latents)` to materialize and free
   the graph. (Separately, the real 77 GB load needs ~100 GB free RAM — stop the
   fusion-mlx server via `start.sh stop` before real-model runs.)

## P9 corrections (1344×768 near-black)

768p (1344×768) T2V at 41 frames decoded to a near-black mp4 (frame mean 12.6,
vs the 768×448 baseline 189.6). Two code bugs and one config constraint found;
both bugs fixed and real-model verified, 130 H3 tests pass.

7. **Imagenet denormalize missing** (`generate.py::_to_frames`). The VAE
   decoder outputs *normalized* pixel space, not raw `[0,1]`. The official
   `vae_processor.revert_tensor = transform_rev(x).clamp(0,1)` applies
   `x*std+mean` (imagenet mean=(0.485,0.456,0.406),
   std=(0.229,0.224,0.225)) *before* the clamp. The early port did `clip(0,1)`
   only. At 1344×768 the decoded DC was −1.15, so `clip` alone sent ~99% of
   pixels to 0 → near-black. 768×448 happened to decode DC=+0.81 (positive)
   so the same defect "looked normal." Fixed: `_to_frames` now calls
   `_denormalize_pixel` (channel axis 1, shape `(1,3,1,1,1)`) before the clip.
8. **VAE decoder spatial tiling missing** (`vae.py::decode`). The official
   config sets `vae_decoder_tiling=1`, `vae_tile_size=256`,
   `vae_tile_overlap_min=64` (pixel space). `klvae.tiled_decode` splits the
   latent spatially (latent = pixel // 16 = 16) and blends tile overlaps in
   pixel space. The ViT3D decoder goes out-of-distribution at large spatial
   token counts (1344×768 = 4032 tokens vs 768×448 = 1344), producing the
   negative DC. The early port did a single full-pass decode. Fixed: `decode`
   now tiles when `latent_h > tile` or `latent_w > tile`, porting
   `klvae._split_tiles` / `klvae.blend` / `klvae.tiled_decode` (sp_size=1,
   linear cross-fade, concat on dim −2/−1).
9. **768p needs ≥243 frames (config constraint, not a bug).** 41 frames at
   1344×768 is out-of-distribution for the temporal axis — the decoded DC is
   intrinsic to the latent content at that frame count, not a VAE structural
   defect (DiT and VAE are both resolution- and length-agnostic; RoPE is
   length-normalized; latent stats match across resolutions). Real-model
   trend: 41f DC=−1.15 (mp4 mean 12.6→52.2 after the two fixes), 89f
   DC=−0.63 (mean 76.2). The official `reproducible-768p-t2va-request.sh`
   uses 243 frames (10s @ 24fps, t=61), which fully normalizes the DC. Use
   the official frame count at 768p; 41f was only a memory shortcut.

## P10 corrections (native audio #588)

Real-model audio E2E found one wiring bug plus the joint denoise path:

1. **Engine dropped the `audio` kwarg** (`engines/video.py`).
   `VideoGenEngine.generate()` built `VideoGenParams` from `**kwargs` but never
   forwarded `audio`, so the backend always received the default `False` and ran
   video-only — the API's `audio=true` was silently discarded at the engine
   boundary. Fix: `audio=kwargs.get("audio", False)` in the params constructor
   (False default keeps backward compat).
2. **Joint denoise + AudioVAE + mux** (`generate.py::generate_t2va_av`).
   Dual scheduler (video `shift=12.0`, audio `shift=3.0`, shared step count),
   `build_t2va_av_packed` (audio rows, separate audio timestep, audio position
   grid), DiT returns `(video_output, audio_output)`, each latent stepped on its
   own scheduler; then `denormalize → audio_vae.decode → _save_audio (wav) →
   _mux_av` (ffmpeg `-c:v copy -c:a aac -shortest`). Temp video/wav removed
   after mux, final MP4 kept.
3. **AudioVAE decode-only port** (`audio_vae/`). 779 MLX weights from 914
   decode keys (135 `weight_norm` (g,v) pairs reconstructed to flat weights;
   kaiser-sinc fixed filters recomputed at init). Decode `(1,162,32) →
   (1,129600,1)` in 0.2s, finite `[-1,1]`.

## Files

```
fusion_mlx/video/minimax_h3/
├── __init__.py          # exports
├── config.py            # H3Config / H3VAEConfig / H3AudioVAEConfig / H3Partition
├── vae.py               # MiniMaxH3VideoVAE (encode/decode/encode_base)
├── transformer.py       # MiniMaxH3DiTModel (packed-scatter forward)
├── scheduler.py         # MiniMaxH3Scheduler (rectified-flow Euler, data-ward PLUS sign)
├── text_encoder.py      # MiniMaxH3TextEncoder (Qwen3-VL layer 49)
├── condition.py         # P6: packed-sequence assembly + patchify + normalize
├── generate.py          # P6 t2va video-only + P10 generate_t2va_av joint A/V + generate_video
├── audio_vae/           # P10 (#588): decode-only DAC+BigVGAN MLX port
│   ├── __init__.py      # MiniMaxH3AudioVAE: from_pretrained + decode
│   ├── bigvgan.py amp_block.py activations.py alias_free.py weight_norm.py
│   └── audio_latents.py # 32-dim latents mean/std normalize/denormalize
└── quantize.py          # 运行时量化 (in-place, DiT 8-bit / TE 4-bit)
```

Backend: `fusion_mlx/engines/video_backends/minimax_h3.py` (`MiniMaxH3Backend`).
Engine: `fusion_mlx/engines/video.py` forwards `audio` kwarg → `VideoGenParams` (#588).
Tests: `tests/unit/test_minimax_h3_{config,vae,transformer,scheduler,text_encoder,condition,generate,quantize,backend}.py` + `test_minimax_h3_audio_generate.py` (#588 joint A/V denoise loop).

## Usage

```bash
# t2va (text → video)
curl -X POST http://localhost:8000/v1/videos/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"minimax-h3","prompt":"...","num_frames":97,"width":768,"height":768,"resolution":"768p"}'
```

Partitions: `fl2va` (t2va/i2va/l2va/fl2va) and `ref2va` (multi-reference).
Resolutions: `768p`, `2k`. Max 361 frames (≤15s @24fps), `n=1`.

### Native audio (#588, P10)

`audio=true` runs the **joint t2va audio+video** path instead of video-only:
the DiT audio branch denoises 32-dim audio latents on a separate scheduler
(`shift=3.0` vs video `12.0`, shared step count), the decode-only AudioVAE
(DAC+BigVGAN MLX port, ~605 MB) turns them into a 32 kHz mono waveform, and
ffmpeg muxes video + audio into one MP4 (`h264` + `aac`, `-shortest`). Default
`audio=false` preserves the existing video-only behavior.

```bash
# t2va with native audio (text → video + audio, single muxed MP4)
curl -X POST http://localhost:8000/v1/videos/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"minimax-h3","prompt":"ocean waves crashing on rocks, with sound of water","num_frames":97,"width":768,"height":768,"quantize":"dit8_te4","audio":true}'
```

Requires `audio_vae/` weights on disk (symlink/extract `FL2VA/audio_vae/` from
the `MiniMaxAI/MiniMax-H3` repo via hf-mirror.com). Missing weights with
`audio=true` fails visibly (no silent fallback to video-only). Quantize
`dit8_te4` keeps the 33B footprint under M5 Max physical RAM for A/V runs.

Verification (real 33B, dit8_te4, 25f 512×512): ffprobe reports 2 streams
(`h264` video + `aac` 32 kHz mono audio); audio RMS ≈ 0.26 (seed 42) vs 0.40
(seed 999) — non-silent and seed-dependent, confirming per-run AudioVAE decode
rather than a static/placeholder track.

## Weight layout (expected)

```
<model_dir>/
├── transformer/    # DiT shards (~66 GB, FL2VA partition)
├── text_encoder/   # Qwen3-VL (~67 GB)
├── audio_vae/      # AudioVAE DAC+BigVGAN (~605 MB, #588)
└── video_vae/      # VAE (~11 GB)
```

Single-directory layouts are also accepted (subdir fallback in `generate.py`).

## Memory & quantization

FL2VA total weights ≈ 144 GB (TE ~67 GB + DiT ~66 GB + VAE ~11 GB), exceeding
M5 Max 137 GB physical RAM. `generate_video` uses **staged loading** to fit:
load TE → encode prompt → materialize `text_embeds` → release TE + clear Metal
cache → load DiT + VAE → denoise. `text_embeds` is only a few MB.

For official-scale resolutions (768p+), pass `quantize=` to `generate_video`
for **runtime in-place quantization** (no on-disk format change):

| `quantize`  | TE    | DiT   | ~peak RAM | Use when                       |
|-------------|-------|-------|-----------|--------------------------------|
| `"none"`    | bf16  | bf16  | ~144 GB   | default; only small configs    |
| `"te4"`     | 4-bit | bf16  | ~95 GB    | TE peak is the bottleneck      |
| `"dit8"`    | bf16  | 8-bit | ~85 GB    | DiT peak is the bottleneck     |
| `"dit8_te4`"| 4-bit | 8-bit | ~62 GB    | 768p official-scale configs    |

Quantization scheme (minimal precision loss):

- **TE** (`Qwen3-VL`, 33 B): 4-bit, group_size=64. Only produces `text_embeds`
  (an intermediate representation consumed by DiT denoising), so most
  quantization-robust. Skips `embed_tokens` + all norms.
- **DiT** (33 B): 8-bit, group_size=64. Skips F32 small layers
  (`time_embedder`, `video_patch_proj`, `audio_patch_proj`, `rope`) and output
  projections (`final_layer.video_out/audio_out`, `condition_proj`); quantizes
  `adaln_proj` / `mlp` / `attn` Linear. 8-bit on 33 B is near-lossless.
- **VAE** (2.6 B, all F32): not quantized — small, and directly outputs
  pixels (artifacts risk).

Verified: quantized 256×256 std=112.3 vs bf16 113.6 (1% drift); 512×512 and
768×448 configs that OOM-killed under bf16 run end-to-end with `dit8_te4`.

