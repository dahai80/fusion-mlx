# MiniMax-H3 (33B Omni-Transformer)

MiniMax-H3 is a single-flow packed-sequence DiT that jointly generates video and
audio from text. Unlike cross-attention DiTs, H3 scatters text/video/audio rows
into **one** packed sequence and runs full self-attention over all of it
(`is_causal=False`, no mask). This is a pure-MLX port in `fusion_mlx/video/minimax_h3/`.

> **Status: P0–P6 landed, P8 real-model E2E VERIFIED.** Config / VAE / DiT /
> scheduler / text-encoder / backend+registry are complete, and the **t2va
> video-only** packed-sequence E2E path is wired **and verified against the real
> 33B FL2VA weights** (DiT 534/536 params matched, VAE 559/560; 2-step 256×256
> bf16 smoke produces non-trivial frames: frame0 std=115.7, min=0, max=255).
> The text_encoder (Qwen3-VL, 67 GB) remains deferred; E2E currently uses a
> fake text embedding for the DiT.

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
├── generate.py          # P6: t2va video-only denoise loop + generate_video
└── quantize.py          # 运行时量化 (in-place, DiT 8-bit / TE 4-bit)
```

Backend: `fusion_mlx/engines/video_backends/minimax_h3.py` (`MiniMaxH3Backend`).
Tests: `tests/unit/test_minimax_h3_{config,vae,transformer,scheduler,text_encoder,condition,generate,quantize,backend}.py`.

## Usage

```bash
# t2va (text → video)
curl -X POST http://localhost:8000/v1/videos/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"minimax-h3","prompt":"...","num_frames":97,"width":768,"height":768,"resolution":"768p"}'
```

Partitions: `fl2va` (t2va/i2va/l2va/fl2va) and `ref2va` (multi-reference).
Resolutions: `768p`, `2k`. Max 361 frames (≤15s @24fps), `n=1`.

## Weight layout (expected)

```
<model_dir>/
├── transformer/    # DiT shards (~66 GB, FL2VA partition)
├── text_encoder/   # Qwen3-VL (~67 GB)
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

