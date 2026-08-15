# MiniMax-H3 (33B Omni-Transformer)

MiniMax-H3 is a single-flow packed-sequence DiT that jointly generates video and
audio from text. Unlike cross-attention DiTs, H3 scatters text/video/audio rows
into **one** packed sequence and runs full self-attention over all of it
(`is_causal=False`, no mask). This is a pure-MLX port in `fusion_mlx/video/minimax_h3/`.

> **Status: P0–P6 landed.** Config / VAE / DiT / scheduler / text-encoder /
> backend+registry are complete, and the **t2va video-only** packed-sequence E2E
> path is wired. Real-model E2E (P8) is pending the 144 GB FL2VA weight set
> (transformer 66 GB + text_encoder 67 GB + VAE 11 GB); the code path is
> unblocked once weights are present.
>
> **The packed-layout assembly is inferred (UNVERIFIED)** — see the contract note
> below. It must be corrected against the real model before shipping.

## Phase map

| Phase | Scope | Status |
|---|---|---|
| P0 | H3Config / H3VAEConfig / H3AudioVAEConfig / H3Partition | ✅ |
| P1 | Video VAE (spatial 16×, temporal 4×, z=24, ViT3D decoder) | ✅ |
| P2 | DiT transformer (packed-scatter forward, AdaLN 3-modality table) | ✅ |
| P3 | Rectified-flow scheduler (reversed velocity, two shifts) | ✅ |
| P4 | Text encoder (Qwen3-VL, layer 49 hidden states) | ✅ |
| P5 | Backend + BACKENDS registry + constraints | ✅ |
| **P6** | **t2va video-only packed-sequence assembly + denoise loop** | **✅ (inferred)** |
| P7 | Prompt skill (h3-prompt-writing Context-IR) | deferred |
| P8 | Real-model E2E | pending weights |
| P9 | Tests + real-model E2E validation | pending weights |

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

`MiniMaxH3Scheduler` — rectified-flow Euler, `eta=0`. Three incompatibilities
with `FlowMatchEulerDiscreteScheduler` (all verified against the diffusers
source):

1. **Velocity sign reversed**: `x0 = x_t + sigma·v` (plus, not minus).
2. **`t = 1 - sigma`**, t=1 is clean; `timesteps = 1 - sigmas[:-1]`.
3. **sigma grid** `linspace(1,0,N)` + exponential shift
   `s·σ/(1+(s-1)σ)` + `_unique_consecutive` fold.

Euler step: `denoised = sample + sigma·output; prev = ratio·sample + (1-ratio)·denoised`
where `ratio = sigma_next/sigma`. Two instances: video `shift=12.0`,
audio `shift=3.0`.

## Files

```
fusion_mlx/video/minimax_h3/
├── __init__.py          # exports
├── config.py            # H3Config / H3VAEConfig / H3AudioVAEConfig / H3Partition
├── vae.py               # MiniMaxH3VideoVAE (encode/decode/encode_base)
├── transformer.py       # MiniMaxH3DiTModel (packed-scatter forward)
├── scheduler.py         # MiniMaxH3Scheduler (reversed-velocity Euler)
├── text_encoder.py      # MiniMaxH3TextEncoder (Qwen3-VL layer 49)
├── condition.py         # P6: packed-sequence assembly + patchify + normalize
└── generate.py          # P6: t2va video-only denoise loop + generate_video
```

Backend: `fusion_mlx/engines/video_backends/minimax_h3.py` (`MiniMaxH3Backend`).
Tests: `tests/unit/test_minimax_h3_{config,vae,transformer,scheduler,text_encoder,condition,generate,backend}.py`.

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
