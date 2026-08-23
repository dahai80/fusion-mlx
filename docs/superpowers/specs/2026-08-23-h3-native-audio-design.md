# MiniMax-H3 Native Audio (#588) — Design Spec

**Issue:** #588 — MiniMax-H3 native audio discarded; `generate_video` is t2va video-only, no audio track.
**Date:** 2026-08-23
**Branch:** `feat/588-h3-native-audio`
**Scope:** Full joint audio+video. Mux audio into MP4. Upstream-faithful joint packed layout. Bottom-up build (decoder → DiT → mux).

## 1. Problem

`MiniMaxH3Backend` = 33B Omni-Transformer doing joint video+audio. But `generate_video()` is t2va video-only — DiT fed `mx.zeros((1,0,32))` empty audio tokens, `_write_mp4` writes video only. Native audio branch wired structurally but fed nothing. Output = silent video.

## 2. Upstream Reference

`MiniMaxAI/MiniMax-H3` HF repo ships full audio source + weights:
- `FL2VA/audio_vae/` — `dac_audio_vae.py` (DacAudioVAE), `dac_bigvgan.py` (BigVGAN), `dac_activations.py` (SnakeBeta), `dac_alias_free_*.py` (anti-aliased activations), `dac_attn_proj.py`, `minimax_h3_audio_vae.py` (wrapper), `config.json`, `config.yaml`, `metadata.json`, `model.safetensors` (**577MB**)
- `audio_scheduler/scheduler_config.json` — `MiniMaxH3Scheduler`, shift=3.0

PyTorch/diffusers source — needs MLX port. Decode-only at inference (encoder/mean_proj/logs_proj/attn_proj = 173 unused weights).

## 3. Architecture

New MLX module `fusion_mlx/video/minimax_h3/audio_vae/` — decode-only port of upstream DAC+BigVGAN:

```
audio_vae/
  __init__.py          # MiniMaxH3AudioVAE: from_pretrained (load weights) + decode
  bigvgan.py           # BigVGAN decoder: conv_pre, 7× upsample+AMPBlock, conv_post, tanh/clamp
  amp_block.py         # AMPBlock1: SnakeBeta + anti-aliased resblocks (convs1/convs2)
  activations.py       # SnakeBeta, snake, snakebeta elementwise
  alias_free.py        # Activation1d (up→act→down), UpSample1d, DownSample1d, kaiser_sinc_filter1d
  weight_norm.py       # weight_norm reparam: reconstruct flat weight from (weight_g, weight_v)
  audio_latents.py     # 32-dim latents mean/std normalize/denormalize (config.json)
```

Decode-only: port `dec_in_proj` (Conv1d 32→2048 k1) + `BigVGAN` (914 weights, 577MB). Skip encoder/mean_proj/logs_proj/pre_block (173 unused inference weights).

Modified files:
- `condition.py` — add `build_t2va_av_packed` (audio rows, audio timestep, audio position grid), audio latents normalize/denormalize
- `generate.py` — add `generate_t2va_av` (joint denoise), audio decode, MP4 mux via ffmpeg; `generate_video` gains `audio: bool`
- `config.py` — H3AudioVAEConfig already defined; add audio latents mean/std constants
- `scheduler` — add audio scheduler instance (shift=3.0) alongside video (shift=12.0)
- video gen params / backend / `videos_routes.py` — `audio` knob

Reuse: LTX2 `audio_vae/` differs (16000Hz, different rates) — NOT direct reuse. Write fresh H3-specific module matching upstream H3 structure exactly (line-by-line verifiable).

## 4. Data Flow

Joint packed sequence (single attention document, no mask), `build_t2va_av_packed`:
```
seq = text_rows + video_rows + audio_rows
  text:   tag=1, timestep_idx=0 (video_t), pos=(arange(n_text),0,0)
  video:  tag=0, timestep_idx=0 (video_t), pos=temporal+spatial grid, origin=n_text
  audio:  tag=2, timestep_idx=1 (audio_t), pos=(t_audio_grid, 0, 0)
timestep = [video_t, audio_t]              # 2 dedup noise levels
timestep_indices: text+video→0, audio→1
adaln_indices = timestep_indices * 3 + token_tags
```

DiT `__call__` contract (already wired, transformer.py:355-408):
- `audio_embeds = audio_patch_proj(audio_hidden_states)` (32→5376), scattered into packed at audio_indices
- `adaln_indices = timestep_indices * MODALITY_NUM + token_tags` → audio rows get distinct AdaLN slot
- `final_layer` returns video_out + audio_out for all rows; `mx.take(out, audio_indices)` extracts audio
- Returns `(video_output, audio_output)` — audio_output shape `(b, n_audio, 32)`

Audio latent shape: `audio_latents = (b, 32, T_audio)`, `T_audio = ceil(num_frames/fps * sample_rate / hop_length)`, `hop_length = prod(encoder_rates) = 2·4·4·5·5 = 800`. 97f@24fps → 4.04s → 161 steps. Normalized via 32-dim latents_mean/std before DiT.

Joint denoise loop, `generate_t2va_av`:
```
init: video_latents=noise(24,t,h,w), audio_latents=noise(32,T_audio)
per step i:
  video_t = scheduler_video.sigmas[i]   # shift 12.0
  audio_t = scheduler_audio.sigmas[i]   # shift 3.0
  packed = build_t2va_av_packed(video_latents, audio_latents, text, video_t, audio_t)
  video_out, audio_out = dit(packed...)
  video_latents = scheduler_video.step(video_out, video_latents, video_t)
  audio_latents = scheduler_audio.step(audio_out, audio_latents, audio_t)
after loop:
  video: denormalize → video_vae.decode → frames
  audio: denormalize → audio_vae.decode → waveform (1ch, 32000Hz)
  mux: frames + waveform → MP4 (ffmpeg subprocess, not cv2)
```

MP4 mux: replace `_write_mp4` (cv2 video-only) with ffmpeg: `-i video -i audio.wav -c:v libx264 -c:a aac -shortest output.mp4`. ffmpeg is existing fusion-mlx dependency. Verify audio non-silent via `ffmpeg signalstats`/waveform RMS (per CLAUDE.md "只查日志，不读图片").

Key uncertainty (real-model resolves): whether video+audio share step count/sigma count, or audio needs own step count. config `sample_steps=40` global, `audio_shift=3.0`. Design uses same step count, separate shift — matches upstream dual-scheduler pattern.

## 5. AudioVAE Decoder Port

Upstream `DacAudioVAE.decode(z)`:
- Input `z`: `[B, D=32, T]` continuous latent
- `dec_in_proj`: Conv1d(32→2048, kernel 1)
- `decoder = BigVGAN(h)`:
  - `conv_pre`: Conv1d(2048→1024, k7, pad3)
  - 7× upsample: ConvTranspose1d, rates [5,5,2,2,2,2,2], kernels [9,9,4,4,4,4,4], channels halve each stage 1024→512→256→128→64→32→16→8
  - Each upsample stage: 3× AMPBlock1 (kernels [3,7,11], dilations [1,3,5]) averaged
  - `activation_post`: SnakeBeta + anti-aliased
  - `conv_post`: Conv1d(8→1, k7, pad3), no bias (use_bias_at_final=False)
  - clamp [-1,1] (use_tanh_at_final=False)
- Output: `[B, 1, T·prod(decoder_rates)]` = `[B, 1, T·800]` waveform

Weight form (1087 total, 914 decode):
- `weight_norm` reparam: `weight_g` (norm direction) + `weight_v` → reconstruct `weight = weight_g * weight_v / ||weight_v||` at load
- Fixed buffers (not learned): `upsample.filter`, `downsample.lowpass.filter` = kaiser_sinc_filter1d — recomputed at init from (cutoff, half_width, kernel_size)
- `act.alpha`, `act.beta`: SnakeBeta params `[channels]`

MLX mapping:
- `nn.Conv1d` → `nn.Conv1d` (MLX supports 1d)
- `ConvTranspose1d` → `mx.conv_transpose1d`
- `weight_norm` → custom: load g+v, compute flat weight, store as single `weight` (no runtime reparam needed — inference only)
- `F.pad(mode="replicate")` → `mx.pad` with replicate mode
- `F.conv1d(groups=C)` (depthwise filter) → `mx.conv1d` with groups
- kaiser_window → compute via scipy or manual bessel (fixed filter, computed once at init on CPU)

## 6. Testing

Layer 1 — AudioVAE decoder unit (no model):
- Load 577MB `model.safetensors`, build `MiniMaxH3AudioVAE`, verify all 914 decode weights mapped (weight_norm reconstruct, filters recompute)
- Decode fixed latent `mx.zeros((1,32,161))` → verify shape `(1,1,128800)`, finite, no NaN
- Upsample ratio: T=161 → 161·800 = 128800 samples ≈ 4.04s@32000Hz, matches video duration
- SnakeBeta, kaiser_sinc_filter, weight_norm reconstruct numerical unit tests

Layer 2 — packed sequence unit (no model):
- `build_t2va_av_packed` shape contract: timestep `(2,)`, timestep_indices sum correct, adaln_indices range `[0, 2*3+2]`, no overlap
- audio position grid, latent normalize round-trip (normalize→denormalize = identity)

Layer 3 — real-model E2E (CLAUDE.md mandatory):
- `start.sh start`. Download `audio_vae/model.safetensors` via hf-mirror.com.
- `generate_t2va_av(prompt, num_frames=97, fps=24, audio=True)` → MP4
- Verify: ffprobe shows video+audio streams; ffmpeg signalstats audio RMS > silence threshold (non-zero energy); video brightness check (existing)
- Iterate on failure: if audio silent/noise → investigate audio position grid, scheduler shift, latent normalize. Real-model diagnostic loop.

Layer 4 — API:
- `videos_routes` with `audio=true` → MP4 with audio; `audio=false` → video-only (existing behavior preserved)
- Default `audio=false` (backward compat — no change to existing H3 callers)

Cleanup: after test, remove downloaded test artifacts, keep final MP4 + logs.

## 7. Error Handling & Defaults

- `audio=False` default: existing video-only path untouched. `generate_t2va_video` unchanged. Audio = separate new path.
- Missing `audio_vae/` weights on disk + `audio=True` → fail-visible error with download instructions (hf-mirror.com path). No silent fallback to video-only (Rule 12).
- Audio decode OOM/shape mismatch → fail-visible with latent shape.
- ffmpeg not in PATH → fail at startup, not silent audio loss.
- Audio duration ≠ video duration → truncate/pad audio to video duration (ffmpeg `-shortest`), log mismatch warning.

## 8. Out of Scope

- Audio encoder (inference unused)
- i2va/l2va/fl2va audio (t2va only — matches existing video-only scope)
- audio prompt conditioning / reference audio (ref2va) — separate Ref2VA pipeline
- audio synthesis quality tuning — verify non-silent + correct shape only

Risk register (UNVERIFIED, real-model resolves):
1. audio token count / T_audio derivation (hop=800 assumption)
2. audio position_ids layout
3. whether video+audio share step count/sigma count
4. audio latent normalization (config.json mean/std)

## 9. Implementation & real-model verification (post-implementation)

All four risk-register items resolved by the real 33B run. Resolutions:

1. **T_audio derivation** — correct: `audio_latent_steps(num_frames, fps) = ceil(num_frames/fps * 32000 / 800)`, hop=800 = `prod(encoder_rates)`. 25f@24fps → 4 audio latent steps → 33792 decoded samples.
2. **Shared step count, separate shift** — confirmed. Both schedulers run `num_inference_steps` steps; video `shift=12.0`, audio `shift=3.0`. Note: `sigma[0]=1.0` (linspace starts at 1.0) → `timestep=0.0` for BOTH at step 0; shifts diverge only from step 1 onward.
3. **Latent normalization** — 32-dim `LATENTS_MEAN/STD` from `audio_vae/config.json`; normalize before DiT, denormalize before decode.

**Real bug found and fixed (the actual blocker):** the `audio` knob was plumbed
through Layers 4/5 (VideoGenParams field, backend `audio=params.audio`, API
`request.audio → gen_kwargs["audio"]`), but `VideoGenEngine.generate()` in
`engines/video.py` — the layer that builds `VideoGenParams` from `**kwargs` —
**never forwarded `audio`**. The backend therefore always received the default
`False` and ran video-only; `audio=true` was silently discarded at the
engine boundary. Fix: one line, `audio=kwargs.get("audio", False)` in the
`VideoGenParams(...)` constructor (False default = backward compat). This was
not in the original design's data-flow section — discovered via real-model
E2E producing a video-only MP4.

**Verification (real 33B FL2VA, `quantize=dit8_te4`, 25f 512×512, 5 steps):**
- `ffprobe`: 2 streams — `h264` video (512×512, 25 frames, 24fps) + `aac` audio (32 kHz mono).
- Audio loudness (`ffmpeg volumedetect`): mean_volume ≈ −11.8 dB, max 0.0 dB — non-silent.
- Two seeds differ: seed 42 RMS=0.2572, seed 999 RMS=0.4010 → per-run AudioVAE decode, not a static/placeholder track (per CLAUDE.md "只查日志，不读图片").
- 155 H3 cluster tests pass, 0 regressions. 2 new audio-loop unit tests pass.

Final artifacts kept: the two seed MP4s + server logs; temp video/wav removed.
