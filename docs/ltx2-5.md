# LTX-2.5 Video Generation (MLX)

Native MLX port of **LTX-2.5** (22B distilled/dev audio-video DiT) for
fusion-mlx. Shipped as a fully self-contained module under
`fusion_mlx/video/ltx2_5/` — **no `from ..ltx2 import`** dependency. The 2.5
transformer is its own independent port even though LTX-2's `LTXModel` shares
the same skeleton, because the 22B checkpoint introduces four structural
deltas that a shared module would average and break (Rule 7 — surface
conflicts, don't average them).

## Architecture

LTX-2.5 22B is an **LTX-2 audio-video DiT + 4 deltas**. All deltas below are
code-verified against the real 22B-distilled checkpoint (4349 keys).

| Component | Detail |
|-----------|--------|
| Transformer (`transformer.py`, `ltx2_5_model.py`) | 48 `BasicAVTransformerBlock`s, video 32 heads × 128 d_head = 4096 inner, audio 32 heads × 64 d_head = 2048 inner, timestep_proj_dim=256, caption_channels=3840 |
| Connectors (`embeddings_connector.py`) | **NEW** — `Embeddings1DConnector` × 2 (video dim 4096, audio dim 2048), each 8×`_BasicTransformerBlock1D` + learnable_registers [128, dim] = 129 keys × 2 = 258 keys |
| AdaLN (`adaln.py`) | `adaln_single` coeff=9, `prompt_adaln_single` coeff=2, `audio_adaln_single` coeff=9, `audio_prompt_adaln_single` coeff=2, av_ca scale_shift coeff=4, av_ca gate coeff=1 |
| Feed-forward (`feed_forward.py`) | **ff_bias asymmetry** — video FF `bias=False`, audio FF `bias=True` (96-key difference) |
| Attention (`attention.py`) | `has_gate_logits=True` on all 6 attn modules per block → `to_gate_logits = Linear(query_dim, heads)`, gate = `2.0 * sigmoid(to_gate_logits(x))` = 12 gate keys/block |
| RoPE (`rope.py`) | Shared with LTX-2 (same shapes), interleaved + split variants, no 2.5-specific change |
| Positions | `create_position_grid` shape `(B, 3, num_patches, 2)`, bfloat16 |

### The four deltas vs LTX-2

1. **258 connector keys** — `video_embeddings_connector` (dim 4096) +
   `audio_embeddings_connector` (dim 2048). Each is 8 `_BasicTransformerBlock1D`
   blocks + a `learnable_registers` buffer `[128, inner_dim]`. LTX-2 has none.
2. **`keyframes_abs_pos_embedding`** `[1, 4096]` — a single absolute position
   embedding for keyframes.
3. **ff_bias asymmetry** — video feed-forward has NO bias, audio feed-forward
   HAS bias. Threading `ff_bias` / `audio_ff_bias` through `BasicAVTransformerBlock`
   is the core structural change.
4. **`has_prompt_adaln=True`** — `prompt_adaln_single` (coeff=2) present, and
   `to_gate_logits` on all 6 attention modules per block (12 gate keys/block).

### Latent flatten convention

Latents `(b, c, f, h, w)` are flattened to token sequence before entering
`patchify_proj`, matching LTX-2's `denoise.py`:

```
latents (b, c, f, h, w)
  → reshape (b, c, -1)
  → transpose (0, 2, 1)
  → (b, num_tokens, c)   # fed to patchify_proj
```

## Weights

| Asset | Source repo | Path |
|-------|-------------|------|
| Transformer | `Lightricks/LTX-2.5` | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` |

Single-file `.safetensors`, all 4349 keys under `model.diffusion_model.` prefix.
Download via the mirror site `https://hf-mirror.com` per project convention.

## Key-loading remaps (`sanitize`)

PyTorch checkpoint naming → MLX module naming. **Critical 2.5 delta: connectors
are NOT skipped** (LTX-2's `sanitize` skips connector keys — 2.5 must keep them).

| Checkpoint key | MLX key |
|----------------|---------|
| `model.diffusion_model.` prefix | stripped |
| `attn1.to_out.0.weight` (Sequential) | `attn1.to_out.weight` (Linear) |
| `ff.net.0.proj.` | `ff.proj_in.` |
| `ff.net.2.` | `ff.proj_out.` |
| `timestep_embedder.linear_1.` | `timestep_embedder.linear1.` |
| `timestep_embedder.linear_2.` | `timestep_embedder.linear2.` |
| `video_embeddings_connector.*` | kept (NOT skipped) |
| `audio_embeddings_connector.*` | kept (NOT skipped) |
| `keyframes_abs_pos_embedding` | kept |

Silent zero-init on key mismatch is the #1 risk — `from_pretrained(..., strict=True)`
audits unmatched/missing and raises on any mismatch. Verified: 0 unmatched,
0 missing against the real 22B-distilled checkpoint.

## Cross-module enum caveat

`ltx2_5.LTXModelType.AudioVideo == ltx2.LTXModelType.AudioVideo` is **False** —
they are different classes with the same value strings. Tests and downstream
code MUST import enums from `fusion_mlx.video.ltx2_5.config`, not
`fusion_mlx.video.ltx2.config`.

## Status

- **Structural port: LANDED.** Strict-load 0 unmatched / 0 missing verified
  against real 22B-distilled weights. Forward smoke finite (min -2.92,
  max 3.03, no NaN). 134 tests pass, ruff clean.
- **E2E generation: PENDING.** `backend.generate` raises `NotImplementedError`
  (fail visible — Rule 12). Wiring the connectors into the conditioning
  pipeline, the Gemma4-12b text encoder, and the duration-head is a later phase.

## Tests

| File | Covers |
|------|--------|
| `tests/unit/test_ltx2_5_transformer.py` | block key count = 84, ff_bias asymmetry, 12 gate logits, connector 129×2, model 4349, sanitize remaps (incl. no-skip connectors), real-weight strict-load (skipped if checkpoint absent) |
| `tests/unit/test_ltx2_5_config.py` | config deltas (has_prompt_adaln, ff_bias, connectors) |
| `tests/unit/test_ltx2_5_reuse.py` | independence assertion (NOT subclass of `LTXModel`) |
| `tests/unit/test_ltx2_5_backend.py` | backend registration + aliases |
| `tests/unit/test_ltx2_5_duration_head.py` | duration-head exp()=seconds |
| `tests/unit/test_ltx2_5_text_encoder.py` | text-encoder wiring |
| `tests/unit/test_ltx2_5_upsampler.py` | upsampler |

Real-weight tests are marked `@pytest.mark.realmodel` and skip if the 68GB
checkpoint is not present locally.
