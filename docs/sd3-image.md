# SD3-Medium Image Generation (MLX)

Native MLX port of **Stable Diffusion 3 Medium** txt2img for fusion-mlx.
Unlike the Flux variants (which wrap `mflux`), SD3 ships as a from-scratch
MLX pipeline under `fusion_mlx/image/sd3/`.

## Architecture

SD3-Medium is a **Multimodal Diffusion Transformer (MMDiT)** with a
**rectified-flow** scheduler and a **CLIP-L + CLIP-G + T5-XXL** tri-encoder
text-conditioning stack.

| Component | Detail |
|-----------|--------|
| Transformer (`mmdit.py`) | 24 joint MMDiT blocks, inner_dim=1536, 24 heads × 64, joint_attention_dim=4096 (T5), pooled_projection_dim=2048 (CLIP-L 768 ⊕ CLIP-G 1280), patch_size=2, in/out_channels=16 |
| VAE (`vae.py`) | AutoencoderKL, block_out_channels=[128,256,512,512], scaling=1.5305, shift=0.0609, spatial_scale=8, latent_channels=16 |
| Scheduler (`scheduler.py`) | FlowMatchEuler with dynamic exponential time-shift (`base_shift=0.5`, `max_shift=1.15`) |
| Text encoders (`text_encoder.py`, `generate.py`) | CLIP-L (768/12/12, quick_gelu), CLIP-G (1280/32/**20 heads**/5120, gelu), T5-v1_1-XXL (4096, 24 blocks) |

### Text-encoder reuse strategy

- **CLIP-L** — reuses `mflux`'s `CLIPEncoder` (hardcoded 768/12, EOS pooled).
- **T5-XXL** — reuses `mflux`'s `T5Encoder` (hardcoded 4096, 24 blocks, 64-dim heads). HF `encoder.block.{i}.*` weights are remapped to mflux's `t5_blocks.{i}.*` naming by `_map_t5` in `generate.py`.
- **CLIP-G** — written from scratch in `text_encoder.py` (`CLIPTextModel`, parametrized) because mflux's CLIP is hardcoded to 768 dims. CLIP-G uses **20 heads** (head_dim=64) per the StabilityAI diffusers config.

### Pooled / context construction

- `pooled = concat(CLIP-L EOS_pooled [768], CLIP-G EOS_pooled [1280])` → **2048** (feeds `y_embedder`).
- `context = T5-XXL last_hidden_state [4096]` → `context_embedder` (Linear 4096→1536).

## Weights

| Asset | Source repo | Notes |
|-------|-------------|-------|
| Transformer + VAE | `argmaxinc/mlx-stable-diffusion-3-medium` (`sd3_medium.safetensors`) | ComfyUI naming (`model.diffusion_model.*`, `first_stage_model.*`). Loaded via `weights.load_transformer` / `weights.load_vae`. |
| CLIP-L | `frankjoshua/stable-diffusion-3-medium-diffusers` (`text_encoder/`) | HF `text_model.*` naming, loads directly. Cast to fp32 (mflux CLIPEncoder float32 mask must promote under fp16 weights). |
| CLIP-G | same repo (`text_encoder_2/`) | HF `text_model.*` naming. |
| T5-XXL | same repo (`text_encoder_3/`) | Sharded `model-0000{1,2}-of-00002.safetensors` + index (fp16). |

### fp8 T5 (single-file)

T5-XXL is also supported as a single **fp8 e4m3fn** file
(`t5xxl_fp8_e4m3fn.safetensors`, ~4.9 GB — the common ComfyUI distribution).
When present it is decoded via `torch.float8_e4m3fn → float32 → bfloat16` and
preferred over the sharded fp16 path; if absent the loader falls back to
shards automatically. This is what the validated E2E used (SD3.5-medium
cache ships the same encoders as SD3-medium).

### Local / offline weight override

All encoder + tokenizer resolutions honor `SD3_LOCAL_DIR`: when set, files
are read directly from that directory (plain folder layout, no HF blob
cache) before falling back to `hf_hub_download`. Additional env overrides:

| Env var | Purpose |
|---------|---------|
| `SD3_LOCAL_DIR` | Root dir of a local SD3 weight tree (encoders + tokenizers) |
| `SD3_ENCODERS_REPO` | Override the HF encoders repo |
| `SD3_CLIP_L_SUBFOLDER` / `SD3_CLIP_L_FILE` | CLIP-L weight location |
| `SD3_CLIP_G_SUBFOLDER` / `SD3_CLIP_G_FILE` | CLIP-G weight location |
| `SD3_T5_SUBFOLDER` | T5 shard subfolder (fp16 fallback) |
| `SD3_T5_FP8_SUBFOLDER` / `SD3_T5_FP8_FILE` | fp8 T5 single-file location |

Conv weights are transposed PyTorch `(out,in,kH,kW)` → MLX `(out,kH,kW,in)`;
attention 1×1 convs `(c,c,1,1)` are squeezed to `(c,c)` for `nn.Linear`.

## Usage

### Via the image engine (`/v1/images/generate`)

Set `model` to an SD3 identifier; the variant is auto-detected (`sd3`,
`stable-diffusion-3`, etc.):

```json
{
  "model": "sd3-medium",
  "prompt": "a photo of a cat",
  "width": 1024,
  "height": 1024,
  "steps": 28,
  "guidance": 4.0,
  "negative_prompt": "blurry, low quality"
}
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `steps` | 28 recommended | SD3 needs more steps than Flux-schnell |
| `guidance` | 4.0 | CFG scale |
| `negative_prompt` | `""` | Supported (unlike Flux); used for CFG |
| `shift` | auto (dynamic) | Override the rectified-flow time-shift |

### Direct pipeline

```python
from fusion_mlx.image.sd3.generate import SD3Pipeline

pipe = SD3Pipeline(model_path="sd3-medium", quantize=None)
result = pipe.generate_image(
    seed=42,
    prompt="a photo of a cat",
    num_inference_steps=28,
    height=1024,
    width=1024,
    guidance=4.0,
    negative_prompt="blurry",
)
result.image.save("out.png")  # PIL.Image
```

`quantize=8` quantizes the transformer (and T5) to 8-bit.

## File map

```
fusion_mlx/image/sd3/
├── config.py        # SD3Config, SD3ModelPaths, ClipL/GConfig, VARIANTS
├── mmdit.py         # MMDiT transformer (24 joint blocks)
├── vae.py           # AutoencoderKL (SD3-specific up-block order)
├── scheduler.py     # FlowMatchEuler + calculate_shift
├── text_encoder.py  # Parametrized CLIPTextModel (CLIP-G)
├── weights.py       # remap_transformer_weights / remap_vae_weights / load_*
└── generate.py      # SD3Pipeline (loads all, encode, denoise loop, decode)
```

Integration: `fusion_mlx/engines/image_gen.py` `VARIANT_MAP["sd3"]` →
`SD3Pipeline`; native (non-mflux) variants skip the mflux `ModelConfig`
factory and carry their own config object.

## Gotchas

- **CLIP-G has 20 heads** (not 16) — head_dim=64, per the StabilityAI
  diffusers `text_encoder_2/config.json`. mflux's CLIP (hardcoded 768/12) cannot be reused for CLIP-G.
- **SD3 VAE up-block order is reversed vs Flux VAE**: SD3 `up.0` is the
  shallowest (no upsample), `up.3` is the deepest (with upsample). Cannot
  reuse the mflux/Flux VAE class.
- **`safe_open` iteration**: use `f.keys()` (not `for k in f`) — `safe_open`
  objects are not iterable in safetensors ≥ 0.8.
- **CLIP-L fp16 mask promotion**: mflux's `CLIPEncoder` builds a float32
  causal mask; with fp16 weights `scaled_dot_product_attention` raises
  "Mask type must promote to output type float16". CLIP-L weights are cast
  to fp32 on load (196 keys, ~235 MB — negligible). CLIP-G uses the
  in-tree `CLIPTextModel` whose mask is cast to `hidden.dtype`, so it stays
  native precision.
- **fp8 T5 has no native MLX dtype**: MLX reads `F8_E4M3` tensors as raw
  `uint8`; decoding goes through `torch.float8_e4m3fn → float32` (torch is
  the authority) then to `bfloat16` for the mflux `T5Encoder`.
- **Rectified-flow runs N+1 steps**: `FlowMatchEulerScheduler` produces
  `N+1` timesteps (sigmas `linspace(1,0,N+1)` time-shifted + appended 0);
  the denoise loop iterates all of them — `step {N+1}/{N+1}` is expected,
  not an off-by-one.
- **fp16 mask**: the CLIP causal mask must be cast to `hidden.dtype` for
  `scaled_dot_product_attention` (weights ship as fp16).
