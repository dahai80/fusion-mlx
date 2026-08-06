# SDXL Image Generation (MLX)

Native MLX port of **Stable Diffusion XL** (SDXL 1.0 base) txt2img for
fusion-mlx. Like SD3, SDXL ships as a from-scratch MLX pipeline under
`fusion_mlx/image/sdxl/` — it does **not** wrap `mflux`.

## Architecture

SDXL is a **UNet2DConditionModel** (convolutional, not transformer-backbone)
with an **epsilon-prediction Euler-discrete** scheduler and a **dual
CLIP text-encoder** stack.

| Component | Detail |
|-----------|--------|
| UNet (`unet.py`) | block_out_channels=(320,640,1280), attention_head_dim=(5,10,20)→64 ch/head, cross_attention_dim=2048, transformer_layers_per_block=(1,2,10), in/out_channels=4, sample_size=128, addition_time_embed_dim=256, projection_class_embeddings_input_dim=2816, use_linear_projection=true |
| VAE (`vae.py`) | AutoencoderKL, block_out_channels=(128,256,512,512), latent_channels=4, scaling_factor=0.13025, **no shift_factor** (unlike SD3) |
| Scheduler (`scheduler.py`) | EulerDiscrete, beta_schedule=scaled_linear (0.00085→0.012), 1000 train steps, prediction_type=epsilon, timestep_spacing=leading, steps_offset=1 |
| Text encoders (`text_encoder.py`) | CLIP-L (768/12 heads/12 layers/3072/quick_gelu), OpenCLIP-G (1280/20 heads/32 layers/5120/gelu, projection_dim=1280) |

### Dual text-encoder conditioning

- **cross-attn context** = `concat(CLIP-L hidden [768], CLIP-G hidden [1280])` → **2048** (feeds every `attn2` cross-attention in the UNet).
- **pooled / add embed** = CLIP-G EOS pooled [1280] only (CLIP-L pool is not used for SDXL).
- **add_text_embed** = `concat(pooled [1280], time_embeds [6×256=1536])` → **2816** → `add_embedding` (TimestepEmbedding).

`time_embeds` comes from the SDXL `add_time_proj` over the 6-value
`original_size / crop / target_size` time-ids vector, embedded at
`addition_time_embed_dim=256` each.

### Linear-projection transformer blocks

SDXL uses `use_linear_projection=true`, so each `Transformer2D` block's
`proj_in`/`proj_out` are **`nn.Linear`** (not Conv1x1). The HF weight keys
`ff.net.0.proj.*` / `ff.net.2.*` (GEGLU) are remapped to the module's
`ff.net_0_proj` / `ff.net_2` naming by `_map_key` in `weights.py`.

## Weights

| Asset | Source repo | File | Notes |
|-------|-------------|------|-------|
| UNet | `stabilityai/stable-diffusion-xl-base-1.0` | `unet/diffusion_pytorch_model.fp16.safetensors` (5.14 GB) | diffusers `down_blocks.*` / `up_blocks.*` / `mid_block` naming; loaded via `weights.load_unet`. |
| VAE | same repo | `vae/diffusion_pytorch_model.fp16.safetensors` (~335 MB) | diffusers `down_blocks` / `up_blocks` / `resnets` / `downsamplers` / `upsamplers` / `conv_shortcut` naming (NOT mflux's `down`/`up`/`nin_shortcut`). |
| CLIP-L | same repo | `text_encoder/model.fp16.safetensors` (~246 MB) | HF `text_model.*` naming; cast to fp32 (causal-mask promotion under fp16 weights). |
| OpenCLIP-G | same repo | `text_encoder_2/model.fp16.safetensors` (1.39 GB) | HF `text_model.*` + `text_projection` (Linear, bias=False, 1280→1280). |
| Tokenizers | same repo | `tokenizer/`, `tokenizer_2/` | Two CLIPTokenizers; both tokenized at max_length=77. |

Conv weights are transposed PyTorch `(out,in,kH,kW)` → MLX
`(out,kH,kW,in)`. The top-level `conv_in.weight` / `conv_out.weight` keys
(handled specially — they lack a leading `.` separator that the nested
conv suffixes have) plus all nested conv keys are covered by
`_is_conv_key`.

### Local / offline weight override

All component resolutions honor `SDXL_LOCAL_DIR`: when set, files are read
directly from that directory before falling back to `hf_hub_download`.
Additional env overrides:

| Env var | Purpose |
|---------|---------|
| `SDXL_LOCAL_DIR` | Root dir of a local SDXL weight tree |
| `SDXL_REPO` | Override the HF repo id |
| `SDXL_UNET_SUBFOLDER` / `SDXL_UNET_FILE` | UNet weight location |
| `SDXL_VAE_SUBFOLDER` / `SDXL_VAE_FILE` | VAE weight location |
| `SDXL_CLIP_L_SUBFOLDER` / `SDXL_CLIP_L_FILE` | CLIP-L weight location |
| `SDXL_CLIP_G_SUBFOLDER` / `SDXL_CLIP_G_FILE` | OpenCLIP-G weight location |
| `SDXL_TOKENIZER_SUBFOLDER` / `SDXL_TOKENIZER_2_SUBFOLDER` | Tokenizer subfolders |

## Variants

The SDXL pipeline covers three model-family variants, all served by the
same `SDXLPipeline` class:

| Variant | Auto-detected from model name | Default guidance |
|---------|-------------------------------|------------------|
| `sdxl` | `sdxl`, `stable-diffusion-xl` | 7.5 |
| `cosxl` | `cosxl` (CosXL-Edit) | 7.5 |
| `sdxs` | `sdxs` (SDXS one-step distill) | 4.0 |

## Usage

### Via the image engine (`/v1/images/generate`)

Set `model` to an SDXL-family identifier; the variant is auto-detected:

```json
{
  "model": "stable-diffusion-xl-base-1.0",
  "prompt": "a photo of a cat",
  "width": 1024,
  "height": 1024,
  "steps": 30,
  "guidance": 7.5,
  "negative_prompt": "blurry, low quality"
}
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `steps` | 30 recommended | Euler-discrete denoise steps |
| `guidance` | 7.5 (sdxl/cosxl) / 4.0 (sdxs) | CFG scale |
| `negative_prompt` | `""` | Supported; used for CFG (unlike Flux) |
| `width` / `height` | 1024 / 1024 | Must be multiples of 8 |

### Direct pipeline

```python
from fusion_mlx.image.sdxl.generate import SDXLPipeline

pipe = SDXLPipeline(model_path="stable-diffusion-xl-base-1.0", quantize=None)
result = pipe.generate_image(
    seed=42,
    prompt="a photo of a cat",
    num_inference_steps=30,
    height=1024,
    width=1024,
    guidance=7.5,
    negative_prompt="blurry",
)
result.image.save("out.png")  # PIL.Image
```

`quantize=8` quantizes the UNet to 8-bit (group_size=64).

## File map

```
fusion_mlx/image/sdxl/
├── config.py        # SDXLConfig, SDXLUNetConfig, SDXLVAEConfig,
│                    #   SDXLTextEncoderConfig, SDXLModelPaths
├── unet.py          # SDXLUNet (UNet2DConditionModel, linear-projection)
├── vae.py           # SDXLVAE (AutoencoderKL, diffusers naming)
├── scheduler.py     # SDXLEulerDiscreteScheduler (epsilon, leading)
├── text_encoder.py  # SDXLCLIPTextModel (parametrized; CLIP-L + CLIP-G)
├── weights.py       # remap_unet_weights / remap_vae_weights / load_*
└── generate.py      # SDXLPipeline (dual-encode, CFG loop, decode)
```

Integration: `fusion_mlx/engines/image_gen.py` `VARIANT_MAP["sdxl"|"cosxl"|"sdxs"]`
→ `SDXLPipeline`; native (non-mflux) variants skip the mflux `ModelConfig`
factory and carry their own config object.

## Gotchas

- **`use_linear_projection=true`**: SDXL `Transformer2D` `proj_in`/`proj_out`
  are `nn.Linear`, so the latents are reshaped `(b,h,w,c)→(b,h*w,c)` before
  projection. Do not use Conv1x1 here (unlike older SD 1.x UNets).
- **VAE naming**: the SDXL VAE mirrors **diffusers** naming
  (`down_blocks`/`up_blocks`/`resnets`/`downsamplers`/`conv_shortcut`),
  **not** mflux's Flux-VAE naming (`down`/`up`/`nin_shortcut`/`block`).
  Reusing the SD3/Flux VAE class would silently mis-load weights.
- **No VAE shift_factor**: unlike SD3, the SDXL VAE decode is
  `latents / scaling_factor` only (no `* (1+shift) + shift`).
- **CLIP-L fp32 cast**: the CLIP-L causal mask is float32; under fp16
  weights `scaled_dot_product_attention` raises "Mask type must promote to
  output type float16", so CLIP-L weights are cast to fp32 on load.
  CLIP-G's in-tree mask is cast to `hidden.dtype` so it stays native.
- **OpenCLIP-G text_projection**: CLIP-G has an extra `text_projection`
  Linear (1280→1280, bias=False) applied to the pooled output; CLIP-L has
  none. `SDXLCLIPTextModel` only builds it when `projection_dim` is set.
- **`safe_open` iteration**: use `f.keys()` (not `for k in f`) — `safe_open`
  objects are not iterable in safetensors ≥ 0.8.
- **Top-level conv keys**: `conv_in.weight` / `conv_out.weight` lack the
  leading `.` that nested conv suffixes have, so `_is_conv_key` checks
  both the dotted suffixes and the bare top-level keys for transpose.
