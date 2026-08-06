# Stable Cascade Image Generation (MLX)

Native MLX port of **Stable Cascade** (Würstchen 3-stage) txt2img for
fusion-mlx. Like SDXL/SD3, Stable Cascade ships as a from-scratch MLX
pipeline under `fusion_mlx/image/cascade/` — it does **not** wrap
`mflux`.

## Architecture

Stable Cascade is a **3-stage cascade** (not a single UNet): a
**prior** produces a 16-channel low-resolution latent, a **decoder**
turns it into a 4-channel latent conditioned on an effnet embedding,
and a **VQGAN** (PaellaVQModel) decodes that to pixels. Both prior and
decoder share **one** unified `StableCascadeUNet` class, parameterised
by `switch_level`.

| Stage | Detail |
|-------|--------|
| Prior (`StableCascadeUNet(PriorConfig)`) | in/out=16, block_out_channels=(2048,2048), heads=(32,32), down=(8,24), up=(24,8), conditioning_dim=2048, clip_text_in=1280, clip_image_in=768, switch_level=(False,) → `UpDownBlock2d` scalers (1×1 mapping, **no** spatial down/up) |
| Decoder (`StableCascadeUNet(DecoderConfig)`) | in/out=4, block_out_channels=(320,640,1280,1280), heads=(0,0,20,20), down=(2,6,28,6), up=(6,28,6,2), conditioning_dim=1280, effnet_in=16, pixel_mapper_in=3, patch_size=2, switch_level=None → plain `Conv2d(k=2,s=2)` down / `ConvTranspose2d(k=2,s=2)` up |
| VQGAN (`vqgan.py`) | PaellaVQModel, embed_dim=384, latent_channels=4, levels=2, bottleneck_blocks=12, scale_factor=0.3764 (decode-only; `vquantizer` dropped) |
| Scheduler (`scheduler.py`) | DDPMWuerstchen, s=0.008, cosine `_alpha_cumprod`, `linspace(1.0,0.0,steps+1)` timesteps (last dropped), DDPM posterior step, init_noise_sigma=1.0 |
| Text encoder (`text_encoder.py`) | CLIP-ViT-bigG (dims=1280, 32 layers, 20 heads, intermediate=5120, gelu, projection_dim=1280, vocab=49408, max_pos=77) |

### Sampling flow

```
prior latents (1,16,ceil(h/42.67),ceil(w/42.67))
  → DDPMWuerstchen denoise (timestep_ratio=t, timesteps[:-1])
  → image_embeddings (16-ch)
decoder latents (1,4,ceil(prior_h*10.67),ceil(prior_w*10.67))
  effnet = image_embeddings  → DDPMWuerstchen denoise → 4-ch latents
vqgan.decode(latents * 0.3764) → pixel
```

CFG (when `guidance > 1`): `pred = pred_un + guidance * (pred_cond - pred_un)`,
computed by concatenating cond/uncond batches. The prior additionally
receives `clip_img = zeros((b,1,768))`.

### NHWC throughout

The UNet main flow is **NHWC** (MLX `conv2d` is NHWC-native: weight
`(out,k,k,in)`, input `(N,H,W,C)`). All feature-map blocks
(`WuerstchenLayerNorm`, `ResBlock`, `AttnBlock`, `TimestepBlock`)
operate on the last axis directly — no NCHW↔NHWC permutes inside the
UNet. effnet/pixel conditioning arrive NCHW and are converted to NHWC
at the mapper boundary.

## Weights

| Asset | Source repo | File | Keys |
|-------|-------------|------|------|
| Prior | `stabilityai/stable-cascade-prior` | `prior/diffusion_pytorch_model.bf16.safetensors` (7.2 GB) | 1550/1550 verified |
| CLIP bigG | same repo | `text_encoder/model.bf16.safetensors` (1.4 GB) | 517/517 verified |
| Tokenizer | same repo | `tokenizer/` (vocab.json + merges.txt) | CLIPTokenizer |
| Decoder | `stabilityai/stable-cascade` | `decoder/diffusion_pytorch_model.bf16.safetensors` (3.1 GB) | 1726/1726 verified |
| VQGAN | same repo | `vqgan/diffusion_pytorch_model.safetensors` (74 MB) | 121/122 (`vquantizer.embedding.weight` dropped — unused, `force_not_quantize=True`) |

### Weight layout conversion

| Layer type | PyTorch layout | MLX layout | Transpose |
|------------|----------------|------------|-----------|
| `Conv2d` / `DepthwiseConv2d` | `(out,in,k,k)` OIHW | `(out,k,k,in)` OHWI | `(0,2,3,1)` |
| `ConvTranspose2d` | `(in,out,k,k)` | `(out,k,k,in)` OHWI | `(1,2,3,0)` |
| `nn.Linear` | `(out,in)` | `(out,in)` | none (MLX stores `(out,in)`, same as PyTorch) |
| `nn.Embedding` | `(vocab,dim)` | `(vocab,dim)` | none |

`nn.Linear`/`nn.Embedding` weights are **left as-is** — MLX `nn.Linear`
stores `(out,in)` matching PyTorch (its `__call__` applies `.T`
internally), so transposing them is a bug. ConvTranspose2d is detected
by key (`up_upscalers.{i}.1.weight` in the decoder; `up_blocks.{N}.weight`
direct-attached in the VQGAN) since its `(in,out,k,k)` shape is
indistinguishable from a plain Conv2d `(out,in,k,k)` of the same
dimensions.

### Local / offline weight override

Component resolution honours `CASCADE_LOCAL_DIR`: when set, files are
read directly from that directory before falling back to
`hf_hub_download`. The HF cache (`HUGGINGFACE_HUB_CACHE`) is also used
automatically. Additional env overrides:

| Env var | Purpose |
|---------|---------|
| `CASCADE_LOCAL_DIR` | Root dir of a local cascade weight tree |
| `CASCADE_PRIOR_REPO` / `CASCADE_PRIOR_FILE` | Prior repo id / filename |
| `CASCADE_DECODER_REPO` / `CASCADE_DECODER_FILE` | Decoder repo id / filename |
| `CASCADE_VQGAN_REPO` / `CASCADE_VQGAN_FILE` | VQGAN repo id / filename |
| `CASCADE_TEXT_REPO` / `CASCADE_TEXT_FILE` | CLIP text-encoder repo id / filename |

## Usage

### Via the image engine (`/v1/images/generate`)

Set `model` to a cascade-family identifier; the variant is
auto-detected from the name (`cascade` / `wuerstchen`):

```json
{
  "model": "stable-cascade",
  "prompt": "a photo of a cat",
  "width": 1024,
  "height": 1024,
  "steps": 20,
  "guidance": 4.0,
  "negative_prompt": "blurry, low quality"
}
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `steps` | 20 recommended | Prior denoise steps |
| `guidance` | 4.0 | CFG scale (prior) |
| `negative_prompt` | `""` | Supported; used for CFG |
| `width` / `height` | 1024 / 1024 | Must be multiples of 8; prior latent = `ceil(dim/42.67)`, decoder latent = `prior * 10.67` |

`decoder_steps` (default 10) and `decoder_guidance` (default 0.0) are
cascade-specific overrides available through the direct pipeline and
the engine `**kwargs` path.

### Direct pipeline

```python
from fusion_mlx.image.cascade.generate import CascadePipeline

pipe = CascadePipeline(model_path="stable-cascade", quantize=None)
result = pipe.generate_image(
    seed=42,
    prompt="a photo of a cat",
    num_inference_steps=20,
    height=1024,
    width=1024,
    guidance=4.0,
    negative_prompt="blurry",
    decoder_steps=10,
    decoder_guidance=0.0,
)
result.image.save("out.png")  # PIL.Image
```

`quantize=8` quantizes the prior + decoder UNets to 8-bit
(group_size=64).

## File map

```
fusion_mlx/image/cascade/
├── config.py        # PriorConfig, DecoderConfig, VQGANConfig,
│                    #   CascadeConfig, CascadeModelPaths
├── common.py        # WuerstchenLayerNorm, GlobalResponseNorm, ResBlock,
│                    #   TimestepBlock, AttnBlock, _Attention
├── unet.py          # StableCascadeUNet (prior + decoder), Conv2d,
│                    #   ConvTranspose2d, UpDownBlock2d, PixelShuffle
├── vqgan.py         # PaellaVQModel (decode-only), MixingResidualBlock
├── scheduler.py     # DDPMWuerstchenScheduler
├── text_encoder.py  # CascadeCLIPTextModel (CLIP-ViT-bigG)
├── weights.py       # remap_unet_weights / remap_vqgan_weights /
│                    #   remap_clip_weights / load_*
└── generate.py      # CascadePipeline (prior→decoder→vqgan, CFG loop)
```

Integration: `fusion_mlx/engines/image_gen.py`
`VARIANT_MAP["stable_cascade"]` → `CascadePipeline`; native (non-mflux)
variant, auto-detected when the model name contains `cascade` or
`wuerstchen`. `fusion_mlx/pool/model_discovery.py` maps
`StableCascadePriorPipeline` / `StableCascadeDecoderPipeline` to
`text-to-image`.

## Gotchas

- **`switch_level` selects the scaler type**: prior
  (`switch_level=(False,)`, not `None`) uses `UpDownBlock2d`
  (`[Conv2d 1×1 mapping, Identity]` → **no** spatial change); decoder
  (`switch_level=None`) uses plain `Conv2d(k=2,s=2)` /
  `ConvTranspose2d(k=2,s=2)`. Reusing one scaler for both mis-loads
  keys.
- **Do not transpose `nn.Linear` weights**: MLX `nn.Linear` stores
  `(out,in)` like PyTorch and applies `.T` in `__call__`. Transposing
  2D weights on load inverts the projection and silently breaks the
  encoder (e.g. CLIP `fc1` shape mismatch). Only conv weights are
  transposed; `nn.Embedding` is also left as-is.
- **ConvTranspose2d uses a different transpose** `(1,2,3,0)`, not
  `(0,2,3,1)`: PyTorch stores it as `(in,out,k,k)` but MLX
  `conv_transpose2d` expects `(out,k,k,in)` OHWI — the same layout as
  `conv2d`, despite the reversed PyTorch convention.
- **MLX 0.32 missing APIs**: no `mx.norm` (use
  `sqrt(sum(x*x, axis, keepdims)+1e-6)`), no `mx.dropout` (use an
  `nn.Dropout(0.0)` module with dynamic `.p`), no `mx.interpolate`
  (bilinear via `mx.take` on lo/hi indices + linear blend,
  `align_corners=True`).
- **Pixel shuffle/unshuffle (NHWC)**: unshuffle
  `(b,h,w,c)→(b,h/r,w/r,c*r²)` reshapes `(b,h//r,r,w//r,r,c)` then
  transposes `(0,1,3,4,5,2)`; shuffle reshapes
  `(b,h,w,c/r²,r,r)` then transposes `(0,1,4,2,5,3)`.
- **VQGAN is decode-only**: `vquantizer.embedding.weight` is dropped on
  load (`force_not_quantize=True`); the encoder half of PaellaVQModel
  is unused — Stable Cascade feeds decoder latents directly.
- **`safe_open` iteration**: use `f.keys()` (not `for k in f`) —
  `safe_open` objects are not iterable in safetensors ≥ 0.8.
