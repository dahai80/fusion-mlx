import logging
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from fusion_mlx.image.sd3.config import ClipGConfig, SD3Config, SD3ModelPaths
from fusion_mlx.image.sd3.mmdit import MMDiT
from fusion_mlx.image.sd3.scheduler import FlowMatchEulerScheduler
from fusion_mlx.image.sd3.text_encoder import CLIPTextModel
from fusion_mlx.image.sd3.vae import SD3VAE
from fusion_mlx.image.sd3.weights import load_transformer, load_vae

logger = logging.getLogger(__name__)


class GenResult:
    def __init__(self, image: Image.Image):
        self.image = image


def _pil_to_nchw(image: Image.Image, height: int, width: int) -> mx.array:
    # Resize to target then normalize to [-1, 1] NCHW for VAE encode (img2img).
    if image.size != (width, height):
        image = image.resize((width, height), Image.LANCZOS)
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    arr = (arr * 2.0) - 1.0
    arr = np.transpose(arr, (2, 0, 1))
    return mx.array(arr)[None]


def _local_path(subfolder: str, filename: str) -> str | None:
    base = os.environ.get("SD3_LOCAL_DIR")
    if not base:
        return None
    cand = (
        os.path.join(base, subfolder, filename)
        if subfolder
        else os.path.join(base, filename)
    )
    if os.path.exists(cand):
        logger.info("SD3 local resolve %s", cand)
        return cand
    return None


def _resolve(repo: str, subfolder: str, filename: str) -> str:
    local = _local_path(subfolder, filename)
    if local:
        return local
    from huggingface_hub import hf_hub_download

    path_in_repo = f"{subfolder}/{filename}" if subfolder else filename
    return hf_hub_download(repo, path_in_repo)


def _load_safetensors(paths: list[str]) -> dict:
    from safetensors import safe_open

    raw = {}
    for p in paths:
        with safe_open(p, "pt") as f:
            keys = list(f.keys())
            for k in keys:
                raw[k] = mx.array(f.get_tensor(k).numpy())
    return raw


def _resolve_dir(repo: str, subfolder: str) -> str:
    base = os.environ.get("SD3_LOCAL_DIR")
    if base and os.path.isdir(os.path.join(base, subfolder)):
        logger.info("SD3 local resolve_dir %s/%s", base, subfolder)
        return os.path.join(base, subfolder)
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo, allow_patterns=[f"{subfolder}/*"], endpoint=os.environ.get("HF_ENDPOINT")
    )


def _clip_pairs_direct(
    repo: str, subfolder: str, filename: str = "model.safetensors", dtype=None
) -> list:
    path = _resolve(repo, subfolder, filename)
    raw = _load_safetensors([path])
    pairs = []
    for k, v in raw.items():
        if k.startswith("text_model."):
            pairs.append((k, v.astype(dtype) if dtype is not None else v))
    logger.info("SD3 clip %s/%s loaded %d keys", subfolder, filename, len(pairs))
    return pairs


def _load_safetensors_fp8(path: str) -> dict:
    import torch
    from safetensors import safe_open

    raw = {}
    with safe_open(path, "pt") as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            if t.dtype.is_floating_point and str(t.dtype) == "torch.float8_e4m3fn":
                t = t.to(torch.float32)
            raw[k] = mx.array(t.numpy())
    return raw


def _t5_pairs(repo: str, subfolder: str) -> list:
    fp8_name = os.environ.get("SD3_T5_FP8_FILE", "t5xxl_fp8_e4m3fn.safetensors")
    fp8_sub = os.environ.get("SD3_T5_FP8_SUBFOLDER", "text_encoders")
    try:
        path = (
            _resolve(repo, "", fp8_name)
            if fp8_sub == ""
            else _resolve(repo, fp8_sub, fp8_name)
        )
        raw = _load_safetensors_fp8(path)
        pairs = []
        for k, v in raw.items():
            new = _map_t5(k)
            if new is not None:
                pairs.append((new, v.astype(mx.bfloat16)))
        logger.info("SD3 t5 fp8 decoded+remapped %d / %d keys", len(pairs), len(raw))
        return pairs
    except Exception as exc:
        logger.info("SD3 t5 fp8 path unavailable (%s), falling back to shards", exc)
    return _t5_pairs_shards(repo, subfolder)


def _t5_pairs_shards(repo: str, subfolder: str) -> list:
    index_path = _resolve(repo, subfolder, "model.safetensors.index.json")
    import json

    with open(index_path) as f:
        idx = json.load(f)
    files = sorted(set(idx["weight_map"].values()))
    paths = [_resolve(repo, subfolder, fn) for fn in files]
    raw = _load_safetensors(paths)
    pairs = []
    for k, v in raw.items():
        new = _map_t5(k)
        if new is not None:
            pairs.append((new, v))
    logger.info("SD3 t5 remapped %d / %d keys", len(pairs), len(raw))
    return pairs


def _map_t5(k: str):
    if k == "shared.weight":
        return "shared.weight"
    if k == "encoder.final_layer_norm.weight":
        return "final_layer_norm.weight"
    if k.startswith("encoder.block."):
        parts = k.split(".")
        block = parts[2]
        rest = ".".join(parts[3:])
        if rest.startswith("layer.0."):
            return (
                "t5_blocks."
                + block
                + ".attention."
                + _map_t5_attn(rest[len("layer.0.") :])
            )
        if rest.startswith("layer.1."):
            return "t5_blocks." + block + ".ff." + _map_t5_ff(rest[len("layer.1.") :])
    return None


def _map_t5_attn(sub: str) -> str:
    table = {
        "layer_norm.weight": "layer_norm.weight",
        "SelfAttention.q.weight": "SelfAttention.q.weight",
        "SelfAttention.k.weight": "SelfAttention.k.weight",
        "SelfAttention.v.weight": "SelfAttention.v.weight",
        "SelfAttention.o.weight": "SelfAttention.o.weight",
        "SelfAttention.relative_attention_bias.weight": "SelfAttention.relative_attention_bias.weight",
    }
    return table.get(sub, "unknown." + sub)


def _map_t5_ff(sub: str) -> str:
    table = {
        "layer_norm.weight": "layer_norm.weight",
        "DenseReluDense.wi_0.weight": "DenseReluDense.wi_0.weight",
        "DenseReluDense.wi_1.weight": "DenseReluDense.wi_1.weight",
        "DenseReluDense.wo.weight": "DenseReluDense.wo.weight",
    }
    return table.get(sub, "unknown." + sub)


class SD3Pipeline:
    def __init__(self, model_config=None, model_path: str | None = None, quantize=None):
        self.config = SD3Config()
        self.paths = SD3ModelPaths()
        self.quantize = quantize
        self.transformer = None
        self.vae = None
        self.clip_l = None
        self.clip_g = None
        self.t5 = None
        self.clip_l_tokenizer = None
        self.clip_g_tokenizer = None
        self.t5_tokenizer = None
        self._loaded = False

    def _load_all(self) -> None:
        if self._loaded:
            return
        logger.info("SD3 pipeline loading components")
        self._load_encoders()
        self._load_transformer_and_vae()
        self._loaded = True
        logger.info("SD3 pipeline ready")

    def _load_encoders(self) -> None:
        from mflux.models.flux.model.flux_text_encoder.clip_encoder.clip_encoder import (
            CLIPEncoder,
        )
        from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import (
            T5Encoder,
        )
        from transformers import CLIPTokenizer, T5TokenizerFast

        repo = self.paths.encoders_repo
        clip_l_file = os.environ.get("SD3_CLIP_L_FILE", "model.safetensors")
        clip_g_file = os.environ.get("SD3_CLIP_G_FILE", "model.safetensors")
        self.clip_l = CLIPEncoder()
        self.clip_l.load_weights(
            _clip_pairs_direct(
                repo, self.paths.clip_l_subfolder, clip_l_file, mx.float32
            ),
            strict=False,
        )
        cg = ClipGConfig()
        self.clip_g = CLIPTextModel(
            dims=cg.dims,
            num_layers=cg.num_layers,
            num_heads=cg.num_heads,
            intermediate=cg.intermediate,
            act=cg.act,
            vocab=cg.vocab,
            max_pos=cg.max_pos,
        )
        self.clip_g.load_weights(
            _clip_pairs_direct(repo, self.paths.clip_g_subfolder, clip_g_file),
            strict=False,
        )
        self.t5 = T5Encoder()
        self.t5.load_weights(_t5_pairs(repo, self.paths.t5_subfolder), strict=False)
        self.clip_l_tokenizer = CLIPTokenizer.from_pretrained(
            _resolve_dir(repo, self.paths.clip_l_tokenizer_subfolder)
        )
        self.clip_g_tokenizer = CLIPTokenizer.from_pretrained(
            _resolve_dir(repo, self.paths.clip_g_tokenizer_subfolder)
        )
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(
            _resolve_dir(repo, self.paths.t5_tokenizer_subfolder)
        )
        if self.quantize:
            self._maybe_quantize_text()
        logger.info("SD3 text encoders loaded (clip-l/clip-g/t5)")

    def _load_transformer_and_vae(self) -> None:
        ckpt_path = _resolve(
            self.paths.transformer_ckpt, "", self.paths.transformer_file
        )
        raw = _load_safetensors([ckpt_path])
        self.transformer = MMDiT(self.config)
        load_transformer(self.transformer, raw, self.config.num_layers)
        self.vae = SD3VAE()
        load_vae(self.vae, raw)
        if self.quantize:
            has_quant_meta = any(
                k.startswith("model.diffusion_model.")
                and (
                    k.endswith(".scales")
                    or k.endswith(".biases")
                    or k.endswith(".qweight")
                )
                for k in raw
            )
            nn.quantize(self.transformer, group_size=64, bits=8)
            if has_quant_meta:
                load_transformer(self.transformer, raw, self.config.num_layers)
            else:
                logger.info(
                    "SD3 transformer quantized in-memory (ckpt is fp16, no "
                    "scales/biases to reload); skipping post-quant reload to "
                    "avoid overwriting uint32 weights with fp16"
                )
        mx.eval(self.transformer.parameters())
        logger.info("SD3 transformer + vae loaded from %s", ckpt_path)

    def _maybe_quantize_text(self) -> None:
        try:
            nn.quantize(self.t5, group_size=64, bits=8)
            logger.info("SD3 t5 quantized to 8-bit")
        except Exception:
            logger.debug("SD3 t5 quantize skipped", exc_info=True)

    def _encode_prompt(self, prompt: str, max_t5_len: int = 256):
        clip_l_tok = self.clip_l_tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )["input_ids"].astype(np.int32)
        clip_g_tok = self.clip_g_tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )["input_ids"].astype(np.int32)
        t5_tok = self.t5_tokenizer(
            prompt,
            padding="max_length",
            max_length=max_t5_len,
            truncation=True,
            return_tensors="np",
        )["input_ids"].astype(np.int32)
        pooled_l = self.clip_l(mx.array(clip_l_tok))
        pooled_g = self.clip_g(mx.array(clip_g_tok))
        pooled = mx.concatenate([pooled_l, pooled_g], axis=-1)
        context = self.t5(mx.array(t5_tok))
        mx.eval(pooled, context)
        return pooled, context

    def generate_image(
        self,
        seed: int = 0,
        prompt: str = "",
        num_inference_steps: int = 28,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 4.0,
        shift: float | None = None,
        negative_prompt: str = "",
        max_t5_len: int = 256,
        image_path: str | None = None,
        image_strength: float | None = None,
        **kwargs,
    ) -> GenResult:
        self._load_all()
        cfg = self.config
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError("SD3 height/width must be multiples of 16")
        h_lat = height // 8
        w_lat = width // 8
        # img2img: image_strength (a.k.a. denoise fraction) controls how many
        # steps run. strength=1.0 -> full txt2img from noise; strength=0.5 ->
        # run half the steps from a noised init image (#480). When image_path
        # is None, fall back to pure txt2img.
        is_img2img = image_path is not None
        strength = float(image_strength) if image_strength is not None else 1.0
        if is_img2img:
            if strength <= 0.0 or strength > 1.0:
                logger.warning(
                    "SD3 image_strength=%.3f out of (0,1], clamping", strength
                )
                strength = max(min(strength, 1.0), 1e-3)
            eff_steps = max(1, int(round(num_inference_steps * strength)))
            logger.info(
                "SD3 img2img prompt=%r steps=%d eff_steps=%d strength=%.3f %dx%d",
                prompt[:60],
                num_inference_steps,
                eff_steps,
                strength,
                width,
                height,
            )
        else:
            eff_steps = num_inference_steps
            logger.info(
                "SD3 generate prompt=%r steps=%d %dx%d guidance=%.2f shift=%s seed=%d",
                prompt[:60],
                num_inference_steps,
                width,
                height,
                guidance,
                shift,
                seed,
            )
        mx.random.seed(seed)
        pooled, context = self._encode_prompt(prompt, max_t5_len)
        if guidance > 1:
            pooled_un, context_un = self._encode_prompt(
                negative_prompt or "", max_t5_len
            )
        else:
            pooled_un = None
            context_un = None

        image_seq_len = (h_lat // cfg.patch_size) * (w_lat // cfg.patch_size)
        scheduler = FlowMatchEulerScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            base_shift=cfg.base_shift,
            max_shift=cfg.max_shift,
            base_image_seq_len=cfg.base_image_seq_len,
            max_image_seq_len=cfg.max_image_seq_len,
            shift=shift,
        )
        scheduler.set_timesteps(num_inference_steps, image_seq_len)
        timesteps = scheduler.timesteps
        # Slice the last eff_steps timesteps (strongest denoise near the end),
        # matching diffusers SD3 img2img skip.
        if is_img2img and eff_steps < len(timesteps):
            timesteps = timesteps[len(timesteps) - eff_steps :]
            scheduler.sigmas = scheduler.sigmas[len(scheduler.sigmas) - 1 - eff_steps :]
            scheduler._step_index = 0
        if is_img2img:
            init_img = Image.open(image_path).convert("RGB")
            init_lat = self.vae.encode(_pil_to_nchw(init_img, height, width))
            init_lat = init_lat.astype(mx.float32)
            # flow-match convention: at sigma s, noisy = (1-s)*init + s*noise.
            # sigma_start ~ 1.0 -> mostly noise; ~0.0 -> mostly init image.
            sigma_start = float(scheduler.sigmas[0])
            noise = mx.random.normal(init_lat.shape, dtype=mx.float32)
            latent = (1.0 - sigma_start) * init_lat + sigma_start * noise
            mx.eval(latent)
            logger.info(
                "SD3 img2img init latent encoded, sigma_start=%.4f", sigma_start
            )
        else:
            latent = mx.random.normal(
                (1, cfg.in_channels, h_lat, w_lat), dtype=mx.float32
            )
        for i, t in enumerate(timesteps):
            t_arr = mx.array([float(t)])
            if guidance > 1:
                latent_in = mx.concatenate([latent, latent], axis=0)
                pooled_in = mx.concatenate([pooled, pooled_un], axis=0)
                context_in = mx.concatenate([context, context_un], axis=0)
                t_in = mx.concatenate([t_arr, t_arr], axis=0)
                noise = self.transformer(
                    latent_in, t_in, pooled_in, context_in, h_lat, w_lat
                )
                noise_cond, noise_un = mx.split(noise, 2, axis=0)
                noise = noise_un + guidance * (noise_cond - noise_un)
            else:
                noise = self.transformer(latent, t_arr, pooled, context, h_lat, w_lat)
            latent = scheduler.step(noise, latent)
            mx.eval(latent)
            logger.info("SD3 step %d/%d done", i + 1, len(timesteps))
        image = self.vae.decode(latent)
        mx.eval(image)
        return GenResult(image=_to_pil(image))


def _to_pil(image: mx.array) -> Image.Image:
    arr = np.array(image[0].astype(mx.float32))
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr / 2 + 0.5).clip(0, 1)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)
