import logging
import os

import mlx.core as mx
import numpy as np
from PIL import Image

from fusion_mlx.image.sdxl.config import (
    SDXLConfig,
    SDXLModelPaths,
)
from fusion_mlx.image.sdxl.scheduler import SDXLEulerDiscreteScheduler
from fusion_mlx.image.sdxl.text_encoder import SDXLCLIPTextModel
from fusion_mlx.image.sdxl.unet import SDXLUNet
from fusion_mlx.image.sdxl.vae import SDXLVAE
from fusion_mlx.image.sdxl.weights import load_clip, load_unet, load_vae

logger = logging.getLogger(__name__)


class GenResult:
    def __init__(self, image: Image.Image):
        self.image = image


def _local_path(subfolder: str, filename: str) -> str | None:
    base = os.environ.get("SDXL_LOCAL_DIR")
    if not base:
        return None
    cand = (
        os.path.join(base, subfolder, filename)
        if subfolder
        else os.path.join(base, filename)
    )
    if os.path.exists(cand):
        logger.info("SDXL local resolve %s", cand)
        return cand
    return None


def _resolve(repo: str, subfolder: str, filename: str) -> str:
    local = _local_path(subfolder, filename)
    if local:
        return local
    from huggingface_hub import hf_hub_download

    path_in_repo = f"{subfolder}/{filename}" if subfolder else filename
    return hf_hub_download(repo, path_in_repo)


def _resolve_dir(repo: str, subfolder: str) -> str:
    base = os.environ.get("SDXL_LOCAL_DIR")
    if base and os.path.isdir(os.path.join(base, subfolder)):
        logger.info("SDXL local resolve_dir %s/%s", base, subfolder)
        return os.path.join(base, subfolder)
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo, allow_patterns=[f"{subfolder}/*"], endpoint=os.environ.get("HF_ENDPOINT")
    )


def _load_safetensors(path: str) -> dict:
    from safetensors import safe_open

    raw = {}
    with safe_open(path, "pt") as f:
        # safe_open objects expose .keys() but are NOT iterable in
        # safetensors >= 0.8 (ruff SIM118 is a false positive here).
        for k in f.keys():  # noqa: SIM118
            raw[k] = mx.array(f.get_tensor(k).numpy())
    return raw


class SDXLPipeline:
    def __init__(self, model_config=None, model_path: str | None = None, quantize=None):
        self.config = SDXLConfig().unet
        self.text_cfg = SDXLConfig().text
        self.paths = SDXLModelPaths()
        self.quantize = quantize
        self.unet = None
        self.vae = None
        self.clip_l = None
        self.clip_g = None
        self.tokenizer_l = None
        self.tokenizer_g = None
        self._loaded = False

    def _load_all(self) -> None:
        if self._loaded:
            return
        logger.info("SDXL pipeline loading components")
        self._load_text_encoders()
        self._load_unet_and_vae()
        self._loaded = True
        logger.info("SDXL pipeline ready")

    def _load_text_encoders(self) -> None:
        from transformers import CLIPTokenizer

        tc = self.text_cfg
        self.clip_l = SDXLCLIPTextModel(
            dims=tc.clip_l_dims,
            num_layers=tc.clip_l_layers,
            num_heads=tc.clip_l_heads,
            intermediate=tc.clip_l_intermediate,
            act=tc.clip_l_act,
            vocab=tc.vocab,
            max_pos=tc.max_pos,
        )
        clip_l_path = _resolve(
            self.paths.repo, self.paths.clip_l_subfolder, self.paths.clip_l_file
        )
        load_clip(self.clip_l, _load_safetensors(clip_l_path))
        self.clip_g = SDXLCLIPTextModel(
            dims=tc.clip_g_dims,
            num_layers=tc.clip_g_layers,
            num_heads=tc.clip_g_heads,
            intermediate=tc.clip_g_intermediate,
            act=tc.clip_g_act,
            vocab=tc.vocab,
            max_pos=tc.max_pos,
            projection_dim=tc.clip_g_projection_dim,
        )
        clip_g_path = _resolve(
            self.paths.repo, self.paths.clip_g_subfolder, self.paths.clip_g_file
        )
        load_clip(self.clip_g, _load_safetensors(clip_g_path))
        self.tokenizer_l = CLIPTokenizer.from_pretrained(
            _resolve_dir(self.paths.repo, self.paths.tokenizer_subfolder)
        )
        self.tokenizer_g = CLIPTokenizer.from_pretrained(
            _resolve_dir(self.paths.repo, self.paths.tokenizer_2_subfolder)
        )
        logger.info("SDXL text encoders loaded (clip-l fp32 / clip-g)")

    def _load_unet_and_vae(self) -> None:
        unet_path = _resolve(
            self.paths.repo, self.paths.unet_subfolder, self.paths.unet_file
        )
        self.unet = SDXLUNet(self.config)
        load_unet(self.unet, _load_safetensors(unet_path))
        vae_path = _resolve(
            self.paths.repo, self.paths.vae_subfolder, self.paths.vae_file
        )
        self.vae = SDXLVAE()
        load_vae(self.vae, _load_safetensors(vae_path))
        if self.quantize:
            try:
                import mlx.nn as nn

                nn.quantize(self.unet, group_size=64, bits=8)
                logger.info("SDXL unet quantized to 8-bit")
            except Exception:
                logger.debug("SDXL unet quantize skipped", exc_info=True)
        mx.eval(self.unet.parameters(), self.vae.parameters())
        logger.info("SDXL unet + vae loaded")

    def _encode_prompt(self, prompt: str):
        tok_l = self.tokenizer_l(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )["input_ids"].astype(np.int32)
        tok_g = self.tokenizer_g(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )["input_ids"].astype(np.int32)
        hidden_l, _ = self.clip_l(mx.array(tok_l))
        hidden_l = hidden_l.astype(mx.float32)
        hidden_g, pooled_g = self.clip_g(mx.array(tok_g))
        hidden_g = hidden_g.astype(mx.float32)
        pooled_g = pooled_g.astype(mx.float32)
        context = mx.concatenate([hidden_l, hidden_g], axis=-1)
        mx.eval(context, pooled_g)
        return context, pooled_g

    def _default_time_ids(self, height: int, width: int) -> mx.array:
        # SDXL original_size, crop_top_left, target_size
        return mx.array([[height, width, 0, 0, height, width]], dtype=mx.float32)

    def generate_image(
        self,
        seed: int = 0,
        prompt: str = "",
        num_inference_steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 7.5,
        negative_prompt: str = "",
        **kwargs,
    ) -> GenResult:
        self._load_all()
        cfg = self.config
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("SDXL height/width must be multiples of 8")
        h_lat = height // 8
        w_lat = width // 8
        logger.info(
            "SDXL generate prompt=%r steps=%d %dx%d guidance=%.2f seed=%d",
            prompt[:60],
            num_inference_steps,
            width,
            height,
            guidance,
            seed,
        )
        mx.random.seed(seed)
        context, pooled = self._encode_prompt(prompt)
        time_ids = self._default_time_ids(height, width)
        if guidance > 1:
            context_un, pooled_un = self._encode_prompt(negative_prompt or "")
            time_ids_un = self._default_time_ids(height, width)
        else:
            context_un = None
            pooled_un = None
            time_ids_un = None

        latent = mx.random.normal((1, cfg.in_channels, h_lat, w_lat), dtype=mx.float32)
        scheduler = SDXLEulerDiscreteScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
            timestep_spacing="leading",
            steps_offset=1,
        )
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps
        latent = latent * scheduler.init_noise_sigma
        for i, t in enumerate(timesteps):
            t_arr = mx.array([float(t)])
            if guidance > 1:
                latent_in = mx.concatenate([latent, latent], axis=0)
                context_in = mx.concatenate([context, context_un], axis=0)
                pooled_in = mx.concatenate([pooled, pooled_un], axis=0)
                tids_in = mx.concatenate([time_ids, time_ids_un], axis=0)
                t_in = mx.concatenate([t_arr, t_arr], axis=0)
                noise = self.unet(
                    scheduler.scale_model_input(latent_in), t_in, context_in,
                    pooled_in, tids_in,
                )
                noise_cond, noise_un = mx.split(noise, 2, axis=0)
                noise = noise_un + guidance * (noise_cond - noise_un)
            else:
                noise = self.unet(
                    scheduler.scale_model_input(latent), t_arr, context, pooled,
                    time_ids,
                )
            latent = scheduler.step(noise, latent)
            mx.eval(latent)
            logger.info("SDXL step %d/%d done", i + 1, len(timesteps))
        image = self.vae.decode(latent)
        mx.eval(image)
        return GenResult(image=_to_pil(image))


def _to_pil(image: mx.array) -> Image.Image:
    arr = np.array(image[0].astype(mx.float32))
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr / 2 + 0.5).clip(0, 1)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)
