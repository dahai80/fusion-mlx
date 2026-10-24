import logging

import mlx.core as mx
import numpy as np

from fusion_mlx.image.pipeline_base import (
    GenResult,
    denoise_loop,
    init_latent_from_image,
    init_noise_latent,
    load_safetensors,
    prepare_schedule,
    resolve_dir,
    resolve_path,
    slice_timesteps,
    to_pil,
)
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

_LOCAL_ENV = "SDXL_LOCAL_DIR"
_LABEL = "SDXL"


def _resolve(repo: str, subfolder: str, filename: str) -> str:
    return resolve_path(_LOCAL_ENV, repo, subfolder, filename, _LABEL)


def _resolve_dir(repo: str, subfolder: str) -> str:
    return resolve_dir(_LOCAL_ENV, repo, subfolder, _LABEL)


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
        load_clip(self.clip_l, load_safetensors(clip_l_path))
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
        load_clip(self.clip_g, load_safetensors(clip_g_path))
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
        load_unet(self.unet, load_safetensors(unet_path))
        vae_path = _resolve(
            self.paths.repo, self.paths.vae_subfolder, self.paths.vae_file
        )
        self.vae = SDXLVAE()
        load_vae(self.vae, load_safetensors(vae_path))
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

    def _make_scheduler(self, num_inference_steps: int) -> SDXLEulerDiscreteScheduler:
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
        return scheduler

    def generate_image(
        self,
        seed: int = 0,
        prompt: str = "",
        num_inference_steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 7.5,
        negative_prompt: str = "",
        image_path: str | None = None,
        image_strength: float | None = None,
        **kwargs,
    ) -> GenResult:
        self._load_all()
        cfg = self.config
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("SDXL height/width must be multiples of 8")
        h_lat = height // 8
        w_lat = width // 8

        is_img2img, eff_steps, _strength = prepare_schedule(
            _LABEL,
            prompt,
            num_inference_steps,
            image_path,
            image_strength,
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
        else:
            context_un = None
            pooled_un = None

        scheduler = self._make_scheduler(num_inference_steps)
        timesteps = scheduler.timesteps
        if is_img2img:
            timesteps = slice_timesteps(scheduler, timesteps, eff_steps)

        if is_img2img:
            latent = init_latent_from_image(
                _LABEL, self.vae, scheduler, image_path, height, width
            )
        else:
            latent = init_noise_latent(cfg, h_lat, w_lat, scheduler)

        cond = {"context": context, "pooled": pooled, "time_ids": time_ids}
        uncond = (
            {"context": context_un, "pooled": pooled_un, "time_ids": time_ids}
            if context_un is not None
            else None
        )
        latent = denoise_loop(
            _LABEL,
            scheduler,
            timesteps,
            latent,
            guidance,
            cond,
            uncond,
            lambda lat, t, args: self.unet(
                lat, t, args["context"], args["pooled"], args["time_ids"]
            ),
        )
        image = self.vae.decode(latent)
        mx.eval(image)
        return GenResult(image=to_pil(image))
