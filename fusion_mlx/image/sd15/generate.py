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
from fusion_mlx.image.sd15.config import SD15Config, SD15ModelPaths
from fusion_mlx.image.sd15.scheduler import SD15EulerDiscreteScheduler
from fusion_mlx.image.sd15.text_encoder import SD15CLIPTextModel
from fusion_mlx.image.sd15.unet import SD15UNet
from fusion_mlx.image.sd15.vae import SD15VAE
from fusion_mlx.image.sd15.weights import load_clip, load_unet, load_vae

logger = logging.getLogger(__name__)

_LOCAL_ENV = "SD15_LOCAL_DIR"
_LABEL = "SD15"


def _resolve(repo: str, subfolder: str, filename: str) -> str:
    return resolve_path(_LOCAL_ENV, repo, subfolder, filename, _LABEL)


def _resolve_dir(repo: str, subfolder: str) -> str:
    return resolve_dir(_LOCAL_ENV, repo, subfolder, _LABEL)


class SD15Pipeline:
    def __init__(self, model_config=None, model_path: str | None = None, quantize=None):
        self.config = SD15Config().unet
        self.text_cfg = SD15Config().text
        self.vae_cfg = SD15Config().vae
        self.paths = SD15ModelPaths()
        self.quantize = quantize
        self.unet = None
        self.vae = None
        self.clip_l = None
        self.tokenizer_l = None
        self._loaded = False

    def _load_all(self) -> None:
        if self._loaded:
            return
        logger.info("SD15 pipeline loading components")
        self._load_text_encoder()
        self._load_unet_and_vae()
        self._loaded = True
        logger.info("SD15 pipeline ready")

    def _load_text_encoder(self) -> None:
        from transformers import CLIPTokenizer

        tc = self.text_cfg
        self.clip_l = SD15CLIPTextModel(
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
        self.tokenizer_l = CLIPTokenizer.from_pretrained(
            _resolve_dir(self.paths.repo, self.paths.tokenizer_subfolder)
        )
        logger.info("SD15 text encoder loaded (clip-l 768/12/12)")

    def _load_unet_and_vae(self) -> None:
        unet_path = _resolve(
            self.paths.repo, self.paths.unet_subfolder, self.paths.unet_file
        )
        self.unet = SD15UNet(self.config)
        load_unet(self.unet, load_safetensors(unet_path))
        vae_path = _resolve(
            self.paths.repo, self.paths.vae_subfolder, self.paths.vae_file
        )
        self.vae = SD15VAE()
        load_vae(self.vae, load_safetensors(vae_path))
        if self.quantize:
            try:
                import mlx.nn as nn

                nn.quantize(self.unet, group_size=64, bits=8)
                logger.info("SD15 unet quantized to 8-bit")
            except Exception:
                logger.debug("SD15 unet quantize skipped", exc_info=True)
        mx.eval(self.unet.parameters(), self.vae.parameters())
        logger.info("SD15 unet + vae loaded")

    def _encode_prompt(self, prompt: str) -> mx.array:
        # SD1.5 uses a single CLIP-L; return last hidden state (full sequence).
        # Pooled output is unused (no add_embedding).
        tok_l = self.tokenizer_l(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )["input_ids"].astype(np.int32)
        hidden_l, _ = self.clip_l(mx.array(tok_l))
        hidden_l = hidden_l.astype(mx.float32)
        mx.eval(hidden_l)
        return hidden_l

    def _make_scheduler(self, num_inference_steps: int) -> SD15EulerDiscreteScheduler:
        scheduler = SD15EulerDiscreteScheduler(
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
        height: int = 512,
        width: int = 512,
        guidance: float = 7.5,
        negative_prompt: str = "",
        image_path: str | None = None,
        image_strength: float | None = None,
        **kwargs,
    ) -> GenResult:
        self._load_all()
        cfg = self.config
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("SD15 height/width must be multiples of 8")
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
        context = self._encode_prompt(prompt)
        context_un = (
            self._encode_prompt(negative_prompt or "") if guidance > 1 else None
        )

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

        cond = {"context": context}
        uncond = {"context": context_un} if context_un is not None else None
        latent = denoise_loop(
            _LABEL,
            scheduler,
            timesteps,
            latent,
            guidance,
            cond,
            uncond,
            lambda lat, t, args: self.unet(lat, t, args["context"]),
        )
        image = self.vae.decode(latent)
        mx.eval(image)
        return GenResult(image=to_pil(image))
