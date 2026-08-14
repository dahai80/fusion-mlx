import logging
import math
import os

import mlx.core as mx
import numpy as np
from PIL import Image

from fusion_mlx.image.cascade.config import (
    CascadeConfig,
    CascadeModelPaths,
    DecoderConfig,
    PriorConfig,
)
from fusion_mlx.image.cascade.scheduler import DDPMWuerstchenScheduler
from fusion_mlx.image.cascade.text_encoder import CascadeCLIPTextModel
from fusion_mlx.image.cascade.unet import StableCascadeUNet
from fusion_mlx.image.cascade.vqgan import PaellaVQModel
from fusion_mlx.image.cascade.weights import load_clip, load_unet, load_vqgan

logger = logging.getLogger(__name__)


class GenResult:
    def __init__(self, image: Image.Image):
        self.image = image


def _resolve_dir(repo: str, subfolder: str) -> str:
    base = os.environ.get("CASCADE_LOCAL_DIR")
    if base and os.path.isdir(os.path.join(base, subfolder)):
        logger.info("Cascade local resolve_dir %s/%s", base, subfolder)
        return os.path.join(base, subfolder)
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo,
        allow_patterns=[f"{subfolder}/*"],
        endpoint=os.environ.get("HF_ENDPOINT"),
    )


def _resolve(repo: str, subfolder: str, filename: str) -> str:
    base = os.environ.get("CASCADE_LOCAL_DIR")
    if base:
        cand = (
            os.path.join(base, subfolder, filename)
            if subfolder
            else os.path.join(base, filename)
        )
        if os.path.exists(cand):
            logger.info("Cascade local resolve %s", cand)
            return cand
    from huggingface_hub import hf_hub_download

    path_in_repo = f"{subfolder}/{filename}" if subfolder else filename
    return hf_hub_download(repo, path_in_repo, endpoint=os.environ.get("HF_ENDPOINT"))


def _load_safetensors(path: str) -> dict:
    from safetensors import safe_open

    raw = {}
    with safe_open(path, framework="pt") as f:
        # safetensors >= 0.8 objects are NOT iterable; .keys() is the API.
        for k in f.keys():  # noqa: SIM118
            t = f.get_tensor(k)
            raw[k] = mx.array(
                t.detach().cpu().float().numpy()
                if hasattr(t, "detach")
                else np.array(t)
            )
    logger.info("Cascade loaded %d tensors from %s", len(raw), os.path.basename(path))
    return raw


class CascadePipeline:
    def __init__(self, model_config=None, model_path: str | None = None, quantize=None):
        self.config = CascadeConfig()
        self.paths = CascadeModelPaths()
        self.quantize = quantize
        self.prior = None
        self.decoder = None
        self.vqgan = None
        self.text_encoder = None
        self.tokenizer = None
        self._loaded = False

    def _load_all(self) -> None:
        if self._loaded:
            return
        logger.info("Cascade pipeline loading components")
        self._load_text_encoder()
        self._load_prior()
        self._load_decoder()
        self._load_vqgan()
        self._loaded = True
        logger.info("Cascade pipeline ready")

    def _load_text_encoder(self) -> None:
        from transformers import CLIPTokenizer

        tc = self.config.text
        self.text_encoder = CascadeCLIPTextModel(
            dims=tc.hidden_size,
            num_layers=tc.num_hidden_layers,
            num_heads=tc.num_attention_heads,
            intermediate=tc.intermediate_size,
            act=tc.hidden_act,
            vocab=tc.vocab_size,
            max_pos=tc.max_position_embeddings,
            projection_dim=tc.projection_dim,
        )
        te_path = _resolve(
            self.paths.text_encoder_repo,
            self.paths.text_encoder_subfolder,
            self.paths.text_encoder_file,
        )
        load_clip(self.text_encoder, _load_safetensors(te_path))
        tok_dir = _resolve_dir(
            self.paths.tokenizer_repo, self.paths.tokenizer_subfolder
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(tok_dir)
        logger.info("Cascade text encoder + tokenizer loaded")

    def _load_prior(self) -> None:
        self.prior = StableCascadeUNet(PriorConfig())
        prior_path = _resolve(
            self.paths.prior_repo, self.paths.prior_subfolder, self.paths.prior_file
        )
        load_unet(self.prior, _load_safetensors(prior_path))
        if self.quantize:
            try:
                import mlx.nn as nn

                nn.quantize(self.prior, group_size=64, bits=8)
                logger.info("Cascade prior quantized to 8-bit")
            except Exception:
                logger.debug("Cascade prior quantize skipped", exc_info=True)
        mx.eval(self.prior.parameters())
        logger.info("Cascade prior loaded")

    def _load_decoder(self) -> None:
        self.decoder = StableCascadeUNet(DecoderConfig())
        dec_path = _resolve(
            self.paths.decoder_repo,
            self.paths.decoder_subfolder,
            self.paths.decoder_file,
        )
        load_unet(self.decoder, _load_safetensors(dec_path))
        if self.quantize:
            try:
                import mlx.nn as nn

                nn.quantize(self.decoder, group_size=64, bits=8)
                logger.info("Cascade decoder quantized to 8-bit")
            except Exception:
                logger.debug("Cascade decoder quantize skipped", exc_info=True)
        mx.eval(self.decoder.parameters())
        logger.info("Cascade decoder loaded")

    def _load_vqgan(self) -> None:
        vc = self.config.vqgan
        self.vqgan = PaellaVQModel(
            in_channels=vc.in_channels,
            out_channels=vc.out_channels,
            up_down_scale_factor=vc.up_down_scale_factor,
            levels=vc.levels,
            bottleneck_blocks=vc.bottleneck_blocks,
            embed_dim=vc.embed_dim,
            latent_channels=vc.latent_channels,
            scale_factor=vc.scale_factor,
        )
        vq_path = _resolve(
            self.paths.vqgan_repo, self.paths.vqgan_subfolder, self.paths.vqgan_file
        )
        load_vqgan(self.vqgan, _load_safetensors(vq_path))
        mx.eval(self.vqgan.parameters())
        logger.info("Cascade vqgan loaded")

    def _encode_prompt(self, prompt: str):
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.config.text.max_position_embeddings,
            truncation=True,
            return_tensors="np",
        )["input_ids"].astype(np.int32)
        hidden, pooled = self.text_encoder(mx.array(tokens))
        hidden = hidden.astype(mx.float32)
        pooled = pooled.astype(mx.float32)
        if pooled.ndim == 2:
            pooled = pooled[:, None, :]
        mx.eval(hidden, pooled)
        return hidden, pooled

    def generate_image(
        self,
        seed: int = 0,
        prompt: str = "",
        num_inference_steps: int = 20,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 4.0,
        negative_prompt: str = "",
        decoder_steps: int = 10,
        decoder_guidance: float = 0.0,
        **kwargs,
    ) -> GenResult:
        self._load_all()
        cfg = self.config
        resolution_multiple = cfg.resolution_multiple
        latent_dim_scale = cfg.latent_dim_scale
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("Cascade height/width must be multiples of 8")
        prior_h = max(1, math.ceil(height / resolution_multiple))
        prior_w = max(1, math.ceil(width / resolution_multiple))
        dec_h = max(1, int(prior_h * latent_dim_scale))
        dec_w = max(1, int(prior_w * latent_dim_scale))
        logger.info(
            "Cascade generate prompt=%r steps=%d %dx%d guidance=%.2f seed=%d prior=%dx%d decoder=%dx%d",
            prompt[:60],
            num_inference_steps,
            width,
            height,
            guidance,
            seed,
            prior_w,
            prior_h,
            dec_w,
            dec_h,
        )
        mx.random.seed(seed)
        text_hidden, text_pooled = self._encode_prompt(prompt)
        neg_hidden, neg_pooled = (None, None)
        if guidance > 1 and negative_prompt:
            neg_hidden, neg_pooled = self._encode_prompt(negative_prompt)
        elif guidance > 1:
            neg_hidden = mx.zeros_like(text_hidden)
            neg_pooled = mx.zeros_like(text_pooled)
        image_embeddings = self._run_prior(
            text_hidden,
            text_pooled,
            neg_hidden,
            neg_pooled,
            prior_h,
            prior_w,
            num_inference_steps,
            guidance,
        )
        latents = self._run_decoder(
            image_embeddings,
            text_pooled,
            neg_pooled,
            dec_h,
            dec_w,
            decoder_steps,
            decoder_guidance,
        )
        latents = self.config.vqgan.scale_factor * latents
        image = self.vqgan.decode(latents)
        mx.eval(image)
        return GenResult(image=_to_pil(image))

    def _run_prior(
        self, text_hidden, text_pooled, neg_hidden, neg_pooled, h, w, steps, guidance
    ):
        prior_cfg = self.config.prior
        scheduler = DDPMWuerstchenScheduler(
            s=self.config.scheduler_s, scaler=self.config.scheduler_scaler
        )
        scheduler.set_timesteps(steps)
        timesteps = scheduler.timesteps[:-1]
        latent = mx.random.normal((1, prior_cfg.in_channels, h, w), dtype=mx.float32)
        latent = latent * scheduler.init_noise_sigma
        clip_img_ch = prior_cfg.clip_image_in_channels
        do_cfg = guidance > 1
        for i, t in enumerate(timesteps):
            t_arr = mx.array([float(t)], dtype=mx.float32)
            if do_cfg:
                sample_in = mx.concatenate([latent, latent], axis=0)
                t_in = mx.concatenate([t_arr, t_arr], axis=0)
                pooled_in = mx.concatenate([text_pooled, neg_pooled], axis=0)
                hidden_in = mx.concatenate([text_hidden, neg_hidden], axis=0)
                img_in = mx.zeros((2, 1, clip_img_ch), dtype=mx.float32)
            else:
                sample_in = latent
                t_in = t_arr
                pooled_in = text_pooled
                hidden_in = text_hidden
                img_in = mx.zeros((1, 1, clip_img_ch), dtype=mx.float32)
            pred = self.prior(
                sample=sample_in,
                timestep_ratio=t_in,
                clip_text_pooled=pooled_in,
                clip_text=hidden_in,
                clip_img=img_in,
            )
            if do_cfg:
                pred_cond, pred_un = mx.split(pred, 2, axis=0)
                pred = pred_un + guidance * (pred_cond - pred_un)
            latent = scheduler.step(pred, t_arr, latent)
            mx.eval(latent)
            logger.info("Cascade prior step %d/%d done", i + 1, len(timesteps))
        return latent

    def _run_decoder(
        self, image_embeddings, text_pooled, neg_pooled, h, w, steps, guidance
    ):
        dec_cfg = self.config.decoder
        scheduler = DDPMWuerstchenScheduler(
            s=self.config.scheduler_s, scaler=self.config.scheduler_scaler
        )
        scheduler.set_timesteps(steps)
        timesteps = scheduler.timesteps[:-1]
        latent = mx.random.normal((1, dec_cfg.in_channels, h, w), dtype=mx.float32)
        latent = latent * scheduler.init_noise_sigma
        do_cfg = guidance > 1
        for i, t in enumerate(timesteps):
            t_arr = mx.array([float(t)], dtype=mx.float32)
            if do_cfg:
                sample_in = mx.concatenate([latent, latent], axis=0)
                t_in = mx.concatenate([t_arr, t_arr], axis=0)
                pooled_in = mx.concatenate([text_pooled, neg_pooled], axis=0)
                effnet_in = mx.concatenate(
                    [image_embeddings, mx.zeros_like(image_embeddings)], axis=0
                )
            else:
                sample_in = latent
                t_in = t_arr
                pooled_in = text_pooled
                effnet_in = image_embeddings
            pred = self.decoder(
                sample=sample_in,
                timestep_ratio=t_in,
                clip_text_pooled=pooled_in,
                effnet=effnet_in,
                sca=t_in,
            )
            if do_cfg:
                pred_cond, pred_un = mx.split(pred, 2, axis=0)
                pred = pred_un + guidance * (pred_cond - pred_un)
            latent = scheduler.step(pred, t_arr, latent)
            mx.eval(latent)
            logger.info("Cascade decoder step %d/%d done", i + 1, len(timesteps))
        return latent


def _to_pil(image: mx.array) -> Image.Image:
    arr = np.array(image[0].astype(mx.float32))
    arr = np.transpose(arr, (1, 2, 0))
    arr = arr.clip(0, 1)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)
