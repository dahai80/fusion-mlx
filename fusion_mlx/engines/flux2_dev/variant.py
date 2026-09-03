import logging
import os
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from fusion_mlx.engines.flux2_dev.text_encoder import Mistral3TextEncoder
from fusion_mlx.engines.flux2_dev.tokenizer import load_mistral_tokenizer
from fusion_mlx.engines.flux2_dev.weights import load_dit, load_text_encoder, load_vae

logger = logging.getLogger(__name__)

_DEV_TRANSFORMER_OVERRIDES = dict(
    patch_size=1,
    in_channels=128,
    num_layers=8,
    num_single_layers=48,
    attention_head_dim=128,
    num_attention_heads=48,
    joint_attention_dim=15360,
    timestep_guidance_channels=256,
    mlp_ratio=3.0,
    axes_dims_rope=(32, 32, 32, 32),
    rope_theta=2000,
    guidance_embeds=True,
)

_TEXT_ENCODER_REPO = "Comfy-Org/flux2-dev"
_TEXT_ENCODER_FILE = "split_files/text_encoders/mistral_3_small_flux2_bf16.safetensors"


@dataclass
class _DevModelConfig:
    num_train_steps: int = 1000
    max_sequence_length: int = 512
    supports_guidance: bool = True
    requires_sigma_shift: bool = True
    sigma_base_shift: float = 0.5
    sigma_max_shift: float = 1.15
    sigma_base_seq_len: int = 256
    sigma_max_seq_len: int = 4096
    sigma_shift_terminal: float | None = None
    precision: mx.Dtype = mx.bfloat16


def _resolve_text_encoder_path():
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from huggingface_hub import snapshot_download

    local_root = os.path.expanduser("~/.fusion-mlx/models")
    snap = snapshot_download(
        repo_id=_TEXT_ENCODER_REPO,
        allow_patterns=[_TEXT_ENCODER_FILE],
        cache_dir=local_root,
    )
    te_path = Path(snap) / _TEXT_ENCODER_FILE
    logger.info(
        "flux2_dev variant: text encoder resolved to %s exists=%s",
        te_path,
        te_path.exists(),
    )
    return te_path


class Flux2Dev(nn.Module):
    vae: nn.Module
    transformer: nn.Module
    text_encoder: nn.Module

    def __init__(
        self,
        model_config=None,
        model_path=None,
        quantize=None,
        lora_paths=None,
        lora_scales=None,
    ):
        super().__init__()
        from mflux.callbacks.callback_registry import CallbackRegistry
        from mflux.models.flux2.model.flux2_transformer.transformer import (
            Flux2Transformer,
        )
        from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE

        self.model_config = _DevModelConfig()
        self.callbacks = CallbackRegistry()
        self.tiling_config = None
        self.prompt_cache = {}
        self.bits = quantize
        self.lora_paths = lora_paths or []
        self.lora_scales = lora_scales or []
        logger.info(
            "flux2_dev variant init: model_path=%s quantize=%s",
            model_path,
            quantize,
        )
        self.transformer = Flux2Transformer(**_DEV_TRANSFORMER_OVERRIDES)
        self.vae = Flux2VAE()
        self.text_encoder = Mistral3TextEncoder(
            hidden_size=5120,
            num_hidden_layers=30,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            intermediate_size=32768,
            vocab_size=131072,
            rope_theta=1_000_000_000.0,
            rms_norm_eps=1e-5,
        )
        transformer_dir = Path(model_path) / "transformer"
        vae_dir = Path(model_path) / "vae"
        bits = int(quantize) if quantize else 8
        load_dit(self.transformer, transformer_dir, quantize_bits=bits)
        load_vae(self.vae, vae_dir, quantize_bits=bits)
        te_path = _resolve_text_encoder_path()
        load_text_encoder(self.text_encoder, te_path)
        self.tokenizers = {"mistral3": load_mistral_tokenizer(max_length=512)}
        logger.info("flux2_dev variant: all components loaded")

    def _encode_prompt_pair(self, *, prompt, negative_prompt, guidance):
        from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import (
            Flux2PromptEncoder,
        )

        prompt_embeds, text_ids = Flux2PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=self.tokenizers["mistral3"],
            text_encoder=self.text_encoder,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )
        negative_prompt_embeds = None
        negative_text_ids = None
        if guidance is not None and guidance > 1.0 and negative_prompt is not None:
            neg_embeds, neg_ids = Flux2PromptEncoder.encode_prompt(
                prompt=negative_prompt,
                tokenizer=self.tokenizers["mistral3"],
                text_encoder=self.text_encoder,
                num_images_per_prompt=1,
                max_sequence_length=512,
                text_encoder_out_layers=(9, 18, 27),
            )
            negative_prompt_embeds = neg_embeds
            negative_text_ids = neg_ids
        return (
            prompt_embeds,
            text_ids,
            negative_prompt_embeds,
            negative_text_ids,
        )

    def _predict(self, transformer):
        def predict(
            latents,
            latent_ids,
            prompt_embeds,
            text_ids,
            negative_prompt_embeds,
            negative_text_ids,
            guidance,
            timestep,
        ):
            noise = transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                img_ids=latent_ids,
                txt_ids=text_ids,
                guidance=guidance,
            )
            if negative_prompt_embeds is not None and negative_text_ids is not None:
                negative_noise = transformer(
                    hidden_states=latents,
                    encoder_hidden_states=negative_prompt_embeds,
                    timestep=timestep,
                    img_ids=latent_ids,
                    txt_ids=negative_text_ids,
                    guidance=guidance,
                )
                noise = negative_noise + guidance * (noise - negative_noise)
            return noise

        return predict

    def _prepare_generation_latents(self, *, seed, config):
        from mflux.models.flux2.latent_creator.flux2_latent_creator import (
            Flux2LatentCreator,
        )

        return Flux2LatentCreator.prepare_packed_latents(
            seed=seed,
            height=config.height,
            width=config.width,
            batch_size=1,
        )

    def generate_image(
        self,
        seed,
        prompt,
        num_inference_steps=20,
        height=1024,
        width=1024,
        guidance=3.5,
        image_path=None,
        image_strength=None,
        scheduler="flow_match_euler_discrete",
    ):
        from mflux.models.common.config.config import Config
        from mflux.utils.exceptions import StopImageGenerationException
        from mflux.utils.image_util import ImageUtil

        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=image_path,
            image_strength=image_strength,
            scheduler=scheduler,
        )
        logger.info(
            "flux2_dev generate: steps=%d %dx%d guidance=%s",
            num_inference_steps,
            height,
            width,
            guidance,
        )
        prompt_embeds, text_ids, neg_embeds, neg_ids = self._encode_prompt_pair(
            prompt=prompt,
            negative_prompt=" ",
            guidance=guidance,
        )
        latents, latent_ids, latent_height, latent_width = (
            self._prepare_generation_latents(seed=seed, config=config)
        )
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        predict = self._predict(self.transformer)
        for t in config.time_steps:
            try:
                noise = predict(
                    latents=latents,
                    latent_ids=latent_ids,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    negative_prompt_embeds=neg_embeds,
                    negative_text_ids=neg_ids,
                    guidance=guidance,
                    timestep=config.scheduler.timesteps[t],
                )
                latents = config.scheduler.step(
                    noise=noise,
                    timestep=t,
                    latents=latents,
                    sigmas=config.scheduler.sigmas,
                )
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )
        ctx.after_loop(latents)
        packed_latents = latents.reshape(
            latents.shape[0],
            latent_height,
            latent_width,
            latents.shape[-1],
        ).transpose(0, 3, 1, 2)
        decoded = self.vae.decode_packed_latents(packed_latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            negative_prompt=None,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            image_path=config.image_path,
            image_strength=config.image_strength,
            generation_time=config.time_steps.format_dict["elapsed"],
        )
