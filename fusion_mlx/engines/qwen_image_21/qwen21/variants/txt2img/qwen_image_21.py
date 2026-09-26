# SPDX-License-Identifier: Apache-2.0
# Vendored from mflux 0.20.0 (https://github.com/filipstrand/mflux) PR#736
# MIT-licensed pure-MLX Qwen-Image-2.1 reference (txt2img + 4-ch RGBA VAE).
# Pinned to fusion-mlx's mlx stack. Upstream-tracked; do not modify forward
# numerics without parity-testing against mflux qwen21.

from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from ...latent_creator.qwen21_latent_creator import Qwen21LatentCreator
from ...model.qwen21_text_encoder.qwen21_prompt_encoder import Qwen21PromptEncoder
from ...model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
from ...model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from ...model.qwen21_vae.qwen21_vae import Qwen21VAE
from ...qwen21_initializer import Qwen21Initializer
from ...weights.qwen21_weight_definition import Qwen21WeightDefinition
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class QwenImage21(nn.Module):
    vae: Qwen21VAE
    transformer: Qwen21Transformer
    text_encoder: Qwen21TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = ModelConfig.qwen_image_21(),
    ):
        super().__init__()
        # fusion-mlx ImageGenEngine native-load branch passes model_config=None;
        # fall back to the vendored qwen_image_21 config (4-ch RGBA VAE) so the
        # mflux weight mapping + sigma-shift scheduler resolve correctly.
        if model_config is None:
            model_config = ModelConfig.qwen_image_21()
        Qwen21Initializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            model_config=model_config,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 40,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 1.0,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str = "linear",
        negative_prompt: str | None = None,
        transparent: bool = False,
    ) -> GeneratedImage:
        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            image_path=image_path,
            image_strength=image_strength,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
        )

        latents = LatentCreator.create_for_txt2img_or_img2img(
            seed=seed,
            width=config.width,
            height=config.height,
            img2img=Img2Img(
                vae=self.vae,
                latent_creator=Qwen21LatentCreator,
                sigmas=config.scheduler.sigmas,
                init_time_step=config.init_time_step,
                image_path=config.image_path,
                tiling_config=self.tiling_config,
            ),
        )
        # img2img noise interpolation promotes to fp32; keep the stream in model precision
        latents = latents.astype(ModelConfig.precision)

        prompt_embeds, prompt_mask = Qwen21PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache=self.prompt_cache,
            tokenizer=self.tokenizers["qwen21"],
            text_encoder=self.text_encoder,
        )
        # like the reference pipeline, a missing (or empty) negative prompt disables true CFG
        do_true_cfg = config.guidance > 1.0 and bool(negative_prompt)
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_mask = Qwen21PromptEncoder.encode_prompt(
                prompt=negative_prompt,
                prompt_cache=self.prompt_cache,
                tokenizer=self.tokenizers["qwen21"],
                text_encoder=self.text_encoder,
            )

        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)

        for t in config.time_steps:
            try:
                latents = config.scheduler.scale_model_input(latents, t)
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_mask,
                )
                if do_true_cfg:
                    noise_negative = self.transformer(
                        t=t,
                        config=config,
                        hidden_states=latents,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_mask=negative_prompt_mask,
                    )
                    noise = noise_negative + config.guidance * (noise - noise_negative)

                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
                ctx.in_loop(t, latents)
                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        ctx.after_loop(latents)

        latents = Qwen21LatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
        # fusion-mlx transparent-PNG: tell the VAE to keep the 4th (alpha)
        # channel so the decoded RGBA latents become an RGBA PIL image.
        self.vae.transparent_output = bool(transparent)
        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            generation_time=config.time_steps.format_dict["elapsed"],
            negative_prompt=negative_prompt,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=Qwen21WeightDefinition,
        )
