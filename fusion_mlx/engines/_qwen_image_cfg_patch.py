# SPDX-License-Identifier: Apache-2.0
# Qwen-Image CFG-free monkeypatch (R2/S4, audit 0910 §3.2/§6.2-5).
#
# Upstream mflux qwen_image.py::generate_image ALWAYS runs 2 DiT forwards
# per step (positive + negative), even when negative_prompt is None (it
# defaults to a placeholder space). Qwen-Image official inference is
# CFG-free / guidance-embedding (single forward). 30 steps x 2 forwards
# = 60 DiT forwards = the "半小时不出图" + 118GB root cause.
#
# This patch replaces QwenImage.generate_image with a CFG-free version
# when FUSION_QWEN_IMAGE_CFG=0: skips the negative DiT forward entirely
# (guided_noise = noise), halving time + peak activation. Upstream issue
# must be filed for a native guidance-embedding path.
#
# Applied only for variant in (qwen_image, qwen_image_edit).
import logging
import os
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)

_CFG_FREE_SENTINEL = "__fusion_qwen_cfg_free_patched__"


def cfg_free_enabled() -> bool:
    raw = os.environ.get("FUSION_QWEN_IMAGE_CFG", "1").strip().lower()
    return raw in ("0", "off", "false", "no")


def _generate_image_cfg_free(
    self,
    seed: int,
    prompt: str,
    num_inference_steps: int = 4,
    height: int = 1024,
    width: int = 1024,
    guidance: float = 4.0,
    image_path: Path | str | None = None,
    image_strength: float | None = None,
    scheduler: str = "linear",
    negative_prompt: str | None = None,
):
    from mflux.models.common.config.config import Config
    from mflux.models.common.latent_creator.latent_creator import (
        Img2Img,
        LatentCreator,
    )
    from mflux.models.common.vae.vae_util import VAEUtil
    from mflux.models.qwen.latent_creator.qwen_latent_creator import (
        QwenLatentCreator,
    )
    from mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import (
        QwenPromptEncoder,
    )
    from mflux.utils.exceptions import StopImageGenerationException
    from mflux.utils.image_util import ImageUtil

    QwenImage = type(self)

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
            latent_creator=QwenLatentCreator,
            sigmas=config.scheduler.sigmas,
            init_time_step=config.init_time_step,
            image_path=config.image_path,
            tiling_config=self.tiling_config,
        ),
    )

    # CFG-free: encode prompt only (skip negative). Pass negative_prompt=None
    # so encoder skips the negative path when it supports that shortcut;
    # otherwise the negative embeds are computed but never fed to DiT.
    prompt_embeds, prompt_mask, _neg_embeds, _neg_mask = (
        QwenPromptEncoder.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            prompt_cache=self.prompt_cache,
            qwen_tokenizer=self.tokenizers["qwen"],
            qwen_text_encoder=self.text_encoder,
        )
    )

    ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
    ctx.before_loop(latents)

    for t in config.time_steps:
        try:
            latents = config.scheduler.scale_model_input(latents, t)

            # Single DiT forward (CFG-free). No negative forward.
            noise = self.transformer(
                t=t,
                config=config,
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_mask,
            )
            guided_noise = noise

            latents = config.scheduler.step(
                noise=guided_noise, timestep=t, latents=latents
            )
            ctx.in_loop(t, latents)
            mx.eval(latents)
        except KeyboardInterrupt:  # noqa: PERF203
            ctx.interruption(t, latents)
            raise StopImageGenerationException(
                f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
            )

    ctx.after_loop(latents)

    latents = QwenLatentCreator.unpack_latents(
        latents=latents, height=config.height, width=config.width
    )
    decoded = VAEUtil.decode(
        vae=self.vae, latent=latents, tiling_config=self.tiling_config
    )
    return ImageUtil.to_image(
        decoded_latents=decoded,
        config=config,
        seed=seed,
        prompt=prompt,
        quantization=self.bits,
        lora_paths=self.lora_paths,
        lora_scales=self.lora_scales,
        image_path=config.image_path,
        image_strength=config.image_strength,
        generation_time=config.time_steps.format_dict["elapsed"],
        negative_prompt=negative_prompt,
    )


def apply_qwen_image_cfg_patch(flux_obj) -> bool:
    # Patch QwenImage.generate_image to CFG-free single-forward version.
    # Returns True if patched. Idempotent (sentinel-tagged).
    cls = type(flux_obj)
    cls_name = cls.__name__
    if cls_name not in ("QwenImage", "QwenImageEdit"):
        return False
    if not cfg_free_enabled():
        return False
    existing = getattr(cls.generate_image, "__doc__", "") or ""
    if _CFG_FREE_SENTINEL in existing:
        return True
    logger.info(
        "Qwen-Image CFG-free patch: replacing %s.generate_image with "
        "single-forward version (FUSION_QWEN_IMAGE_CFG=0). ~2x speedup, "
        "halved peak activation. Upstream issue pending for native "
        "guidance-embedding path.",
        cls_name,
    )
    _generate_image_cfg_free.__doc__ = (
        f"{_CFG_FREE_SENTINEL} CFG-free patched generate_image."
    )
    cls.generate_image = _generate_image_cfg_free
    return True
