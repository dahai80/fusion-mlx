# SPDX-License-Identifier: Apache-2.0
import logging
import os
import time

import mlx.core as mx

from .config import MageFlowConfig
from .scheduler import FlowMatchScheduler
from .transformer import MageFlowTransformer
from .vae import MageVAE

logger = logging.getLogger(__name__)


class MageFlowPipeline:
    def __init__(self, config: MageFlowConfig | None = None):
        self.config = config or MageFlowConfig()
        self.transformer: MageFlowTransformer | None = None
        self.vae: MageVAE | None = None
        self.scheduler: FlowMatchScheduler | None = None
        self._loaded = False

    def load(self) -> bool:
        if self._loaded:
            return True
        try:
            t0 = time.perf_counter()
            model_path = self.config.model_path

            if not os.path.isdir(model_path):
                logger.warning(
                    "mage_flow: model path %s not found, skipping load", model_path
                )
                return False

            self.scheduler = FlowMatchScheduler(
                num_steps=self.config.num_steps,
                shift=self.config.static_shift,
            )

            self.transformer = MageFlowTransformer()
            self.vae = MageVAE()

            weights_path = os.path.join(model_path, "transformer", "model.safetensors")
            if os.path.exists(weights_path):
                from mlx.utils import load_safetensors

                weights = load_safetensors(weights_path)
                self.transformer.load_weights(list(weights.items()))
                logger.info("mage_flow: loaded transformer weights from %s", weights_path)
            else:
                logger.warning(
                    "mage_flow: transformer weights not found at %s", weights_path
                )

            vae_path = os.path.join(model_path, "vae", "model.safetensors")
            if os.path.exists(vae_path):
                from mlx.utils import load_safetensors

                weights = load_safetensors(vae_path)
                self.vae.load_weights(list(weights.items()))
                logger.info("mage_flow: loaded VAE weights from %s", vae_path)
            else:
                logger.warning("mage_flow: VAE weights not found at %s", vae_path)

            dt = time.perf_counter() - t0
            self._loaded = True
            logger.info("mage_flow: pipeline loaded in %.1fs from %s", dt, model_path)
            return True
        except Exception as e:
            logger.warning("mage_flow: failed to load: %s", e)
            return False

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        height: int = 1024,
        width: int = 1024,
        num_steps: int | None = None,
        cfg_scale: float | None = None,
        seed: int | None = None,
    ) -> mx.array:
        if not self._loaded or self.transformer is None or self.vae is None:
            raise RuntimeError("MageFlow pipeline not loaded. Call load() first.")

        num_steps = num_steps or self.config.num_steps
        cfg_scale = cfg_scale or self.config.cfg_scale

        if seed is not None:
            mx.random.seed(seed)

        if self.scheduler is None or self.scheduler.num_steps != num_steps:
            self.scheduler = FlowMatchScheduler(
                num_steps=num_steps, shift=self.config.static_shift
            )

        logger.info(
            "mage_flow: generate prompt=%r size=%dx%d steps=%d cfg=%.1f",
            prompt[:80],
            height,
            width,
            num_steps,
            cfg_scale,
        )

        latent_h = height // 8
        latent_w = width // 8

        noise = mx.random.normal(shape=(1, latent_h, latent_w, 16), dtype=mx.float32)
        latent = noise

        txt_emb = self._encode_text(prompt)
        neg_emb = (
            self._encode_text(negative_prompt) if negative_prompt else mx.zeros_like(txt_emb)
        )
        vec = txt_emb.mean(axis=1, keepdims=True)

        for step in range(num_steps):
            sigma = self.scheduler.sigmas[step]
            timestep = mx.array([sigma] * latent.shape[0], dtype=mx.float32)

            img_ids = self._build_img_ids(latent_h, latent_w)

            noise_pred = self.transformer(
                img=latent,
                img_ids=img_ids,
                txt=txt_emb,
                timesteps=timestep,
                vec=vec,
            )

            if cfg_scale > 1.0:
                neg_pred = self.transformer(
                    img=latent,
                    img_ids=img_ids,
                    txt=neg_emb,
                    timesteps=timestep,
                    vec=vec,
                )
                noise_pred = neg_pred + cfg_scale * (noise_pred - neg_pred)

            latent = self.scheduler.step(noise_pred, step, latent)

            if step % 5 == 0:
                logger.debug("mage_flow: step %d/%d", step, num_steps)

        image = self.vae.decode(latent)
        logger.info("mage_flow: generation complete")
        return image

    def _encode_text(self, prompt: str) -> mx.array:
        logger.info("mage_flow: text encoding stub for prompt=%r", prompt[:60])
        return mx.zeros(
            (1, 64, self.transformer.dim if self.transformer else 3072),
            dtype=mx.float32,
        )

    def _build_img_ids(self, h: int, w: int) -> mx.array:
        ids = mx.arange(h * w, dtype=mx.float32).reshape(1, h * w)
        return ids
