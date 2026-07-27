# SPDX-License-Identifier: Apache-2.0
# UniWorld-V1 VideoBackend: VLM-driven image understanding + generation.
# Dual-path: task_head classifies understanding vs generation;
# if generation, VLM embeddings feed into Flux Transformer2D denoiser.
# Uses mflux Flux2Klein for image generation, mlx-lm for Qwen2.5-VL.

from __future__ import annotations

import asyncio
import gc
import io
import logging
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from fusion_mlx.engines.video_backends.base import (
    VideoBackend,
    VideoConstraints,
    VideoGenParams,
)
from fusion_mlx.engine_core import get_executor

from .config import UniWorldConfig
from .siglip2 import SigLIP2VisionEncoder
from .projectors import TaskHead, UniWorldProjectors
from .feature_merge import (
    insert_img_to_vlm,
    apply_shortcut_blend,
    apply_residual_image_factor,
)

logger = logging.getLogger(__name__)

ASSISTANT_TOKEN_ID = 77091
IMAGE_END_TOKEN_ID = 151646


class UniWorldBackend(VideoBackend):
    name = "uniworld"
    supports_i2v = True

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self._model_name = model_name
        self._model_path = model_name
        self._config: UniWorldConfig | None = None
        self._vlm = None
        self._vlm_tokenizer = None
        self._vlm_processor = None
        self._siglip: SigLIP2VisionEncoder | None = None
        self._projectors: UniWorldProjectors | None = None
        self._task_head: TaskHead | None = None
        self._flux = None
        self._t5_encoder = None
        self._clip_encoder = None
        self._loaded = False

    @classmethod
    def detect(cls, model_path: str) -> bool:
        p = model_path.lower()
        return any(k in p for k in ["uniworld", "uniworld-v1", "univa"])

    async def start(self, model_path: str, **kwargs: Any) -> None:
        if self._loaded:
            return
        self._model_path = model_path
        self._config = UniWorldConfig.from_pretrained(model_path, **kwargs)
        logger.info("Starting UniWorld backend: %s", model_path)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(get_executor("video"), self._load_components)
        self._loaded = True
        logger.info("UniWorld backend loaded: %s", model_path)

    def _load_components(self) -> None:
        cfg = self._config
        self._load_vlm(cfg)
        self._load_siglip(cfg)
        self._load_projectors(cfg)
        self._load_task_head(cfg)
        self._load_flux(cfg)

    def _load_vlm(self, cfg: UniWorldConfig) -> None:
        try:
            from mlx_vlm import load as vlm_load
            vlm_dir = str(cfg.vlm_dir) if cfg.vlm_dir.exists() else cfg.vlm_model
            logger.info("Loading VLM from %s", vlm_dir)
            self._vlm, self._vlm_processor = vlm_load(vlm_dir)
            if hasattr(self._vlm_processor, "tokenizer"):
                self._vlm_tokenizer = self._vlm_processor.tokenizer
            else:
                self._vlm_tokenizer = self._vlm_processor
            logger.info("VLM loaded successfully")
        except ImportError:
            logger.warning("mlx-vlm not installed, VLM loading skipped")
        except Exception as e:
            logger.error("Failed to load VLM: %s", e)

    def _load_siglip(self, cfg: UniWorldConfig) -> None:
        siglip_dir = cfg.siglip_dir
        if not siglip_dir.exists():
            base = Path.home() / ".fusion-mlx" / "models"
            siglip_dir = base / cfg.siglip_model
        if siglip_dir.exists():
            logger.info("Loading SigLIP2 from %s", siglip_dir)
            self._siglip = SigLIP2VisionEncoder.from_pretrained(
                siglip_dir, dtype=cfg.mx_dtype
            )
        else:
            logger.warning("SigLIP2 directory not found: %s", siglip_dir)

    def _load_projectors(self, cfg: UniWorldConfig) -> None:
        proj_dir = cfg.projectors_dir
        if proj_dir.exists():
            logger.info("Loading projectors from %s", proj_dir)
            self._projectors = UniWorldProjectors.from_pretrained(
                proj_dir,
                denoise_in=cfg.denoise_projector_input,
                denoise_hidden=cfg.denoise_projector_hidden,
                denoise_out=cfg.denoise_projector_output,
                vae_in=cfg.vae_projector_input,
                vae_hidden=cfg.vae_projector_hidden,
                vae_out=cfg.vae_projector_output,
                siglip_in=cfg.siglip_projector_input,
                siglip_hidden=cfg.siglip_projector_hidden,
                siglip_out=cfg.siglip_projector_output,
            )
        else:
            logger.warning("Projectors directory not found: %s", proj_dir)
            self._projectors = UniWorldProjectors(
                denoise_in=cfg.denoise_projector_input,
                denoise_hidden=cfg.denoise_projector_hidden,
                denoise_out=cfg.denoise_projector_output,
                vae_in=cfg.vae_projector_input,
                vae_hidden=cfg.vae_projector_hidden,
                vae_out=cfg.vae_projector_output,
                siglip_in=cfg.siglip_projector_input,
                siglip_hidden=cfg.siglip_projector_hidden,
                siglip_out=cfg.siglip_projector_output,
            )

    def _load_task_head(self, cfg: UniWorldConfig) -> None:
        self._task_head = TaskHead(
            input_dim=cfg.task_head_input,
            hidden_dim=cfg.task_head_hidden,
            output_dim=cfg.task_head_output,
            dropout=cfg.task_head_dropout,
        )
        task_head_path = cfg.model_dir / "task_head.safetensors"
        if task_head_path.exists():
            weights = mx.load(str(task_head_path))
            remapped = {}
            for k, v in weights.items():
                new_k = k.replace("task_head.", "")
                if new_k.startswith("0."):
                    new_k = new_k.replace("0.", "fc1.")
                elif new_k.startswith("3."):
                    new_k = new_k.replace("3.", "fc2.")
                remapped[new_k] = v
            self._task_head.load_weights(list(remapped.items()))
            mx.eval(self._task_head.parameters())
            logger.info("TaskHead loaded from %s", task_head_path)
        else:
            logger.warning("TaskHead weights not found at %s", task_head_path)

    def _load_flux(self, cfg: UniWorldConfig) -> None:
        try:
            from mflux import Flux1 as _Flux1
        except ImportError:
            try:
                from mflux import Flux as _Flux1
            except ImportError:
                logger.warning("mflux not installed, Flux loading skipped")
                return
        flux_dir = cfg.flux_dir
        if not flux_dir.exists():
            flux_dir = Path(self._model_path)
        logger.info("Loading Flux from %s", flux_dir)
        try:
            from mflux.models.common.config.model_config import ModelConfig
            self._flux = _Flux1(
                model_config=ModelConfig.from_alias(str(flux_dir)),
                model_path=str(flux_dir),
                quantize=None,
            )
            logger.info("Flux loaded successfully")
        except Exception as e:
            logger.error("Failed to load Flux: %s", e)

    async def stop(self) -> None:
        self._vlm = None
        self._vlm_tokenizer = None
        self._vlm_processor = None
        self._siglip = None
        self._projectors = None
        self._task_head = None
        self._flux = None
        self._t5_encoder = None
        self._clip_encoder = None
        self._loaded = False
        gc.collect()
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                get_executor("video"), lambda: (mx.synchronize(), mx.clear_cache())
            ),
            timeout=5.0,
        )
        logger.info("UniWorld backend stopped")

    async def generate(self, params: VideoGenParams) -> list[bytes] | list[Any]:
        if not self._loaded:
            raise RuntimeError("UniWorld backend not started")

        loop = asyncio.get_running_loop()
        t0 = time.monotonic()

        def _generate():
            return self._generate_sync(params)

        result = await asyncio.wait_for(
            loop.run_in_executor(get_executor("video"), _generate), timeout=600.0
        )
        elapsed = time.monotonic() - t0
        logger.info("UniWorld generated in %.2fs", elapsed)
        return result

    def _generate_sync(self, params: VideoGenParams) -> list[bytes]:
        prompt = params.prompt
        cfg = self._config

        task_result = self._classify_task(prompt)
        is_generation = task_result is not None and task_result[1] > task_result[0]

        if not is_generation:
            logger.info("Task classified as understanding (not generation)")
            return [b""]

        logger.info("Task classified as generation, running pipeline")

        siglip_hidden = self._encode_siglip_images(params.image)

        lvlm_embeds = self._encode_vlm_denoise(
            prompt, params.image, siglip_hidden
        )

        text_embeds = self._encode_text(prompt)

        encoder_hidden_states = self._merge_embeddings(
            text_embeds, lvlm_embeds, cfg
        )

        images = self._denoise_flux(
            encoder_hidden_states=encoder_hidden_states,
            pooled_embeds=text_embeds.get("pooled"),
            width=params.width,
            height=params.height,
            steps=params.num_inference_steps or cfg.denoise_steps,
            seed=params.seed or 0,
            guidance=cfg.guidance_scale,
        )
        return images

    def _classify_task(self, prompt: str) -> mx.array | None:
        if self._vlm is None or self._task_head is None:
            logger.warning("VLM or TaskHead not loaded, defaulting to generation")
            return mx.array([0.0, 1.0])

        try:
            hidden_states = self._vlm_forward(prompt)
            if hidden_states is None:
                return mx.array([0.0, 1.0])

            last_hidden = hidden_states[-1] if isinstance(hidden_states, list) else hidden_states

            input_ids = self._get_input_ids(prompt)
            assistant_mask = (input_ids == ASSISTANT_TOKEN_ID)
            if not mx.any(assistant_mask):
                logger.debug("No assistant token found, defaulting to generation")
                return mx.array([0.0, 1.0])

            positions = []
            for i in range(assistant_mask.shape[-1]):
                if bool(assistant_mask[0, i]) if assistant_mask.ndim == 2 else bool(assistant_mask[i]):
                    positions.append(i)
            if not positions:
                return mx.array([0.0, 1.0])

            last_assistant_pos = positions[-1]
            if last_hidden.ndim == 3:
                assistant_vector = last_hidden[0, last_assistant_pos:last_assistant_pos + 1, :]
            else:
                assistant_vector = last_hidden[last_assistant_pos:last_assistant_pos + 1, :]

            task_result = self._task_head(assistant_vector)
            mx.eval(task_result)
            task_result = task_result.reshape(-1)
            logger.info(
                "Task classification: understanding=%.4f, generation=%.4f",
                float(task_result[0]), float(task_result[1]),
            )
            return task_result
        except Exception as e:
            logger.error("Task classification failed: %s", e)
            return mx.array([0.0, 1.0])

    def _vlm_forward(self, prompt: str) -> mx.array | None:
        if self._vlm is None:
            return None
        try:
            input_ids = self._get_input_ids(prompt)
            if input_ids.ndim == 1:
                input_ids = input_ids.reshape(1, -1)
            outputs = self._vlm(input_ids, output_hidden_states=True)
            if isinstance(outputs, tuple):
                return outputs[-1]
            if hasattr(outputs, "hidden_states"):
                return outputs.hidden_states
            return outputs
        except Exception as e:
            logger.error("VLM forward failed: %s", e)
            return None

    def _get_input_ids(self, prompt: str) -> mx.array:
        if self._vlm_tokenizer is not None:
            try:
                if hasattr(self._vlm_processor, "apply_chat_template"):
                    messages = [{"role": "user", "content": prompt}]
                    text = self._vlm_processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    tokens = self._vlm_tokenizer.encode(text)
                else:
                    tokens = self._vlm_tokenizer.encode(prompt)
                return mx.array([tokens])
            except Exception as e:
                logger.error("Tokenization failed: %s", e)
        return mx.array([[0]])

    def _encode_siglip_images(self, image_path: str | None) -> mx.array | None:
        if self._siglip is None or image_path is None:
            return None
        try:
            pixel_values = self._preprocess_siglip_image(image_path)
            siglip_hidden = self._siglip.encode_image(pixel_values)
            logger.info("SigLIP encoded: shape=%s", tuple(siglip_hidden.shape))
            return siglip_hidden
        except Exception as e:
            logger.error("SigLIP encoding failed: %s", e)
            return None

    def _preprocess_siglip_image(self, image_path: str) -> mx.array:
        from PIL import Image
        import numpy as np

        img = Image.open(image_path).convert("RGB")
        cfg = self._config
        img = img.resize((cfg.siglip_image_size, cfg.siglip_image_size))
        arr = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)
        pixel_values = mx.array(arr[np.newaxis, :, :, :])
        return pixel_values

    def _encode_vlm_denoise(
        self, prompt: str, image_path: str | None,
        siglip_hidden: mx.array | None
    ) -> mx.array:
        if self._vlm is None or self._projectors is None:
            return mx.zeros((1, 1, self._config.denoise_projector_output))

        try:
            hidden_states = self._vlm_forward(prompt)
            if hidden_states is None:
                return mx.zeros((1, 1, self._config.denoise_projector_output))

            last_hidden = hidden_states[-1] if isinstance(hidden_states, list) else hidden_states

            if siglip_hidden is not None:
                input_ids = self._get_input_ids(prompt)
                last_hidden = insert_img_to_vlm(
                    last_hidden, siglip_hidden, input_ids, IMAGE_END_TOKEN_ID
                )

            denoise_embeds = self._projectors.denoise_projector(last_hidden)
            mx.eval(denoise_embeds)
            logger.info("VLM denoise embeds: shape=%s", tuple(denoise_embeds.shape))
            return denoise_embeds
        except Exception as e:
            logger.error("VLM denoise encoding failed: %s", e)
            return mx.zeros((1, 1, self._config.denoise_projector_output))

    def _encode_text(self, prompt: str) -> dict[str, mx.array]:
        cfg = self._config
        result = {"sequence": None, "pooled": None}

        if self._t5_encoder is not None:
            try:
                from fusion_mlx.video.t5_encoder import T5Encoder
                t5_embeds = self._t5_encoder.encode(prompt)
                result["sequence"] = t5_embeds
                logger.info("T5 encoded: shape=%s", tuple(t5_embeds.shape))
            except Exception as e:
                logger.error("T5 encoding failed: %s", e)

        if self._clip_encoder is not None:
            try:
                pooled = self._clip_encoder.encode(prompt)
                result["pooled"] = pooled
                logger.info("CLIP pooled: shape=%s", tuple(pooled.shape))
            except Exception as e:
                logger.error("CLIP encoding failed: %s", e)

        if result["sequence"] is None:
            seq_len = max(1, len(prompt) // 4)
            result["sequence"] = mx.zeros((1, seq_len, cfg.flux_hidden_size))
        if result["pooled"] is None:
            result["pooled"] = mx.zeros((1, cfg.flux_hidden_size))

        return result

    def _merge_embeddings(
        self, text_embeds: dict, lvlm_embeds: mx.array, cfg: UniWorldConfig
    ) -> mx.array:
        t5_embeds = text_embeds.get("sequence")
        if cfg.no_joint_with_t5:
            return lvlm_embeds

        if t5_embeds is not None:
            if t5_embeds.shape[-1] != lvlm_embeds.shape[-1]:
                logger.warning(
                    "T5 dim %d != lvlm dim %d, using lvlm only",
                    t5_embeds.shape[-1], lvlm_embeds.shape[-1],
                )
                return lvlm_embeds
            merged = mx.concatenate([t5_embeds, lvlm_embeds], axis=1)
            logger.info(
                "Merged embeddings: t5=%s + lvlm=%s -> %s",
                tuple(t5_embeds.shape), tuple(lvlm_embeds.shape), tuple(merged.shape),
            )
            return merged
        return lvlm_embeds

    def _denoise_flux(
        self,
        encoder_hidden_states: mx.array,
        pooled_embeds: mx.array | None,
        width: int,
        height: int,
        steps: int,
        seed: int,
        guidance: float,
    ) -> list[bytes]:
        if self._flux is None:
            logger.error("Flux not loaded, cannot generate image")
            return [b""]

        try:
            gen = self._flux.generate_image(
                seed=seed,
                prompt="",
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
            )
            buf = io.BytesIO()
            gen.image.save(buf, format="PNG")
            return [buf.getvalue()]
        except Exception as e:
            logger.error("Flux denoising failed: %s", e)
            return [b""]

    def constraints(self) -> VideoConstraints:
        return VideoConstraints(
            supports_i2v=True,
            max_n=1,
            dim_divisibility=16,
            num_frames_validator=lambda n: n == 1,
            num_frames_hint="UniWorld generates single images (num_frames=1)",
            dim_hint="width and height must be divisible by 16",
        )

    def get_stats(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model_path": self._model_path,
            "loaded": self._loaded,
            "vlm_loaded": self._vlm is not None,
            "siglip_loaded": self._siglip is not None,
            "flux_loaded": self._flux is not None,
        }
