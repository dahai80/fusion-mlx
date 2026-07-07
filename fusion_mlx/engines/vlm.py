# SPDX-License-Identifier: Apache-2.0
"""VLM (Vision-Language Model) engine with continuous batching."""

import asyncio
import copy
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..engine_core import AsyncEngineCore, EngineConfig, get_executor
from ..models.vlm import VLMModelAdapter
from ..utils.image import (
    compute_image_hash,
    compute_per_image_hashes,
    extract_images_from_messages,
)
from ..utils.video import (
    DEFAULT_FPS,
    MAX_FRAMES,
    extract_video_frames_smart,
    process_video_input,
    save_frames_to_temp,
)
from .base import BaseEngine, GenerationOutput, _fallback_parse_tool_calls

logger = logging.getLogger(__name__)


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


OCR_MODEL_TYPES = {"deepseekocr", "deepseekocr_2", "dots_ocr", "glm_ocr"}

OCR_MODEL_PROMPTS: dict[str, str] = {
    "deepseekocr": "Convert the document to markdown.",
    "deepseekocr_2": "Convert the document to markdown.",
    "dots_ocr": "Convert this page to clean Markdown while preserving reading order.",
    "glm_ocr": "Text Recognition:",
}

OCR_EXTRA_STOP_SEQUENCES = ["<|user|>", "##", "\n", "<|endofassistant|>"]

OCR_MODEL_GENERATION_DEFAULTS: dict[str, dict[str, Any]] = {
    "glm_ocr": {"temperature": 0.0, "repetition_penalty": 1.1, "max_tokens": 4096},
    "deepseekocr": {"temperature": 0.0, "max_tokens": 8192},
    "deepseekocr_2": {"temperature": 0.0, "max_tokens": 8192},
    "dots_ocr": {"temperature": 0.0, "max_tokens": 8192},
}

_SINGLE_IMAGE_ONLY = {
    "llava_next",
    "llava-qwen2",
    "bunny-llama",
    "paligemma",
    "multi_modality",
    "mllama",
}

_QWEN_VISION_MODELS = {
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen2_vl",
    "qwen2_5_vl",
}


class VLMBatchedEngine(BaseEngine):
    """VLM engine with continuous batching, tiered KV cache, and vision feature caching."""

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = False,
        scheduler_config: Any | None = None,
        stream_interval: int = 1,
        enable_thinking: bool | None = None,
        preserve_thinking: bool | None = None,
        model_settings: Any | None = None,
        prefill_eviction_callback: Any | None = None,
    ):
        super().__init__()
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._scheduler_config = scheduler_config
        self._stream_interval = stream_interval
        self._enable_thinking = enable_thinking
        self._preserve_thinking = preserve_thinking
        self._model_settings = model_settings
        self._prefill_eviction_callback = prefill_eviction_callback

        self._vlm_model = None
        self._processor = None
        self._tokenizer = None
        self._adapter = None
        self._engine = None
        self._loaded = False
        self._vision_cache = None
        self._vision_cache_enabled = True
        self._grammar_compiler = None
        self._grammar_compiler_init_attempted = False
        self._vlm_mtp_drafter = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def is_mllm(self) -> bool:
        return True

    @property
    def model_type(self) -> str | None:
        if self._vlm_model is not None and hasattr(self._vlm_model, "config"):
            cfg = self._vlm_model.config
            if hasattr(cfg, "model_type"):
                return cfg.model_type
        return None

    @property
    def is_ocr_model(self) -> bool:
        return (self.model_type or "") in OCR_MODEL_TYPES

    @property
    def prefix_cache_enabled(self) -> bool:
        if self._engine is None:
            return False
        try:
            return self._engine.engine.scheduler.block_aware_cache is not None
        except AttributeError:
            return False

    @property
    def grammar_compiler(self):
        if self._grammar_compiler is not None:
            return self._grammar_compiler
        if self._grammar_compiler_init_attempted:
            return None
        self._grammar_compiler_init_attempted = True
        try:
            from ..api.grammar import create_grammar_compiler

            self._grammar_compiler = create_grammar_compiler(
                self._tokenizer, self._vlm_model
            )
            logger.info("GrammarCompiler initialized for %s", self._model_name)
        except Exception as e:
            logger.info("GrammarCompiler init failed for %s: %s", self._model_name, e)
        return self._grammar_compiler

    @property
    def message_extractor(self):
        try:
            from ..parsers.output_parser import detect_message_extractor

            model_config = getattr(self._vlm_model, "config", None)
            return detect_message_extractor(self._model_name, model_config)
        except Exception:
            return None

    @property
    def supports_tool_calling(self) -> bool:
        return bool(getattr(self._tokenizer, "has_tool_calling", False))

    def set_vlm_mtp_drafter(self, drafter: Any) -> None:
        self._vlm_mtp_drafter = drafter
        if self._engine is not None:
            try:
                scheduler = self._engine.engine.scheduler
                block_size = None
                if self._model_settings:
                    block_size = getattr(
                        self._model_settings, "vlm_mtp_draft_block_size", None
                    )
                if hasattr(scheduler, "set_vlm_mtp_drafter"):
                    scheduler.set_vlm_mtp_drafter(drafter, draft_block_size=block_size)
            except Exception as e:
                logger.warning("Failed to set VLM MTP drafter on scheduler: %s", e)

    @property
    def vlm_mtp_drafter(self) -> Any | None:
        return self._vlm_mtp_drafter

    def _inject_tool_calling(self, tokenizer) -> None:
        chat_template = getattr(tokenizer, "chat_template", None)
        if not chat_template:
            return
        try:
            from mlx_vlm.tool_parsers import (
                _infer_tool_parser,
                load_tool_module,
            )

            tool_parser_type = _infer_tool_parser(chat_template)
            if tool_parser_type is None:
                return
            try:
                tool_module = load_tool_module(tool_parser_type)
            except ImportError:
                logger.warning("VLM tool parser module not found: %s", tool_parser_type)
                return
        except ImportError:
            try:
                import importlib

                from mlx_lm.tokenizer_utils import (
                    _infer_tool_parser as _mlx_lm_infer,
                )
            except ImportError:
                return
            tool_parser_type = _mlx_lm_infer(chat_template)
            if tool_parser_type is None:
                return
            try:
                tool_module = importlib.import_module(
                    f"mlx_lm.tool_parsers.{tool_parser_type}"
                )
            except ImportError:
                logger.warning("VLM tool parser module not found: %s", tool_parser_type)
                return

        tool_call_start = tool_module.tool_call_start
        tool_call_end = tool_module.tool_call_end
        vocab = tokenizer.get_vocab()
        if (tool_call_start and tool_call_start not in vocab) or (
            tool_call_end and tool_call_end not in vocab
        ):
            return
        tokenizer.has_tool_calling = True
        tokenizer.tool_call_start = tool_call_start
        tokenizer.tool_call_end = tool_call_end
        tokenizer.tool_parser = tool_module.parse_tool_call
        logger.info("VLM tool calling enabled: parser=%s", tool_parser_type)

    async def start(self) -> None:
        if self._loaded:
            return

        from mlx_vlm.utils import load as vlm_load

        def _load_vlm_sync():
            start = time.monotonic()
            logger.info("Loading VLM model: %s", self._model_name)
            model, processor = vlm_load(
                self._model_name, trust_remote_code=self._trust_remote_code
            )
            elapsed = time.monotonic() - start
            # Estimate model size
            try:
                from mlx.utils import tree_flatten

                params = tree_flatten(model.parameters())
                total_bytes = sum(arr.size * arr.itemsize for _, arr in params)
            except Exception:
                total_bytes = 0
            size_str = _human_size(total_bytes)
            logger.info(
                "VLM model loaded in %.1fs | %s | %s",
                elapsed,
                size_str,
                self._model_name,
            )
            return model, processor

        loop = asyncio.get_running_loop()
        self._vlm_model, self._processor = await asyncio.wait_for(
            loop.run_in_executor(get_executor("io"), _load_vlm_sync), timeout=120.0
        )

        # Vision feature cache
        vision_ssd_dir = None
        if self._scheduler_config and getattr(
            self._scheduler_config, "paged_ssd_cache_dir", None
        ):
            vision_ssd_dir = (
                Path(self._scheduler_config.paged_ssd_cache_dir) / "vision_features"
            )
        try:
            from ..cache.vision_feature_cache import VisionFeatureSSDCache

            self._vision_cache = VisionFeatureSSDCache(
                cache_dir=vision_ssd_dir, max_memory_entries=20
            )
        except ImportError:
            logger.debug("VisionFeatureSSDCache not available, vision caching disabled")
        logger.info(
            "Vision feature cache enabled (SSD: %s)", vision_ssd_dir or "disabled"
        )

        # Deep-copy tokenizer for thread safety
        if hasattr(self._processor, "tokenizer"):
            self._tokenizer = copy.deepcopy(self._processor.tokenizer)
        else:
            self._tokenizer = copy.deepcopy(self._processor)

        # Create adapter wrapping language_model
        self._adapter = VLMModelAdapter(self._vlm_model)

        # Scheduler + engine
        scheduler_config = (
            copy.copy(self._scheduler_config) if self._scheduler_config else None
        )
        if scheduler_config:
            scheduler_config.model_name = self._model_name
        engine_config = EngineConfig(
            model_name=self._model_name,
            scheduler_config=scheduler_config,
            stream_interval=self._stream_interval,
        )
        self._engine = AsyncEngineCore(
            model=self._adapter, tokenizer=self._tokenizer, config=engine_config
        )
        await self._engine.engine.start()

        # SpecPrefill: load draft model if configured
        if self._model_settings is not None:
            specprefill_draft = getattr(
                self._model_settings, "specprefill_draft_model", None
            )
            specprefill_enabled = getattr(
                self._model_settings, "specprefill_enabled", False
            )
            if specprefill_enabled and specprefill_draft:
                try:
                    from mlx_lm import load as mlx_lm_load

                    def _load_specprefill_draft():
                        draft_model, _ = mlx_lm_load(
                            specprefill_draft,
                            trust_remote_code=self._trust_remote_code,
                        )
                        return draft_model

                    draft_model = await loop.run_in_executor(
                        get_executor("io"), _load_specprefill_draft
                    )
                    self._engine.engine.scheduler.set_specprefill_draft_model(
                        draft_model, draft_model_name=specprefill_draft
                    )
                    logger.info(
                        "SpecPrefill: draft model loaded (%s)", specprefill_draft
                    )
                except Exception as e:
                    logger.error("SpecPrefill: draft model load failed: %s", e)

        # TurboQuant KV cache
        if self._model_settings is not None:
            tq_enabled = getattr(self._model_settings, "turboquant_kv_enabled", False)
            if tq_enabled:
                try:
                    from mlx_vlm.turboquant import turboquant_attention

                    tq_bits = getattr(self._model_settings, "turboquant_kv_bits", 4)
                    tq_skip = getattr(
                        self._model_settings, "turboquant_skip_last", True
                    )
                    turboquant_attention(
                        self._vlm_model.language_model,
                        kv_bits=tq_bits,
                        skip_last=tq_skip,
                    )
                    logger.info("TurboQuant KV cache enabled for VLM: %s bits", tq_bits)
                except Exception as e:
                    logger.warning("TurboQuant KV init failed: %s", e)

        # Inject tool calling support into VLM tokenizer
        self._inject_tool_calling(self._tokenizer)

        self._loaded = True
        from ..scheduler.helpers import register_llm_engine

        register_llm_engine()
        logger.info("VLMBatchedEngine loaded: %s", self._model_name)

    async def stop(self) -> None:
        if self._engine:
            await self._engine.stop()
            if hasattr(self._engine, "engine") and self._engine.engine is not None:
                try:
                    self._engine.engine.close()
                except Exception as e:
                    logger.warning("Error closing engine: %s", e)
        if self._vision_cache is not None:
            self._vision_cache.close()
            self._vision_cache = None
        self._engine = None
        self._vlm_model = None
        self._processor = None
        self._adapter = None
        self._tokenizer = None
        self._grammar_compiler = None
        self._grammar_compiler_init_attempted = False
        self._vlm_mtp_drafter = None
        if self._loaded:
            from ..scheduler.helpers import unregister_llm_engine

            unregister_llm_engine()
        self._loaded = False

    # -- Vision feature computation --

    def _compute_vision_features(
        self, pixel_values: Any, extra_model_inputs: dict
    ) -> Any | None:
        model = self._vlm_model
        model_type = self.model_type or ""

        # Strategy 1: upstream encode_image
        if hasattr(model, "encode_image"):
            return model.encode_image(pixel_values)

        # Strategy 2: qwen-style (vision_tower + grid_thw)
        if model_type in _QWEN_VISION_MODELS:
            grid_thw = extra_model_inputs.get(
                "image_grid_thw"
            ) or extra_model_inputs.get("video_grid_thw")
            if grid_thw is None:
                return None
            dtype = model.vision_tower.patch_embed.proj.weight.dtype
            pv = (
                mx.array(pixel_values)
                if not isinstance(pixel_values, mx.array)
                else pixel_values
            )
            pv = pv.astype(dtype)
            result = model.vision_tower(pv, grid_thw)
            return result[0] if isinstance(result, tuple) else result

        # Strategy 3: llava-style
        if model_type == "llava":
            pv = (
                mx.array(pixel_values)
                if not isinstance(pixel_values, mx.array)
                else pixel_values
            )
            _, *hidden_states = model.vision_tower(
                pv.transpose(0, 2, 3, 1), output_hidden_states=True
            )
            selected = hidden_states[model.vision_feature_layer]
            if isinstance(model.vision_feature_layer, int):
                if (
                    getattr(model, "vision_feature_select_strategy", "default")
                    == "default"
                ):
                    selected = selected[:, 1:]
            else:
                hs_pool = [hidden_states[idx] for idx in model.vision_feature_layer]
                if (
                    getattr(model, "vision_feature_select_strategy", "default")
                    == "default"
                ):
                    hs_pool = [hs[:, 1:] for hs in hs_pool]
                selected = mx.concatenate(hs_pool, axis=-1)
            return model.multi_modal_projector(selected)

        return None

    def _split_vision_features(
        self, features: mx.array, num_images: int, extra_model_inputs: dict
    ) -> list[mx.array] | None:
        if num_images <= 1:
            return [features]

        model_type = self.model_type or ""
        if features.ndim >= 3 and features.shape[0] == num_images:
            return [features[i : i + 1] for i in range(num_images)]

        if model_type in _QWEN_VISION_MODELS and features.ndim == 2:
            grid_thw = extra_model_inputs.get("image_grid_thw")
            if grid_thw is None:
                return None
            spatial_merge_size = getattr(
                self._vlm_model.vision_tower, "spatial_merge_size", 2
            )
            merge_sq = spatial_merge_size**2
            per_image_tokens = []
            for i in range(num_images):
                t, h, w = int(grid_thw[i, 0]), int(grid_thw[i, 1]), int(grid_thw[i, 2])
                per_image_tokens.append((t * h * w) // merge_sq)
            if sum(per_image_tokens) != features.shape[0]:
                return None
            result, offset = [], 0
            for count in per_image_tokens:
                result.append(features[offset : offset + count])
                offset += count
            return result

        return None

    # -- Vision input preparation --

    def _prepare_vision_inputs(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
    ) -> tuple:
        """Run VLM preprocessing: tokenize, preprocess images, compute embeddings, cache.

        Returns (token_ids, inputs_embeds, extra_kwargs, image_hash, image_cache_key_start, image_cache_key_ranges).
        """
        from mlx_vlm.utils import prepare_inputs

        num_images = len(images)
        model_type = self.model_type or ""

        if num_images > 1 and model_type in _SINGLE_IMAGE_ONLY:
            raise ValueError(
                f"Model {model_type} does not support multi-image chat. Use only 1 image."
            )

        # Apply chat template
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self._enable_thinking is not None:
            template_kwargs["enable_thinking"] = self._enable_thinking

        template_target = self._processor
        if not hasattr(template_target, "apply_chat_template"):
            template_target = getattr(self._processor, "tokenizer", self._processor)

        # Ensure exactly one system message at the beginning
        systems = [m for m in messages if m.get("role") in ("system", "developer")]
        others = [m for m in messages if m.get("role") not in ("system", "developer")]
        if systems:
            sys_text = "\n\n".join(
                m.get("content", "") for m in systems if m.get("content")
            )
            messages = [{"role": "system", "content": sys_text}] + others
        elif others:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."}
            ] + others

        logger.debug(
            "VLM template: roles=%s", [m.get("role", "?") for m in messages[:3]]
        )

        try:
            prompt = template_target.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            prompt = template_target.apply_chat_template(messages, **template_kwargs)

        # Tokenize text and preprocess images
        inputs = prepare_inputs(
            self._processor, images=images or None, prompts=[prompt]
        )
        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")
        attention_mask = inputs.get("attention_mask")

        extra_model_inputs = {
            k: v
            for k, v in inputs.items()
            if k not in ("input_ids", "attention_mask", "pixel_values")
            and v is not None
        }

        if pixel_values is not None and num_images > 0:
            image_hash = compute_image_hash(images)
            call_kwargs = dict(extra_model_inputs)

            # Vision feature cache lookup / compute / store
            if self._vision_cache is not None and self._vision_cache_enabled:
                per_hashes = compute_per_image_hashes(images)
                cached_per_image = [
                    self._vision_cache.get(h, self._model_name) for h in per_hashes
                ]

                if all(f is not None for f in cached_per_image):
                    call_kwargs["cached_image_features"] = mx.concatenate(
                        cached_per_image, axis=0
                    )
                else:
                    cached_whole = self._vision_cache.get(image_hash, self._model_name)
                    if cached_whole is not None:
                        call_kwargs["cached_image_features"] = cached_whole
                    else:
                        try:
                            features = self._compute_vision_features(
                                pixel_values, extra_model_inputs
                            )
                            if features is not None:
                                mx.eval(features)
                                call_kwargs["cached_image_features"] = features
                                per_features = self._split_vision_features(
                                    features, num_images, extra_model_inputs
                                )
                                if per_features is not None:
                                    for h, f in zip(per_hashes, per_features):
                                        self._vision_cache.put(h, self._model_name, f)
                                else:
                                    self._vision_cache.put(
                                        image_hash, self._model_name, features
                                    )
                        except Exception:
                            logger.debug(
                                "Vision feature computation failed, using full pipeline",
                                exc_info=True,
                            )

            # Run vision encoder + embedding merge
            try:
                embed_features = self._vlm_model.get_input_embeddings(
                    input_ids,
                    pixel_values,
                    mask=attention_mask,
                    **call_kwargs,
                )
            except TypeError:
                if "cached_image_features" in call_kwargs:
                    logger.warning(
                        "cached_image_features not supported by %s, disabling",
                        model_type,
                    )
                    self._vision_cache_enabled = False
                    call_kwargs.pop("cached_image_features")
                    embed_features = self._vlm_model.get_input_embeddings(
                        input_ids, pixel_values, mask=attention_mask, **call_kwargs
                    )
                else:
                    raise

            mx.eval(embed_features.inputs_embeds)

            # Extract extra kwargs from embed_features
            extra_kwargs = {}
            if hasattr(embed_features, "to_dict"):
                for k, v in embed_features.to_dict().items():
                    if k != "inputs_embeds" and v is not None:
                        extra_kwargs[k] = v

            # Capture per-request mRoPE state
            lm = getattr(self._vlm_model, "language_model", None)
            if lm is not None:
                pid = getattr(lm, "_position_ids", None)
                if pid is not None and "position_ids" not in extra_kwargs:
                    extra_kwargs["position_ids"] = pid
                rd = getattr(lm, "_rope_deltas", None)
                if rd is not None:
                    extra_kwargs["_captured_rope_deltas"] = rd

            token_ids = (
                input_ids[0].tolist() if input_ids.ndim > 1 else input_ids.tolist()
            )
            return (
                token_ids,
                embed_features.inputs_embeds,
                extra_kwargs,
                image_hash,
                0,
                [],
            )
        else:
            token_ids = (
                input_ids[0].tolist() if input_ids.ndim > 1 else input_ids.tolist()
            )
            return token_ids, None, None, None, 0, []

    def _is_native_video_model(self) -> bool:
        config = getattr(self._vlm_model, "config", None)
        if config is None:
            return False
        return hasattr(config, "video_token_id") or hasattr(config, "video_token_index")

    def _prepare_native_video_inputs(
        self,
        messages: list[dict[str, Any]],
        videos: list[str | dict],
        video_fps: float = DEFAULT_FPS,
        video_max_frames: int = MAX_FRAMES,
    ) -> tuple:
        """Use mlx_vlm native video path for Qwen-family models."""
        from mlx_vlm.utils import prepare_inputs

        from ..utils.video import process_video_input

        # Apply chat template
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self._enable_thinking is not None:
            template_kwargs["enable_thinking"] = self._enable_thinking

        template_target = self._processor
        if not hasattr(template_target, "apply_chat_template"):
            template_target = getattr(self._processor, "tokenizer", self._processor)

        try:
            prompt = template_target.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            prompt = template_target.apply_chat_template(messages, **template_kwargs)

        # Load video frames natively via mlx_vlm.
        # load_video(ele) expects {"video": path, "fps", "max_frames"} and
        # returns (ndarray (T,C,H,W), sample_fps).
        all_images = []

        def _fallback_cv2_frames(path: str) -> None:
            frames = extract_video_frames_smart(
                path, fps=video_fps, max_frames=video_max_frames
            )
            frame_paths = save_frames_to_temp(frames)
            from ..utils.image import load_image

            for fp in frame_paths:
                all_images.append(load_image(fp))
            logger.info("Fallback video: %d frames from %s", len(frame_paths), path)

        for video in videos:
            try:
                video_path = process_video_input(video)
                try:
                    from mlx_vlm.video_generate import load_video

                    ele = {
                        "video": video_path,
                        "fps": video_fps,
                        "max_frames": video_max_frames,
                    }
                    video_np, _sample_fps = load_video(ele)
                    import numpy as np
                    from PIL import Image

                    for t in range(video_np.shape[0]):
                        frame = np.transpose(video_np[t], (1, 2, 0))
                        all_images.append(Image.fromarray(frame.astype(np.uint8)))
                    logger.info(
                        "Native video: %d frames from %s",
                        int(video_np.shape[0]),
                        video_path,
                    )
                except ImportError:
                    logger.warning(
                        "mlx_vlm.video_generate unavailable, fallback to cv2: %s",
                        video_path,
                    )
                    _fallback_cv2_frames(video_path)
                except Exception as e:
                    logger.warning(
                        "Native load_video failed for %s: %s; fallback to cv2",
                        video_path,
                        e,
                    )
                    _fallback_cv2_frames(video_path)
            except Exception as e:
                logger.warning("Video processing failed for %s: %s", video, e)

        if not all_images:
            return self._prepare_vision_inputs(messages, [])

        inputs = prepare_inputs(self._processor, images=all_images, prompts=[prompt])
        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")
        attention_mask = inputs.get("attention_mask")

        extra_model_inputs = {
            k: v
            for k, v in inputs.items()
            if k not in ("input_ids", "attention_mask", "pixel_values")
            and v is not None
        }

        num_images = len(all_images)
        if pixel_values is not None and num_images > 0:
            image_hash = compute_image_hash(all_images)
            call_kwargs = dict(extra_model_inputs)

            if self._vision_cache is not None and self._vision_cache_enabled:
                cached_whole = self._vision_cache.get(image_hash, self._model_name)
                if cached_whole is not None:
                    call_kwargs["cached_image_features"] = cached_whole
                else:
                    try:
                        features = self._compute_vision_features(
                            pixel_values, extra_model_inputs
                        )
                        if features is not None:
                            mx.eval(features)
                            call_kwargs["cached_image_features"] = features
                            self._vision_cache.put(
                                image_hash, self._model_name, features
                            )
                    except Exception:
                        logger.debug("Vision feature computation failed", exc_info=True)

            try:
                embed_features = self._vlm_model.get_input_embeddings(
                    input_ids,
                    pixel_values,
                    mask=attention_mask,
                    **call_kwargs,
                )
            except TypeError:
                if "cached_image_features" in call_kwargs:
                    self._vision_cache_enabled = False
                    call_kwargs.pop("cached_image_features")
                    embed_features = self._vlm_model.get_input_embeddings(
                        input_ids, pixel_values, mask=attention_mask, **call_kwargs
                    )
                else:
                    raise

            mx.eval(embed_features.inputs_embeds)

            extra_kwargs = {}
            if hasattr(embed_features, "to_dict"):
                for k, v in embed_features.to_dict().items():
                    if k != "inputs_embeds" and v is not None:
                        extra_kwargs[k] = v

            lm = getattr(self._vlm_model, "language_model", None)
            if lm is not None:
                pid = getattr(lm, "_position_ids", None)
                if pid is not None and "position_ids" not in extra_kwargs:
                    extra_kwargs["position_ids"] = pid
                rd = getattr(lm, "_rope_deltas", None)
                if rd is not None:
                    extra_kwargs["_captured_rope_deltas"] = rd

            token_ids = (
                input_ids[0].tolist() if input_ids.ndim > 1 else input_ids.tolist()
            )
            return (
                token_ids,
                embed_features.inputs_embeds,
                extra_kwargs,
                image_hash,
                0,
                [],
            )
        else:
            token_ids = (
                input_ids[0].tolist() if input_ids.ndim > 1 else input_ids.tolist()
            )
            return token_ids, None, None, None, 0, []

    def _process_chat_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None,
        kwargs: dict,
    ) -> tuple:
        text_messages, images, _videos, _audio = extract_images_from_messages(messages)

        # Merge videos from message content into kwargs
        if _videos and "videos" not in kwargs:
            kwargs["videos"] = _videos

        if images:
            text_messages = (
                self._apply_ocr_prompt(messages) if self.is_ocr_model else text_messages
            )

        # Video frame extraction: video frames become additional images
        videos = kwargs.get("videos") or []
        if videos and self._is_native_video_model():
            # Qwen native video path (preserves temporal info)
            video_fps = kwargs.get("video_fps", DEFAULT_FPS)
            video_max_frames = kwargs.get("video_max_frames", MAX_FRAMES)
            (
                token_ids,
                vlm_embeds,
                vlm_kwargs,
                image_hash,
                cache_key_start,
                cache_key_ranges,
            ) = self._prepare_native_video_inputs(
                text_messages, videos, video_fps, video_max_frames
            )
            mx.synchronize()
            mx.clear_cache()
            return (
                token_ids,
                vlm_embeds,
                vlm_kwargs,
                image_hash,
                cache_key_start,
                cache_key_ranges,
            )
        elif videos:

            video_fps = kwargs.get("video_fps", DEFAULT_FPS)
            video_max_frames = kwargs.get("video_max_frames", MAX_FRAMES)

            for video in videos:
                try:
                    video_path = process_video_input(video)
                    frames = extract_video_frames_smart(
                        video_path, fps=video_fps, max_frames=video_max_frames
                    )
                    frame_paths = save_frames_to_temp(frames)
                    for fp in frame_paths:
                        try:
                            from ..utils.image import load_image

                            images.append(load_image(fp))
                        except Exception as e:
                            logger.warning("Failed to load video frame %s: %s", fp, e)
                    logger.info(
                        "Video: extracted %d frames from %s",
                        len(frame_paths),
                        video_path,
                    )
                except Exception as e:
                    logger.warning("Video processing failed for %s: %s", video, e)

        (
            token_ids,
            vlm_embeds,
            vlm_kwargs,
            image_hash,
            cache_key_start,
            cache_key_ranges,
        ) = self._prepare_vision_inputs(text_messages, images)
        if images:
            mx.synchronize()
            mx.clear_cache()
        return (
            token_ids,
            vlm_embeds,
            vlm_kwargs,
            image_hash,
            cache_key_start,
            cache_key_ranges,
        )

    def _apply_ocr_prompt(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        model_type = self.model_type or ""
        if model_type not in OCR_MODEL_PROMPTS:
            return messages

        ocr_prompt = OCR_MODEL_PROMPTS[model_type]
        messages = copy.deepcopy(messages)
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                has_image = any(
                    isinstance(p, dict) and p.get("type") == "image_url"
                    for p in content
                )
                if not has_image:
                    break
                user_text = " ".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ).strip()
                if user_text:
                    break
                new_content = [{"type": "text", "text": ocr_prompt}]
                new_content.extend(
                    p
                    for p in content
                    if not (isinstance(p, dict) and p.get("type") == "text")
                )
                msg["content"] = new_content
            break
        return messages

    # -- Generation APIs --

    def _resolve_ocr_stop_token_ids(self) -> list[int]:
        if hasattr(self, "_ocr_stop_ids_cache"):
            return self._ocr_stop_ids_cache
        ids = []
        if self._tokenizer is None:
            return ids
        unk_id = getattr(self._tokenizer, "unk_token_id", None)
        for seq in OCR_EXTRA_STOP_SEQUENCES:
            try:
                token_id = self._tokenizer.convert_tokens_to_ids(seq)
                if token_id is not None and token_id != unk_id:
                    ids.append(token_id)
            except (AttributeError, KeyError, TypeError):
                pass
        self._ocr_stop_ids_cache = ids
        return ids

    def _build_sampling_params(
        self,
        max_tokens,
        temperature,
        top_p,
        top_k,
        min_p,
        repetition_penalty,
        presence_penalty,
        stop,
        **kwargs,
    ):
        from ..request import SamplingParams

        extra_stop_ids = self._resolve_ocr_stop_token_ids() if self.is_ocr_model else []
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            xtc_probability=kwargs.get("xtc_probability", 0.0),
            xtc_threshold=kwargs.get("xtc_threshold", 0.1),
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            stop=stop or [],
            stop_token_ids=extra_stop_ids or None,
            thinking_budget=kwargs.get("thinking_budget"),
            compiled_grammar=kwargs.get("compiled_grammar"),
            seed=kwargs.get("seed"),
        )

    async def generate(
        self,
        prompt: str | list[int],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        vlm_inputs_embeds: Any = None,
        vlm_extra_kwargs: dict[str, Any] | None = None,
        vlm_image_hash: str | None = None,
        vlm_cache_key_start: int = 0,
        vlm_cache_key_ranges: list[tuple[int, str]] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        if not self._loaded:
            await self.start()
        sampling_params = self._build_sampling_params(
            max_tokens,
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            presence_penalty,
            stop,
            **kwargs,
        )
        output = await self._engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            vlm_inputs_embeds=vlm_inputs_embeds,
            vlm_extra_kwargs=vlm_extra_kwargs,
            vlm_image_hash=vlm_image_hash,
            vlm_cache_key_start=vlm_cache_key_start,
            vlm_cache_key_ranges=vlm_cache_key_ranges,
        )
        text = output.output_text
        return GenerationOutput(
            text=text,
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            finish_reason=output.finish_reason,
            tool_calls=output.tool_calls,
            cached_tokens=output.cached_tokens,
        )

    async def stream_generate(
        self,
        prompt: str | list[int],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        vlm_inputs_embeds: Any = None,
        vlm_extra_kwargs: dict[str, Any] | None = None,
        vlm_image_hash: str | None = None,
        vlm_cache_key_start: int = 0,
        vlm_cache_key_ranges: list[tuple[int, str]] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        if not self._loaded:
            await self.start()
        sampling_params = self._build_sampling_params(
            max_tokens,
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            presence_penalty,
            stop,
            **kwargs,
        )
        engine = self._engine
        request_id = await engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            vlm_inputs_embeds=vlm_inputs_embeds,
            vlm_extra_kwargs=vlm_extra_kwargs,
            vlm_image_hash=vlm_image_hash,
            vlm_cache_key_start=vlm_cache_key_start,
            vlm_cache_key_ranges=vlm_cache_key_ranges,
            streaming=True,
        )
        finished_normally = False
        try:
            async for output in engine.stream_outputs(request_id):
                if output.finished:
                    finished_normally = True
                yield GenerationOutput(
                    text=output.output_text,
                    new_text=output.new_text,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finished=output.finished,
                    finish_reason=output.finish_reason,
                    tool_calls=output.tool_calls,
                    cached_tokens=output.cached_tokens,
                )
        except GeneratorExit:
            logger.info(
                "[vlm_stream_generate] GeneratorExit for request %s", request_id
            )
        finally:
            if not finished_normally:
                await engine.abort_request(request_id)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        if not self._loaded:
            await self.start()
        loop = asyncio.get_running_loop()
        (
            prompt,
            vlm_embeds,
            vlm_kwargs,
            image_hash,
            cache_key_start,
            cache_key_ranges,
        ) = await asyncio.wait_for(
            loop.run_in_executor(
                self._engine._mlx_executor,
                self._process_chat_messages,
                messages,
                tools,
                kwargs,
            ),
            timeout=30.0,
        )
        gen = await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            vlm_inputs_embeds=vlm_embeds,
            vlm_extra_kwargs=vlm_kwargs,
            vlm_image_hash=image_hash,
            vlm_cache_key_start=cache_key_start,
            vlm_cache_key_ranges=cache_key_ranges,
            **kwargs,
        )
        if tools and not gen.tool_calls:
            gen = _fallback_parse_tool_calls(gen, self._tokenizer, tools)
        return gen

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        if not self._loaded:
            await self.start()
        loop = asyncio.get_running_loop()
        (
            prompt,
            vlm_embeds,
            vlm_kwargs,
            image_hash,
            cache_key_start,
            cache_key_ranges,
        ) = await loop.run_in_executor(
            self._engine._mlx_executor,
            self._process_chat_messages,
            messages,
            tools,
            kwargs,
        )
        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            vlm_inputs_embeds=vlm_embeds,
            vlm_extra_kwargs=vlm_kwargs,
            vlm_image_hash=image_hash,
            vlm_cache_key_start=cache_key_start,
            vlm_cache_key_ranges=cache_key_ranges,
            **kwargs,
        ):
            if output.finished and tools and not output.tool_calls:
                output = _fallback_parse_tool_calls(output, self._tokenizer, tools)
            yield output

    # -- Utilities --

    def count_chat_tokens(
        self, messages: list[dict[str, Any]], tools: list[dict] | None = None
    ) -> int:
        text_messages, _, _, _ = extract_images_from_messages(messages)
        prompt = (
            "\n".join(f"{m['role']}: {m['content']}" for m in text_messages)
            + "\nassistant:"
        )
        return len(self._tokenizer.encode(prompt))

    def has_active_requests(self) -> bool:
        ec = getattr(self, "_engine", None)
        if ec is not None:
            inner = getattr(ec, "engine", None)
            if inner is not None:
                return len(getattr(inner, "_output_collectors", {})) > 0
        return False

    def get_stats(self) -> dict[str, Any]:
        stats = {
            "engine_type": "vlm",
            "model_name": self._model_name,
            "loaded": self._loaded,
            "stream_interval": self._stream_interval,
        }
        if self._engine:
            stats.update(self._engine.get_stats())
        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        return self._engine.get_cache_stats() if self._engine else None

    async def abort_all_requests(self) -> int:
        if self._engine and self._engine.engine:
            return await self._engine.engine.abort_all_requests()
        return 0

    def __repr__(self) -> str:
        status = "running" if self._loaded else "stopped"
        return f"<VLMBatchedEngine model={self._model_name} status={status}>"
