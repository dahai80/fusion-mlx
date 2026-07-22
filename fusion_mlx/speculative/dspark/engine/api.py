from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.generate import wired_limit

from .adapters import LoadedTargetModel, load_target_model
from .draft import DSparkDraftModel, load_draft_model
from .runtime import dspark_generate, dspark_generate_stream

logger = logging.getLogger(__name__)

DEFAULT_TARGET_MODEL = "mlx-community/Qwen3-4B-bf16"
# Local directory produced by `dspark-metal-convert deepseek-ai/dspark_qwen3_4b_block7
# --target mlx-community/Qwen3-4B-bf16` (see dspark_metal.convert).
DEFAULT_DRAFT_MODEL = "models/dspark_qwen3_4b_block7-mlx"


@dataclass
class DSparkResult:
    text: str
    output_tokens: list[int]
    generated_tokens: list[int]
    metrics: dict[str, Any]


@dataclass
class DSparkStreamEvent:
    delta: str
    text: str
    token_ids: list[int]
    output_tokens: list[int]
    generated_tokens: list[int]
    metrics: dict[str, Any] | None = None
    finished: bool = False


class DSparkGenerator:
    def __init__(
        self,
        target_model: str = DEFAULT_TARGET_MODEL,
        draft_model: str = DEFAULT_DRAFT_MODEL,
        draft_attention_mask: str = "auto",
        draft_quant_bits: int | None = None,
        draft_quant_group_size: int = 64,
        draft_reuse_target_embeddings: bool = False,
        seed: int = 0,
        target_model_config: dict[str, Any] | None = None,
    ):
        mx.random.seed(seed)
        self.requested_target_model = target_model
        self.requested_draft_model = draft_model
        self.target: LoadedTargetModel = load_target_model(
            target_model, model_config=target_model_config
        )
        self.draft: DSparkDraftModel
        # Runtime embedding reuse rebinds the draft's embed_tokens/lm_head to
        # the target's bf16 tensors (U2 audit: byte-identical), so they must
        # stay unquantized on the draft side.
        self.draft, self.draft_path = load_draft_model(
            draft_model,
            quantize_bits=draft_quant_bits,
            quantize_group_size=draft_quant_group_size,
            quantize_embeddings=not draft_reuse_target_embeddings,
        )

        # DSpark draft attention is always bidirectional over [ctx; block]
        # (reference convention); the skeleton's causal-mask knob is gone.
        if draft_attention_mask not in ("auto", "none"):
            raise ValueError(
                "DSpark draft attention is always bidirectional; "
                f"draft_attention_mask={draft_attention_mask!r} is not supported."
            )
        self.draft_attention_mask = "none"
        self.draft_quantization = self.draft.draft_quantization
        self.draft_reuse_target_embeddings = draft_reuse_target_embeddings
        if draft_reuse_target_embeddings:
            self._rebind_draft_embeddings_to_target()

    def _rebind_draft_embeddings_to_target(self) -> None:
        """Point the draft's embed_tokens/lm_head at the target's tensors.

        Guarded by the converter's audit (audit.json must prove the draft
        copies byte-identical to the target's) and by a shape/dtype check
        (refuses quantized targets, whose embeddings are not the audited bf16
        tensors). Rebinding dedups the two ~1.2 GB vocab tensors: draft and
        target then read one shared buffer.
        """
        audit_path = self.draft_path / "audit.json"
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        if not audit.get("all_identical"):
            raise ValueError(
                "draft_reuse_target_embeddings requires the converted draft's "
                "audit.json to prove embed_tokens/lm_head byte-identical to "
                f"the target's (audit at {audit_path})."
            )
        target_model = self.target.model
        embed_weight = target_model.model.embed_tokens.weight
        if getattr(target_model.args, "tie_word_embeddings", False):
            lm_head_weight = embed_weight
        else:
            lm_head_weight = target_model.lm_head.weight
        for module, weight, name in (
            (self.draft.embed_tokens, embed_weight, "embed_tokens"),
            (self.draft.lm_head, lm_head_weight, "lm_head"),
        ):
            if (
                weight.shape != module.weight.shape
                or weight.dtype != module.weight.dtype
            ):
                raise ValueError(
                    f"target {name} weight {weight.shape}/{weight.dtype} does "
                    f"not match draft {module.weight.shape}/{module.weight.dtype} "
                    "(quantized target?); cannot reuse target embeddings."
                )
            module.weight = weight

    @property
    def target_model_path(self) -> Path:
        return self.target.resolved_model_path

    def encode_prompt(
        self, prompt_text: str, enable_thinking: bool = False
    ) -> mx.array:
        return self.target.build_prompt(prompt_text, enable_thinking=enable_thinking)

    def generate_from_tokens(
        self,
        prompt_tokens: mx.array,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        confidence_threshold: float = 0.0,
        speculative_tokens: int | None = None,
        verify_mode: str = "full",
        verify_chunk_size: int = 4,
        seed: int | None = None,
        reset_peak_memory: bool = True,
        skip_special_tokens: bool = False,
        profile: bool = False,
        trace_hook=None,
    ) -> DSparkResult:
        with wired_limit(self.target.model):
            if reset_peak_memory:
                mx.reset_peak_memory()
            output_tokens, metrics = dspark_generate(
                target=self.target,
                draft=self.draft,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_token_ids=self.target.stop_token_ids(),
                layer_ids=self.draft.target_layer_ids,
                speculative_tokens=speculative_tokens,
                confidence_threshold=confidence_threshold,
                verify_mode=verify_mode,
                verify_chunk_size=verify_chunk_size,
                seed=seed,
                profile=profile,
                trace_hook=trace_hook,
            )

        generated_tokens = output_tokens[metrics["num_input_tokens"] :]
        text = self._decode(generated_tokens, skip_special_tokens)
        return DSparkResult(
            text=text,
            output_tokens=output_tokens,
            generated_tokens=generated_tokens,
            metrics=metrics,
        )

    def stream_from_tokens(
        self,
        prompt_tokens: mx.array,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        confidence_threshold: float = 0.0,
        speculative_tokens: int | None = None,
        verify_mode: str = "full",
        verify_chunk_size: int = 4,
        seed: int | None = None,
        reset_peak_memory: bool = True,
        skip_special_tokens: bool = False,
        profile: bool = False,
        trace_hook=None,
    ) -> Iterator[DSparkStreamEvent]:
        prompt_len = int(prompt_tokens.shape[0])
        decoded_text = ""
        with wired_limit(self.target.model):
            if reset_peak_memory:
                mx.reset_peak_memory()
            for event in dspark_generate_stream(
                target=self.target,
                draft=self.draft,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_token_ids=self.target.stop_token_ids(),
                layer_ids=self.draft.target_layer_ids,
                speculative_tokens=speculative_tokens,
                confidence_threshold=confidence_threshold,
                verify_mode=verify_mode,
                verify_chunk_size=verify_chunk_size,
                seed=seed,
                profile=profile,
                trace_hook=trace_hook,
            ):
                generated_tokens = event.output_tokens[prompt_len:]
                text = self._decode(generated_tokens, skip_special_tokens)
                if text.startswith(decoded_text):
                    delta = text[len(decoded_text) :]
                else:
                    delta = text
                decoded_text = text
                yield DSparkStreamEvent(
                    delta=delta,
                    text=text,
                    token_ids=list(event.token_ids),
                    output_tokens=list(event.output_tokens),
                    generated_tokens=list(generated_tokens),
                    metrics=event.metrics,
                    finished=event.finished,
                )

    def generate(
        self,
        prompt_text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        confidence_threshold: float = 0.0,
        speculative_tokens: int | None = None,
        verify_mode: str = "full",
        verify_chunk_size: int = 4,
        seed: int | None = None,
        reset_peak_memory: bool = True,
        skip_special_tokens: bool = False,
        profile: bool = False,
        trace_hook=None,
        enable_thinking: bool = False,
    ) -> DSparkResult:
        return self.generate_from_tokens(
            prompt_tokens=self.encode_prompt(
                prompt_text, enable_thinking=enable_thinking
            ),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
            speculative_tokens=speculative_tokens,
            verify_mode=verify_mode,
            verify_chunk_size=verify_chunk_size,
            seed=seed,
            reset_peak_memory=reset_peak_memory,
            skip_special_tokens=skip_special_tokens,
            profile=profile,
            trace_hook=trace_hook,
        )

    def stream(
        self,
        prompt_text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        confidence_threshold: float = 0.0,
        speculative_tokens: int | None = None,
        verify_mode: str = "full",
        verify_chunk_size: int = 4,
        seed: int | None = None,
        reset_peak_memory: bool = True,
        skip_special_tokens: bool = False,
        profile: bool = False,
        trace_hook=None,
        enable_thinking: bool = False,
    ) -> Iterator[DSparkStreamEvent]:
        return self.stream_from_tokens(
            prompt_tokens=self.encode_prompt(
                prompt_text, enable_thinking=enable_thinking
            ),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
            speculative_tokens=speculative_tokens,
            verify_mode=verify_mode,
            verify_chunk_size=verify_chunk_size,
            seed=seed,
            reset_peak_memory=reset_peak_memory,
            skip_special_tokens=skip_special_tokens,
            profile=profile,
            trace_hook=trace_hook,
        )

    # ------------------------------------------------------------------
    # Multimodal (VLM) generation. Vision arrays are threaded to the
    # target adapter's prefill via a side-channel (set_vision_inputs);
    # the adapter consumes them once on the prefill forward and clears
    # them, so decode is text-only. Only qwen3_vl-family targets support
    # vision; text targets raise NotImplementedError on set_vision_inputs.
    # ------------------------------------------------------------------

    def _decode(self, tokens, skip_special_tokens: bool = False) -> str:
        tok = getattr(self.target.tokenizer, "tokenizer", self.target.tokenizer)
        return tok.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _is_vlm(self) -> bool:
        return self.target.adapter.family == "qwen3_vl"

    def _set_vision(
        self,
        pixel_values,
        image_grid_thw,
        video_grid_thw,
    ) -> None:
        if pixel_values is None and image_grid_thw is None and video_grid_thw is None:
            self.target.clear_vision_inputs()
            return
        if not self._is_vlm():
            raise NotImplementedError(
                f"Multimodal generation requires a VLM target, got "
                f"family={self.target.adapter.family!r}."
            )
        self.target.set_vision_inputs(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )

    @staticmethod
    def _input_field(inputs, key):
        if hasattr(inputs, key):
            return getattr(inputs, key)
        if isinstance(inputs, dict):
            return inputs.get(key)
        return None

    def _build_multimodal_inputs(
        self,
        prompt_text: str,
        images=None,
        videos=None,
    ):
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs

        processor = self.target.tokenizer
        config = self.target.model.config
        n_images = 0
        if images is not None:
            n_images = len(images) if isinstance(images, (list, tuple)) else 1
        prompt = apply_chat_template(
            processor,
            config,
            prompt_text,
            add_generation_prompt=True,
            num_images=n_images,
        )
        inputs = prepare_inputs(
            processor,
            images=images if images is not None else None,
            videos=videos if videos is not None else None,
            prompts=prompt,
        )
        logger.info(
            "dspark-vlm build_inputs: images=%s videos=%s",
            n_images,
            (
                0
                if videos is None
                else (len(videos) if isinstance(videos, (list, tuple)) else 1)
            ),
        )
        return inputs

    def generate_multimodal_from_tokens(
        self,
        prompt_tokens: mx.array,
        pixel_values=None,
        image_grid_thw=None,
        video_grid_thw=None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        confidence_threshold: float = 0.0,
        speculative_tokens: int | None = None,
        verify_mode: str = "full",
        verify_chunk_size: int = 4,
        seed: int | None = None,
        reset_peak_memory: bool = True,
        skip_special_tokens: bool = False,
        profile: bool = False,
        trace_hook=None,
    ) -> DSparkResult:
        self._set_vision(pixel_values, image_grid_thw, video_grid_thw)
        try:
            return self.generate_from_tokens(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                confidence_threshold=confidence_threshold,
                speculative_tokens=speculative_tokens,
                verify_mode=verify_mode,
                verify_chunk_size=verify_chunk_size,
                seed=seed,
                reset_peak_memory=reset_peak_memory,
                skip_special_tokens=skip_special_tokens,
                profile=profile,
                trace_hook=trace_hook,
            )
        finally:
            self.target.clear_vision_inputs()

    def stream_multimodal_from_tokens(
        self,
        prompt_tokens: mx.array,
        pixel_values=None,
        image_grid_thw=None,
        video_grid_thw=None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        confidence_threshold: float = 0.0,
        speculative_tokens: int | None = None,
        verify_mode: str = "full",
        verify_chunk_size: int = 4,
        seed: int | None = None,
        reset_peak_memory: bool = True,
        skip_special_tokens: bool = False,
        profile: bool = False,
        trace_hook=None,
    ) -> Iterator[DSparkStreamEvent]:
        self._set_vision(pixel_values, image_grid_thw, video_grid_thw)
        try:
            yield from self.stream_from_tokens(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                confidence_threshold=confidence_threshold,
                speculative_tokens=speculative_tokens,
                verify_mode=verify_mode,
                verify_chunk_size=verify_chunk_size,
                seed=seed,
                reset_peak_memory=reset_peak_memory,
                skip_special_tokens=skip_special_tokens,
                profile=profile,
                trace_hook=trace_hook,
            )
        finally:
            self.target.clear_vision_inputs()

    def generate_multimodal(
        self,
        prompt_text: str,
        images=None,
        videos=None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        confidence_threshold: float = 0.0,
        speculative_tokens: int | None = None,
        verify_mode: str = "full",
        verify_chunk_size: int = 4,
        seed: int | None = None,
        reset_peak_memory: bool = True,
        skip_special_tokens: bool = False,
        profile: bool = False,
        trace_hook=None,
    ) -> DSparkResult:
        if not self._is_vlm():
            raise NotImplementedError(
                f"generate_multimodal requires a VLM target, got "
                f"family={self.target.adapter.family!r}."
            )
        inputs = self._build_multimodal_inputs(
            prompt_text, images=images, videos=videos
        )
        input_ids = self._input_field(inputs, "input_ids")
        if input_ids.ndim == 2:
            input_ids = input_ids[0]
        return self.generate_multimodal_from_tokens(
            prompt_tokens=input_ids,
            pixel_values=self._input_field(inputs, "pixel_values"),
            image_grid_thw=self._input_field(inputs, "image_grid_thw"),
            video_grid_thw=self._input_field(inputs, "video_grid_thw"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
            speculative_tokens=speculative_tokens,
            verify_mode=verify_mode,
            verify_chunk_size=verify_chunk_size,
            seed=seed,
            reset_peak_memory=reset_peak_memory,
            skip_special_tokens=skip_special_tokens,
            profile=profile,
            trace_hook=trace_hook,
        )

    def stream_multimodal(
        self,
        prompt_text: str,
        images=None,
        videos=None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        confidence_threshold: float = 0.0,
        speculative_tokens: int | None = None,
        verify_mode: str = "full",
        verify_chunk_size: int = 4,
        seed: int | None = None,
        reset_peak_memory: bool = True,
        skip_special_tokens: bool = False,
        profile: bool = False,
        trace_hook=None,
    ) -> Iterator[DSparkStreamEvent]:
        if not self._is_vlm():
            raise NotImplementedError(
                f"stream_multimodal requires a VLM target, got "
                f"family={self.target.adapter.family!r}."
            )
        inputs = self._build_multimodal_inputs(
            prompt_text, images=images, videos=videos
        )
        input_ids = self._input_field(inputs, "input_ids")
        if input_ids.ndim == 2:
            input_ids = input_ids[0]
        yield from self.stream_multimodal_from_tokens(
            prompt_tokens=input_ids,
            pixel_values=self._input_field(inputs, "pixel_values"),
            image_grid_thw=self._input_field(inputs, "image_grid_thw"),
            video_grid_thw=self._input_field(inputs, "video_grid_thw"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
            speculative_tokens=speculative_tokens,
            verify_mode=verify_mode,
            verify_chunk_size=verify_chunk_size,
            seed=seed,
            reset_peak_memory=reset_peak_memory,
            skip_special_tokens=skip_special_tokens,
            profile=profile,
            trace_hook=trace_hook,
        )
