# SPDX-License-Identifier: Apache-2.0
"""Embedding engine for fusion-mlx."""

import asyncio
import gc
import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..engine_core import get_executor
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)

_DEFAULT_EMBEDDING_MAX_LENGTH = 512
_TOKENIZER_MAX_LENGTH_SENTINEL = 10**18
_CONTEXT_LENGTH_ATTRS = (
    "max_position_embeddings",
    "max_seq_len",
    "max_seq_length",
    "seq_length",
    "n_positions",
)


@dataclass
class EmbeddingOutput:
    embeddings: list[list[float]]
    total_tokens: int
    dimensions: int


class MLXEmbeddingModel:
    def __init__(self, model_name: str, trust_remote_code: bool = False):
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._model = None
        self._processor = None
        self._loaded = False
        self._hidden_size: int | None = None
        self._using_native = False
        self._is_compiled = False
        self._compiled_embed = None
        self._remap_input_ids_to_inputs = False

    def _load_native(self) -> bool:
        from transformers import AutoTokenizer

        model_path = Path(self._model_name)
        config_path = model_path / "config.json"
        if not config_path.exists():
            logger.debug("No config.json at %s, native loading skipped", model_path)
            return False

        try:
            with open(config_path) as f:
                config_dict = json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.debug("Failed to read config.json, native loading skipped")
            return False

        architectures = config_dict.get("architectures", [])
        arch = architectures[0] if architectures else ""

        native_arch_modules = {
            "XLMRobertaModel": "xlm_roberta",
            "BertModel": "xlm_roberta",
            "BertForMaskedLM": "xlm_roberta",
            "Qwen2ForCausalLM": "qwen2_embedding",
        }
        module_name = native_arch_modules.get(arch)
        if module_name is None:
            logger.debug(
                "Architecture '%s' not natively supported for embedding, "
                "trying mlx-embeddings",
                arch,
            )
            return False

        try:
            from importlib import import_module

            native_module = import_module(f"{__package__}.{module_name}")
            Model = native_module.Model
            ModelArgs = native_module.ModelArgs

            known_fields = {f.name for f in ModelArgs.__dataclass_fields__.values()}
            model_config = {k: v for k, v in config_dict.items() if k in known_fields}
            model_config["architectures"] = architectures

            config = ModelArgs(**model_config)
            model_instance = Model(config)

            weights = {}
            weight_files = list(model_path.glob("*.safetensors"))
            if not weight_files:
                logger.debug("No safetensors files found in %s", model_path)
                return False

            for wf in weight_files:
                weights.update(mx.load(str(wf)))

            weights = model_instance.sanitize(weights)
            self._validate_native_weights(model_instance, weights)
            model_instance.load_weights(list(weights.items()), strict=False)
            mx.eval(model_instance.parameters())
            model_instance.train(False)

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    use_fast=False,
                    trust_remote_code=self._trust_remote_code,
                )
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    trust_remote_code=self._trust_remote_code,
                )

            self._model = model_instance
            self._processor = tokenizer
            self._hidden_size = config.hidden_size
            self._loaded = True
            self._using_native = True
            self._is_compiled = False
            self._compiled_embed = None
            logger.info(
                "Embedding model loaded natively: %s (arch=%s, hidden_size=%s)",
                self._model_name,
                arch,
                config.hidden_size,
            )
            return True

        except Exception as e:
            logger.debug("Native loading failed for %s: %s", self._model_name, e)
            return False

    def load(self):
        if self._loaded:
            return

        if self._load_native():
            return

        try:
            from ..models.mlx_embeddings_compat import (
                patch_qwen3_vl_processor_for_torch_free_image_loading,
            )

            patch_qwen3_vl_processor_for_torch_free_image_loading()
            from mlx_embeddings import load

            logger.info(
                "Loading embedding model via mlx-embeddings: %s", self._model_name
            )
            self._model, self._processor = load(
                self._model_name,
                tokenizer_config={"trust_remote_code": self._trust_remote_code},
            )

            if hasattr(self._model, "config"):
                config = self._model.config
                self._hidden_size = getattr(config, "hidden_size", None)
                if self._hidden_size is None and hasattr(config, "text_config"):
                    self._hidden_size = getattr(config.text_config, "hidden_size", None)

            self._using_native = False
            self._detect_input_key_remapping()
            self._is_compiled = self._try_compile()
            self._loaded = True
            logger.info(
                "Embedding model loaded successfully: %s (hidden_size=%s, compiled=%s)",
                self._model_name,
                self._hidden_size,
                self._is_compiled,
            )

        except ImportError:
            raise ImportError(
                "mlx-embeddings is required for embedding generation. "
                "Install with: pip install mlx-embeddings"
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No safetensors weight files found for '{self._model_name}'. "
                f"Embedding models require weights in safetensors format. "
                f"If this is a PyTorch model, use an MLX-converted version "
                f"(e.g., from mlx-community on HuggingFace)."
            )
        except Exception as e:
            logger.error("Failed to load embedding model: %s", e)
            raise

    def _extract_embeddings_array(self, outputs):
        if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            embeddings = outputs.text_embeds
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        elif (
            hasattr(outputs, "last_hidden_state")
            and outputs.last_hidden_state is not None
        ):
            embeddings = outputs.last_hidden_state
        else:
            raise ValueError(
                "Model output does not contain expected embedding fields "
                "(text_embeds, pooler_output, or last_hidden_state)"
            )
        if embeddings.ndim == 3:
            embeddings = mx.mean(embeddings, axis=1)
        return embeddings

    def _validate_native_weights(self, model_instance, weights: dict[str, Any]) -> None:
        from mlx.utils import tree_flatten

        expected_weights = dict(tree_flatten(model_instance.parameters()))
        expected_weight_names = set(expected_weights.keys())
        provided_weight_names = set(weights.keys())
        missing_weight_names = expected_weight_names - provided_weight_names

        optional_missing_prefixes = ("pooler.",)
        required_missing = sorted(
            name
            for name in missing_weight_names
            if not name.startswith(optional_missing_prefixes)
        )
        if required_missing:
            preview = ", ".join(required_missing[:10])
            suffix = "..." if len(required_missing) > 10 else ""
            raise ValueError(
                "Native embedding checkpoint is missing required weights: "
                f"{preview}{suffix}"
            )

        shape_mismatches = []
        for name in expected_weight_names & provided_weight_names:
            expected_shape = tuple(expected_weights[name].shape)
            provided_shape = tuple(weights[name].shape)
            if expected_shape != provided_shape:
                shape_mismatches.append((name, expected_shape, provided_shape))

        if shape_mismatches:
            preview = ", ".join(
                f"{name}: expected {expected_shape}, got {provided_shape}"
                for name, expected_shape, provided_shape in shape_mismatches[:5]
            )
            suffix = "..." if len(shape_mismatches) > 5 else ""
            raise ValueError(
                "Native embedding checkpoint has incompatible weight shapes: "
                f"{preview}{suffix}"
            )

    def _uses_custom_embedding_inputs(self, processor) -> bool:
        for attr_name in ("prepare_embedding_inputs", "prepare_model_inputs"):
            try:
                inspect.getattr_static(processor, attr_name)
                return True
            except AttributeError:
                continue
        return False

    def _normalize_embedding_inputs(
        self,
        inputs: str | dict[str, str] | list[str] | list[dict[str, str]],
    ) -> list[dict[str, str]]:
        if not inputs:
            return []
        if isinstance(inputs, str):
            return [{"text": inputs}]
        if isinstance(inputs, dict):
            return [dict(inputs)]
        first = inputs[0]
        if isinstance(first, str):
            return [{"text": text} for text in inputs]
        return [dict(item) for item in inputs]

    @staticmethod
    def _positive_context_length(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        if 0 < value < _TOKENIZER_MAX_LENGTH_SENTINEL:
            return value
        return None

    @classmethod
    def _get_config_value(cls, config: Any, key: str) -> int | None:
        if config is None:
            return None
        if isinstance(config, dict):
            return cls._positive_context_length(config.get(key))
        return cls._positive_context_length(getattr(config, key, None))

    @classmethod
    def _context_length_from_config(cls, config: Any) -> int | None:
        for key in _CONTEXT_LENGTH_ATTRS:
            value = cls._get_config_value(config, key)
            if value is not None:
                return value

        for nested_key in ("text_config", "language_config"):
            nested = None
            if isinstance(config, dict):
                nested = config.get(nested_key)
            elif config is not None:
                nested = getattr(config, nested_key, None)
            for key in _CONTEXT_LENGTH_ATTRS:
                value = cls._get_config_value(nested, key)
                if value is not None:
                    return value

        return None

    def _resolve_max_length(self, max_length: int | None) -> int:
        if max_length is not None:
            value = self._positive_context_length(max_length)
            if value is None:
                raise ValueError("max_length must be a positive integer")
            return value

        for config in (
            getattr(self._model, "config", None),
            getattr(self._processor, "config", None),
        ):
            value = self._context_length_from_config(config)
            if value is not None:
                return value

        processor = self._processor
        tokenizers = [
            processor,
            getattr(processor, "tokenizer", None),
            getattr(processor, "_tokenizer", None),
        ]
        for tokenizer in tokenizers:
            for attr_name in ("model_max_length", "max_length"):
                value = self._positive_context_length(
                    getattr(tokenizer, attr_name, None)
                )
                if value is not None:
                    return value

        return _DEFAULT_EMBEDDING_MAX_LENGTH

    def _prepare_embedding_inputs(
        self,
        processor,
        inputs: list[str] | list[dict[str, str]],
        max_length: int,
        padding: bool,
        truncation: bool,
    ):
        normalized_inputs = self._normalize_embedding_inputs(inputs)

        if self._uses_custom_embedding_inputs(processor):
            if hasattr(processor, "prepare_embedding_inputs"):
                return processor.prepare_embedding_inputs(
                    normalized_inputs, return_tensors="mlx"
                )
            return processor.prepare_model_inputs(
                normalized_inputs, return_tensors="mlx"
            )

        if any("image" in item for item in normalized_inputs):
            raise ValueError(
                f"Embedding model '{self._model_name}' does not support image inputs"
            )

        from mlx_embeddings.utils import prepare_inputs

        return prepare_inputs(
            processor,
            None,
            [item.get("text", "") for item in normalized_inputs],
            max_length,
            padding,
            truncation,
            None,
        )

    def _detect_input_key_remapping(self) -> None:
        try:
            params = inspect.signature(self._model.__call__).parameters
            self._remap_input_ids_to_inputs = (
                "input_ids" not in params and "inputs" in params
            )
        except (TypeError, ValueError):
            self._remap_input_ids_to_inputs = False

    def _adapt_model_inputs_for_call(
        self, model_inputs: dict[str, Any]
    ) -> dict[str, Any]:
        adapted_inputs = dict(model_inputs)
        if self._remap_input_ids_to_inputs and "input_ids" in adapted_inputs:
            adapted_inputs["inputs"] = adapted_inputs.pop("input_ids")
        return adapted_inputs

    def _try_compile(self) -> bool:
        base_model = self._model

        try:

            def _compiled_embed(inputs):
                outputs = base_model(**self._adapt_model_inputs_for_call(inputs))
                return self._extract_embeddings_array(outputs)

            self._compiled_embed = mx.compile(_compiled_embed)

            test_inputs = {"input_ids": mx.zeros((1, 4), dtype=mx.int32)}
            _ = self._compiled_embed(test_inputs)

            logger.info(
                "mx.compile enabled for %s (primitive embedding path)",
                self._model_name,
            )
            return True
        except Exception as e:
            logger.info("mx.compile unavailable for %s: %s", self._model_name, e)
            self._compiled_embed = None
            return False

    def embed(
        self,
        inputs: str | list[str] | list[dict[str, str]],
        max_length: int | None = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> EmbeddingOutput:
        if not self._loaded:
            self.load()

        max_length = self._resolve_max_length(max_length)
        normalized_inputs = self._normalize_embedding_inputs(inputs)
        input_texts = [item["text"] for item in normalized_inputs if "text" in item]
        has_image_inputs = any("image" in item for item in normalized_inputs)

        processor = self._processor
        uses_custom_embedding_inputs = self._uses_custom_embedding_inputs(processor)
        if hasattr(processor, "_tokenizer") and not uses_custom_embedding_inputs:
            processor = processor._tokenizer

        if has_image_inputs and (
            self._using_native or not uses_custom_embedding_inputs
        ):
            raise ValueError(
                f"Embedding model '{self._model_name}' does not support image inputs"
            )

        embeddings_array = None
        total_tokens: int | None = None

        if self._using_native:
            if hasattr(processor, "__call__"):
                encoded = processor(
                    input_texts,
                    padding=padding,
                    truncation=truncation,
                    max_length=max_length,
                    return_tensors="np",
                )
                input_ids = mx.array(encoded["input_ids"])
                attention_mask = mx.array(encoded["attention_mask"])
            else:
                encoded_ids = []
                masks = []
                for text in input_texts:
                    enc = processor.encode(text, add_special_tokens=True)
                    ids = list(enc.ids)
                    if truncation:
                        ids = ids[:max_length]
                    encoded_ids.append(ids)
                max_len = max(len(ids) for ids in encoded_ids)
                padded = []
                for ids in encoded_ids:
                    pad_len = max_len - len(ids)
                    padded.append(ids + [0] * pad_len)
                    masks.append([1] * len(ids) + [0] * pad_len)
                input_ids = mx.array(padded)
                attention_mask = mx.array(masks)

            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings_array = self._extract_embeddings_array(outputs)
            total_tokens = self._count_prepared_tokens(
                {"attention_mask": attention_mask, "input_ids": input_ids}
            )
        else:
            if self._is_compiled and self._compiled_embed is not None:
                try:
                    prepared = self._prepare_embedding_inputs(
                        processor,
                        normalized_inputs,
                        max_length,
                        padding,
                        truncation,
                    )
                    if not isinstance(prepared, dict):
                        prepared = dict(prepared)
                    total_tokens = self._count_prepared_tokens(prepared)
                    embeddings_array = self._compiled_embed(prepared)
                except Exception as e:
                    logger.warning(
                        "compiled embedding path failed for %s: %s; "
                        "disabling compile and falling back to eager generate()",
                        self._model_name,
                        e,
                    )
                    self._is_compiled = False
                    self._compiled_embed = None
                    total_tokens = None

            if embeddings_array is None:
                if uses_custom_embedding_inputs:
                    prepared = self._prepare_embedding_inputs(
                        processor,
                        normalized_inputs,
                        max_length,
                        padding,
                        truncation,
                    )
                    if not isinstance(prepared, dict):
                        prepared = dict(prepared)
                    outputs = self._model(**self._adapt_model_inputs_for_call(prepared))
                    total_tokens = self._count_prepared_tokens(prepared)
                else:
                    from mlx_embeddings import generate

                    outputs = generate(
                        self._model,
                        processor,
                        input_texts,
                        max_length=max_length,
                        padding=padding,
                        truncation=truncation,
                    )
                embeddings_array = self._extract_embeddings_array(outputs)

        mx.eval(embeddings_array)
        embeddings = embeddings_array.tolist()
        if total_tokens is None:
            total_tokens = self._count_tokens(normalized_inputs)
        dimensions = len(embeddings[0]) if embeddings else 0

        return EmbeddingOutput(
            embeddings=embeddings,
            total_tokens=total_tokens,
            dimensions=dimensions,
        )

    def _count_tokens(self, inputs: list[str] | list[dict[str, str]]) -> int:
        total = 0
        processor = self._processor

        for item in self._normalize_embedding_inputs(inputs):
            text = item.get("text")
            if not text:
                continue
            if hasattr(processor, "encode"):
                tokens = processor.encode(text, add_special_tokens=True)
                if isinstance(tokens, list):
                    total += len(tokens)
                elif hasattr(tokens, "shape"):
                    total += tokens.shape[-1] if tokens.ndim > 0 else 1
                elif hasattr(tokens, "ids"):
                    total += len(tokens.ids)
                else:
                    total += len(tokens)
            elif hasattr(processor, "tokenizer"):
                tokens = processor.tokenizer.encode(text, add_special_tokens=True)
                total += len(tokens) if isinstance(tokens, list) else len(list(tokens))
            elif hasattr(processor, "_tokenizer"):
                tokens = processor._tokenizer.encode(text, add_special_tokens=True)
                total += len(tokens) if isinstance(tokens, list) else len(list(tokens))
            else:
                total += len(text.split()) + 2

        return total

    def _count_prepared_tokens(self, prepared_inputs: dict[str, Any]) -> int:
        attention_mask = prepared_inputs.get("attention_mask")
        if attention_mask is not None:
            try:
                return int(mx.sum(attention_mask).item())
            except (TypeError, ValueError):
                pass
            if isinstance(attention_mask, list):
                return int(
                    sum(
                        sum(row) if isinstance(row, list) else row
                        for row in attention_mask
                    )
                )
            if hasattr(attention_mask, "tolist"):
                values = attention_mask.tolist()
                if values and isinstance(values[0], list):
                    return int(sum(sum(row) for row in values))
                return int(sum(values))

        input_ids = prepared_inputs.get("input_ids")
        if input_ids is None:
            return 0
        if hasattr(input_ids, "shape"):
            if len(input_ids.shape) == 0:
                return 1
            if len(input_ids.shape) == 1:
                return int(input_ids.shape[0])
            return int(input_ids.shape[0] * input_ids.shape[1])
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                return int(sum(len(row) for row in input_ids))
            return int(len(input_ids))
        return 0

    @property
    def processor(self):
        return self._processor

    @property
    def hidden_size(self) -> int | None:
        return self._hidden_size

    def get_model_info(self) -> dict[str, Any]:
        if not self._loaded:
            return {"loaded": False, "model_name": self._model_name}

        info = {
            "loaded": True,
            "model_name": self._model_name,
            "hidden_size": self._hidden_size,
            "native_implementation": self._using_native,
            "compiled": self._is_compiled,
        }

        if hasattr(self._model, "config"):
            config = self._model.config
            info.update(
                {
                    "model_type": getattr(config, "model_type", None),
                    "vocab_size": getattr(config, "vocab_size", None),
                    "max_position_embeddings": getattr(
                        config, "max_position_embeddings", None
                    ),
                }
            )

        return info

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        impl = "native" if self._using_native else "mlx-embeddings"
        return (
            f"<MLXEmbeddingModel model={self._model_name} "
            f"status={status} impl={impl}>"
        )


class EmbeddingEngine(BaseNonStreamingEngine):
    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = False,
        batch_size: int | None = None,
        *,
        scheduler_config: Any | None = None,
    ):
        super().__init__()
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        if batch_size is None:
            batch_size = (
                getattr(scheduler_config, "embedding_batch_size", 32)
                if scheduler_config is not None
                else 32
            )
        self._batch_size = max(1, int(batch_size))
        self._model: MLXEmbeddingModel | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def processor(self) -> Any:
        return self._model.processor if self._model else None

    @property
    def hidden_size(self) -> int | None:
        return self._model.hidden_size if self._model else None

    async def start(self) -> None:
        if self._model is not None:
            return
        logger.info("Starting embedding engine: %s", self._model_name)
        self._model = MLXEmbeddingModel(
            self._model_name, trust_remote_code=self._trust_remote_code
        )
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(get_executor("llm"), self._model.load), timeout=120.0
        )

    async def stop(self) -> None:
        if self._model is None:
            return
        self._model = None
        gc.collect()
        loop = asyncio.get_running_loop()
        from ..scheduler.helpers import _safe_clear_cache_for_non_llm

        await asyncio.wait_for(
            loop.run_in_executor(get_executor("llm"), _safe_clear_cache_for_non_llm),
            timeout=5.0,
        )

    async def embed(
        self,
        texts: list[str] | list[dict[str, str]],
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
    ) -> EmbeddingOutput:
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")
        model = self._model
        input_items = [texts] if isinstance(texts, str) else list(texts)
        if not input_items:
            return EmbeddingOutput(embeddings=[], total_tokens=0, dimensions=0)
        batch_size = self._batch_size
        activity_id = self._begin_activity(
            "embedding", detail="Embedding", total_items=len(input_items)
        )
        try:
            loop = asyncio.get_running_loop()
            embeddings: list[list[float]] = []
            total_tokens = 0
            dimensions = 0
            for start in range(0, len(input_items), batch_size):
                batch = input_items[start : start + batch_size]

                def _embed_sync(b=batch):
                    return model.embed(
                        inputs=b,
                        max_length=max_length,
                        padding=padding,
                        truncation=truncation,
                    )

                output = await asyncio.wait_for(
                    loop.run_in_executor(get_executor("llm"), _embed_sync), timeout=30.0
                )
                embeddings.extend(output.embeddings)
                total_tokens += output.total_tokens
                if output.dimensions:
                    dimensions = output.dimensions
            return EmbeddingOutput(
                embeddings=embeddings, total_tokens=total_tokens, dimensions=dimensions
            )
        finally:
            self._end_activity(activity_id)

    def get_stats(self) -> dict[str, Any]:
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
            "hidden_size": self.hidden_size,
            "batch_size": self._batch_size,
        }

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<EmbeddingEngine model={self._model_name} status={status}>"
