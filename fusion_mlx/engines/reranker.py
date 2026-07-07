# SPDX-License-Identifier: Apache-2.0
"""Reranker engine for fusion-mlx."""

import asyncio
import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..engine_core import get_executor
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)

_THINK_BLOCK = "<think>\n\n</think>\n\n"


def _coerce_item_to_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("text", "") or ""
    return str(item)


@dataclass
class RerankOutput:
    scores: list[float]
    indices: list[int]
    total_tokens: int


class MLXRerankerModel:
    _CAUSAL_LM_SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the "
        "Query and the Instruct provided. Note that the answer can only be "
        '"yes" or "no".'
    )
    _CAUSAL_LM_DEFAULT_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    _DEFAULT_MAX_LENGTH_SEQ_CLASSIFICATION = 512
    _DEFAULT_MAX_LENGTH_CAUSAL_LM = 8192

    def __init__(self, model_name: str, trust_remote_code: bool = False):
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._model = None
        self._processor = None
        self._loaded = False
        self._num_labels: int | None = None
        self._is_causal_lm = False
        self._is_jina_reranker = False
        self._is_vl_reranker = False
        self._token_true_id: int | None = None
        self._token_false_id: int | None = None
        self._doc_embed_token_id: int | None = None
        self._query_embed_token_id: int | None = None
        self._jina_projector = None
        self._prefix_tokens: list[int] | None = None
        self._suffix_tokens: list[int] | None = None
        self._is_compiled = False
        self._compiled_seq_logits = None

    @property
    def processor(self):
        return self._processor

    @property
    def num_labels(self) -> int | None:
        return self._num_labels

    def _get_architecture(self) -> str | None:
        config_path = Path(self._model_name) / "config.json"
        if not config_path.exists():
            return None
        try:
            with open(config_path) as f:
                config = json.load(f)
            architectures = config.get("architectures", [])
            return architectures[0] if architectures else None
        except (json.JSONDecodeError, OSError):
            return None

    def _load_xlm_roberta(self) -> tuple[Any, Any]:
        from transformers import AutoTokenizer

        from ..models.xlm_roberta import Model, ModelArgs

        model_path = Path(self._model_name)
        with open(model_path / "config.json") as f:
            config_dict = json.load(f)
        config = ModelArgs(
            **{
                k: v
                for k, v in config_dict.items()
                if k in ModelArgs.__dataclass_fields__
            }
        )
        model = Model(config)
        weights = {}
        weight_files = list(model_path.glob("*.safetensors"))
        for wf in weight_files:
            weights.update(mx.load(str(wf)))
        weights = model.sanitize(weights)
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())
        model.train(False)
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=self._trust_remote_code
        )
        return model, tokenizer

    def _load_vl_reranker(self) -> tuple[Any, Any]:
        from ..models.mlx_embeddings_compat import (
            patch_qwen3_vl_processor_for_torch_free_image_loading,
        )

        patch_qwen3_vl_processor_for_torch_free_image_loading()
        from mlx_embeddings import load as mlx_emb_load

        return mlx_emb_load(
            str(self._model_name),
            tokenizer_config={"trust_remote_code": self._trust_remote_code},
        )

    def _build_vl_item(self, item: "str | dict[str, Any]") -> dict[str, Any]:
        from ..utils.image import load_image

        if isinstance(item, str):
            return {"text": item}
        if not isinstance(item, dict):
            return {"text": str(item)}
        result: dict[str, Any] = {}
        text = item.get("text")
        if text:
            result["text"] = text
        image_ref = item.get("image")
        if image_ref:
            if isinstance(image_ref, str):
                result["image"] = load_image(image_ref)
            else:
                result["image"] = image_ref
        if not result:
            raise ValueError("VL reranker item must have at least 'text' or 'image'.")
        return result

    def _rerank_vl(
        self,
        query: "str | dict[str, Any]",
        documents: "list[str] | list[dict[str, Any]]",
        max_length: int,
    ) -> RerankOutput:
        query_item = self._build_vl_item(query)
        doc_items = [self._build_vl_item(d) for d in documents]
        inputs = {
            "instruction": self._CAUSAL_LM_DEFAULT_INSTRUCTION,
            "query": query_item,
            "documents": doc_items,
        }
        scores = self._model.process(inputs, processor=self._processor)
        mx.eval(scores)
        scores_list = [float(s) for s in scores.tolist()]
        indices = sorted(
            range(len(scores_list)),
            key=lambda i: scores_list[i],
            reverse=True,
        )
        return RerankOutput(
            scores=scores_list,
            indices=indices,
            total_tokens=0,
        )

    def _load_causal_lm(self) -> tuple[Any, Any]:
        from mlx_lm import load as mlx_lm_load

        from ..utils.model_loading import maybe_load_custom_quantization

        model_path = str(self._model_name)
        tokenizer_config = {"trust_remote_code": self._trust_remote_code}
        custom_loaded = maybe_load_custom_quantization(
            model_path,
            is_vlm=False,
        )
        if custom_loaded is not None:
            model, tokenizer_wrapper = custom_loaded
        else:
            loaded = mlx_lm_load(
                model_path,
                tokenizer_config=tokenizer_config,
                trust_remote_code=self._trust_remote_code,
            )
            model = loaded[0]
            tokenizer_wrapper = loaded[1]
        tokenizer = getattr(tokenizer_wrapper, "_tokenizer", tokenizer_wrapper)
        self._token_true_id = tokenizer.convert_tokens_to_ids("yes")
        self._token_false_id = tokenizer.convert_tokens_to_ids("no")
        if self._token_true_id is None or self._token_false_id is None:
            raise ValueError(
                "Could not find 'yes'/'no' token IDs in tokenizer. "
                "This model may not be a compatible CausalLM reranker."
            )
        _SENTINEL = "<<__CONTENT_SENTINEL__>>"
        messages = [
            {"role": "system", "content": self._CAUSAL_LM_SYSTEM_PROMPT},
            {"role": "user", "content": _SENTINEL},
        ]
        template_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        parts = template_str.split(_SENTINEL)
        if len(parts) != 2:
            raise ValueError(
                "Chat template produced unexpected format; "
                f"could not split on sentinel. Template: {template_str!r}"
            )
        prefix = parts[0]
        suffix = parts[1] + _THINK_BLOCK
        self._prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        self._suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        logger.info(
            "CausalLM reranker tokens: yes=%d, no=%d, prefix_len=%d, suffix_len=%d",
            self._token_true_id,
            self._token_false_id,
            len(self._prefix_tokens),
            len(self._suffix_tokens),
        )
        return model, tokenizer

    def _load_jina_reranker(self) -> tuple[Any, Any]:
        from mlx_lm import load as mlx_lm_load

        from ..utils.model_loading import maybe_load_custom_quantization

        model_path = str(self._model_name)
        tokenizer_config = {"trust_remote_code": self._trust_remote_code}
        custom_loaded = maybe_load_custom_quantization(
            model_path,
            is_vlm=False,
        )
        if custom_loaded is not None:
            model, tokenizer_wrapper = custom_loaded
        else:
            loaded = mlx_lm_load(
                model_path,
                tokenizer_config=tokenizer_config,
                trust_remote_code=self._trust_remote_code,
            )
            model = loaded[0]
            tokenizer_wrapper = loaded[1]
        tokenizer = getattr(tokenizer_wrapper, "_tokenizer", tokenizer_wrapper)
        doc_embed_token_id = self._resolve_token_id(tokenizer, "<|embed_token|>")
        query_embed_token_id = self._resolve_token_id(tokenizer, "<|rerank_token|>")
        if doc_embed_token_id is None or query_embed_token_id is None:
            raise ValueError(
                "Could not resolve required Jina special tokens "
                "('<|embed_token|>', '<|rerank_token|>'). "
                "This model may not be a compatible Jina v3 reranker."
            )
        self._doc_embed_token_id = doc_embed_token_id
        self._query_embed_token_id = query_embed_token_id
        self._jina_projector = self._load_jina_projector(self._model_name)
        logger.info(
            "Jina reranker tokens: embed_token=%d, rerank_token=%d",
            doc_embed_token_id,
            query_embed_token_id,
        )
        return model, tokenizer

    def _resolve_token_id(self, tokenizer: Any, token_text: str) -> int | None:
        added_tokens = getattr(tokenizer, "added_tokens_decoder", {}) or {}
        for tid, tinfo in added_tokens.items():
            content = ""
            if isinstance(tinfo, str):
                content = tinfo
            elif hasattr(tinfo, "content"):
                content = tinfo.content
            elif isinstance(tinfo, dict):
                content = tinfo.get("content", "")
            if content == token_text:
                return int(tid)
        convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
        if callable(convert_tokens_to_ids):
            try:
                token_id = convert_tokens_to_ids(token_text)
            except Exception:
                token_id = None
            if isinstance(token_id, int) and token_id >= 0:
                unk_token_id = getattr(tokenizer, "unk_token_id", None)
                if unk_token_id is None or token_id != unk_token_id:
                    return token_id
        get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
        if callable(get_added_vocab):
            try:
                added_vocab = get_added_vocab() or {}
            except Exception:
                added_vocab = {}
            token_id = added_vocab.get(token_text)
            if isinstance(token_id, int):
                return token_id
        get_vocab = getattr(tokenizer, "get_vocab", None)
        if callable(get_vocab):
            try:
                vocab = get_vocab() or {}
            except Exception:
                vocab = {}
            token_id = vocab.get(token_text)
            if isinstance(token_id, int):
                return token_id
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            try:
                encoded = encode(token_text, add_special_tokens=False)
            except TypeError:
                encoded = encode(token_text)
            except Exception:
                encoded = None
            if hasattr(encoded, "ids"):
                encoded = encoded.ids
            if (
                isinstance(encoded, list)
                and len(encoded) == 1
                and isinstance(encoded[0], int)
            ):
                return encoded[0]
        return None

    def _load_jina_projector(self, model_dir: str | Path):
        model_path = Path(model_dir)
        projector_path = model_path / "projector.safetensors"
        if not projector_path.exists():
            raise FileNotFoundError(
                f"Missing Jina projector file: {projector_path}. "
                "Expected projector.safetensors for JinaForRanking models."
            )
        weights = mx.load(str(projector_path))
        required_keys = ("linear1.weight", "linear2.weight")
        missing_keys = [key for key in required_keys if key not in weights]
        if missing_keys:
            raise ValueError(
                f"Jina projector is malformed: missing keys {missing_keys} in {projector_path}. "
                f"Available keys: {sorted(weights.keys())}"
            )
        linear1_weight = weights["linear1.weight"]
        linear2_weight = weights["linear2.weight"]
        if len(linear1_weight.shape) != 2 or len(linear2_weight.shape) != 2:
            raise ValueError(
                "Jina projector weights must be 2D matrices: "
                f"linear1.weight={linear1_weight.shape}, linear2.weight={linear2_weight.shape}."
            )
        if linear1_weight.shape != (512, 1024) or linear2_weight.shape != (512, 512):
            raise ValueError(
                "Unexpected Jina projector shapes. Expected "
                "linear1.weight=(512, 1024) and linear2.weight=(512, 512), "
                f"got linear1.weight={linear1_weight.shape}, linear2.weight={linear2_weight.shape}."
            )

        def _project(x):
            if x.shape[-1] != linear1_weight.shape[1]:
                raise ValueError(
                    f"Jina projector input dim mismatch for linear1: "
                    f"input={x.shape[-1]}, expected={linear1_weight.shape[1]}."
                )
            hidden = x @ mx.transpose(linear1_weight)
            hidden = mx.maximum(hidden, 0)
            return hidden @ mx.transpose(linear2_weight)

        return _project

    def _sanitize_jina_text(self, text: str) -> str:
        sanitized = str(text)
        sanitized = sanitized.replace("<|embed_token|>", " ")
        sanitized = sanitized.replace("<|rerank_token|>", " ")
        sanitized = sanitized.replace("<|score_token|>", " ")
        sanitized = sanitized.replace("<|im_start|>", " ")
        sanitized = sanitized.replace("<|im_end|>", " ")
        return sanitized.strip()

    def _format_jina_prompt(
        self,
        query: str,
        documents: list[str],
        instruction: str | None = None,
    ) -> str:
        sanitized_query = self._sanitize_jina_text(query)
        sanitized_docs = [self._sanitize_jina_text(doc) for doc in documents]
        sanitized_instruction = (
            self._sanitize_jina_text(instruction) if instruction is not None else None
        )
        user_content = (
            f"I will provide you with {len(sanitized_docs)} passages, each indicated "
            f"by a numerical identifier. Rank the passages based on their relevance "
            f"to query: {sanitized_query}\n"
        )
        if sanitized_instruction:
            user_content += f"<instruct>\n{sanitized_instruction}\n</instruct>\n"
        doc_prompts = [
            f'<passage id="{idx}">\n{doc}<|embed_token|>\n</passage>'
            for idx, doc in enumerate(sanitized_docs)
        ]
        user_content += "\n".join(doc_prompts) + "\n"
        user_content += f"<query>\n{sanitized_query}<|rerank_token|>\n</query>"
        system_prompt = (
            "You are a search relevance expert who can determine a ranking of the "
            "passages based on how relevant they are to the query. If the query is "
            "a question, how relevant a passage is depends on how well it answers the "
            "question. If not, try to analyze the intent of the query and "
            "assess how well each passage satisfies the intent. If an instruction "
            "is provided, you should follow the instruction when determining the "
            "ranking."
        )
        return (
            "<|im_start|>system\n"
            + system_prompt
            + "<|im_end|>\n"
            + "<|im_start|>user\n"
            + user_content
            + "<|im_end|>\n"
            + "<|im_start|>assistant\n"
            + _THINK_BLOCK
        )

    def _get_jina_hidden_states(self, input_ids):
        backbone = getattr(self._model, "model", None)
        if backbone is None or not callable(backbone):
            model_type = type(self._model).__name__ if self._model is not None else "None"
            raise ValueError(
                "Could not find Jina model backbone (model.model). "
                f"The mlx-lm model wrapper may have changed: {model_type}."
            )
        hidden_states = backbone(input_ids)
        if not hasattr(hidden_states, "shape"):
            raise ValueError("Jina backbone did not return hidden states as a tensor.")
        if len(hidden_states.shape) == 2:
            return mx.expand_dims(hidden_states, axis=0)
        if len(hidden_states.shape) != 3:
            raise ValueError(
                f"Jina hidden states must be rank 2 or 3. Got shape: {hidden_states.shape}"
            )
        return hidden_states

    def _cosine_similarity(self, query_vec, doc_vecs, eps: float = 1e-8):
        if len(query_vec.shape) == 2:
            query_vec = query_vec[0]
        if len(doc_vecs.shape) == 1:
            doc_vecs = mx.expand_dims(doc_vecs, axis=0)
        query_norm = mx.linalg.norm(query_vec)
        doc_norms = mx.linalg.norm(doc_vecs, axis=-1)
        denom = mx.maximum(doc_norms * query_norm, eps)
        numer = mx.sum(doc_vecs * query_vec, axis=-1)
        return numer / denom

    def load(self) -> None:
        from ..pool.model_discovery import (
            CAUSAL_LM_RERANKER_ARCHITECTURES,
            MULTIMODAL_RERANKER_ARCHITECTURES,
        )

        if self._loaded:
            return
        self._validate_architecture()
        arch = self._get_architecture()
        logger.info("Loading reranker model: %s (arch=%s)", self._model_name, arch)
        try:
            if arch in MULTIMODAL_RERANKER_ARCHITECTURES:
                self._model, self._processor = self._load_vl_reranker()
                self._is_vl_reranker = True
                self._num_labels = 1
            elif arch == "JinaForRanking":
                self._model, self._processor = self._load_jina_reranker()
                self._is_jina_reranker = True
                self._num_labels = 1
            elif arch in CAUSAL_LM_RERANKER_ARCHITECTURES:
                self._model, self._processor = self._load_causal_lm()
                self._is_causal_lm = True
                self._num_labels = 2
            elif arch == "XLMRobertaForSequenceClassification":
                self._model, self._processor = self._load_xlm_roberta()
                self._num_labels = getattr(self.model, "config", None)
                if self._num_labels is not None:
                    self._num_labels = getattr(self._num_labels, "num_labels", None)
            else:
                from ..models.mlx_embeddings_compat import (
                    patch_qwen3_vl_processor_for_torch_free_image_loading,
                )

                patch_qwen3_vl_processor_for_torch_free_image_loading()
                from mlx_embeddings import load

                self._model, self._processor = load(
                    self._model_name,
                    tokenizer_config={"trust_remote_code": self._trust_remote_code},
                )
                if hasattr(self._model, "config"):
                    config = self._model.config
                    self._num_labels = getattr(config, "num_labels", None)
            self._is_compiled = self._try_compile()
            self._loaded = True
            logger.info(
                "Reranker model loaded: %s (arch=%s, num_labels=%s, "
                "causal_lm=%s, vl=%s, jina=%s, compiled=%s)",
                self._model_name,
                arch,
                self._num_labels,
                self._is_causal_lm,
                self._is_vl_reranker,
                self._is_jina_reranker,
                self._is_compiled,
            )
        except ImportError as e:
            raise ImportError(
                "mlx-lm, mlx-embeddings, or transformers is required for reranking. "
                "Install with: pip install mlx-lm mlx-embeddings transformers"
            ) from e
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No safetensors weight files found for '{self._model_name}'. "
                "Reranker models require weights in safetensors format. "
                "If this is a PyTorch model, use an MLX-converted version "
                "(e.g., from mlx-community on HuggingFace)."
            )
        except Exception:
            logger.exception("Failed to load reranker model: %s", self._model_name)
            raise

    def _try_compile(self) -> bool:
        if self._is_causal_lm or self._is_vl_reranker or self._is_jina_reranker:
            logger.info("mx.compile skipped for %s", self._model_name)
            self._compiled_seq_logits = None
            return False
        base_model = self._model
        if not callable(base_model):
            return False
        try:

            def _compiled_seq_logits(inputs):
                outputs = base_model(**inputs)
                if (
                    hasattr(outputs, "pooler_output")
                    and outputs.pooler_output is not None
                ):
                    return outputs.pooler_output
                raise ValueError(
                    "Model output does not contain pooler_output. "
                    "Ensure the model is a SequenceClassification model."
                )

            self._compiled_seq_logits = mx.compile(_compiled_seq_logits)
            test_inputs = {
                "input_ids": mx.zeros((1, 4), dtype=mx.int32),
                "attention_mask": mx.ones((1, 4), dtype=mx.int32),
            }
            _ = self._compiled_seq_logits(test_inputs)
            logger.info(
                "mx.compile enabled for %s (primitive reranker logits path)",
                self._model_name,
            )
            return True
        except Exception as e:
            logger.info("mx.compile unavailable for %s: %s", self._model_name, e)
            self._compiled_seq_logits = None
            return False

    def rerank(
        self,
        query: "str | dict",
        documents: "list[str] | list[dict]",
        max_length: int | None = None,
    ) -> RerankOutput:
        if not self._loaded:
            self.load()
        if not documents:
            return RerankOutput(scores=[], indices=[], total_tokens=0)
        if self._is_vl_reranker:
            effective_max_length = (
                max_length if max_length is not None else self._DEFAULT_MAX_LENGTH_CAUSAL_LM
            )
            return self._rerank_vl(query, documents, effective_max_length)
        query_str = _coerce_item_to_text(query)
        docs_str = [_coerce_item_to_text(d) for d in documents]
        if self._is_jina_reranker:
            effective_max_length = (
                max_length if max_length is not None else self._DEFAULT_MAX_LENGTH_CAUSAL_LM
            )
            return self._rerank_jina(query_str, docs_str, effective_max_length)
        elif self._is_causal_lm:
            effective_max_length = (
                max_length if max_length is not None else self._DEFAULT_MAX_LENGTH_CAUSAL_LM
            )
            return self._rerank_causal_lm(query_str, docs_str, effective_max_length)
        else:
            effective_max_length = (
                max_length
                if max_length is not None
                else self._DEFAULT_MAX_LENGTH_SEQ_CLASSIFICATION
            )
            return self._rerank_seq_classification(
                query_str, docs_str, effective_max_length
            )

    def _rerank_causal_lm(
        self,
        query: str,
        documents: list[str],
        max_length: int = 8192,
    ) -> RerankOutput:
        tokenizer = self._processor
        prefix_tokens = self._prefix_tokens
        suffix_tokens = self._suffix_tokens
        if not callable(tokenizer):
            raise ValueError("CausalLM reranker tokenizer is not initialized.")
        if prefix_tokens is None or suffix_tokens is None:
            raise ValueError("CausalLM reranker prompt tokens are not initialized.")
        if not callable(self._model):
            raise ValueError("CausalLM reranker model is not initialized.")
        max_content_tokens = max_length - len(prefix_tokens) - len(suffix_tokens)
        pairs_text = []
        for doc in documents:
            content = (
                f"<Instruct>: {self._CAUSAL_LM_DEFAULT_INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"
            )
            pairs_text.append(content)
        content_encodings = tokenizer(
            pairs_text,
            padding=False,
            truncation=True,
            return_attention_mask=False,
            max_length=max_content_tokens,
            add_special_tokens=False,
        )
        all_input_ids = []
        for content_ids in content_encodings["input_ids"]:
            full_ids = prefix_tokens + content_ids + suffix_tokens
            all_input_ids.append(full_ids)
        scores = []
        total_tokens = 0
        for ids in all_input_ids:
            input_ids = mx.array([ids])
            logits = self._model(input_ids)
            last_logits = logits[0, -1, :]
            true_logit = last_logits[self._token_true_id]
            false_logit = last_logits[self._token_false_id]
            paired = mx.array([false_logit, true_logit])
            probs = mx.softmax(paired)
            mx.eval(probs)
            scores.append(probs[1].item())
            total_tokens += len(ids)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_indices = [idx for idx, _ in indexed_scores]
        return RerankOutput(
            scores=scores,
            indices=sorted_indices,
            total_tokens=total_tokens,
        )

    def _rerank_jina(
        self,
        query: str,
        documents: list[str],
        max_length: int = 8192,
    ) -> RerankOutput:
        tokenizer = self._processor
        doc_embed_token_id = self._doc_embed_token_id
        query_embed_token_id = self._query_embed_token_id
        projector = self._jina_projector
        if tokenizer is None:
            raise ValueError("Jina reranker tokenizer is not initialized.")
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            raise ValueError("Jina reranker tokenizer does not provide encode().")
        if (
            doc_embed_token_id is None
            or query_embed_token_id is None
            or projector is None
        ):
            raise ValueError(
                "Jina reranker is not fully initialized. "
                "Missing special-token IDs or projector."
            )

        def _to_token_ids(text: str) -> list[int]:
            encoded = encode(text, add_special_tokens=False)
            if hasattr(encoded, "ids"):
                return list(encoded.ids)
            return list(encoded)

        decode = getattr(tokenizer, "decode", None)

        def _truncate_doc_to_fit(
            query_text: str, doc_text: str
        ) -> tuple[str, list[int]]:
            doc_token_ids = _to_token_ids(doc_text)
            if not doc_token_ids:
                prompt = self._format_jina_prompt(query_text, [""])
                prompt_ids = _to_token_ids(prompt)[:max_length]
                return "", prompt_ids
            best_doc = ""
            best_ids: list[int] = []
            lo = 0
            hi = len(doc_token_ids)
            while lo <= hi:
                mid = (lo + hi) // 2
                if callable(decode):
                    candidate_doc = decode(
                        doc_token_ids[:mid], skip_special_tokens=False
                    )
                else:
                    candidate_doc = doc_text[:mid]
                prompt = self._format_jina_prompt(query_text, [candidate_doc])
                prompt_ids = _to_token_ids(prompt)
                if len(prompt_ids) <= max_length:
                    best_doc = candidate_doc
                    best_ids = prompt_ids
                    lo = mid + 1
                else:
                    hi = mid - 1
            if not best_ids:
                raise ValueError(
                    f"Could not fit even a minimally truncated document into max_length. "
                    f"max_length={max_length}"
                )
            return best_doc, best_ids

        sanitized_query = self._sanitize_jina_text(query)
        sanitized_docs = [self._sanitize_jina_text(doc) for doc in documents]
        scores = [0.0] * len(documents)
        total_tokens = 0
        start = 0
        while start < len(sanitized_docs):
            chunk_doc_indices: list[int] = []
            chunk_docs: list[str] = []
            chunk_input_ids: list[int] | None = None
            cursor = start
            while cursor < len(sanitized_docs):
                candidate_docs = chunk_docs + [sanitized_docs[cursor]]
                candidate_prompt = self._format_jina_prompt(
                    sanitized_query, candidate_docs
                )
                candidate_ids = _to_token_ids(candidate_prompt)
                if len(candidate_ids) <= max_length:
                    chunk_docs = candidate_docs
                    chunk_doc_indices.append(cursor)
                    chunk_input_ids = candidate_ids
                    cursor += 1
                    continue
                if chunk_docs:
                    break
                truncated_doc, truncated_ids = _truncate_doc_to_fit(
                    sanitized_query,
                    sanitized_docs[cursor],
                )
                chunk_docs = [truncated_doc]
                chunk_doc_indices = [cursor]
                chunk_input_ids = truncated_ids
                cursor += 1
                break
            if chunk_input_ids is None or not chunk_doc_indices:
                raise ValueError("Failed to create a valid Jina reranker chunk.")
            input_array = mx.array([chunk_input_ids])
            hidden_states = self._get_jina_hidden_states(input_array)
            query_positions = [
                pos
                for pos, token_id in enumerate(chunk_input_ids)
                if token_id == query_embed_token_id
            ]
            if not query_positions:
                raise ValueError(
                    "Jina prompt does not contain '<|rerank_token|>' in tokenized input."
                )
            doc_positions = [
                pos
                for pos, token_id in enumerate(chunk_input_ids)
                if token_id == doc_embed_token_id
            ]
            if len(doc_positions) < len(chunk_docs):
                raise ValueError(
                    "Jina prompt/doc mismatch: detected fewer '<|embed_token|>' "
                    "positions than documents in chunk."
                )
            selected_doc_positions = doc_positions[: len(chunk_docs)]
            query_hidden = hidden_states[0, query_positions[0], :]
            doc_hidden = hidden_states[0, selected_doc_positions, :]
            query_vec = projector(query_hidden)
            doc_vecs = projector(doc_hidden)
            similarities = self._cosine_similarity(query_vec, doc_vecs)
            mx.eval(similarities)
            chunk_scores = similarities.tolist()
            for original_idx, score in zip(chunk_doc_indices, chunk_scores):
                scores[original_idx] = float(score)
            total_tokens += len(chunk_input_ids)
            start = cursor
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_indices = [idx for idx, _ in indexed_scores]
        return RerankOutput(
            scores=scores,
            indices=sorted_indices,
            total_tokens=total_tokens,
        )

    def _rerank_seq_classification(
        self,
        query: str,
        documents: list[str],
        max_length: int = 512,
    ) -> RerankOutput:
        processor = self._processor
        processor_class = type(processor).__name__
        if processor_class == "TokenizerWrapper" and hasattr(processor, "_tokenizer"):
            processor = processor._tokenizer
        if not callable(processor):
            raise ValueError("SequenceClassification processor is not initialized.")
        pairs = [(query, doc) for doc in documents]
        inputs = processor(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        input_ids = mx.array(inputs["input_ids"])
        attention_mask = mx.array(inputs["attention_mask"])
        logits = None
        if self._is_compiled and self._compiled_seq_logits is not None:
            try:
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                logits = self._compiled_seq_logits(model_inputs)
            except Exception as e:
                logger.warning(
                    "compiled reranker path failed for %s: %s; "
                    "disabling compile and falling back to eager forward()",
                    self._model_name,
                    e,
                )
                self._is_compiled = False
                self._compiled_seq_logits = None
        if logits is None:
            if not callable(self._model):
                raise ValueError("SequenceClassification model is not initialized.")
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                logits = outputs.pooler_output
            else:
                raise ValueError(
                    "Model output does not contain pooler_output. "
                    "Ensure the model is a SequenceClassification model."
                )
        mx.eval(logits)
        if logits.shape[-1] == 1:
            scores = logits.squeeze(-1).tolist()
        else:
            scores = logits[:, -1].tolist()
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_indices = [idx for idx, _ in indexed_scores]
        total_tokens = self._count_tokens(query, documents)
        return RerankOutput(
            scores=scores,
            indices=sorted_indices,
            total_tokens=total_tokens,
        )

    def _count_tokens(self, query: str, documents: list[str]) -> int:
        total = 0
        processor = self._processor
        processor_class = type(processor).__name__
        if processor_class == "TokenizerWrapper" and hasattr(processor, "_tokenizer"):
            processor = processor._tokenizer

        def get_token_count(text: str, add_special: bool = True) -> int:
            if hasattr(processor, "encode"):
                tokens = processor.encode(text, add_special_tokens=add_special)
                if isinstance(tokens, list):
                    return len(tokens)
                elif hasattr(tokens, "ids"):
                    return len(tokens.ids)
                else:
                    return len(tokens)
            else:
                return len(text.split()) + (2 if add_special else 0)

        query_len = get_token_count(query, add_special=True)
        for doc in documents:
            doc_len = get_token_count(doc, add_special=False)
            total += query_len + doc_len + 3
        return total

    def _validate_architecture(self) -> None:
        from ..pool.model_discovery import (
            CAUSAL_LM_RERANKER_ARCHITECTURES,
            MULTIMODAL_RERANKER_ARCHITECTURES,
            SUPPORTED_RERANKER_ARCHITECTURES,
            _is_causal_lm_reranker,
        )

        config_path = Path(self._model_name) / "config.json"
        if not config_path.exists():
            return
        try:
            with open(config_path) as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read config.json: %s", e)
            return
        architectures = config.get("architectures", [])
        if not architectures:
            return
        arch = architectures[0]
        if arch in CAUSAL_LM_RERANKER_ARCHITECTURES:
            if not _is_causal_lm_reranker(Path(self._model_name)):
                raise ValueError(
                    f"Architecture {arch} is a CausalLM that can be used as a "
                    "reranker, but the model directory name "
                    f"'{Path(self._model_name).name}' does not contain "
                    "'reranker' or 'rerank'. Please rename the directory or "
                    "use the correct model."
                )
            return
        if arch in MULTIMODAL_RERANKER_ARCHITECTURES:
            if not _is_causal_lm_reranker(Path(self._model_name)):
                raise ValueError(
                    f"Architecture {arch} is a VLM that can be used as a "
                    "reranker, but the model directory name "
                    f"'{Path(self._model_name).name}' does not contain "
                    "'reranker' or 'rerank'. Please rename the directory or "
                    "use the correct model."
                )
            return
        if arch not in SUPPORTED_RERANKER_ARCHITECTURES:
            supported_list = ", ".join(
                sorted(
                    SUPPORTED_RERANKER_ARCHITECTURES
                    | CAUSAL_LM_RERANKER_ARCHITECTURES
                    | MULTIMODAL_RERANKER_ARCHITECTURES
                )
            )
            raise ValueError(
                f"Unsupported reranker architecture: {arch}. "
                f"Currently supported architectures: {supported_list}."
            )

    def get_model_info(self) -> dict[str, Any]:
        if not self._loaded:
            return {"loaded": False, "model_name": self._model_name}
        info: dict[str, Any] = {
            "loaded": True,
            "model_name": self._model_name,
            "num_labels": self._num_labels,
        }
        if hasattr(self._model, "config"):
            config = self._model.config
            info.update(
                {
                    "model_type": getattr(config, "model_type", None),
                    "hidden_size": getattr(config, "hidden_size", None),
                    "max_position_embeddings": getattr(
                        config, "max_position_embeddings", None
                    ),
                }
            )
        return info

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"<MLXRerankerModel model={self._model_name} status={status}>"


class RerankerEngine(BaseNonStreamingEngine):
    def __init__(self, model_name: str, trust_remote_code: bool = False):
        super().__init__()
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._model: MLXRerankerModel | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def processor(self) -> Any:
        return self._model.processor if self._model else None

    @property
    def num_labels(self) -> int | None:
        return self._model.num_labels if self._model else None

    async def start(self) -> None:
        if self._model is not None:
            return
        logger.info("Starting reranker engine: %s", self._model_name)
        self._model = MLXRerankerModel(
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

    async def rerank(
        self,
        query: "str | dict",
        documents: "list[str] | list[dict]",
        top_n: int | None = None,
        max_length: int | None = None,
    ) -> RerankOutput:
        if self._model is None:
            raise RuntimeError("Engine not started.")
        activity_id = self._begin_activity("reranking", total_items=len(documents))
        try:
            loop = asyncio.get_running_loop()

            def _rerank():
                return self._model.rerank(
                    query=query, documents=documents, max_length=max_length
                )

            output = await asyncio.wait_for(
                loop.run_in_executor(get_executor("llm"), _rerank), timeout=30.0
            )
            self._update_activity(activity_id, token_count=output.total_tokens)
            if top_n is not None and top_n < len(output.indices):
                return RerankOutput(
                    scores=output.scores,
                    indices=output.indices[:top_n],
                    total_tokens=output.total_tokens,
                )
            return output
        finally:
            await self._finish_activity(activity_id)

    def get_stats(self) -> dict[str, Any]:
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
            "num_labels": self.num_labels,
        }

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<RerankerEngine model={self._model_name} status={status}>"
