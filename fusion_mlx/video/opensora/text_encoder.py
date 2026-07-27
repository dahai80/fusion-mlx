# SPDX-License-Identifier: Apache-2.0
# Open-Sora V2 dual text encoder: T5-XXL + CLIP-ViT-Large-patch14.

import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class T5TextEncoder(nn.Module):
    # UMT5 or T5-XXL encoder for context embeddings (4096-dim).

    def __init__(self, model_path: str | None = None, max_length: int = 512):
        super().__init__()
        self.model_path = model_path
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import T5EncoderModel, T5Tokenizer

        logger.info(f"Loading T5 encoder from {self.model_path}")
        self._model = T5EncoderModel.from_pretrained(self.model_path)
        self._tokenizer = T5Tokenizer.from_pretrained(self.model_path)
        logger.info("T5 encoder loaded")

    def encode(self, text: str | list[str], max_length: int | None = None):
        self._ensure_loaded()
        ml = max_length or self.max_length
        if isinstance(text, str):
            text = [text]
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=ml,
            truncation=True,
        )
        import torch

        with torch.no_grad():
            outputs = self._model(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
            )
        hidden_states = mx.array(outputs.last_hidden_state.numpy())
        attention_mask = mx.array(inputs.attention_mask.numpy())
        mask = attention_mask[:, :, None].astype(hidden_states.dtype)
        hidden_states = hidden_states * mask
        return hidden_states


class CLIPTextEncoder(nn.Module):
    # CLIP-ViT-Large-patch14 for vector embeddings (768-dim pooler_output).

    def __init__(self, model_path: str | None = None, max_length: int = 77):
        super().__init__()
        self.model_path = model_path
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import CLIPTextModel, CLIPTokenizer

        logger.info(f"Loading CLIP encoder from {self.model_path}")
        self._model = CLIPTextModel.from_pretrained(self.model_path)
        self._tokenizer = CLIPTokenizer.from_pretrained(self.model_path)
        logger.info("CLIP encoder loaded")

    def encode(self, text: str | list[str], max_length: int | None = None):
        self._ensure_loaded()
        ml = max_length or self.max_length
        if isinstance(text, str):
            text = [text]
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=ml,
            truncation=True,
        )
        import torch

        with torch.no_grad():
            outputs = self._model(input_ids=inputs.input_ids)
        pooler = mx.array(outputs.pooler_output.numpy())
        return pooler


class DualTextEncoder(nn.Module):
    # Combines T5 (context) + CLIP (vector) encoders.

    def __init__(
        self,
        t5_path: str | None = None,
        clip_path: str | None = None,
        t5_max_length: int = 512,
        clip_max_length: int = 77,
    ):
        super().__init__()
        self.t5 = T5TextEncoder(t5_path, t5_max_length)
        self.clip = CLIPTextEncoder(clip_path, clip_max_length)

    def encode(self, text: str | list[str]):
        if isinstance(text, str):
            text = [text]
        context = self.t5.encode(text)
        y_vec = self.clip.encode(text)
        return context, y_vec
