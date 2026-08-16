# SPDX-License-Identifier: Apache-2.0
# LTX-2.5 Gemma4-12b text encoder + aggregate feature extractor.
# Stubs mirror Gemma4LanguageModel.__call__(inputs, attention_mask,
# output_hidden_states) -> (final, [hidden_states...]) and
# GemmaFeaturesExtractorV2.__call__(hidden_states, attention_mask, mode).
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from fusion_mlx.video.ltx2_5.text_encoder import (
    LTX2_5_CAPTION_CHANNELS,
    LTX2_5TextEncoder,
    LTX2_5TextProjection,
)


class TestLTX2_5TextProjection:
    def test_output_dim_3840(self):
        proj = LTX2_5TextProjection(in_features=16, out_features=3840)
        x = mx.zeros((1, 5, 16))
        out = proj(x)
        assert out.shape == (1, 5, 3840)

    def test_default_caption_channels(self):
        proj = LTX2_5TextProjection(in_features=16)
        assert proj.linear2.weight.shape[0] == LTX2_5_CAPTION_CHANNELS

    def test_custom_hidden(self):
        proj = LTX2_5TextProjection(in_features=8, out_features=32, hidden_size=64)
        assert proj.linear1.weight.shape[0] == 64
        assert proj.linear2.weight.shape[0] == 32

    def test_forward_shape(self):
        proj = LTX2_5TextProjection(in_features=16, out_features=20)
        out = proj(mx.random.normal((2, 7, 16)))
        assert out.shape == (2, 7, 20)


class _StubFeatureExtractor(nn.Module):
    # Mimics GemmaFeaturesExtractorV2.__call__: takes hidden_states list +
    # attention_mask, returns (b, t, video_output_dim) or audio.
    def __init__(self, video_output_dim: int = 4096, audio_output_dim: int = 2048):
        super().__init__()
        self.video_output_dim = video_output_dim
        self.audio_output_dim = audio_output_dim
        self.video_aggregate_embed = nn.Linear(16, video_output_dim)
        self.audio_aggregate_embed = nn.Linear(16, audio_output_dim)

    def __call__(self, hidden_states, attention_mask, mode="video"):
        out_dim = self.video_output_dim if mode == "video" else self.audio_output_dim
        b, t = hidden_states[0].shape[0], hidden_states[0].shape[1]
        return mx.zeros((b, t, out_dim))


class _StubLanguageModel(nn.Module):
    # Mimics Gemma4LanguageModel.__call__(inputs, attention_mask,
    # output_hidden_states) -> (final, [hidden_states...]).
    def __init__(self, hidden_size: int = 16, num_hidden_states: int = 49):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_states = num_hidden_states

    def __call__(self, inputs, attention_mask=None, output_hidden_states=False):
        b, t = inputs.shape
        h = mx.zeros((b, t, self.hidden_size))
        if not output_hidden_states:
            return h
        hidden_states = [
            mx.zeros((b, t, self.hidden_size)) for _ in range(self.num_hidden_states)
        ]
        return h, hidden_states


class _StubTokenizer:
    # Mimics transformers tokenizer __call__: returns input_ids + attention_mask
    # padded to max_length.
    def __call__(
        self,
        prompt,
        *,
        return_tensors="np",
        max_length=1024,
        truncation=True,
        padding="max_length",
    ):
        import numpy as np

        tokens = [ord(c) % 100 for c in prompt[:max_length]]
        ids = tokens + [0] * (max_length - len(tokens))
        mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
        return {"input_ids": np.array([ids]), "attention_mask": np.array([mask])}

    @property
    def padding_side(self):
        return "left"

    @padding_side.setter
    def padding_side(self, v):
        pass


class TestLTX2_5TextEncoder:
    def _make_encoder(self, **kw):
        lm = _StubLanguageModel(hidden_size=16)
        fe = _StubFeatureExtractor(video_output_dim=4096, audio_output_dim=2048)
        tok = _StubTokenizer()
        return LTX2_5TextEncoder(lm, fe, tokenizer=tok, **kw)

    def test_encode_video_audio(self):
        enc = self._make_encoder()
        video, audio = enc.encode("hi", max_length=8, return_audio_embeddings=True)
        assert video.shape == (1, 8, 4096)
        assert audio.shape == (1, 8, 2048)

    def test_encode_video_only_returns_additive_mask(self):
        enc = self._make_encoder()
        video, additive_mask = enc.encode(
            "hi", max_length=8, return_audio_embeddings=False
        )
        assert video.shape == (1, 8, 4096)
        assert additive_mask.shape == (1, 1, 1, 8)

    def test_additive_mask_is_float(self):
        enc = self._make_encoder()
        _, additive_mask = enc.encode("hi", max_length=8, return_audio_embeddings=False)
        assert additive_mask.dtype in (mx.float32, mx.bfloat16, mx.float16)

    def test_call_alias(self):
        enc = self._make_encoder()
        v1, a1 = enc("hi", max_length=8)
        v2, a2 = enc.encode("hi", max_length=8)
        assert v1.shape == v2.shape
        assert a1.shape == a2.shape

    def test_caption_channels_field(self):
        enc = self._make_encoder(caption_channels=3840)
        assert enc.caption_channels == 3840

    def test_has_prompt_adaln_true(self):
        enc = self._make_encoder()
        assert enc.has_prompt_adaln is True

    def test_encode_no_tokenizer_raises(self):
        lm = _StubLanguageModel(hidden_size=16)
        fe = _StubFeatureExtractor()
        enc = LTX2_5TextEncoder(lm, fe, tokenizer=None)
        with pytest.raises(RuntimeError, match="tokenizer not loaded"):
            enc.encode("hi")

    def test_attention_mask_padding_propagates(self):
        # padded tokens (mask=0) should produce zeros via norm_and_concat masking
        # at the feature extractor; verify additive_mask shape + sign for padded.
        enc = self._make_encoder()
        _, additive_mask = enc.encode("hi", max_length=8, return_audio_embeddings=False)
        mx.eval(additive_mask)
        padded_val = float(additive_mask[0, 0, 0, 2].item())
        real_val = float(additive_mask[0, 0, 0, 0].item())
        assert padded_val < real_val


class TestSplitProjectionWeights:
    def test_split(self):
        from fusion_mlx.video.ltx2_5.text_encoder import _split_projection_weights

        weights = {
            "projection.linear1.weight": mx.zeros((4, 8)),
            "projection.linear2.weight": mx.zeros((4, 4)),
            "model.embed_tokens.weight": mx.zeros((10, 8)),
            "model.layers.0.weight": mx.zeros((8, 8)),
        }
        lang, proj = _split_projection_weights(weights)
        assert "model.embed_tokens.weight" in lang
        assert "model.layers.0.weight" in lang
        assert "linear1.weight" in proj
        assert "linear2.weight" in proj
        assert len(proj) == 2
        assert len(lang) == 2

    def test_no_projection_keys(self):
        from fusion_mlx.video.ltx2_5.text_encoder import _split_projection_weights

        lang, proj = _split_projection_weights({"model.weight": mx.zeros((2, 2))})
        assert len(proj) == 0
        assert len(lang) == 1


class TestLoadTextEncoderMissingFiles:
    def test_missing_weights_raises(self, tmp_path):
        from fusion_mlx.video.ltx2_5.text_encoder import load_text_encoder

        with pytest.raises(FileNotFoundError, match="weights not found"):
            load_text_encoder(tmp_path / "nope.safetensors")

    def test_missing_config_falls_back_to_builtin(self, tmp_path):
        # 新架构: config_path 缺省时用内置 Gemma4-12b 默认配置, 不 raise。
        from fusion_mlx.video.ltx2_5.text_encoder import _load_text_config

        cfg = _load_text_config(None)
        assert getattr(cfg, "hidden_size", 0) == 3840
        assert getattr(cfg, "num_hidden_layers", 0) == 48
