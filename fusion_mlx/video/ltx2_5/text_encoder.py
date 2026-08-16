# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2.5 text encoder: Gemma4-12b + aggregate projection.
#
# LTX-2.5 单文件 text-encoder checkpoint
# (gemma4-12b-with-proj-ltx-2.5-bf16.safetensors) 含:
#   - model.* (666 keys)             Gemma4-12b 语言模型 (48 层, hidden 3840)
#   - text_embedding_projection.*    {video,audio}_aggregate_embed (188160->4096/2048)
#   - tokenizer_json + hf_asset__*   内嵌 tokenizer / configs (无需额外下载)
#
# 188160 = Gemma4 hidden_size 3840 × 49. 49 个 hidden state 的堆叠约定与
# ltx2 LanguageModel 一致: embedding h + layer 0..46 输出 + final RMSNorm 输出。
# feature_extractor_v2 (GemmaFeaturesExtractorV2) 做 per-token RMSNorm + concat +
# rescale + aggregate_embed (Linear)。connectors 归 transformer (LTX2_5Model) 持有,
# 在 generate_video 中显式运行, 这里只输出 pre-connector 的 4096/2048 维特征。
#
# Gemma4 config.json 不在 checkpoint 内, 从 google/gemma-4-12b-it (hf-mirror) 获取。
from __future__ import annotations

import json
import logging
import math
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# Gemma4-12b 文本编码维度 (AR doc §2.1)。
LTX2_5_CAPTION_CHANNELS = 3840
GEMMA4_HIDDEN_SIZE = 3840
GEMMA4_NUM_LAYERS = 48
# 49 = embedding + layer0..46 + final-norm, 与 ltx2 aggregate 约定一致。
GEMMA4_NUM_HIDDEN_STATES = GEMMA4_NUM_LAYERS + 1
# aggregate 线性层输入维度 = hidden × 49。
AGGREGATE_FLAT_DIM = GEMMA4_HIDDEN_SIZE * GEMMA4_NUM_HIDDEN_STATES
# text-encoder checkpoint 顶层键前缀。
_GEMMA4_PREFIX = "model."
_PROJ_PREFIX = "text_embedding_projection."
TOKENIZER_ASSET = "tokenizer_json"


def norm_and_concat_per_token_rms(
    encoded_text: mx.array,
    attention_mask: mx.array,
) -> mx.array:
    # 按 token 做 RMSNorm 后沿 hidden 维 concat (188160 = 3840 × 49)。
    b, t, d, num_layers = encoded_text.shape
    dtype = encoded_text.dtype

    variance = mx.mean(encoded_text.astype(mx.float32) ** 2, axis=2, keepdims=True)
    normed = encoded_text.astype(mx.float32) * mx.rsqrt(variance + 1e-6)
    normed = normed.astype(dtype)

    normed = mx.reshape(normed, (b, t, d * num_layers))

    mask_3d = attention_mask[:, :, None].astype(mx.bool_)
    normed = mx.where(mask_3d, normed, mx.zeros_like(normed))

    return normed


def _rescale_norm(x: mx.array, target_dim: int, source_dim: int) -> mx.array:
    return x * math.sqrt(target_dim / source_dim)


class GemmaFeaturesExtractorV2(nn.Module):
    # 49 层 hidden state -> per-token RMSNorm+concat -> rescale -> aggregate Linear。
    def __init__(
        self,
        flat_dim: int,
        embedding_dim: int,
        video_output_dim: int,
        audio_output_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.video_aggregate_embed = nn.Linear(flat_dim, video_output_dim, bias=bias)
        self.audio_aggregate_embed = nn.Linear(flat_dim, audio_output_dim, bias=bias)

    def __call__(
        self,
        hidden_states: list[mx.array],
        attention_mask: mx.array,
        mode: str = "video",
    ) -> mx.array:
        encoded = mx.stack(hidden_states, axis=-1)
        normed = norm_and_concat_per_token_rms(encoded, attention_mask)
        normed = normed.astype(encoded.dtype)

        if mode == "video":
            target_dim = self.video_aggregate_embed.weight.shape[0]
            return self.video_aggregate_embed(
                _rescale_norm(normed, target_dim, self.embedding_dim)
            )
        target_dim = self.audio_aggregate_embed.weight.shape[0]
        return self.audio_aggregate_embed(
            _rescale_norm(normed, target_dim, self.embedding_dim)
        )


class Gemma4LanguageModel(nn.Module):
    # 包装 Gemma4TextModel, 收集 49 个 hidden state (ltx2 约定)。
    # Gemma4 原生 __call__ 的 capture_layer_ids 收集所有 48 层 (无 embedding/final-norm),
    # 与 ltx2 aggregate 训练约定不同, 因此手写 forward: embedding h + layer0..46 + final-norm。
    # 掩码: 复用 ltx2 的 padding-aware 因果掩码 (sliding 层不加窗, 与 ltx2 Gemma3 一致)。
    def __init__(self, config):
        super().__init__()
        self.config = config
        from mlx_vlm.models.gemma4.language import Gemma4TextModel

        self.model = Gemma4TextModel(config)

    def _create_causal_mask_with_padding(
        self,
        seq_len: int,
        attention_mask: mx.array | None,
        dtype: mx.Dtype,
    ) -> mx.array:
        causal_mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))
        if attention_mask is not None:
            padding_mask = attention_mask.astype(mx.bool_)
            combined = causal_mask[None, :, :] & padding_mask[:, None, :]
            min_val = (
                mx.finfo(dtype).min if dtype in (mx.float16, mx.bfloat16) else -1e9
            )
            return mx.where(
                combined,
                mx.zeros(combined.shape, dtype=dtype),
                mx.full(combined.shape, min_val, dtype=dtype),
            )[:, None, :, :]
        min_val = mx.finfo(dtype).min if dtype in (mx.float16, mx.bfloat16) else -1e9
        return mx.where(
            causal_mask,
            mx.zeros((seq_len, seq_len), dtype=dtype),
            mx.full((seq_len, seq_len), min_val, dtype=dtype),
        )[None, None, :, :]

    def __call__(
        self,
        inputs: mx.array,
        attention_mask: mx.array | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[mx.array, list[mx.array]]:
        h = self.model.embed_tokens(inputs)
        h = h * self.model.embed_scale
        mx.eval(h)

        all_hidden_states = [h] if output_hidden_states else []

        cache = [None] * len(self.model.layers)
        full_mask = self._create_causal_mask_with_padding(
            inputs.shape[1], attention_mask, h.dtype
        )

        num_layers = len(self.model.layers)
        for i, layer in enumerate(self.model.layers):
            # Gemma4 DecoderLayer 返回 (h, shared_kv, offset); 12b 无 kv-shared。
            h, _kvs, _offset = layer(h, full_mask, cache[i])
            mx.eval(h)
            if output_hidden_states and i < num_layers - 1:
                all_hidden_states.append(h)

        hidden_states = self.model.norm(h)
        mx.eval(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            return hidden_states, all_hidden_states
        return hidden_states

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        # 去除 model. 前缀, float32->bfloat16。
        sanitized: dict[str, mx.array] = {}
        for key, value in weights.items():
            if key.startswith(_GEMMA4_PREFIX):
                nk = key[len(_GEMMA4_PREFIX) :]
                if hasattr(value, "dtype") and value.dtype == mx.float32:
                    sanitized[nk] = value.astype(mx.bfloat16)
                else:
                    sanitized[nk] = value
        return sanitized

    @property
    def layers(self) -> list[nn.Module]:
        return self.model.layers

    @classmethod
    def from_checkpoint(
        cls,
        weights_path: str | Path,
        config: TextConfig,
    ) -> Gemma4LanguageModel:
        weights_path = Path(weights_path)
        raw = mx.load(str(weights_path))
        lm = cls(config=config)
        lang_weights = lm.sanitize(raw)
        # 仅保留 Gemma4 语言模型键, 丢弃 projection/tokenizer/assets。
        lang_weights = {
            k: v
            for k, v in lang_weights.items()
            if not k.startswith("text_embedding_projection")
            and not k.startswith("vision_model")
            and not k.startswith("audio_projector")
            and not k.startswith("multi_modal_projector")
            and not k.startswith("hf_asset")
            and k != "tokenizer_json"
        }
        lm.load_weights(list(lang_weights.items()), strict=False)
        logger.info(
            "Gemma4LanguageModel.from_checkpoint: loaded %d keys from %s",
            len(lang_weights),
            weights_path.name,
        )
        return lm


# 为避免在类型注解处 import mlx_vlm, 用字符串前向引用。
TextConfig = object


def _extract_embedded_tokenizer(weights: dict, cache_dir: Path) -> Path:
    # tokenizer_json 以 uint8 字节数组存储, 写入临时文件供 AutoTokenizer 加载。
    import numpy as np

    blob = weights.get(TOKENIZER_ASSET)
    if blob is None:
        raise FileNotFoundError(
            "embedded tokenizer_json not found in text-encoder checkpoint"
        )
    data = bytes(np.array(blob.astype(mx.uint8)).tolist())
    tokenizer_path = cache_dir / "tokenizer.json"
    tokenizer_path.write_bytes(data)
    # 同步写入 tokenizer_config (hf_asset__tokenizer_config.json)。
    cfg_blob = weights.get("hf_asset__tokenizer_config.json")
    if cfg_blob is not None:
        cfg_data = bytes(np.array(cfg_blob.astype(mx.uint8)).tolist())
        (cache_dir / "tokenizer_config.json").write_bytes(cfg_data)
    return tokenizer_path


def _split_weights(weights: dict) -> tuple[dict, dict, dict]:
    # 返回 (lang, proj, assets)。lang = Gemma4 语言模型键 (model.*)。
    lang: dict = {}
    proj: dict = {}
    assets: dict = {}
    for k, v in weights.items():
        if k.startswith(_PROJ_PREFIX):
            proj[k[len(_PROJ_PREFIX) :]] = v
        elif k.startswith(_GEMMA4_PREFIX):
            lang[k] = v
        else:
            assets[k] = v
    logger.info(
        "LTX-2.5 text-encoder weights split: lang=%d proj=%d assets=%d",
        len(lang),
        len(proj),
        len(assets),
    )
    return lang, proj, assets


def _build_default_text_config() -> TextConfig:
    # Gemma4-12b TextConfig (来自 google/gemma-4-12b-it config.json)。
    # checkpoint 不含 config.json, 这里硬编码经过验证的 12b 配置。
    from mlx_vlm.models.gemma4.config import TextConfig

    return TextConfig(
        hidden_size=3840,
        num_hidden_layers=48,
        intermediate_size=15360,
        num_attention_heads=16,
        head_dim=256,
        global_head_dim=512,
        num_key_value_heads=8,
        num_global_key_value_heads=1,
        num_kv_shared_layers=0,
        sliding_window=1024,
        sliding_window_pattern=6,
        vocab_size=262144,
        vocab_size_per_layer_input=262144,
        hidden_size_per_layer_input=0,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        rope_traditional=False,
        partial_rotary_factor=1.0,
        global_partial_rotary_factor=0.25,
        tie_word_embeddings=True,
        use_double_wide_mlp=False,
        attention_bias=False,
        final_logit_softcapping=30.0,
        pad_token_id=0,
    )


def _load_text_config(config_path: str | Path | None) -> TextConfig:
    # 优先本地 config.json (text_config), 否则用 12b 默认配置。
    if config_path and Path(config_path).exists():
        from mlx_vlm.models.gemma4.config import TextConfig

        with open(config_path) as f:
            cfg = json.load(f)
        text_cfg = cfg.get("text_config", cfg)
        logger.info("LTX-2.5 text config loaded from %s", config_path)
        return TextConfig.from_dict(text_cfg)
    logger.info("LTX-2.5 text config: using built-in Gemma4-12b defaults")
    return _build_default_text_config()


class LTX2_5TextEncoder(nn.Module):
    # Gemma4-12b 语言模型 + feature_extractor_v2 (aggregate projection)。
    # connectors 不在此处 (归 LTX2_5Model), encode 输出 pre-connector 特征。
    def __init__(
        self,
        language_model: Gemma4LanguageModel,
        feature_extractor: GemmaFeaturesExtractorV2,
        *,
        tokenizer=None,
        caption_channels: int = LTX2_5_CAPTION_CHANNELS,
    ):
        super().__init__()
        self.language_model = language_model
        self.feature_extractor_v2 = feature_extractor
        self.processor = tokenizer
        self.caption_channels = caption_channels
        self.has_prompt_adaln = True

    def _make_additive_mask(
        self, attention_mask: mx.array, dtype: mx.Dtype
    ) -> mx.array:
        additive_mask = (attention_mask - 1).astype(dtype)
        return additive_mask.reshape(attention_mask.shape[0], 1, 1, -1) * 1e9

    def encode(
        self,
        prompt: str,
        *,
        max_length: int = 1024,
        return_audio_embeddings: bool = True,
    ) -> tuple[mx.array, mx.array]:
        if self.processor is None:
            raise RuntimeError("LTX2_5TextEncoder: tokenizer not loaded")
        inputs = self.processor(
            prompt,
            return_tensors="np",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = mx.array(inputs["input_ids"])
        attention_mask = mx.array(inputs["attention_mask"])

        _, all_hidden_states = self.language_model(
            inputs=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        video_features = self.feature_extractor_v2(
            all_hidden_states, attention_mask, mode="video"
        )
        additive_mask = self._make_additive_mask(attention_mask, video_features.dtype)

        if return_audio_embeddings:
            audio_features = self.feature_extractor_v2(
                all_hidden_states, attention_mask, mode="audio"
            )
            audio_mask = self._make_additive_mask(attention_mask, audio_features.dtype)
            logger.debug(
                "LTX2_5TextEncoder.encode: video=%s audio=%s",
                video_features.shape,
                audio_features.shape,
            )
            return video_features, audio_features
        logger.debug(
            "LTX2_5TextEncoder.encode: video=%s (audio skipped)",
            video_features.shape,
        )
        return video_features, additive_mask

    def __call__(self, prompt: str, **kwargs) -> tuple[mx.array, mx.array]:
        return self.encode(prompt, **kwargs)


# --- 兼容旧测试的最小存根 (将在后续测试重写中移除) ---
class LTX2_5TextProjection(nn.Module):
    # 已废弃: 真实 checkpoint 无 projection.* 键, 保留仅为向后兼容旧测试。
    def __init__(
        self,
        in_features: int,
        out_features: int = LTX2_5_CAPTION_CHANNELS,
        hidden_size: int | None = None,
        bias: bool = True,
    ):
        super().__init__()
        hidden = hidden_size or out_features
        self.linear1 = nn.Linear(in_features, hidden, bias=bias)
        self.act = nn.GELU(approx="tanh")
        self.linear2 = nn.Linear(hidden, out_features, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


def _split_projection_weights(weights: dict) -> tuple[dict, dict]:
    # 已废弃兼容包装: 用新 _split_weights 的 lang/proj 语义。
    lang, _proj, _assets = _split_weights(weights)
    legacy_proj = {
        k[len("projection.") :]: v
        for k, v in weights.items()
        if k.startswith("projection.")
    }
    return lang, legacy_proj


def load_text_encoder(
    weights_path: str | Path,
    *,
    config_path: str | Path | None = None,
    tokenizer_cache_dir: str | Path | None = None,
) -> LTX2_5TextEncoder:
    # 从单文件 checkpoint 加载 Gemma4-12b + aggregate projection + 内嵌 tokenizer。
    # fail visible (Rule 12): 权重/config 缺失直接 raise, 不静默零初始化。
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"LTX-2.5 text encoder weights not found: {weights_path}"
        )

    text_config = _load_text_config(config_path)

    logger.info(
        "load_text_encoder: hidden_size=%d layers=%d caption=%d",
        getattr(text_config, "hidden_size", GEMMA4_HIDDEN_SIZE),
        getattr(text_config, "num_hidden_layers", GEMMA4_NUM_LAYERS),
        LTX2_5_CAPTION_CHANNELS,
    )

    raw = mx.load(str(weights_path))
    _lang_weights, proj_weights, _assets = _split_weights(raw)

    language_model = Gemma4LanguageModel.from_checkpoint(weights_path, text_config)

    feature_extractor = GemmaFeaturesExtractorV2(
        flat_dim=AGGREGATE_FLAT_DIM,
        embedding_dim=GEMMA4_HIDDEN_SIZE,
        video_output_dim=4096,
        audio_output_dim=2048,
        bias=True,
    )
    if proj_weights:
        for attr, prefix in [
            ("video_aggregate_embed", "video_aggregate_embed"),
            ("audio_aggregate_embed", "audio_aggregate_embed"),
        ]:
            w_key = f"{prefix}.weight"
            b_key = f"{prefix}.bias"
            submodule = getattr(feature_extractor, attr)
            if w_key in proj_weights:
                submodule.weight = proj_weights[w_key]
            if b_key in proj_weights:
                submodule.bias = proj_weights[b_key]
        logger.info(
            "LTX2_5TextEncoder: loaded aggregate projection (%d keys)",
            len(proj_weights),
        )
    else:
        logger.warning(
            "LTX2_5TextEncoder: no text_embedding_projection keys found — "
            "feature extractor will use UNINITIALIZED weights (fail visible at encode)"
        )

    # 内嵌 tokenizer 写入临时目录供 AutoTokenizer 加载。
    tokenizer = None
    cache_dir = (
        Path(tokenizer_cache_dir)
        if tokenizer_cache_dir
        else Path(tempfile.mkdtemp(prefix="ltx2_5_tok_"))
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        tok_path = _extract_embedded_tokenizer(raw, cache_dir)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(cache_dir), trust_remote_code=True
        )
        tokenizer.padding_side = "left"
        logger.info("LTX2_5TextEncoder: loaded embedded tokenizer from %s", tok_path)
    except Exception as exc:
        logger.warning(
            "LTX2_5TextEncoder: embedded tokenizer load failed (%s); "
            "encode() will raise",
            exc,
        )

    return LTX2_5TextEncoder(language_model, feature_extractor, tokenizer=tokenizer)
