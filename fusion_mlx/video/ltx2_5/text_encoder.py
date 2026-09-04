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

import dataclasses
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
        *,
        raw_weights: dict | None = None,
    ) -> tuple[Gemma4LanguageModel, dict]:
        # 返回 (language_model, projection_weights)。
        # 两类布局:
        #   1. canon Comfy: 顶层键 model.* + text_embedding_projection.* +
        #      内嵌 tokenizer_json / hf_asset__*。sanitize 剥 model. 前缀, fuzzy
        #      load (bf16, 非量化)。
        #   2. flat diffusers (#762, dgrauet q8): 顶层键 text_encoder.model.* +
        #      text_encoder.text_embedding_projection.*, 量化 (.scales/.biases)。
        #      需剥 text_encoder. 文件前缀但保留 model. 前缀 (量化键须精确匹配
        #      model.* 参数树), nn.quantize 后 load model.* 键。
        weights_path = Path(weights_path)
        raw = raw_weights if raw_weights is not None else mx.load(str(weights_path))
        lm = cls(config=config)

        flat_prefix = "text_encoder."
        mlxcomm_prefix = "language_model."
        is_flat = any(k.startswith(flat_prefix) for k in raw)
        is_mlxcomm = any(k.startswith(mlxcomm_prefix) for k in raw)
        if is_flat:
            stripped = {
                k[len(flat_prefix) :]: v
                for k, v in raw.items()
                if k.startswith(flat_prefix)
            }
        elif is_mlxcomm:
            # #786: mlx-community shard keys language_model.model.* (含 projection
            # 时 language_model.text_embedding_projection.*, 但投影通常在 connector)。
            # 剥 language_model. 前缀 -> 剩 model.* (走 canon 分支)。
            stripped = {
                k[len(mlxcomm_prefix) :]: v
                for k, v in raw.items()
                if k.startswith(mlxcomm_prefix)
            }
        else:
            stripped = raw

        proj_weights = {
            k[len(_PROJ_PREFIX) :]: v
            for k, v in stripped.items()
            if k.startswith(_PROJ_PREFIX)
        }

        is_quantized = any(k.endswith(".scales") for k in stripped)
        if is_quantized:
            # 保留 model. 前缀: QuantizedLinear .scales/.biases 键须精确匹配参数树
            # (Gemma4TextModel 持有 self.model -> 参数树为 model.layers.*)。
            lang_weights = {
                k: v for k, v in stripped.items() if k.startswith(_GEMMA4_PREFIX)
            }
            lang_weights = {
                k: (
                    v.astype(mx.bfloat16)
                    if hasattr(v, "dtype") and v.dtype == mx.float32
                    else v
                )
                for k, v in lang_weights.items()
            }
            lang_keys = set(lang_weights.keys())

            def _quant_predicate(path, module):
                return isinstance(module, nn.Linear) and f"{path}.scales" in lang_keys

            group_size, bits = _read_te_quant_config(weights_path.parent)
            nn.quantize(
                lm,
                group_size=group_size,
                bits=bits,
                class_predicate=_quant_predicate,
            )
            lm.load_weights(list(lang_weights.items()), strict=False)
            logger.info(
                "Gemma4LanguageModel.from_checkpoint: flat q8 loaded %d keys "
                "(quantized group=%d bits=%d) from %s",
                len(lang_weights),
                group_size,
                bits,
                weights_path.name,
            )
        else:
            lang_weights = lm.sanitize(stripped)
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
        return lm, proj_weights


# 为避免在类型注解处 import mlx_vlm, 用字符串前向引用。
TextConfig = object


def _read_te_quant_config(model_dir: Path) -> tuple[int, int]:
    # #762: flat diffusers text-encoder 量化参数。split_model.json 缺失时回退
    # q8 默认 (group_size=64, bits=8), 与 dgrauet/ltx-2.5-mlx-q8 一致。
    split_model = model_dir / "split_model.json"
    group_size = 64
    bits = 8
    if split_model.exists():
        try:
            with open(split_model) as f:
                cfg = json.load(f)
            if cfg.get("quantization_group_size"):
                group_size = int(cfg["quantization_group_size"])
            if cfg.get("quantization_bits"):
                bits = int(cfg["quantization_bits"])
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("ltx2_5 te quant config read failed: %s", exc)
    return group_size, bits


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

    # #792: 三处显式设置, 不依赖 TextConfig.__post_init__ 默认推导 (旧版 mlx_vlm
    # 默认 layer_types=full_attention 会导致所有层用 global_head_dim=512 崩溃):
    #   1. attention_k_eq_v=True: full_attention 层 (5/11/.../47) checkpoint 仅含
    #      k_proj=512 (1*512), 无 v_proj -> v 复用 k。默认 False 会用
    #      num_key_value_heads=8 -> 期望 k_proj=4096, 与 checkpoint 512 不匹配
    #      -> reshape ValueError。
    #   2. layer_types: 5 sliding + 1 full 重复 8 次 = 48 层 (sliding head_dim=256,
    #      full head_dim=512)。
    #   3. rope_parameters: full (theta=1e6, proportional, partial_rotary=0.25),
    #      sliding (theta=1e4, default)。
    # 验证来源: dgrauet/ltx-2.5-mlx-q8 text_encoder.safetensors 逐层 k/q/v 形状
    # + mlx-community/ltx-2.5-mlx-q8 gemma4-12b-ltx-v1/config.json 权威配置。
    layer_types = []
    for _ in range(8):
        layer_types.extend(["sliding_attention"] * 5 + ["full_attention"])
    rope_parameters = {
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "proportional",
        },
        "sliding_attention": {
            "rope_theta": 10000.0,
            "rope_type": "default",
        },
    }
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
        attention_k_eq_v=True,
        sliding_window=1024,
        sliding_window_pattern=6,
        layer_types=layer_types,
        rope_parameters=rope_parameters,
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
    # #762: flat diffusers (dgrauet) text_encoder_config.json 的 text_config 缺
    # sliding_window_pattern，TextConfig 默认值=5 会错建 v_proj 层 (应=6，
    # Gemma4-12b 每 6 层省 v_proj: 5/11/17/23/29/35/41/47)。必须用默认 config
    # 兜底缺字段，强制 pattern=6，避免 8 个 v_proj 找不到权重。
    from mlx_vlm.models.gemma4.config import TextConfig

    default_cfg = _build_default_text_config()
    if not (config_path and Path(config_path).exists()):
        logger.info("LTX-2.5 text config: using built-in Gemma4-12b defaults")
        return default_cfg

    with open(config_path) as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    # 仅取 TextConfig dataclass 字段，丢弃 bos_token_id/eos_token_id 等无关键。
    field_names = {fd.name for fd in dataclasses.fields(TextConfig)}
    merged = {
        fd.name: getattr(default_cfg, fd.name) for fd in dataclasses.fields(default_cfg)
    }
    for k, v in text_cfg.items():
        if k in field_names:
            merged[k] = v
    # sliding_window_pattern 可由 layer_types 数量推导；若文件未显式给出则强制
    # 默认值 6 (12b 验证过)，覆盖 TextConfig 默认 5。
    if "sliding_window_pattern" not in text_cfg:
        merged["sliding_window_pattern"] = 6
    logger.info(
        "LTX-2.5 text config loaded from %s (merged, pattern=%s)",
        config_path,
        merged.get("sliding_window_pattern"),
    )
    return TextConfig(**merged)


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


def _load_sharded_weights(shard_dir: Path) -> dict:
    # #786: 多分片 text_encoder 目录 (gemma4-12b-ltx-v1/model-*.safetensors)。
    # 按 model.safetensors.index.json 顺序合并所有分片 -> 单一 weights dict。
    # mlx-community 分片键 language_model.model.* (在 from_checkpoint 剥前缀)。
    index = shard_dir / "model.safetensors.index.json"
    if index.exists():
        with open(index) as f:
            idx = json.load(f)
        shard_files = sorted(set(idx.get("weight_map", {}).values()))
    else:
        shard_files = sorted(
            p.name
            for p in shard_dir.iterdir()
            if p.name.startswith("model-") and p.name.endswith(".safetensors")
        )
    if not shard_files:
        raise FileNotFoundError(f"no text-encoder shards found in {shard_dir}")
    merged: dict = {}
    for shard_name in shard_files:
        shard_path = shard_dir / shard_name
        part = mx.load(str(shard_path))
        merged.update(part)
        logger.info("ltx2_5 te shard loaded: %s (%d keys)", shard_name, len(part))
    logger.info(
        "ltx2_5 te sharded weights merged: %d keys from %d shards",
        len(merged),
        len(shard_files),
    )
    return merged


def _load_connector_projection(connector_path: Path) -> dict:
    # #786: mlx-community connector.safetensors 含 connector.text_embedding_projection.*
    # (video/audio_aggregate_embed)。剥 connector. 前缀 -> text_embedding_projection.*
    # 供 feature_extractor 装载。
    raw = mx.load(str(connector_path))
    proj_prefix = "connector." + _PROJ_PREFIX
    proj_weights = {
        k[len(proj_prefix) :]: v for k, v in raw.items() if k.startswith(proj_prefix)
    }
    logger.info(
        "ltx2_5 te connector projection: %d keys from %s",
        len(proj_weights),
        connector_path.name,
    )
    return proj_weights


def load_text_encoder(
    weights_path: str | Path,
    *,
    config_path: str | Path | None = None,
    tokenizer_cache_dir: str | Path | None = None,
    projection_weights_path: str | Path | None = None,
) -> LTX2_5TextEncoder:
    # 加载 Gemma4-12b + aggregate projection + tokenizer。三类布局:
    #   canon Comfy: 单文件, 内嵌 tokenizer_json + text_embedding_projection.*。
    #   flat diffusers (#762, dgrauet): 单文件 text_encoder.safetensors,
    #     tokenizer.json 在仓根 standalone。
    #   mlx-community (#786): weights_path 为分片目录 (gemma4-12b-ltx-v1/),
    #     projection 在独立 connector.safetensors (须传 projection_weights_path),
    #     tokenizer.json 在分片子目录 standalone。
    # fail visible (Rule 12): 权重/config 缺失直接 raise, 不静默零初始化。
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"LTX-2.5 text encoder weights not found: {weights_path}"
        )
    is_sharded = weights_path.is_dir()

    # config 探测: flat 仓根 text_encoder_config.json / mlxcomm 分片子目录
    # config.json (gemma4-12b-ltx-v1/config.json) / canon 同级 config.json。
    if config_path is None:
        cand_dirs = [weights_path.parent] if is_sharded else [weights_path.parent]
        for cand_dir in cand_dirs:
            for name in ("text_encoder_config.json", "config.json"):
                cand = cand_dir / name
                if cand.exists():
                    config_path = cand
                    break
            if config_path is not None:
                break

    text_config = _load_text_config(config_path)

    logger.info(
        "load_text_encoder: hidden_size=%d layers=%d caption=%d sharded=%s",
        getattr(text_config, "hidden_size", GEMMA4_HIDDEN_SIZE),
        getattr(text_config, "num_hidden_layers", GEMMA4_NUM_LAYERS),
        LTX2_5_CAPTION_CHANNELS,
        is_sharded,
    )

    if is_sharded:
        raw = _load_sharded_weights(weights_path)
    else:
        raw = mx.load(str(weights_path))

    language_model, proj_weights = Gemma4LanguageModel.from_checkpoint(
        weights_path, text_config, raw_weights=raw
    )

    # #786: mlx-community projection 在 connector.safetensors (非 TE 分片)。
    if not proj_weights and projection_weights_path is not None:
        proj_weights = _load_connector_projection(Path(projection_weights_path))

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

    # tokenizer: 内嵌 (canon) 或仓根 standalone (flat #762)。
    tokenizer = None
    cache_dir = (
        Path(tokenizer_cache_dir)
        if tokenizer_cache_dir
        else Path(tempfile.mkdtemp(prefix="ltx2_5_tok_"))
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        if TOKENIZER_ASSET in raw:
            tok_path = _extract_embedded_tokenizer(raw, cache_dir)
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(cache_dir), trust_remote_code=True
            )
            logger.info(
                "LTX2_5TextEncoder: loaded embedded tokenizer from %s", tok_path
            )
        else:
            # standalone tokenizer.json + tokenizer_config.json。flat #762 在仓根
            # (weights_path.parent); mlx-community #786 在分片子目录 (weights_path
            # 本身即为目录) 或其上级。
            tok_roots = (
                [weights_path, weights_path.parent]
                if is_sharded
                else [weights_path.parent]
            )
            tok_root = next(
                (r for r in tok_roots if (r / "tokenizer.json").exists()), None
            )
            if tok_root is not None:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    str(tok_root), trust_remote_code=True
                )
                logger.info(
                    "LTX2_5TextEncoder: loaded standalone tokenizer from %s",
                    tok_root,
                )
            else:
                raise FileNotFoundError(
                    "no embedded tokenizer_json and no standalone tokenizer.json "
                    f"at {tok_root}"
                )
        tokenizer.padding_side = "left"
    except Exception as exc:
        logger.warning(
            "LTX2_5TextEncoder: tokenizer load failed (%s); encode() will raise",
            exc,
        )

    return LTX2_5TextEncoder(language_model, feature_extractor, tokenizer=tokenizer)
