# SPDX-License-Identifier: Apache-2.0
# MiniMax H3 文本编码器：基于 Qwen3-VL，读取第 50 层 decoder hidden states。
# 契约（AR doc + 源码 modular_blocks_minimax_h3.py:150-151）：
#   - text_encoder = Qwen3VLForConditionalGeneration
#   - 输出 prompt_embeds shape (1, num_text_tokens, 5120)
#   - "read after the 50th decoder layer"（即 layers[49] 输出，不接 final norm）
#
# 移植策略：复用 mlx-vlm qwen3_vl 的 Qwen3VLModel 层实现，自定义 forward 在第 50 层截断。
# 上游验证：mlx-vlm 0.5.0 已支持 qwen3_vl（无上游阻塞，无需提 issue）。
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# H3 读取的 decoder 层索引（第 50 层，0-indexed = 49）。
H3_TEXT_ENCODER_LAYER = 49


class MiniMaxH3TextEncoder(nn.Module):
    # 包装 Qwen3-VL language model，暴露第 50 层 hidden states 给 Omni-Transformer。

    def __init__(self, language_model, layer: int = H3_TEXT_ENCODER_LAYER):
        super().__init__()
        self.language_model = language_model
        self.layer = int(layer)
        text_cfg = getattr(language_model, "args", None)
        hidden_size = getattr(text_cfg, "hidden_size", None)
        num_layers = getattr(text_cfg, "num_hidden_layers", None)
        if num_layers is not None and self.layer >= num_layers:
            raise ValueError(
                f"H3 text encoder layer {self.layer} (50th) >= "
                f"num_hidden_layers {num_layers}"
            )
        self.hidden_size = hidden_size
        logger.info(
            "minimax_h3 text_encoder: layer=%d hidden_size=%s",
            self.layer,
            hidden_size,
        )

    def _forward_layers(
        self, input_ids, inputs_embeds=None, mask=None, position_ids=None
    ):
        # 复刻 mlx-vlm Qwen3VLModel.__call__ 的层循环，在第 self.layer 层截断。
        # 不接 final norm（self.language_model.model.norm）—— H3 读 raw hidden states。
        model = self.language_model.model
        if inputs_embeds is None:
            h = model.embed_tokens(input_ids)
        else:
            h = inputs_embeds
        cache = [None] * len(model.layers)
        for layer_idx, (layer, c) in enumerate(zip(model.layers, cache)):
            h = layer(h, mask, c, position_ids)
            if layer_idx == self.layer:
                return h
        # layer 越界已在 __init__ 拦截，理论不可达。
        return h

    def __call__(self, input_ids, attention_mask=None, position_ids=None):
        # t2va 文本路径：纯文本 token，无视觉输入。
        # input_ids: (1, seq); attention_mask: (1, seq) int 或 None。
        mask = _mask_from_attention(attention_mask, input_ids)
        h = self._forward_layers(input_ids, mask=mask, position_ids=position_ids)
        seq_len = int(input_ids.shape[1])
        out = h[:, :seq_len, :]
        logger.info(
            "minimax_h3 text_encoder: out shape=%s dtype=%s",
            out.shape,
            out.dtype,
        )
        return out


def _mask_from_attention(attention_mask, input_ids):
    # attention_mask (1,seq) 1=keep 0=pad → 返回 mlx-vlm 层期望的 additive 4D mask 或 None。
    # H3 t2va 通常无 padding，None 即可；有 padding 时构造 4D additive mask。
    # mask dtype 须与 q/k/v (bfloat16) 可 promote，否则 mx.fast.scaled_dot_product_attention
    # 报 "Mask type must promote to output type bfloat16"。
    if attention_mask is None:
        return None
    b, s = attention_mask.shape
    # 全 1（无 padding）直接返 None，省一次构造。
    am = mx.asarray(attention_mask)
    if int(am.min()) == 1:
        return None
    am = am.astype(mx.bfloat16)
    mask = mx.where(
        am > 0, mx.array(0.0, mx.bfloat16), mx.array(-float("inf"), mx.bfloat16)
    )
    mask = mask[:, None, None, :]
    mask = mx.broadcast_to(mask, (b, 1, s, s))
    return mask


class MiniMaxH3MultimodalTextEncoder(nn.Module):
    # ref2va 多模态文本编码器：保留 Qwen3-VL vision_tower，参考视频经
    # vision_tower → video_pad masked_scatter → deepstack 注入 → 3D mrope
    # position_ids，读取第 50 层（0-indexed=49）hidden states，不接 final norm。
    # 与 MiniMaxH3TextEncoder（纯文本，丢弃 vision_tower）互补。

    def __init__(self, vlm, layer: int = H3_TEXT_ENCODER_LAYER):
        super().__init__()
        self.vlm = vlm
        self.layer = int(layer)
        language_model = getattr(vlm, "language_model", vlm)
        self.language_model = language_model
        text_cfg = getattr(language_model, "args", None)
        hidden_size = getattr(text_cfg, "hidden_size", None)
        num_layers = getattr(text_cfg, "num_hidden_layers", None)
        if num_layers is not None and self.layer >= num_layers:
            raise ValueError(
                f"H3 multimodal text encoder layer {self.layer} (50th) >= "
                f"num_hidden_layers {num_layers}"
            )
        self.hidden_size = hidden_size
        logger.info(
            "minimax_h3 ref2va text_encoder: layer=%d hidden_size=%s vision_tower=%s",
            self.layer,
            hidden_size,
            type(getattr(vlm, "vision_tower", None)).__name__,
        )

    def __call__(
        self,
        input_ids,
        pixel_values_videos=None,
        video_grid_thw=None,
        attention_mask=None,
    ):
        # 1. 视觉路径：复用 mlx-vlm Model.get_input_embeddings，内部完成
        #    vision_tower 前向 + video_pad masked_scatter + deepstack 特征 +
        #    get_rope_index 计算 3D mrope position_ids。
        features = self.vlm.get_input_embeddings(
            input_ids,
            pixel_values=None,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mask=attention_mask,
        )
        inputs_embeds = features.inputs_embeds
        deepstack_visual_embeds = features.deepstack_visual_embeds
        visual_pos_masks = features.visual_pos_masks
        # get_input_embeddings 已把 3D position_ids 缓存到 language_model._position_ids。
        position_ids = self.language_model._position_ids

        # 2. 截断层循环：复刻 Qwen3VLModel.__call__，在第 self.layer 层停止，
        #    注入 deepstack visual embeds，不接 final norm（H3 读 raw hidden）。
        model = self.language_model.model
        h = inputs_embeds
        from mlx_lm.models.base import create_attention_mask

        cache = [None] * len(model.layers)
        mask = create_attention_mask(h, cache[0])
        for layer_idx, (layer, c) in enumerate(zip(model.layers, cache)):
            h = layer(h, mask, c, position_ids)
            if deepstack_visual_embeds is not None and layer_idx in range(
                len(deepstack_visual_embeds)
            ):
                h = self.language_model.model._deepstack_process(
                    h, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )
            if layer_idx == self.layer:
                out = h
                logger.info(
                    "minimax_h3 ref2va text_encoder: out shape=%s dtype=%s",
                    out.shape,
                    out.dtype,
                )
                return out
        # layer 越界已在 __init__ 拦截，理论不可达。
        return h


# #948: ddalcu/MiniMax-H3-FL2VA-MLX-Serve-4bit 把 text_encoder 存成预量化
# language_model-only checkpoint，且剥掉了 language_model. 前缀（keys 为
# model.* / visual.*），config.json 也不含 quantization_config。mlx-vlm
# load_model 期望 language_model.model.* / vision_tower.* + quantization config，
# 直接调会 1780 unmatched + 不量化 → 权重形状错配。须手动 remap key + 注入量化。
_DDALCU_QUANT = {"group_size": 64, "bits": 4, "mode": "affine"}


def _is_ddalcu_stripped(weights: dict) -> bool:
    # True = 剥前缀布局：无 language_model.* 但有 model.* 前缀。
    has_model = any(k.startswith("model.") for k in weights)
    has_lm = any(k.startswith("language_model.") for k in weights)
    return has_model and not has_lm


def _remap_ddalcu_keys(weights: dict) -> dict:
    # model.* -> language_model.model.*
    # visual.* -> vision_tower.*
    # 其余 key (lm_head 等) 原样保留，交给 qwen3_vl sanitize 处理。
    remapped = {}
    for key, value in weights.items():
        if key.startswith("model."):
            new_key = "language_model.model." + key[len("model.") :]
        elif key.startswith("visual."):
            new_key = "vision_tower." + key[len("visual.") :]
        else:
            new_key = key
        remapped[new_key] = value
    logger.info(
        "minimax_h3 #948 ddalcu key remap: %d keys (model.->language_model.model., "
        "visual.->vision_tower.)",
        len(remapped),
    )
    return remapped


def _detect_ddalcu(model_path: Path) -> bool:
    # 廉价探测：只读 safetensors key（不加载张量），判断是否 ddalcu 剥前缀布局。
    import glob

    from safetensors import safe_open

    wfs = [
        wf
        for wf in glob.glob(str(model_path / "*.safetensors"))
        if not wf.endswith("consolidated.safetensors")
    ]
    if not wfs:
        return False
    has_model = has_lm = False
    with safe_open(wfs[0], framework="np") as f:
        for k in f.keys():  # noqa: SIM118 - safe_open.keys() is not a dict
            if k.startswith("model."):
                has_model = True
            elif k.startswith("language_model."):
                has_lm = True
            if has_model and not has_lm:
                # 采到一个 model.* 且尚未见 language_model.* 即可判定（ddalcu 全无）。
                return True
            if has_lm:
                return False
    return has_model and not has_lm


def _load_vlm_remapped(model_path: Path, trust_remote_code: bool = True):
    # #948: 复刻 mlx_vlm.utils.load_model 关键路径，但在 build 前注入量化 config +
    # remap key，使 nn.quantize 的 class_predicate 能看到 <path>.scales 而正确量化。
    import glob

    from mlx_vlm.utils import (
        get_model_and_args,
        load_config,
        sanitize_weights,
        skip_multimodal_module,
        update_module_configs,
    )

    cfg = load_config(model_path)
    weight_files = [
        wf
        for wf in glob.glob(str(model_path / "*.safetensors"))
        if not wf.endswith("consolidated.safetensors")
    ]
    if not weight_files:
        raise FileNotFoundError(f"no .safetensors in {model_path} (#948 loader)")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))
    logger.info(
        "minimax_h3 #948 loader: %d weight files, %d raw keys",
        len(weight_files),
        len(weights),
    )

    ddalcu = _is_ddalcu_stripped(weights)
    if ddalcu:
        weights = _remap_ddalcu_keys(weights)
        # config 无 quantization_config -> 注入，否则 nn.quantize 不会跑，
        # 模型保持 nn.Linear 而 checkpoint 是 group-quantized -> 形状错配。
        cfg.setdefault("quantization", dict(_DDALCU_QUANT))
        # ddalcu 只存了 H3 实际读取的层（0..49，第 50 层 0-indexed=49），
        # config 仍声明 64 层 -> 模型多建 14 层无权重 -> load_weights strict 报
        # Missing。按 checkpoint 实际层裁 num_hidden_layers，模型与权重对齐。
        import re

        max_layer = -1
        for k in weights:
            m = re.match(r"language_model\.model\.layers\.(\d+)\.", k)
            if m:
                max_layer = max(max_layer, int(m.group(1)))
        if max_layer >= 0:
            cfg.setdefault("text_config", {})
            cfg["text_config"]["num_hidden_layers"] = max_layer + 1
            logger.info(
                "minimax_h3 #948 loader: ddalcu stripped to %d layers "
                "(H3 reads layer %d) -> num_hidden_layers=%d",
                max_layer + 1,
                H3_TEXT_ENCODER_LAYER,
                max_layer + 1,
            )
        logger.info(
            "minimax_h3 #948 loader: ddalcu layout detected, injected quant %s",
            _DDALCU_QUANT,
        )
    else:
        logger.info("minimax_h3 #948 loader: standard layout, no remap")

    model_class, _ = get_model_and_args(config=cfg)
    cfg.setdefault("text_config", cfg.pop("llm_config", {}))
    cfg.setdefault("vision_config", {})
    cfg.setdefault("audio_config", {})
    model_config = model_class.ModelConfig.from_dict(cfg)
    modules = ["text", "vision", "perceiver", "projector", "audio"]
    model_config = update_module_configs(model_config, model_class, cfg, modules)
    model = model_class.Model(model_config)

    weights = sanitize_weights(model, weights)
    if hasattr(model_class, "VisionModel"):
        weights = sanitize_weights(
            model_class.VisionModel, weights, model_config.vision_config
        )
    if hasattr(model_class, "LanguageModel"):
        weights = sanitize_weights(
            model_class.LanguageModel, weights, model_config.text_config
        )

    quantization = cfg.get("quantization")
    if quantization is not None:
        skip_vision = cfg.get("vision_config", {}).get("skip_vision", False)

        def get_class_predicate(p, m):
            if skip_multimodal_module(p) and skip_vision:
                return False
            if p in cfg["quantization"]:
                return cfg["quantization"][p]
            if not hasattr(m, "to_quantized"):
                return False
            if hasattr(m, "weight") and m.weight.size % 64 != 0:
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=get_class_predicate,
        )
        logger.info(
            "minimax_h3 #948 loader: nn.quantize applied (g%d b%d %s)",
            quantization["group_size"],
            quantization["bits"],
            quantization.get("mode", "affine"),
        )

    # strict=False: ddalcu 只存 H3 读取的层（0..49）+ 无 lm_head/final_norm
    # （H3 读 raw layer-49 hidden，不走 norm/lm_head）。这两类模型有但 checkpoint
    # 无的参数留未初始化，H3 forward 不触碰，fail-visible 在 encode 时才会暴露。
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    model.eval()
    logger.info(
        "minimax_h3 #948 loader: weights applied, model=%s", type(model).__name__
    )
    return model


def load_text_encoder(
    model_path, layer: int = H3_TEXT_ENCODER_LAYER, trust_remote_code: bool = True
):
    # 通过 mlx-vlm 加载 qwen3_vl language_model，包装成 MiniMaxH3TextEncoder。
    # model_path: Qwen3-VL 模型目录（~/.fusion-mlx/models/<h3>/text_encoder/）。
    # #948: ddalcu 4bit 剥前缀布局走 _load_vlm_remapped；标准布局走 load_model。
    from mlx_vlm.utils import load_config, load_model

    model_path = Path(model_path)
    cfg = load_config(model_path)
    text_model_type = cfg.get("text_config", {}).get(
        "model_type", cfg.get("model_type")
    )
    ddalcu = _detect_ddalcu(model_path)
    logger.info(
        "minimax_h3 text_encoder: loading qwen3_vl from %s (text_model_type=%s "
        "ddalcu=%s)",
        model_path,
        text_model_type,
        ddalcu,
    )
    if ddalcu:
        language_model = _load_vlm_remapped(model_path, trust_remote_code)
    else:
        language_model = load_model(
            model_path, lazy=True, trust_remote_code=trust_remote_code
        )
    # load_model 返回完整 VLM（vision_tower + language_model）。
    lm = getattr(language_model, "language_model", language_model)
    encoder = MiniMaxH3TextEncoder(lm, layer=layer)
    logger.info(
        "minimax_h3 text_encoder: loaded, wrapping language_model=%s", type(lm).__name__
    )
    return encoder


def load_multimodal_text_encoder(
    model_path, layer: int = H3_TEXT_ENCODER_LAYER, trust_remote_code: bool = True
):
    # ref2va 多模态文本编码器：加载完整 qwen3_vl VLM（保留 vision_tower），
    # 包装成 MiniMaxH3MultimodalTextEncoder。与 load_text_encoder 不同——后者
    # 丢弃 vision_tower 走纯文本路径；ref2va 参考视频必须经 vision_tower。
    # #948: ddalcu 4bit 剥前缀布局走 _load_vlm_remapped。
    from mlx_vlm.utils import load_config, load_model

    model_path = Path(model_path)
    cfg = load_config(model_path)
    text_model_type = cfg.get("text_config", {}).get(
        "model_type", cfg.get("model_type")
    )
    ddalcu = _detect_ddalcu(model_path)
    logger.info(
        "minimax_h3 ref2va text_encoder: loading qwen3_vl (with vision_tower) "
        "from %s (text_model_type=%s ddalcu=%s)",
        model_path,
        text_model_type,
        ddalcu,
    )
    if ddalcu:
        vlm = _load_vlm_remapped(model_path, trust_remote_code)
    else:
        vlm = load_model(model_path, lazy=True, trust_remote_code=trust_remote_code)
    encoder = MiniMaxH3MultimodalTextEncoder(vlm, layer=layer)
    logger.info(
        "minimax_h3 ref2va text_encoder: loaded, vlm=%s",
        type(vlm).__name__,
    )
    return encoder
