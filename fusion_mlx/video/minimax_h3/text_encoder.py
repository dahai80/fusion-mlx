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


def load_text_encoder(
    model_path, layer: int = H3_TEXT_ENCODER_LAYER, trust_remote_code: bool = True
):
    # 通过 mlx-vlm 加载 qwen3_vl language_model，包装成 MiniMaxH3TextEncoder。
    # model_path: Qwen3-VL 模型目录（~/.fusion-mlx/models/<h3>/text_encoder/）。
    from mlx_vlm.utils import load_config, load_model

    model_path = Path(model_path)
    cfg = load_config(model_path)
    text_model_type = cfg.get("text_config", {}).get(
        "model_type", cfg.get("model_type")
    )
    logger.info(
        "minimax_h3 text_encoder: loading qwen3_vl from %s (text_model_type=%s)",
        model_path,
        text_model_type,
    )
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
    from mlx_vlm.utils import load_config, load_model

    model_path = Path(model_path)
    cfg = load_config(model_path)
    text_model_type = cfg.get("text_config", {}).get(
        "model_type", cfg.get("model_type")
    )
    logger.info(
        "minimax_h3 ref2va text_encoder: loading qwen3_vl (with vision_tower) "
        "from %s (text_model_type=%s)",
        model_path,
        text_model_type,
    )
    vlm = load_model(model_path, lazy=True, trust_remote_code=trust_remote_code)
    encoder = MiniMaxH3MultimodalTextEncoder(vlm, layer=layer)
    logger.info(
        "minimax_h3 ref2va text_encoder: loaded, vlm=%s",
        type(vlm).__name__,
    )
    return encoder
