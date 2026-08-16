# SPDX-License-Identifier: Apache-2.0
# MiniMax H3 运行时量化（in-place，不落盘）。
#
# 设计动机：FL2VA 三组件合计 144GB > M5 Max 137G 物理内存，同时加载 swap
# thrash 致 Metal 前向 OOM。量化降低内存峰值，配合 generate_video 阶段化加载。
#
# 精度策略（针对生成模型，最小化输出影响）：
#   - TE（Qwen3-VL 33B）：4-bit group=64。只产 text_embeds 中间表示，经 DiT
#     多步去噪消化，对量化最不敏感；Qwen3-VL 4-bit 成熟。跳过 embed/norm。
#   - DiT（33B）：8-bit group=64。决定视频主体质量，8-bit 对 33B 几乎无损。
#     跳过 F32 小层（time/patch/rope/final output）与真 norm。
#   - VAE：不量化。仅 10GB 且直接输出像素，量化收益小、伪影风险大。
#
# 跳过层判定（class_predicate path 前缀）：
#   DiT 跳过：time_embedder / video_patch_proj / audio_patch_proj / rope /
#             final_layer.video_out / final_layer.audio_out / condition_proj
#             （这些是 F32 小层或输出投影，精度敏感且体量小）
#   真 norm（RMSNorm，非 Linear）天然不被 nn.quantize 触碰（无 to_quantized）。
import logging

import mlx.nn as nn

logger = logging.getLogger(__name__)

# DiT 中保持原精度（不量化）的 module path 前缀。
_DIT_KEEP_PREFIXES = (
    "time_embedder",
    "video_patch_proj",
    "audio_patch_proj",
    "rope",
    "final_layer.video_out",
    "final_layer.audio_out",
    "condition_proj",
)


def _dit_predicate(path, module):
    # DiT 量化判定：Linear 且不在 keep 前缀内才量化。
    # Embedding 一律跳过（DiT 无大 embedding，保精度）。
    if not hasattr(module, "to_quantized"):
        return False
    if any(path.startswith(p) for p in _DIT_KEEP_PREFIXES):
        return False
    if isinstance(module, nn.Embedding):
        return False
    return True


def quantize_dit(dit, bits=8, group_size=64):
    # DiT 运行时量化，in-place。返回 dit 本身（已量化）。
    before = _param_bytes(dit)
    nn.quantize(dit, group_size=group_size, bits=bits, class_predicate=_dit_predicate)
    after = _param_bytes(dit)
    logger.info(
        "h3 quantize: dit %d-bit g=%d  %.2fGB -> %.2fGB",
        bits,
        group_size,
        before / 1e9,
        after / 1e9,
    )
    return dit


def quantize_text_encoder(text_encoder, bits=4, group_size=64):
    # TE（Qwen3-VL language_model）运行时量化，in-place。
    # 跳过 embed_tokens 与 norm：embed 2.3% 但精度敏感，norm 极小。
    # 仅量化 Linear（ffn gate/up/down + attn q/k/v/o）。
    before = _param_bytes(text_encoder)
    nn.quantize(
        text_encoder,
        group_size=group_size,
        bits=bits,
        class_predicate=_te_predicate,
    )
    after = _param_bytes(text_encoder)
    logger.info(
        "h3 quantize: text_encoder %d-bit g=%d  %.2fGB -> %.2fGB",
        bits,
        group_size,
        before / 1e9,
        after / 1e9,
    )
    return text_encoder


# TE 中保持原精度（不量化）的 module path 前缀。
_TE_KEEP_PREFIXES = (
    "embed_tokens",
    "model.embed_tokens",
    "language_model.embed_tokens",
)


def _te_predicate(path, module):
    if not hasattr(module, "to_quantized"):
        return False
    if isinstance(module, nn.Embedding):
        return False
    if any(p in path for p in _TE_KEEP_PREFIXES):
        return False
    return True


def _param_bytes(model):
    # 统计模型参数总字节数（粗估，用于日志）。
    from mlx.utils import tree_flatten

    total = 0
    for _, v in tree_flatten(model.parameters()):
        total += v.size * v.itemsize if hasattr(v, "itemsize") else 0
    return total


__all__ = ["quantize_dit", "quantize_text_encoder"]
