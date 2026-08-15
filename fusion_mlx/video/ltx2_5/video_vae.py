# SPDX-License-Identifier: Apache-2.0
# LTX-2.5 Video VAE adapter (P1, verified against real conv-VAE weights).
#
# 2.5 仓有两个 video VAE 变体（实测键树 2026-08-15）：
#   1. vae/ltx-2.5-video-vae-bf16.safetensors: transformer 式 "det" decoder
#      (t_embedder/shared_adaln/diff_blocks/det_stages/linear upsamples)，
#      与 ltx2 LTX2VideoDecoder (conv-UNet) 不同架构 - 待独立移植。
#   2. vae/ltx-2.5-video-vae-conv-bf16.safetensors: conv-UNet 族，键树与
#      ltx2 兼容 -> 本文件用它（2.5 结构落地路径）。
#
# 实测 conv 变体键树：encoder 84 键 / decoder 84 键 + 顶层
# per_channel_statistics.{mean,std}-of-means。encoder 键剥 'encoder.' 前缀后
# 与 MLX VideoEncoder 模块树直接同名（CausalConv3d.conv=nn.Conv3d）；差异仅
# down_blocks.4 为 4 个 res_blocks（ltx2 默认 6）。decoder 剥 'decoder.' 后
# 需经 ltx2 sanitize（'vae.decoder.' 前缀 + .conv.weight->.conv.conv.weight
# + 5D 转置）。块推断必须读 PyTorch 布局 shape[1] 为 in_channels（ltx2
# _infer_blocks 读 shape[-1]=kernel W=3，对 PyTorch 原始权重是错的）。
from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from ..ltx2.config import VideoEncoderModelConfig
from ..ltx2.video_vae import VideoEncoder
from ..ltx2.video_vae.decoder import LTX2VideoDecoder

logger = logging.getLogger(__name__)

# 2.5 conv VAE encoder 块结构：与 ltx2 默认差异为 down_blocks.4 层数 6->4，
# down_blocks.7 compress_all_res multiplier 2->1（实测 conv (128,1024)：通道
# 保持 1024，conv_out=1024/8=128；db8 res 仍 1024）。
LTX2_5_ENCODER_BLOCKS = [
    ("res_x", {"num_layers": 4}),
    ("compress_space_res", {"multiplier": 2}),
    ("res_x", {"num_layers": 6}),
    ("compress_time_res", {"multiplier": 2}),
    ("res_x", {"num_layers": 4}),
    ("compress_all_res", {"multiplier": 2}),
    ("res_x", {"num_layers": 2}),
    ("compress_all_res", {"multiplier": 1}),
    ("res_x", {"num_layers": 2}),
]

_STATS_MEAN_KEY = "per_channel_statistics.mean-of-means"
_STATS_STD_KEY = "per_channel_statistics.std-of-means"


def ltx2_5_encoder_config() -> VideoEncoderModelConfig:
    # 2.5 conv VAE encoder 配置（实测 down_blocks.4 = 4 res_blocks）。
    return VideoEncoderModelConfig(
        encoder_blocks=[list(b) for b in LTX2_5_ENCODER_BLOCKS]
    )


def _split_vae_weights(
    single_file: Path,
) -> tuple[dict[str, mx.array], dict[str, mx.array], dict[str, mx.array]]:
    # 单文件拆 encoder./decoder./顶层 stats，剥去前缀。
    weights = mx.load(str(single_file))
    enc: dict[str, mx.array] = {}
    dec: dict[str, mx.array] = {}
    stats: dict[str, mx.array] = {}
    other = 0
    for k, v in weights.items():
        if k.startswith("encoder."):
            enc[k[len("encoder.") :]] = v
        elif k.startswith("decoder."):
            dec[k[len("decoder.") :]] = v
        elif k in (_STATS_MEAN_KEY, _STATS_STD_KEY):
            stats[k] = v
        else:
            other += 1
    if other:
        logger.warning(
            "ltx2_5 VAE split: %d unrecognized keys dropped (first 10: %s)",
            other,
            sorted(
                k
                for k in weights
                if not k.startswith(("encoder.", "decoder."))
                and k not in (_STATS_MEAN_KEY, _STATS_STD_KEY)
            )[:10],
        )
    logger.info(
        "ltx2_5 VAE split: %s total=%d encoder=%d decoder=%d stats=%d other=%d",
        single_file.name,
        len(weights),
        len(enc),
        len(dec),
        len(stats),
        other,
    )
    return enc, dec, stats


def _audit_and_load(model, sanitized: dict[str, mx.array], name: str, strict: bool):
    # 键树审计 + 加载。unmatched/missing 全量打出（AR §4.2 静默零初始化防线）。
    model_params = dict(tree_flatten(model.parameters()))
    model_keys = set(model_params.keys())
    sanitized_keys = set(sanitized.keys())
    unmatched = sorted(sanitized_keys - model_keys)
    missing = sorted(model_keys - sanitized_keys)
    logger.info(
        "ltx2_5 %s audit: weights=%d model_params=%d unmatched=%d missing=%d",
        name,
        len(sanitized_keys),
        len(model_keys),
        len(unmatched),
        len(missing),
    )
    if unmatched:
        logger.warning("ltx2_5 %s unmatched (first 30): %s", name, unmatched[:30])
    if missing:
        logger.warning("ltx2_5 %s missing (first 30): %s", name, missing[:30])
    model.load_weights(list(sanitized.items()), strict=strict)
    mx.eval(model.parameters())
    model.eval()
    if strict and (unmatched or missing):
        raise RuntimeError(
            f"ltx2_5 {name} key-tree mismatch: unmatched={len(unmatched)} "
            f"missing={len(missing)} (see warnings above)"
        )
    return model


def _infer_conv_decoder_blocks(dec: dict[str, mx.array]) -> list:
    # 2.5 conv decoder 块推断。与 ltx2 _infer_blocks 逻辑同构，但 in_channels
    # 读 PyTorch 5D 布局 shape[1]（ltx2 读 shape[-1] 对原始权重是 kernel W）。
    idxs = sorted(
        {
            int(k.split(".")[1])
            for k in dec
            if k.startswith("up_blocks.") and k.split(".")[1].isdigit()
        }
    )
    raw = []
    for idx in idxs:
        res_n = {
            int(k.split(".res_blocks.")[1].split(".")[0])
            for k in dec
            if k.startswith(f"up_blocks.{idx}.res_blocks.")
        }
        if res_n:
            ch = int(dec[f"up_blocks.{idx}.res_blocks.0.conv1.conv.weight"].shape[0])
            raw.append(("res", ch, max(res_n) + 1))
            continue
        ckey = f"up_blocks.{idx}.conv.conv.weight"
        if ckey in dec:
            v = dec[ckey]
            in_ch = int(v.shape[1]) if v.ndim == 5 else int(v.shape[1])
            raw.append(("d2s", in_ch, int(v.shape[0])))
    blocks = []
    strides = []
    for i, b in enumerate(raw):
        if b[0] == "res":
            blocks.append(b)
            continue
        in_ch, out_ch = b[1], b[2]
        next_ch = next((r[1] for r in raw[i + 1 :] if r[0] == "res"), in_ch // 2)
        reduction = max(1, in_ch // next_ch)
        mult = out_ch // next_ch if next_ch else 8
        stride = {8: (2, 2, 2), 4: (1, 2, 2), 2: (2, 1, 1)}.get(mult, (2, 2, 2))
        strides.append(stride)
        blocks.append(("d2s", in_ch, reduction, stride))
    mixed = len(set(strides)) > 1
    nonstd = any(b[0] == "d2s" and b[2] != 2 for b in blocks)
    residual = not mixed and not nonstd
    result = [
        ("d2s", b[1], b[2], b[3], residual) if b[0] == "d2s" else b for b in blocks
    ]
    logger.info("ltx2_5 conv-decoder blocks: %s (residual=%s)", result, residual)
    return result


def load_video_encoder(weights_path: str | Path) -> VideoEncoder:
    # 2.5 conv VAE -> MLX VideoEncoder。默认用 conv 变体。
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"LTX-2.5 video VAE weights not found: {weights_path}")
    logger.info("ltx2_5 load_video_encoder: %s", weights_path.name)
    enc, _dec, stats = _split_vae_weights(weights_path)
    if not enc:
        raise RuntimeError(f"no encoder.* keys in {weights_path.name}")
    model = VideoEncoder(ltx2_5_encoder_config())
    # ltx2 VideoEncoder.sanitize 吃 'vae.encoder.' 前缀 + 'vae.' 前缀 stats，
    # 负责 5D/4D conv 转置与键名映射。
    prefixed = {f"vae.encoder.{k}": v for k, v in enc.items()}
    prefixed.update({f"vae.{k}": v for k, v in stats.items()})
    sanitized = model.sanitize(prefixed)
    return _audit_and_load(model, sanitized, "video-encoder", strict=True)


def load_video_decoder(weights_path: str | Path) -> LTX2VideoDecoder:
    # 2.5 conv VAE -> MLX LTX2VideoDecoder（conv 变体；det 变体待移植）。
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"LTX-2.5 video VAE weights not found: {weights_path}")
    if "conv" not in weights_path.name:
        raise ValueError(
            f"LTX-2.5 main VAE ({weights_path.name}) uses a transformer 'det' "
            f"decoder not yet ported; use the -conv variant "
            f"(vae/ltx-2.5-video-vae-conv-bf16.safetensors)"
        )
    logger.info("ltx2_5 load_video_decoder: %s", weights_path.name)
    _enc, dec, stats = _split_vae_weights(weights_path)
    if not dec:
        raise RuntimeError(f"no decoder.* keys in {weights_path.name}")
    blocks = _infer_conv_decoder_blocks(dec)
    model = LTX2VideoDecoder(
        in_channels=128,
        out_channels=3,
        patch_size=4,
        timestep_conditioning=False,
        decoder_blocks=blocks,
    )
    # ltx2 decoder.sanitize 吃 'vae.decoder.' 前缀（含 .conv.weight->
    # .conv.conv.weight 重映射 + 5D 转置）+ 'vae.' 前缀 stats。
    prefixed = {f"vae.decoder.{k}": v for k, v in dec.items()}
    prefixed.update({f"vae.{k}": v for k, v in stats.items()})
    sanitized = model.sanitize(prefixed)
    return _audit_and_load(model, sanitized, "video-decoder", strict=True)


__all__ = [
    "VideoEncoder",
    "LTX2VideoDecoder",
    "load_video_encoder",
    "load_video_decoder",
    "ltx2_5_encoder_config",
    "_split_vae_weights",
    "_infer_conv_decoder_blocks",
]
