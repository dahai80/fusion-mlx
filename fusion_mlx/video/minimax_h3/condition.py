# SPDX-License-Identifier: Apache-2.0
# MiniMax H3 多模态 packed-sequence 组装（condition）。
#
# 上游权威缺失说明（UNVERIFIED，需真实模型校正）：
#   - diffusers `MiniMaxH3Pipeline` / `MiniMaxH3ModularPipeline`（packed 序列组装源码）
#     在未发布的 diffusers minimax-h3 分支，HF 仓库只含 VAE 源码 + 权重 + config，
#     PyPI 无 minimax-h3 包。本组装逻辑从 transformer `forward` 契约
#     （diffusers main `transformer_minimax_h3.py` docstring + 参数表）+ AR 文档推断。
#   - transformer docstring 明确：「The caller is responsible for building the
#     packed layout」。token_tags 0=video/1=text/2=audio，timestep = 去重噪声水平，
#     timestep_indices 每行指向其在 timestep 数组中的下标，position_ids 每行 (t,h,w)。
#   - t2va video-only（无音频、无条件帧）：text 行（tag=1，clean t=1）+ 目标 video 行
#     （tag=0，noisy t=1-sigma）。timestep=[1.0, t_video]：text→idx0，video→idx1。
#
# 约束（来自 docstring）：
#   - 无 padding，单一 attention document，无 mask。
#   - text 行 position (0,0,0)；video 行 = patchify 后的 (t,h,w) 网格。
#   - batch 轴纯复制：结构参数描述一个 layout，每个 batch item 共享。
import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

# 视频 latent 归一化常量（FL2VA/video_vae/config.json latents_mean/std，24 维）。
# 去噪前 (z - mean)/std，解码前还原。与 AR doc §2.4 一致。
_H3_VIDEO_LATENTS_MEAN = [
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127610969543,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933399173279,
    -0.04225143790245056,
    0.2604829967041922,
    0.2286409288864906,
    0.7056031823158264,
]
_H3_VIDEO_LATENTS_STD = [
    1.2223774194717407,
    1.2767263650894165,
    1.6831774711608887,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.9653137922286987,
    1.0569885969161987,
    0.8419489264483822,
    0.7729952931403834,
    1.8955937623977661,
    0.9468418359756487,
    0.7996809482574463,
    0.44988900423049927,
    0.7197399735450745,
    0.6936293244361877,
    2.961095094680676,
    2.7694199908523596,
    3.0496185825897217,
    2.1088054180145264,
    3.276226582119521,
    3.1627357006073,
    2.2816812992094947,
    2.6127843856811523,
]


def video_latents_mean_std(dtype=mx.float32):
    # 返回 (mean, std) (24,) 用于 latent 归一化/反归一化。
    mean = mx.array(_H3_VIDEO_LATENTS_MEAN, dtype=dtype).reshape(1, -1, 1, 1, 1)
    std = mx.array(_H3_VIDEO_LATENTS_STD, dtype=dtype).reshape(1, -1, 1, 1, 1)
    return mean, std


def normalize_latents(z):
    # (b,c,t,h,w) → (z - mean)/std，channel 维对齐。
    mean, std = video_latents_mean_std(z.dtype)
    return (z - mean) / std


def denormalize_latents(z):
    # (b,c,t,h,w) → z*std + mean。
    mean, std = video_latents_mean_std(z.dtype)
    return z * std + mean


def patchify_video_latents(z, patch_size=(1, 2, 2)):
    # 视频 latent (b, c=24, t, h, w) → token (b, n, c*pt*ph*pw)。
    # patch_size=(pt, ph, pw)。n = (t//pt)*(h//ph)*(w//pw)。
    # 行顺序：t 外、h 中、w 内（与 VAE ViT decoder _pack_tensors_3d 一致）。
    pt, ph, pw = patch_size
    b, c, t, h, w = z.shape
    if t % pt or h % ph or w % pw:
        raise ValueError(f"latent {z.shape} not divisible by patch {patch_size}")
    z = z.reshape(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
    # 行优先 (t,h,w)，每行展平 (c,pt,ph,pw)。
    z = z.transpose(0, 2, 4, 6, 1, 3, 5, 7)
    return z.reshape(b, (t // pt) * (h // ph) * (w // pw), c * pt * ph * pw)


def unpatchify_video_tokens(tokens, latent_shape, patch_size=(1, 2, 2)):
    # token (b, n, c*pt*ph*pw) → latent (b, c, t, h, w)。patchify 的逆。
    pt, ph, pw = patch_size
    b, c, t, h, w = latent_shape
    nt, nh, nw = t // pt, h // ph, w // pw
    tokens = tokens.reshape(b, nt, nh, nw, c, pt, ph, pw)
    tokens = tokens.transpose(0, 4, 1, 5, 2, 6, 3, 7)
    return tokens.reshape(b, c, t, h, w)


def video_position_grid(latent_shape, patch_size=(1, 2, 2)):
    # patchify 后 token 的 (t,h,w) 坐标，返回 (n, 3)。
    # 行顺序与 patchify_video_latents 一致（t 外 h 中 w 内）。
    pt, ph, pw = patch_size
    _b, _c, t, h, w = latent_shape
    nt, nh, nw = t // pt, h // ph, w // pw
    # 各轴独立坐标，通过 broadcast 拼到 (nt,nh,nw,3)。
    t_ids = mx.arange(nt, dtype=mx.float32).reshape(nt, 1, 1)
    h_ids = mx.arange(nh, dtype=mx.float32).reshape(1, nh, 1)
    w_ids = mx.arange(nw, dtype=mx.float32).reshape(1, 1, nw)
    t_b = mx.broadcast_to(t_ids, (nt, nh, nw))
    h_b = mx.broadcast_to(h_ids, (nt, nh, nw))
    w_b = mx.broadcast_to(w_ids, (nt, nh, nw))
    grid = mx.stack([t_b, h_b, w_b], axis=-1)  # (nt,nh,nw,3)
    return grid.reshape(nt * nh * nw, 3)


# token_tags 常量（对齐 transformer MINIMAX_H3_MODALITY_NUM：0=video,1=text,2=audio）。
TAG_VIDEO = 0
TAG_TEXT = 1
TAG_AUDIO = 2


def build_t2va_packed(
    video_latents,
    text_embeds,
    timestep_video,
):
    # t2va video-only packed-sequence 组装（UNVERIFIED，无上游参考）。
    #
    # video_latents: (b, 24, t, h, w) 已归一化 latent（含噪声）。
    # text_embeds: (b, n_text, 5120) text_encoder 第 50 层输出。
    # timestep_video: 标量 t∈[0,1]（1=clean，=1-sigma），当前 video 噪声水平。
    #
    # 返回 transformer __call__ 所需全部结构参数（batch 共享一个 layout）：
    #   hidden_states (b, n_video, patch_dim)
    #   audio_hidden_states (b, 0, audio_dim)   # video-only：空
    #   encoder_hidden_states (b, n_text, 5120)
    #   timestep (num_timesteps,)
    #   timestep_indices (seq,)
    #   token_tags (seq,)
    #   position_ids (seq, 3)
    #   video_indices (n_video,)
    #   audio_indices (0,)
    #   text_indices (n_text,)
    patch_size = (1, 2, 2)
    b, _c, _t, _h, _w = video_latents.shape
    video_tokens = patchify_video_latents(video_latents, patch_size)
    n_video = video_tokens.shape[1]
    n_text = text_embeds.shape[1]
    seq_len = n_text + n_video

    # 序列顺序：text 在前，video 在后（顺序仅由 *_indices 决定，不影响正确性）。
    text_indices = mx.arange(n_text, dtype=mx.int32)
    video_indices = mx.arange(n_text, n_text + n_video, dtype=mx.int32)
    audio_indices = mx.zeros((0,), dtype=mx.int32)

    # timestep：t2va 无 condition 行，官方 build_row_timesteps 给全序列（含 text）
    # 赋 video_timestep（before_denoise.py:1195 "text rows inherit the video timestep"）。
    # 旧实现把 text 钉在 1.0(clean) 致 AdaLN 错位、输出发暗（#602）。单一 video_timestep。
    timestep = mx.array([float(timestep_video)], dtype=mx.float32)
    timestep_indices = mx.zeros((seq_len,), dtype=mx.int32)
    token_tags = mx.concatenate(
        [
            mx.full((n_text,), TAG_TEXT, dtype=mx.int32),
            mx.full((n_video,), TAG_VIDEO, dtype=mx.int32),
        ]
    )

    # position_ids：text 全 (0,0,0)；video = patch 网格 (t,h,w)。
    text_pos = mx.zeros((n_text, 3), dtype=mx.float32)
    video_pos = video_position_grid(video_latents.shape, patch_size)
    position_ids = mx.concatenate([text_pos, video_pos], axis=0)

    # audio 空（video-only）。
    audio_hidden_states = mx.zeros((b, 0, 32), dtype=text_embeds.dtype)

    logger.info(
        "h3 t2va packed: seq=%d (text=%d video=%d) timestep=%s t_video=%.4f",
        seq_len,
        n_text,
        n_video,
        [float(x) for x in timestep],
        float(timestep_video),
    )
    return {
        "hidden_states": video_tokens,
        "audio_hidden_states": audio_hidden_states,
        "encoder_hidden_states": text_embeds,
        "timestep": timestep,
        "timestep_indices": timestep_indices,
        "token_tags": token_tags,
        "position_ids": position_ids,
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "text_indices": text_indices,
        "latent_shape": video_latents.shape,
    }


__all__ = [
    "normalize_latents",
    "denormalize_latents",
    "video_latents_mean_std",
    "patchify_video_latents",
    "unpatchify_video_tokens",
    "video_position_grid",
    "build_t2va_packed",
    "TAG_VIDEO",
    "TAG_TEXT",
    "TAG_AUDIO",
]
