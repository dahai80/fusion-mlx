# SPDX-License-Identifier: Apache-2.0
# MiniMax H3 多模态 packed-sequence 组装（condition）。
#
# 权威对照：diffusers MiniMaxH3 FL2VA pipeline（before_denoise.py 已发布）。
#   - transformer docstring 明确：「The caller is responsible for building the
#     packed layout」。token_tags 0=video/1=text/2=audio，timestep = 去重噪声水平，
#     timestep_indices 每行指向其在 timestep 数组中的下标，position_ids 每行 (t,h,w)。
#   - t2va：text 行（tag=1）+ 目标 video 行（tag=0，noisy）。
#   - t2va-av：text + audio + video（build_row_timesteps 三类噪声水平）。
#   - fl2va：text + keyframe 条件行 + video（i2va/l2va）。条件行 tag=0，
#     timestep 钉 max(video_t, keyframe_noise_aug)，每步不变（rides through）。
#     行序 [text | keyframe conditions | (audio) | target video]。
#
# 约束（来自 docstring + before_denoise.py）：
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


# RoPE position 常量（对照 diffusers before_denoise.py）。
# 空间轴 aspect-normalized 到固定 [0,32)，分辨率无关。
# 时间轴非均匀：(5/3)*(1,4,4,4,4) 重复，从 origin=n_text 起算。
# 旧实现用裸 arange(n) 致空间范围随分辨率线性增长 → DiT 行为分辨率相关
# → 16x16 latent 解码偏暗(YAVG 9.9) vs 32x32 偏亮(146)，同 seed（#605）。
_ROPE_SPATIAL_SCALE = 32.0
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)


def _spatial_position_grid(dim, patch, sqrt_area):
    # 单空间轴坐标：aspect-normalized linspace，范围 [0, 32)。
    # ratio = dim/sqrt_area，正方形 ratio=1 → 均匀 [0,32)；
    # 矩形短轴 ratio<1 居中收缩，长轴 ratio>1 居中拉伸。
    n = dim // patch
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    step = ratio / n
    idx = mx.arange(n, dtype=mx.float32)
    grid = (left + idx * step) * _ROPE_SPATIAL_SCALE
    return grid


def _temporal_position_grid(num_latent_frames, origin):
    # 非均匀时间坐标：spans = 5/3*(1,4,4,4,4) 重复，origin + 累积和（首帧 0 span）。
    n = num_latent_frames
    spans = mx.array(
        [_ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % 5] for i in range(n)],
        dtype=mx.float32,
    )
    cum = mx.cumsum(spans)
    # 首帧坐标=origin，后续 = origin + spans[:-1].cumsum()（去掉末尾 span）。
    grid = mx.zeros((n,), dtype=mx.float32)
    grid = mx.array([origin], dtype=mx.float32)
    if n > 1:
        prev_cum = cum[:-1]
        rest = origin + prev_cum
        grid = mx.concatenate([grid, rest])
    return grid


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


def video_position_grid(latent_shape, patch_size=(1, 2, 2), origin=0.0):
    # patchify 后 token 的 (t,h,w) 坐标，返回 (n, 3)。
    # 行顺序与 patchify_video_latents 一致（t 外 h 中 w 内）。
    # 对照官方 before_denoise.py：
    #   空间 h/w = _spatial_position_grid(dim, patch, sqrt_area)（aspect-normalized [0,32)）。
    #   时间 t = _temporal_position_grid(nt, origin)（非均匀 5/3*(1,4,4,4,4) 从 origin 起算）。
    pt, ph, pw = patch_size
    _b, _c, t, h, w = latent_shape
    nt, nh, nw = t // pt, h // ph, w // pw
    # sqrt_area 用 patch 前的 latent 尺寸（对照 before_denoise.py
    # _frame_position_grid: sqrt_area=sqrt(latent_height*latent_width)）。
    sqrt_area = float(h * w) ** 0.5
    h_grid = _spatial_position_grid(h, ph, sqrt_area)
    w_grid = _spatial_position_grid(w, pw, sqrt_area)
    t_grid = _temporal_position_grid(nt, float(origin))
    # broadcast 拼到 (nt,nh,nw,3)，行顺序 t 外 h 中 w 内。
    t_b = mx.broadcast_to(t_grid.reshape(nt, 1, 1), (nt, nh, nw))
    h_b = mx.broadcast_to(h_grid.reshape(1, nh, 1), (nt, nh, nw))
    w_b = mx.broadcast_to(w_grid.reshape(1, 1, nw), (nt, nh, nw))
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

    # position_ids（对照 before_denoise.py:318-356）：
    #   text 行 time = arange(n_text)（非零），h/w = 0。
    #   video 行 time = _temporal_position_grid(nt, origin=n_text)，
    #   h/w = aspect-normalized [0,32) 空间网格。
    text_time = mx.arange(n_text, dtype=mx.float32)
    text_zero = mx.zeros((n_text,), dtype=mx.float32)
    text_pos = mx.stack([text_time, text_zero, text_zero], axis=-1)  # (n_text,3)
    video_pos = video_position_grid(
        video_latents.shape, patch_size, origin=float(n_text)
    )
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


def build_t2va_av_packed(
    video_latents,
    audio_latents,
    text_embeds,
    timestep_video,
    timestep_audio,
):
    # t2va joint audio+video packed-sequence 组装（UNVERIFIED，上游 diffusers 管线源码未发布）。
    #
    # video_latents: (b, 24, t, h, w) 已归一化 latent（含噪声）。
    # audio_latents: (b, T_audio, 32) 已归一化 audio latent（含噪声，通道 last）。
    # text_embeds: (b, n_text, 5120)。
    # timestep_video / timestep_audio: 标量 t∈[0,1]，video/audio 各自噪声水平（不同 shift）。
    #
    # 序列：text + video + audio（顺序仅由 *_indices 决定）。
    # timestep = [video_t, audio_t]（2 个去重噪声水平）。
    #   text/video → idx0，audio → idx1。
    #   adaln_indices = timestep_indices * 3 + token_tags。
    # audio 行：无 patchify，n_audio = T_audio，每行 (b, 1, 32)。
    #   audio patch_proj（32→5376）在 transformer 内，这里喂原始 32 维。
    # position_ids：
    #   text time=arange(n_text)，h/w=0。
    #   video time=_temporal_position_grid(nt, origin=n_text)，h/w 空间网格。
    #   audio time=arange(T_audio) + (n_text+n_video)，h/w=0（连续时间轴，无空间）。
    patch_size = (1, 2, 2)
    b, _c, _t, _h, _w = video_latents.shape
    video_tokens = patchify_video_latents(video_latents, patch_size)
    n_video = video_tokens.shape[1]
    n_text = text_embeds.shape[1]
    n_audio = audio_latents.shape[1]
    seq_len = n_text + n_video + n_audio

    text_indices = mx.arange(n_text, dtype=mx.int32)
    video_indices = mx.arange(n_text, n_text + n_video, dtype=mx.int32)
    audio_indices = mx.arange(n_text + n_video, seq_len, dtype=mx.int32)

    # timestep：2 水平。text 继承 video_t（同 t2va video-only，#602 fix）。
    timestep = mx.array(
        [float(timestep_video), float(timestep_audio)], dtype=mx.float32
    )
    timestep_indices = mx.concatenate(
        [
            mx.zeros((n_text + n_video,), dtype=mx.int32),  # text+video → idx0
            mx.ones((n_audio,), dtype=mx.int32),  # audio → idx1
        ]
    )
    token_tags = mx.concatenate(
        [
            mx.full((n_text,), TAG_TEXT, dtype=mx.int32),
            mx.full((n_video,), TAG_VIDEO, dtype=mx.int32),
            mx.full((n_audio,), TAG_AUDIO, dtype=mx.int32),
        ]
    )

    # position_ids。
    text_time = mx.arange(n_text, dtype=mx.float32)
    text_zero = mx.zeros((n_text,), dtype=mx.float32)
    text_pos = mx.stack([text_time, text_zero, text_zero], axis=-1)  # (n_text,3)
    video_pos = video_position_grid(
        video_latents.shape, patch_size, origin=float(n_text)
    )
    # audio：连续时间轴，从 video 末尾续。无空间（h/w=0）。
    audio_origin = float(n_text + n_video)
    audio_time = mx.arange(n_audio, dtype=mx.float32) + audio_origin
    audio_zero = mx.zeros((n_audio,), dtype=mx.float32)
    audio_pos = mx.stack([audio_time, audio_zero, audio_zero], axis=-1)  # (n_audio,3)
    position_ids = mx.concatenate([text_pos, video_pos, audio_pos], axis=0)

    audio_hidden_states = audio_latents.astype(text_embeds.dtype)  # (b, n_audio, 32)

    logger.info(
        "h3 t2va-av packed: seq=%d (text=%d video=%d audio=%d) t_video=%.4f t_audio=%.4f",
        seq_len,
        n_text,
        n_video,
        n_audio,
        float(timestep_video),
        float(timestep_audio),
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
        "audio_shape": audio_latents.shape,
    }


_H3_KEYFRAME_NOISE_AUG = 0.999


def _last_anchor_time(num_latent_frames, num_text_tokens):
    # 'last' anchor 的 rotary time：n_text + spans.sum() - _ROPE_FRAME_RESCALE。
    # spans = 5/3*(1,4,4,4,4) 重复，pairwise sum（对照 before_denoise.py:333-336）。
    spans = [
        _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % len(_ROPE_FRAMES_PER_LATENT)]
        for i in range(num_latent_frames)
    ]
    return float(num_text_tokens) + float(sum(spans)) - _ROPE_FRAME_RESCALE


def build_fl2va_packed(
    video_latents,
    condition_latents,
    text_embeds,
    timestep_video,
    keyframe_anchors,
    keyframe_noise_aug=_H3_KEYFRAME_NOISE_AUG,
):
    # fl2va packed-sequence 组装（i2va/l2va/fl2va 场景连续性）。
    # 对照 diffusers before_denoise.py:268 build_packed_sequence + L940-964 条件编码。
    #
    # video_latents: (b, 24, t, h, w) 已归一化目标 latent（含噪声，生成行）。
    # condition_latents: (b, 24, k, h, w) 已归一化条件 latent（keyframe_anchors 数量帧），
    #   调用前已 scale_noise(keyframe_noise_aug) 加噪（generate_fl2va_video 负责）。
    # text_embeds: (b, n_text, 5120)。
    # timestep_video: 标量 t∈[0,1]，生成行当前噪声水平。
    # keyframe_anchors: tuple[str,...]，每条件帧 'first'/'last'，packed 顺序。
    # keyframe_noise_aug: 条件行钉定噪声水平（官方 0.999）。
    #
    # 行序 [text | keyframe conditions | target video]（无音频：fl2va 视频路径）。
    # 条件行 tag=TAG_VIDEO（0），timestep = max(video_t, keyframe_noise_aug)，
    # 每步不变（denoise loop 只写生成行，条件行 rides through，L970-1009）。
    # 条件行 position：'first'→n_text；'last'→n_text+spans.sum()-rescale；空间=target 同 grid。
    patch_size = (1, 2, 2)
    pt, ph, pw = patch_size
    b, _c, nt, h, w = video_latents.shape
    _, _, k_cond, h_c, w_c = condition_latents.shape

    if k_cond != len(keyframe_anchors):
        raise ValueError(
            f"condition_latents has {k_cond} frames but keyframe_anchors has "
            f"{len(keyframe_anchors)} entries; they must agree."
        )
    rows_per_frame = (h // ph) * (w // pw)
    if (h_c, w_c) != (h, w):
        raise ValueError(
            f"condition canvas ({h_c}x{w_c}) disagrees with target ({h}x{w}); "
            f"encode the keyframe at the target latent size."
        )

    video_tokens = patchify_video_latents(video_latents, patch_size)  # (b, n_gen, dim)
    condition_tokens = patchify_video_latents(condition_latents, patch_size)  # (b, n_cond, dim)
    n_gen = video_tokens.shape[1]
    n_cond = condition_tokens.shape[1]
    n_text = text_embeds.shape[1]
    seq_len = n_text + n_cond + n_gen

    # hidden_states = [condition | generated]（FL2VAPrepareLatentsStep cat 顺序）。
    hidden_states = mx.concatenate([condition_tokens, video_tokens], axis=1)

    # 行索引：text 0..n_text-1，video(含条件) n_text..seq-1。
    text_indices = mx.arange(n_text, dtype=mx.int32)
    video_indices = mx.arange(n_text, seq_len, dtype=mx.int32)
    audio_indices = mx.zeros((0,), dtype=mx.int32)

    # timestep：生成行 video_t，条件行 max(video_t, keyframe_noise_aug)（钉定）。
    # 去重：若 video_t == cond_t → 单一水平（生成行 noisier 时）。
    gen_t = float(timestep_video)
    cond_t = max(gen_t, float(keyframe_noise_aug))
    unique_ts = sorted({gen_t, cond_t})
    timestep = mx.array(unique_ts, dtype=mx.float32)
    gen_idx = unique_ts.index(gen_t)
    cond_idx = unique_ts.index(cond_t)
    # timestep_indices：text → gen_idx（继承 video_t，#602），条件行 → cond_idx，生成行 → gen_idx。
    timestep_indices = mx.concatenate(
        [
            mx.full((n_text,), gen_idx, dtype=mx.int32),
            mx.full((n_cond,), cond_idx, dtype=mx.int32),
            mx.full((n_gen,), gen_idx, dtype=mx.int32),
        ]
    )
    # token_tags：text=1，条件+生成 video=0。
    token_tags = mx.concatenate(
        [
            mx.full((n_text,), TAG_TEXT, dtype=mx.int32),
            mx.full((n_cond + n_gen,), TAG_VIDEO, dtype=mx.int32),
        ]
    )

    # position_ids：text time=arange(n_text)，h/w=0。
    text_time = mx.arange(n_text, dtype=mx.float32)
    text_zero = mx.zeros((n_text,), dtype=mx.float32)
    text_pos = mx.stack([text_time, text_zero, text_zero], axis=-1)

    # 条件行 position：每 anchor 块 anchor_time + target 同 frame_grid（空间）。
    # 单帧空间网格（t=1），time 列覆盖为 anchor_time。
    single_frame_shape = (b, _c, 1, h, w)
    frame_spatial = video_position_grid(single_frame_shape, patch_size, origin=0.0)
    # (rows_per_frame, 3)，time 列全 0，空间 h/w 即 target 同 grid。
    cond_rows_pos = []
    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(n_text)
        elif anchor == "last":
            anchor_time = _last_anchor_time(nt, n_text)
        else:
            raise ValueError(
                f"A keyframe anchor must be 'first' or 'last', got {anchor!r}."
            )
        times = mx.full((rows_per_frame,), anchor_time, dtype=mx.float32)
        block = mx.stack([times, frame_spatial[:, 1], frame_spatial[:, 2]], axis=-1)
        cond_rows_pos.append(block)
    cond_pos = mx.concatenate(cond_rows_pos, axis=0)  # (n_cond, 3)

    # 生成行 position：origin=n_text 的时间网格 + 空间网格。
    gen_pos = video_position_grid(video_latents.shape, patch_size, origin=float(n_text))

    position_ids = mx.concatenate([text_pos, cond_pos, gen_pos], axis=0)

    audio_hidden_states = mx.zeros((b, 0, 32), dtype=text_embeds.dtype)

    logger.info(
        "h3 fl2va packed: seq=%d (text=%d cond=%d gen=%d) anchors=%s t_video=%.4f cond_t=%.4f",
        seq_len,
        n_text,
        n_cond,
        n_gen,
        list(keyframe_anchors),
        float(timestep_video),
        cond_t,
    )
    return {
        "hidden_states": hidden_states,
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
        "num_condition_rows": n_cond,
    }


__all__ = [
    "normalize_latents",
    "denormalize_latents",
    "video_latents_mean_std",
    "patchify_video_latents",
    "unpatchify_video_tokens",
    "video_position_grid",
    "build_t2va_packed",
    "build_t2va_av_packed",
    "build_fl2va_packed",
    "audio_latent_steps",
    "TAG_VIDEO",
    "TAG_TEXT",
    "TAG_AUDIO",
]


# Audio latent 步数 hop_length = prod(encoder_rates) = 2*4*4*5*5 = 800（metadata.json）。
# T_audio = ceil(num_frames/fps * sample_rate / hop_length)。
_H3_AUDIO_HOP_LENGTH = 800
_H3_AUDIO_SAMPLE_RATE = 32000


def audio_latent_steps(
    num_frames, fps, sample_rate=_H3_AUDIO_SAMPLE_RATE, hop_length=_H3_AUDIO_HOP_LENGTH
):
    # 视频时长 → audio latent 步数。97f@24fps → 4.04s → 161 步。
    import math

    duration = num_frames / float(fps)
    steps = math.ceil(duration * sample_rate / hop_length)
    return steps
