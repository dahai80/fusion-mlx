# SPDX-License-Identifier: Apache-2.0
# MiniMax H3 t2va/fl2va 视频推理循环（generate）。
#
# 权威对照：diffusers MiniMaxH3 FL2VA pipeline（before_denoise.py 已发布）。
#   - t2va：text→video。build_t2va_packed → transformer → unpatchify → step。
#   - t2va-av：text→audio+video（joint）。双 scheduler（video shift=12 / audio shift=3）。
#   - fl2va：text→keyframe 条件 + video（i2va/l2va 场景连续性）。
#     build_fl2va_packed → transformer → 拆生成行 → step（条件行 rides through）。
#
# 流程：text_encoder(prompt)→(b,n_t,5120)；noise→normalize→patchify→packed；
# 每 step：build_*_packed(latents, text, t_video)→transformer→video_output；
# unpatchify→step；循环结束 denormalize→vae.decode→[0,1]→frames。
import logging
import os

import mlx.core as mx

from fusion_mlx.engines.video_backends._inpaint import apply_inpaint_mask

from .audio_vae.audio_latents import (
    denormalize_audio_latents,
    normalize_audio_latents,
)
from .condition import (
    audio_latent_steps,
    build_fl2va_packed,
    build_t2va_av_packed,
    build_t2va_packed,
    denormalize_latents,
    normalize_latents,
    unpatchify_video_tokens,
)
from .scheduler import MiniMaxH3Scheduler

logger = logging.getLogger(__name__)


def _clear_metal_cache():
    # 兼容 mlx 版本：新版本 mx.clear_cache，旧版本 mx.metal.clear_cache。
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()


def _latents_shape(num_frames, height, width, vae_ratio, vae_ratio_t, z_channels):
    # 视频 latent 形状：空间 /vae_ratio，时间 /(vae_ratio_t*4) 之外的 4 倍因子由
    # VAE encoder 处理；此处按 AR doc：t'=(num_frames-1)//vae_ratio_t+1 近似。
    # 实际 encode_base 给出真实 t'，这里仅用于随机噪声初始化的形状推断。
    t = max(1, (num_frames - 1) // vae_ratio_t + 1)
    h = height // vae_ratio
    w = width // vae_ratio
    return (1, z_channels, t, h, w)


def generate_t2va_video(
    *,
    dit,
    vae,
    text_embeds,
    num_frames,
    height,
    width,
    seed=None,
    num_inference_steps=20,
    guide_scale=5.0,
    z_channels=24,
    vae_ratio=16,
    vae_ratio_t=4,
    compute_dtype=mx.bfloat16,
):
    # t2va video-only 去噪（UNVERIFIED）。
    #
    # dit: 已加载 MiniMaxH3DiTModel。
    # vae: 已加载 MiniMaxH3VideoVAE。
    # text_embeds: (1, n_text, 5120) 已编码文本嵌入。
    # 返回：list[np.ndarray frames] (H,W,3) uint8。
    if seed is not None:
        mx.random.seed(int(seed))

    latent_shape = _latents_shape(
        num_frames, height, width, vae_ratio, vae_ratio_t, z_channels
    )
    logger.info(
        "h3 t2va generate: latent_shape=%s steps=%d guide=%.1f seed=%s",
        latent_shape,
        num_inference_steps,
        guide_scale,
        seed,
    )

    # CFG：video-only 用空串负条件（无 negative_prompt 路径则跳过 CFG）。
    # 本 video-only 推断实现先不做 CFG（guide_scale 仅记录），保持最小路径。
    noise = mx.random.normal(latent_shape, dtype=compute_dtype)
    latents = noise  # t=0（纯噪声，sigma=1）→ scheduler.scale_noise 在首步不调用。

    scheduler = MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps
    logger.info("h3 t2va generate: timesteps=%s", [float(t) for t in timesteps])

    for i, t in enumerate(timesteps):
        t_video = float(t)
        packed = build_t2va_packed(
            latents.astype(compute_dtype),
            text_embeds.astype(compute_dtype),
            t_video,
        )
        video_output, _audio_output = dit(
            packed["hidden_states"],
            packed["audio_hidden_states"],
            packed["encoder_hidden_states"],
            packed["timestep"],
            packed["timestep_indices"],
            packed["token_tags"],
            packed["position_ids"],
            packed["video_indices"],
            packed["audio_indices"],
            packed["text_indices"],
        )
        # transformer 输出 video token (1, n_video, patch_dim) → latent。
        model_output = unpatchify_video_tokens(
            video_output, packed["latent_shape"], (1, 2, 2)
        )
        latents = scheduler.step(model_output, t, latents)
        # 物化并释放计算图，避免多步累积 OOM（MLX lazy，需显式 eval）。
        mx.eval(latents)
        if i % 10 == 0 or i == len(timesteps) - 1:
            logger.info(
                "h3 t2va generate: step %d/%d t=%.4f", i, len(timesteps), t_video
            )

    # 去噪结束：反归一化 → VAE 解码 → [0,1]。
    latents = denormalize_latents(latents.astype(mx.float32))
    logger.info("h3 t2va generate: decoding latents shape=%s", latents.shape)
    decoded = vae.decode(latents)
    frames = _to_frames(decoded)
    logger.info("h3 t2va generate: done frames=%d", len(frames))
    return frames


def generate_t2va_av(
    *,
    dit,
    vae,
    audio_vae,
    text_embeds,
    num_frames,
    height,
    width,
    fps=24,
    seed=None,
    num_inference_steps=20,
    guide_scale=5.0,
    z_channels=24,
    vae_ratio=16,
    vae_ratio_t=4,
    compute_dtype=mx.bfloat16,
):
    # t2va joint audio+video 去噪（UNVERIFIED，上游 pipeline 源码未发布）。
    #
    # dit: 已加载 MiniMaxH3DiTModel（audio 分支结构性已接）。
    # vae: 已加载 MiniMaxH3VideoVAE（video）。
    # audio_vae: 已加载 MiniMaxH3AudioVAE（audio decode-only）。
    # text_embeds: (1, n_text, 5120)。
    # 返回 (frames, waveform)：frames list[(H,W,3) uint8]；waveform (T_out,) float32 [-1,1]。
    #
    # 双 scheduler：video shift=12.0，audio shift=3.0，共享步数（独立 sigma 网格）。
    # 每 step：video_t=scheduler_video.timesteps[i]，audio_t=scheduler_audio.timesteps[i]，
    # build_t2va_av_packed → dit → (video_output, audio_output) → 各 scheduler.step。
    # 结束：video denormalize→vae.decode→frames；audio denormalize→audio_vae.decode→waveform。
    if seed is not None:
        mx.random.seed(int(seed))

    video_shape = _latents_shape(
        num_frames, height, width, vae_ratio, vae_ratio_t, z_channels
    )
    n_audio = audio_latent_steps(num_frames, fps)
    audio_shape = (1, n_audio, 32)
    logger.info(
        "h3 t2va-av generate: video_shape=%s audio_shape=%s steps=%d seed=%s",
        video_shape,
        audio_shape,
        num_inference_steps,
        seed,
    )

    # 噪声初始化：video (b,c,t,h,w)，audio (b,T_audio,32)。
    video_noise = mx.random.normal(video_shape, dtype=compute_dtype)
    audio_noise = mx.random.normal(audio_shape, dtype=compute_dtype)
    video_latents = normalize_latents(video_noise)
    audio_latents = normalize_audio_latents(audio_noise)

    scheduler_video = MiniMaxH3Scheduler(shift=12.0)
    scheduler_video.set_timesteps(num_inference_steps)
    scheduler_audio = MiniMaxH3Scheduler(shift=3.0)
    scheduler_audio.set_timesteps(num_inference_steps)
    v_timesteps = scheduler_video.timesteps
    a_timesteps = scheduler_audio.timesteps
    n_steps = min(int(v_timesteps.shape[0]), int(a_timesteps.shape[0]))
    logger.info(
        "h3 t2va-av generate: v_steps=%d a_steps=%d n=%d",
        v_timesteps.shape[0],
        a_timesteps.shape[0],
        n_steps,
    )

    for i in range(n_steps):
        t_video = float(v_timesteps[i])
        t_audio = float(a_timesteps[i])
        packed = build_t2va_av_packed(
            video_latents.astype(compute_dtype),
            audio_latents.astype(compute_dtype),
            text_embeds.astype(compute_dtype),
            t_video,
            t_audio,
        )
        video_output, audio_output = dit(
            packed["hidden_states"],
            packed["audio_hidden_states"],
            packed["encoder_hidden_states"],
            packed["timestep"],
            packed["timestep_indices"],
            packed["token_tags"],
            packed["position_ids"],
            packed["video_indices"],
            packed["audio_indices"],
            packed["text_indices"],
        )
        video_model = unpatchify_video_tokens(
            video_output, packed["latent_shape"], (1, 2, 2)
        )
        audio_model = audio_output  # (b, n_audio, 32) 每步 32 维，无需 unpatchify
        video_latents = scheduler_video.step(video_model, v_timesteps[i], video_latents)
        audio_latents = scheduler_audio.step(audio_model, a_timesteps[i], audio_latents)
        mx.eval(video_latents, audio_latents)
        if i % 10 == 0 or i == n_steps - 1:
            logger.info(
                "h3 t2va-av generate: step %d/%d t_v=%.4f t_a=%.4f",
                i,
                n_steps,
                t_video,
                t_audio,
            )

    # video 解码。
    video_latents = denormalize_latents(video_latents.astype(mx.float32))
    logger.info(
        "h3 t2va-av generate: decoding video latents shape=%s", video_latents.shape
    )
    decoded = vae.decode(video_latents)
    frames = _to_frames(decoded)

    # audio 解码。
    audio_latents = denormalize_audio_latents(audio_latents.astype(mx.float32))
    logger.info(
        "h3 t2va-av generate: decoding audio latents shape=%s", audio_latents.shape
    )
    audio_out = audio_vae.decode(audio_latents)  # (b, T_out, 1)
    mx.eval(audio_out)
    waveform = mx.array(audio_out[0, :, 0])  # (T_out,) mono
    mx.eval(waveform)
    logger.info(
        "h3 t2va-av generate: done frames=%d waveform=%s", len(frames), waveform.shape
    )
    return frames, waveform


_H3_KEYFRAME_NOISE_AUG = 0.999

# 条件帧 VAE encode seed（对照 diffusers keyframe_encode_seed=42，
# modular_pipeline.py:278-284：固定 seed 使同一 keyframe 总编到同一锚）。
_H3_KEYFRAME_ENCODE_SEED = 42


def encode_vae_condition(vae, pixels, encode_seed=_H3_KEYFRAME_ENCODE_SEED):
    # 条件帧 → VAE encode → 采样后归一化 latent（对照 diffusers
    # encoders.py:102-136 encode_vae_condition）。
    #   1. 像素 ImageNet 归一化：(x/255 - mean)/std（vae.NORM_*）。
    #   2. VAE encode → DiagonalGaussianDistribution.sample()，用固定 seed
    #      使同一 keyframe 总编到同一锚（不随请求 seed 漂移）。
    #   3. 采样 latent round 到 fp16 再回 fp32（~11 bit，匹配 reference）。
    # 旧实现直接喂 raw [0,1] 给 encode_base，漏掉 ImageNet 归一化 →
    # moments 尺度偏 → 条件行幅度错 → DiT 注意力被污染 → 输出花屏/灰（#657）。
    from .vae import DiagonalGaussianDistribution, _normalize_pixel

    x = _normalize_pixel(pixels.astype(mx.float32))  # ImageNet 归一化
    moments = vae.encode(x)
    # 固定 seed 采样：同一 keyframe 总编到同一锚（对照 diffusers
    # keyframe_encode_seed=42，独立于请求 seed 的 fresh generator）。
    mx.random.seed(int(encode_seed))
    z = DiagonalGaussianDistribution(moments).sample()
    z = z.astype(mx.float16).astype(mx.float32)  # fp16 round-trip
    return z


def _load_image_to_latent(image_path, vae, target_h, target_w, vae_ratio, z_channels):
    # 条件帧 → encode_vae_condition → latent (1, z, 1, h, w)。
    # 对照 diffusers before_denoise.py:949-954：image VisualVAE encode → moments →
    # DiagonalGaussianDistribution.sample() → latent z，patchify 前加噪。
    # 单图 (H,W,3) → (1,3,1,H,W)（5D）→ encode_vae_condition（含 ImageNet 归一化）。
    import numpy as np
    from PIL import Image

    logger.info(
        "h3 fl2va: loading condition image %s -> %dx%d latent",
        image_path,
        target_h,
        target_w,
    )
    img = Image.open(image_path).convert("RGB")
    if img.size != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.BILINEAR)
        logger.info("h3 fl2va: condition image resized to %dx%d", target_w, target_h)
    arr = np.array(img, dtype=np.float32) / 255.0  # (H,W,3) [0,1]
    arr = np.transpose(arr, (2, 0, 1))  # (3,H,W)
    x = mx.array(arr).reshape(1, 3, 1, target_h, target_w)  # (1,3,1,H,W)
    z = encode_vae_condition(vae, x)  # (1, z, 1, h, w) latent
    return z


def _encode_keyframe_latents(image_paths, vae, height, width, vae_ratio, z_channels):
    # 多锚点条件帧编码：每图 encode_base → (1,z,1,h,w) → 沿时间轴拼接 (1,z,k,h,w)。
    # 对照 before_denoise.py:940-964：每锚点一帧图，patchify 后沿行拼接。
    zs = [
        _load_image_to_latent(p, vae, height, width, vae_ratio, z_channels)
        for p in image_paths
    ]
    if len(zs) == 1:
        return zs[0]
    return mx.concatenate(zs, axis=2)  # (1, z, k, h, w)


def generate_fl2va_video(
    *,
    dit,
    vae,
    text_embeds,
    condition_image_paths,
    num_frames,
    height,
    width,
    seed=None,
    num_inference_steps=20,
    guide_scale=5.0,
    z_channels=24,
    vae_ratio=16,
    vae_ratio_t=4,
    compute_dtype=mx.bfloat16,
    keyframe_anchors=("first",),
    # #736 Surface C: inpaint-mask re-composite after each scheduler.step.
    # mask=1 -> reactive (keep denoised); mask=0 -> frozen (restore init).
    # None -> T2V passthrough, bit-identical to pre-#736 behavior.
    inpaint_mask=None,
    init_latent=None,
):
    # fl2va 去噪（i2va/l2va 场景连续性）。
    # 对照 diffusers before_denoise.py:268 build_packed_sequence + L940-1009 条件编码 +
    # FL2VAPrepareLatentsStep（条件行 rides through，denoise loop 只写生成行）。
    #
    # condition_image_paths: 每锚点一帧图路径，与 keyframe_anchors 顺序对齐。
    #   i2va：paths=[img], anchors=('first',)。
    #   l2va：paths=[img], anchors=('last',)。
    #   fl2va 联合：paths=[first_img, last_img], anchors=('first','last')。
    # 条件帧编码：VAE encode_base → latent → normalize → scale_noise(0.999) → patchify。
    # 每 step：build_fl2va_packed(gen_latents, cond_latents, text, t_video, anchors)
    #   → dit → video_output → 拆出生成行（跳过前 n_cond 条件行）→ unpatchify → step。
    # 条件行每步不变（不参与 scheduler.step，只作注意力上下文）。
    if isinstance(condition_image_paths, (str, os.PathLike)):
        condition_image_paths = [condition_image_paths]
    if len(condition_image_paths) != len(keyframe_anchors):
        raise ValueError(
            f"condition_image_paths ({len(condition_image_paths)}) must match "
            f"keyframe_anchors ({len(keyframe_anchors)})"
        )

    if seed is not None:
        mx.random.seed(int(seed))

    latent_shape = _latents_shape(
        num_frames, height, width, vae_ratio, vae_ratio_t, z_channels
    )
    logger.info(
        "h3 fl2va generate: latent_shape=%s steps=%d anchors=%s seed=%s",
        latent_shape,
        num_inference_steps,
        keyframe_anchors,
        seed,
    )

    # 条件帧：每锚点一图 → encode → normalize → 加噪钉 0.999（每步不变）。
    cond_z = _encode_keyframe_latents(
        condition_image_paths, vae, height, width, vae_ratio, z_channels
    )
    cond_latents = normalize_latents(cond_z.astype(mx.float32))  # (1,z,k,h,w)
    # 加噪到 keyframe_noise_aug=0.999。
    cond_noise = mx.random.normal(cond_latents.shape, dtype=mx.float32)
    scheduler_for_noise = MiniMaxH3Scheduler(shift=12.0)
    cond_latents_noised = scheduler_for_noise.scale_noise(
        cond_latents, _H3_KEYFRAME_NOISE_AUG, cond_noise
    )
    logger.info(
        "h3 fl2va: condition noised to t=%.3f shape=%s",
        _H3_KEYFRAME_NOISE_AUG,
        cond_latents_noised.shape,
    )

    # 生成行噪声初始化。
    noise = mx.random.normal(latent_shape, dtype=compute_dtype)
    gen_latents = noise

    scheduler = MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps
    logger.info("h3 fl2va generate: timesteps=%s", [float(t) for t in timesteps])

    patch_size = (1, 2, 2)

    logger.info(
        "minimax_h3 fl2va denoise: inpaint=%s steps=%d",
        inpaint_mask is not None,
        len(timesteps),
    )

    for i, t in enumerate(timesteps):
        t_video = float(t)
        packed = build_fl2va_packed(
            gen_latents.astype(compute_dtype),
            cond_latents_noised.astype(compute_dtype),
            text_embeds.astype(compute_dtype),
            t_video,
            keyframe_anchors=keyframe_anchors,
            keyframe_noise_aug=_H3_KEYFRAME_NOISE_AUG,
        )
        video_output, _audio_output = dit(
            packed["hidden_states"],
            packed["audio_hidden_states"],
            packed["encoder_hidden_states"],
            packed["timestep"],
            packed["timestep_indices"],
            packed["token_tags"],
            packed["position_ids"],
            packed["video_indices"],
            packed["audio_indices"],
            packed["text_indices"],
        )
        # video_output (1, n_cond+n_gen, dim)：拆出生成行（跳过前 n_cond）。
        n_cond = packed["num_condition_rows"]
        gen_output = video_output[:, n_cond:, :]
        model_output = unpatchify_video_tokens(
            gen_output, packed["latent_shape"], patch_size
        )
        gen_latents = scheduler.step(model_output, t, gen_latents)
        mx.eval(gen_latents)
        # #736 Surface C: frozen-region re-composite after each step.
        # DiT-agnostic, latent-space only; None -> passthrough.
        if inpaint_mask is not None and init_latent is not None:
            gen_latents = apply_inpaint_mask(gen_latents, init_latent, inpaint_mask)
            mx.eval(gen_latents)
        if i % 10 == 0 or i == len(timesteps) - 1:
            logger.info(
                "h3 fl2va generate: step %d/%d t=%.4f", i, len(timesteps), t_video
            )

    # 生成行去噪结束 → 反归一化 → VAE 解码。
    gen_latents = denormalize_latents(gen_latents.astype(mx.float32))
    logger.info("h3 fl2va generate: decoding latents shape=%s", gen_latents.shape)
    decoded = vae.decode(gen_latents)
    frames = _to_frames(decoded)
    logger.info("h3 fl2va generate: done frames=%d", len(frames))
    return frames


def _to_frames(decoded):
    # decoded (1, 3, t, h, w) float (可能 bfloat16) → list[(H,W,3) uint8]。
    # np.array 对 MLX bfloat16 会失败/极慢，先转 float32 并显式 eval 物化。
    #
    # VAE decoder 输出为 imagenet 归一化像素空间（mean/std 见 vae.NORM_*），
    # 官方 vae_processor.revert_tensor = transform_rev(x).clamp(0,1)，
    # 即先反归一化 x*std+mean 再 clamp。早期移植漏掉反归一化直接 clip，
    # 致 decoded DC 偏负时（如 1344×768 mean=-1.15）几乎全裁到 0 → 近黑。
    # 768×448 decoded mean=+0.81 偶然为正才"看起来正常"，实为同一缺陷。
    import numpy as np

    from .vae import _denormalize_pixel

    x = decoded[0:1].astype(mx.float32)
    x = _denormalize_pixel(x)  # (1,3,t,h,w) imagenet 反归一化 → [0,1] 像素
    x = x[0]
    x = mx.transpose(x, (1, 2, 3, 0))  # (t,h,w,3)
    mx.eval(x)
    x = mx.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(mx.uint8)
    mx.eval(x)
    arr = np.array(x)
    return [arr[i] for i in range(arr.shape[0])]


def _resolve_subdir(model_path, name):
    # 优先 model_path/<name>，否则回退 model_path（单目录布局）。
    # VAE 权重在真实布局里位于 <name>/source/（config.json source_path=source），
    # 当 <name>/ 无 safetensors 但 <name>/source/ 有时，落到 source 子目录。
    import glob
    import os

    sub = os.path.join(model_path, name)
    if not os.path.isdir(sub):
        return model_path
    if glob.glob(os.path.join(sub, "*.safetensors")):
        return sub
    source_sub = os.path.join(sub, "source")
    if os.path.isdir(source_sub) and glob.glob(
        os.path.join(source_sub, "*.safetensors")
    ):
        logger.info("h3: %s weights resolved to nested source/ subdir", name)
        return source_sub
    return sub


def _encode_prompt(text_encoder, tokenizer, prompt, max_length=256):
    # 用 Qwen3-VL tokenizer 编码 prompt（纯文本，无视觉输入）。
    # 返回 (1, seq, 5120) text_encoder 第 50 层输出。
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # 回退：直接编码裸 prompt（部分 tokenizer 无 chat template）。
        text = prompt
        logger.warning("h3: tokenizer chat_template unavailable, raw prompt")
    enc = tokenizer(
        text,
        return_tensors=None,
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    ids_list = enc["input_ids"]
    input_ids = mx.array([ids_list], dtype=mx.int32)
    am_list = enc.get("attention_mask", [1] * len(ids_list))
    attention_mask = mx.array([am_list], dtype=mx.int32)
    text_embeds = text_encoder(input_ids, attention_mask=attention_mask)
    logger.info(
        "h3 encode_prompt: ids=%s embeds=%s", input_ids.shape, text_embeds.shape
    )
    return text_embeds


def _load_ref2va_frames(reference_paths):
    # 加载参考视频/图像为 Qwen3VLVideoProcessor 期望的 list[(T, C, H, W) uint8]。
    # reference_paths: list[str]（每个是一条参考视频路径，mp4/帧图）。
    # imageio 抽帧得 (T,H,W,C)；转 (T,C,H,W) 供 processor._process_one。
    # 单图视为 1 帧 (1,C,H,W)。返回 list[np.ndarray]，每条一个 (T,C,H,W)。
    import numpy as np

    if not isinstance(reference_paths, (list, tuple)) or not reference_paths:
        raise ValueError(
            "h3 ref2va: reference_paths must be a non-empty list of paths "
            "(issue #688 step 2-3)"
        )
    videos = []
    for i, rp in enumerate(reference_paths):
        rp = os.fspath(rp)
        if rp.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            try:
                import imageio.v3 as iio

                frames = iio.imread(rp, index=None)  # (T, H, W, C) uint8
            except Exception as e:
                logger.error("h3 ref2va: read video %s failed: %s", rp, e)
                raise
            if frames.ndim != 4 or frames.shape[-1] not in (1, 3, 4):
                raise ValueError(
                    f"h3 ref2va: unexpected video shape {frames.shape} for {rp} (issue #688 step 2-3)"
                )
            video = np.transpose(frames, (0, 3, 1, 2))  # (T,H,W,C) -> (T,C,H,W)
        else:
            from PIL import Image

            img = Image.open(rp).convert("RGB")
            video = np.transpose(np.array(img), (2, 0, 1))[None, ...]  # (1,C,H,W)
        logger.info(
            "h3 ref2va: loaded reference %d/%d %s shape=%s",
            i + 1,
            len(reference_paths),
            rp,
            video.shape,
        )
        videos.append(video)
    return videos


def _encode_prompt_ref2va(
    text_encoder, processor, prompt, reference_paths, max_length=2048
):
    # ref2va 多模态编码：参考视频经 Qwen3-VL chat template 包成
    # <|vision_start|><|video_pad|><|vision_end|>，processor 展开占位符 →
    # input_ids + pixel_values_videos + video_grid_thw，多模态 TE 前向到第 50 层。
    # 返回 (1, seq, 5120) vision-conditioned text_embeds。
    frames_list = _load_ref2va_frames(reference_paths)
    # chat template：每条参考视频包 vision token。mlx-vlm processor 的 chat
    # template 已在 from_pretrained 加载；这里手搓多模态 user content 以确保
    # <|video_pad|> 占位符存在（processor 展开它）。
    video_token = "<|video_pad|>"
    content_parts = []
    for i in range(len(reference_paths)):
        content_parts.append({"type": "video", "video": video_token})
    content_parts.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content_parts}]
    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # 回退：拼裸 vision token + prompt（部分 processor chat template 形态不同）。
        vision_blocks = "".join(
            f"<|vision_start|>{video_token}<|vision_end|> Video {i + 1}: "
            for i in range(len(reference_paths))
        )
        text = vision_blocks + prompt
        logger.warning("h3 ref2va: processor chat_template failed, raw vision text")
    logger.info("h3 ref2va encode: prompt text len=%d", len(text))

    batch = processor(text=[text], videos=frames_list, return_tensors=None)
    input_ids = mx.array(batch["input_ids"], dtype=mx.int32)
    pixel_values_videos = batch.get("pixel_values_videos")
    if pixel_values_videos is None:
        # 部分版本 key 名为 pixel_values（视频统一走 vision_tower）。
        pixel_values_videos = batch.get("pixel_values")
    if pixel_values_videos is None:
        raise RuntimeError(
            "h3 ref2va: processor returned no pixel_values/pixel_values_videos — "
            "reference video preprocessing failed (issue #688 step 2-3)"
        )
    pixel_values_videos = mx.array(pixel_values_videos)
    video_grid_thw = batch.get("video_grid_thw")
    if video_grid_thw is None:
        raise RuntimeError(
            "h3 ref2va: processor returned no video_grid_thw (issue #688 step 2-3)"
        )
    video_grid_thw = mx.array(video_grid_thw)
    text_embeds = text_encoder(
        input_ids,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
    )
    logger.info(
        "h3 ref2va encode: ids=%s grid_thw=%s embeds=%s",
        input_ids.shape,
        video_grid_thw.shape,
        text_embeds.shape,
    )
    return text_embeds


def generate_video(
    *,
    model_path,
    prompt,
    num_frames=97,
    width=768,
    height=768,
    fps=24,
    seed=None,
    num_inference_steps=20,
    output_path=None,
    quantize="none",
    audio=False,
    image=None,
    last_frame_image=None,
    keyframe_anchors=None,
    reference_images=None,
    # #736 Surface B+C: ControlNet residual threading (B) and inpaint-mask
    # re-composite (C). All default None -> T2V pure-noise path, bit-identical
    # to pre-#736 behavior. controlnet_image fails visibly (no backend model).
    controlnet_image=None,
    controlnet_adapter=None,
    controlnet_latent=None,
    inpaint_mask=None,
    init_latent=None,
):
    # H3 t2va/fl2va 顶层编排。
    #
    # model_path: H3 模型根目录（含 transformer/ text_encoder/ video_vae/ audio_vae/ 子目录或单目录）。
    # 加载 DiT + VAE + text_encoder，编码 prompt，去噪，写 mp4。
    #
    # image: 条件帧图路径（i2va/l2va 场景连续性）。非 None → fl2va 去噪。
    #   keyframe_anchors 默认 ('first',)（i2va 首帧锚定）；传 ('last',) → l2va 末帧锚定。
    # audio: True → joint audio+video（t2va-av），audio_vae 解码音频 + ffmpeg 合流。
    #   fl2va 当前 video-only（音频连续性非本 PR 范围，audio 与 image 互斥）。
    #
    # audio: True → joint audio+video 去噪（generate_t2va_av），audio_vae 解码音频，
    #   ffmpeg 合流成单 MP4（A/V）；False → video-only 原路径（向后兼容）。
    #
    # quantize: 运行时量化策略（in-place，不落盘）：
    #   "none"     - 不量化（默认，bf16 原精度）。
    #   "te4"      - TE 4-bit（缓解 TE 67G 内存峰值）。
    #   "dit8"     - DiT 8-bit（缓解 DiT 66G）。
    #   "dit8_te4" - DiT 8-bit + TE 4-bit（最大压缩，官方尺度配置推荐）。
    import os
    import tempfile

    from .config import H3Config, H3VAEConfig
    from .text_encoder import load_text_encoder
    from .transformer import load_dit_from_pretrained
    from .vae import MiniMaxH3VideoVAE

    quantize = (quantize or "none").lower()
    do_te_q = quantize in ("te4", "dit8_te4")
    do_dit_q = quantize in ("dit8", "dit8_te4")

    # ref2va (reference-video-to-video) 分支：参考视频经 Qwen3-VL vision_tower
    # 编码成 vision-conditioned text_embeds，DiT 走纯 T2V 去噪（build_t2va_packed）。
    # issue #688 step 2-3：原 NotImplementedError gate 已替换为实现。
    is_ref2va = reference_images is not None
    if is_ref2va:
        logger.info(
            "h3 generate_video: ref2va branch (issue #688 step 2-3) references=%s",
            reference_images,
        )
        if not isinstance(reference_images, (list, tuple)) or not reference_images:
            raise ValueError(
                "h3 ref2va: reference_images must be a non-empty list of paths "
                "(issue #688 step 2-3)"
            )

    logger.info(
        "h3 generate_video: prompt='%s' frames=%d %dx%d fps=%d seed=%s steps=%d quantize=%s audio=%s ref2va=%s",
        prompt[:60],
        num_frames,
        width,
        height,
        fps,
        seed,
        num_inference_steps,
        quantize,
        audio,
        is_ref2va,
    )
    logger.info(
        "minimax_h3 denoise: inpaint=%s controlnet=%s",
        inpaint_mask is not None,
        controlnet_image is not None,
    )

    # #736 Surface B: ControlNet residual injection is NOT fabricatable for
    # minimax_h3 — the shared ControlNet adapter is Wan2-arch and no per-backend
    # ControlNet model exists. controlnet_image must fail visibly (Rule 12)
    # rather than silently degrade to T2V.
    if controlnet_image is not None:
        raise RuntimeError(
            "minimax_h3: ControlNet (Surface B) not available for this backend — "
            "no per-backend ControlNet model (see issue #736 follow-up). "
            "Refusing to silently degrade to T2V (#736)."
        )

    cfg = H3Config()
    if seed is not None:
        mx.random.seed(int(seed))

    # 阶段化加载：FL2VA/Ref2VA 总权重 ~144GB（TE 67G + DiT 66G + VAE 11G）超过
    # M5 Max 137G 物理内存，同时加载会 swap thrash 致 Metal 前向极慢。先加载 TE
    # 编码 prompt，物化 text_embeds 后释放 TE，再加载 DiT+VAE 去噪。text_embeds 仅几 MB。
    import gc

    te_path = _resolve_subdir(model_path, "text_encoder")
    if is_ref2va:
        # ref2va 需要完整 qwen3_vl VLM（保留 vision_tower）+ Qwen3VLProcessor。
        from .text_encoder import load_multimodal_text_encoder

        text_encoder = load_multimodal_text_encoder(te_path)
        if do_te_q:
            # vision_tower 量化不在本 PR 范围；ref2va 仅量化 language_model。
            from .quantize import quantize_text_encoder

            lm = getattr(text_encoder, "language_model", text_encoder)
            quantize_text_encoder(lm)
        # processor：Qwen3VLProcessor.from_pretrained（含 video_processor）。
        try:
            from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import (
                Qwen3VLProcessor,
            )

            processor = Qwen3VLProcessor.from_pretrained(te_path)
        except Exception as e:
            logger.error("h3 ref2va: processor load failed from %s: %s", te_path, e)
            raise
        text_embeds = _encode_prompt_ref2va(
            text_encoder, processor, prompt, reference_images
        )
    else:
        text_encoder = load_text_encoder(te_path)
        if do_te_q:
            from .quantize import quantize_text_encoder

            # TE encode 一次即释放，4-bit 对 text_embeds 影响最小。
            # 量化须在 encode 前（此时权重已物化），encode 后随 TE 一并释放。
            # MiniMaxH3TextEncoder.language_model = mlx-vlm Qwen3VLModel，量化其 Linear。
            lm = getattr(text_encoder, "language_model", text_encoder)
            quantize_text_encoder(lm)

        # tokenizer：优先 transformers AutoTokenizer（qwen3_vl chat template）。
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(te_path, local_files_only=True)
        except Exception as e:
            logger.error("h3: tokenizer load failed from %s: %s", te_path, e)
            raise

        text_embeds = _encode_prompt(text_encoder, tokenizer, prompt)
    # 物化 text_embeds 后释放 TE（~67G）与 tokenizer/processor。
    mx.eval(text_embeds)
    del text_encoder
    if not is_ref2va:
        del tokenizer
    else:
        del processor
    gc.collect()
    _clear_metal_cache()
    logger.info("h3: text_encoder released, loading DiT+VAE")

    dit_path = _resolve_subdir(model_path, "transformer")
    dit = load_dit_from_pretrained(dit_path, config=cfg)
    if do_dit_q:
        from .quantize import quantize_dit

        # DiT 8-bit：跳过 F32 小层与输出投影，adaLN/ffn/attn 大线性层量化。
        quantize_dit(dit, bits=8, group_size=64)

    vae_path = _resolve_subdir(model_path, "video_vae")
    vae = MiniMaxH3VideoVAE.from_pretrained(vae_path, config=H3VAEConfig())

    if output_path is None:
        tmp = tempfile.mkdtemp(prefix="fusion_h3_")
        output_path = os.path.join(tmp, "h3_output.mp4")

    if image is not None and audio:
        raise ValueError(
            "fl2va (image-conditioned) 与 joint audio 当前互斥；"
            "音频连续性非本 PR 范围。请二选一。"
        )

    # ref2va 互斥：参考视频已编码进 text_embeds（vision-conditioned），DiT 走纯
    # T2V 去噪。fl2va 关键帧（image/last_frame_image）与 joint audio 均假设纯文本
    # text_embeds，混用会语义错乱。ref2va 必须只走 generate_t2va_video。
    if is_ref2va and (image is not None or last_frame_image is not None):
        raise ValueError(
            "h3 ref2va: reference_images (ref2va) cannot combine with "
            "image/last_frame_image (fl2va keyframe) — different text_embeds "
            "conditioning (issue #688 step 2-3)."
        )
    if is_ref2va and audio:
        raise ValueError(
            "h3 ref2va: reference_images (ref2va) cannot combine with joint "
            "audio — t2va_av assumes pure-text text_embeds (issue #688 step 2-3)."
        )

    if image is not None or last_frame_image is not None:
        # fl2va：keyframe 条件帧去噪（i2va/l2va 场景连续性）。
        # 锚点与路径推导（遵循 base.py 约定：image=首帧 'first'，
        # last_frame_image=末帧 'last'）。显式 keyframe_anchors 覆盖时用它。
        if keyframe_anchors is not None:
            anchors = tuple(keyframe_anchors)
            paths = []
            if "first" in anchors and image is not None:
                paths.append(image)
            if "last" in anchors and last_frame_image is not None:
                paths.append(last_frame_image)
            if len(paths) != len(anchors):
                raise ValueError(
                    f"keyframe_anchors={anchors} 但缺失对应条件图 "
                    f"(image={image}, last_frame_image={last_frame_image})"
                )
        else:
            anchors = []
            paths = []
            if image is not None:
                anchors.append("first")
                paths.append(image)
            if last_frame_image is not None:
                anchors.append("last")
                paths.append(last_frame_image)
            anchors = tuple(anchors)
        logger.info("h3 generate_video: fl2va mode anchors=%s paths=%s", anchors, paths)
        frames = generate_fl2va_video(
            dit=dit,
            vae=vae,
            text_embeds=text_embeds,
            condition_image_paths=paths,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guide_scale=cfg.guide_scale,
            z_channels=cfg.latents_dim,
            vae_ratio=H3VAEConfig().vae_ratio,
            vae_ratio_t=H3VAEConfig().vae_ratio_t,
            keyframe_anchors=tuple(anchors),
            inpaint_mask=inpaint_mask,
            init_latent=init_latent,
        )
        _write_mp4(frames, output_path, fps)
    elif audio:
        # joint audio+video：额外加载 AudioVAE，去噪得 (frames, waveform)。
        from .audio_vae import MiniMaxH3AudioVAE

        audio_vae_path = _resolve_subdir(model_path, "audio_vae")
        audio_vae = MiniMaxH3AudioVAE.from_pretrained(audio_vae_path)

        frames, waveform = generate_t2va_av(
            dit=dit,
            vae=vae,
            audio_vae=audio_vae,
            text_embeds=text_embeds,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guide_scale=cfg.guide_scale,
            z_channels=cfg.latents_dim,
            vae_ratio=H3VAEConfig().vae_ratio,
            vae_ratio_t=H3VAEConfig().vae_ratio_t,
        )
        # 先写临时 video-only mp4 + wav，再 ffmpeg 合流成最终 MP4。
        tmp_dir = os.path.dirname(output_path)
        tmp_video = os.path.join(tmp_dir, "h3_video_only.mp4")
        tmp_wav = os.path.join(tmp_dir, "h3_audio.wav")
        _write_mp4(frames, tmp_video, fps)
        _save_audio(waveform, tmp_wav)
        _mux_av(tmp_video, tmp_wav, output_path)
        # 清理过程文件（只留最终 MP4 + 日志）。
        try:
            os.remove(tmp_video)
            os.remove(tmp_wav)
        except OSError as e:
            logger.warning("h3: cleanup temp av files failed: %s", e)
    else:
        frames = generate_t2va_video(
            dit=dit,
            vae=vae,
            text_embeds=text_embeds,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guide_scale=cfg.guide_scale,
            z_channels=cfg.latents_dim,
            vae_ratio=H3VAEConfig().vae_ratio,
            vae_ratio_t=H3VAEConfig().vae_ratio_t,
        )
        _write_mp4(frames, output_path, fps)

    logger.info("h3: output saved to %s", output_path)
    return output_path


def _write_mp4(frames, output_path, fps):
    # frames: list[np.ndarray (H,W,3) uint8] -> avc1 mp4。用 cv2 (与 ltx2_5 一致),
    # 避免 PyAV 依赖缺失导致 RuntimeError。
    import cv2
    import numpy as np

    frames_np = np.stack(frames)  # (T,H,W,3)
    h, w = frames_np.shape[1], frames_np.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    try:
        for frame in frames_np:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        out.release()
    logger.info(
        "h3: mp4 written %dx%d %d frames -> %s", w, h, len(frames_np), output_path
    )


def _save_audio(waveform, path, sample_rate=32000):
    # waveform (T,) float32 [-1,1] → mono PCM wav（与 ltx2 audio.save_audio 同模式）。
    import wave

    import numpy as np

    audio = np.clip(np.array(waveform), -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    logger.info(
        "h3: wav written %d samples @%dHz -> %s", len(audio_int16), sample_rate, path
    )


def _mux_av(video_path, audio_path, output_path):
    # ffmpeg 合流 video + audio → 单 MP4（copy video + aac audio，-shortest 截齐）。
    # 与 ltx2 audio.mux_video_audio 同模式。
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "ffmpeg mux failed: " + (e.stderr.decode() if e.stderr else str(e))
        ) from e
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found, install ffmpeg for audio mux") from e
    logger.info("h3: av muxed -> %s", output_path)


__all__ = [
    "generate_fl2va_video",
    "generate_t2va_video",
    "generate_t2va_av",
    "generate_video",
]
