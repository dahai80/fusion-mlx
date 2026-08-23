# SPDX-License-Identifier: Apache-2.0
# MiniMax H3 t2va video-only 推理循环（generate）。
#
# 上游权威缺失说明（UNVERIFIED，需真实模型校正）：
#   - 与 condition.py 同源：diffusers pipeline 源码未发布，本循环从
#     transformer `forward` 契约 + scheduler 契约推断。
#   - video-only：无音频 latent、无音频 scheduler、无条件帧。仅 video scheduler
#     （shift=12.0）去噪。text 行始终 clean（timestep=1.0，idx0）。
#
# 流程：text_encoder(prompt)→(b,n_t,5120)；noise→normalize→patchify→packed；
# 每 step：build_t2va_packed(latents, text, t_video)→transformer→video_output；
# unpatchify→step；循环结束 denormalize→vae.decode→[0,1]→frames。
import logging

import mlx.core as mx

from .audio_vae.audio_latents import (
    denormalize_audio_latents,
    normalize_audio_latents,
)
from .condition import (
    audio_latent_steps,
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
):
    # H3 t2va 顶层编排（UNVERIFIED，需真实模型校正）。
    #
    # model_path: H3 模型根目录（含 transformer/ text_encoder/ video_vae/ audio_vae/ 子目录或单目录）。
    # 加载 DiT + VAE + text_encoder，编码 prompt，去噪，写 mp4。
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

    logger.info(
        "h3 generate_video: prompt='%s' frames=%d %dx%d fps=%d seed=%s steps=%d quantize=%s audio=%s",
        prompt[:60],
        num_frames,
        width,
        height,
        fps,
        seed,
        num_inference_steps,
        quantize,
        audio,
    )

    cfg = H3Config()
    if seed is not None:
        mx.random.seed(int(seed))

    # 阶段化加载：FL2VA 总权重 144GB（TE 67G + DiT 66G + VAE 11G）超过 M5 Max 137G
    # 物理内存，同时加载会 swap thrash 致 Metal 前向极慢。先加载 TE 编码 prompt，
    # 物化 text_embeds 后释放 TE，再加载 DiT+VAE 去噪。text_embeds 仅几 MB。
    import gc

    te_path = _resolve_subdir(model_path, "text_encoder")
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
    # 物化 text_embeds 后释放 TE（~67G）与 tokenizer。
    mx.eval(text_embeds)
    del text_encoder
    del tokenizer
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

    if audio:
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


__all__ = ["generate_t2va_video", "generate_t2va_av", "generate_video"]
