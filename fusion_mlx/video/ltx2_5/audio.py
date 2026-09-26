import logging
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..ltx2.audio_vae.audio_vae import AudioDecoder
from ..ltx2.audio_vae.vocoder import VocoderWithBWE, _load_vocoder_with_bwe
from ..ltx2.config import AudioDecoderModelConfig

logger = logging.getLogger(__name__)

LTX2_5_AUDIO_SAMPLE_RATE = 48000

_VOCODER_CFG = {
    "resblock_kernel_sizes": [3, 7, 11],
    "upsample_rates": [5, 2, 2, 2, 2, 2],
    "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_initial_channel": 1536,
    "stereo": True,
    "resblock": "AMP1",
    "output_sample_rate": 16000,
    "activation": "snakebeta",
    "use_tanh_at_final": True,
    "apply_final_activation": False,
    "use_bias_at_final": False,
}

_BWE_CFG = {
    "resblock_kernel_sizes": [3, 7, 11],
    "upsample_rates": [6, 5, 2, 2, 2],
    "upsample_kernel_sizes": [12, 11, 4, 4, 4],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_initial_channel": 512,
    "stereo": True,
    "resblock": "AMP1",
    "output_sample_rate": 48000,
    "activation": "snakebeta",
    "use_tanh_at_final": True,
    "apply_final_activation": False,
    "use_bias_at_final": False,
    "hop_length": 80,
    "input_sampling_rate": 16000,
    "output_sampling_rate": 48000,
}

_DECODER_CFG = AudioDecoderModelConfig(
    ch=128,
    out_ch=2,
    ch_mult=(1, 2, 4),
    num_res_blocks=2,
    resolution=256,
    z_channels=8,
    mel_bins=64,
    sample_rate=16000,
    mel_hop_length=160,
    is_causal=True,
    mid_block_add_attention=False,
    norm_type="pixel",
    causality_axis="height",
    attn_type="none",
    attn_resolutions=[],
)


def _is_conv1d_weight(key: str) -> bool:
    if not key.endswith(".weight"):
        return False
    if ".ups." in key:
        return False
    if key.endswith("conv_pre.weight") or key.endswith("conv_post.weight"):
        return True
    if ".convs1." in key or ".convs2." in key:
        return True
    return False


def _is_conv_transpose1d_weight(key: str) -> bool:
    return key.endswith(".weight") and ".ups." in key


def load_audio_decoder(weights_path: str | Path) -> AudioDecoder:
    weights_path = Path(weights_path)
    logger.info("load_audio_decoder: %s", weights_path.name)
    raw = mx.load(str(weights_path))
    decoder = AudioDecoder(_DECODER_CFG)
    dec_weights = {
        k: v
        for k, v in raw.items()
        if k.startswith("audio_vae.decoder.")
        or k.startswith("audio_vae.per_channel_statistics.")
    }
    sanitized = decoder.sanitize(dec_weights)
    decoder.load_weights(list(sanitized.items()), strict=False)
    logger.info("load_audio_decoder: %d decoder weights loaded", len(sanitized))
    return decoder


def load_vocoder_model(weights_path: str | Path) -> VocoderWithBWE:
    weights_path = Path(weights_path)
    logger.info("load_vocoder_model: %s", weights_path.name)
    raw = mx.load(str(weights_path))
    mapped = {}
    n_conv1d = 0
    n_convt = 0
    for k, v in raw.items():
        if k.startswith("vocoder."):
            k = k[len("vocoder.") :]
        if k.startswith("audio_vae."):
            continue
        if _is_conv1d_weight(k):
            v = mx.transpose(v, (0, 2, 1))
            n_conv1d += 1
        elif _is_conv_transpose1d_weight(k):
            v = mx.transpose(v, (1, 2, 0))
            n_convt += 1
        mapped[k] = v
    config_dict = {
        "has_bwe_generator": True,
        "vocoder": _VOCODER_CFG,
        "bwe": _BWE_CFG,
    }
    vocoder = _load_vocoder_with_bwe(config_dict, mapped)
    logger.info(
        "load_vocoder_model: %d weights (conv1d transposed=%d, conv_t=%d)",
        len(mapped),
        n_conv1d,
        n_convt,
    )
    return vocoder


def decode_audio(
    audio_latents: mx.array,
    decoder: AudioDecoder,
    vocoder: VocoderWithBWE,
) -> np.ndarray:
    logger.info(
        "decode_audio: latents shape=%s dtype=%s",
        audio_latents.shape,
        audio_latents.dtype,
    )
    mel_spectrogram = decoder(audio_latents)
    mx.eval(mel_spectrogram)
    logger.info(
        "decode_audio: mel shape=%s std=%.4f mean=%.4f",
        mel_spectrogram.shape,
        mel_spectrogram.std().item(),
        mel_spectrogram.mean().item(),
    )
    audio_waveform = vocoder(mel_spectrogram)
    mx.eval(audio_waveform)
    audio_np = np.array(audio_waveform.astype(mx.float32))
    if audio_np.ndim == 3:
        audio_np = audio_np[0]
    logger.info(
        "decode_audio: waveform shape=%s sr=%d",
        audio_np.shape,
        vocoder.output_sampling_rate,
    )
    return audio_np


def save_audio(
    audio: np.ndarray,
    path: str | Path,
    sample_rate: int = LTX2_5_AUDIO_SAMPLE_RATE,
):
    path = Path(path)
    logger.info("save_audio: %s @ %d Hz", path, sample_rate)
    if audio.ndim == 2:
        audio = audio.T
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2 if audio_int16.ndim == 2 else 1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def mux_video_audio(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
) -> bool:
    import subprocess

    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    logger.info(
        "mux_video_audio: %s + %s -> %s",
        video_path.name,
        audio_path.name,
        output_path.name,
    )
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
        return True
    except subprocess.CalledProcessError as e:
        logger.error("FFmpeg error: %s", e.stderr.decode() if e.stderr else e)
        return False
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install ffmpeg.")
        return False
