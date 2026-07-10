import logging
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .generate import PipelineType

logger = logging.getLogger(__name__)

AUDIO_SAMPLE_RATE = 24000


def load_audio_decoder(model_path: Path, pipeline: "PipelineType"):
    from .audio_vae import AudioDecoder

    logger.debug(
        "Loading audio VAE decoder from %s (pipeline=%s)", model_path, pipeline
    )
    decoder = AudioDecoder.from_pretrained(model_path / "audio_vae" / "decoder")
    return decoder


def load_vocoder_model(model_path: Path, pipeline: "PipelineType"):
    from .audio_vae.vocoder import load_vocoder as _load_vocoder

    logger.debug("Loading vocoder from %s (pipeline=%s)", model_path, pipeline)
    return _load_vocoder(model_path / "vocoder")


def save_audio(audio: np.ndarray, path: Path, sample_rate: int = AUDIO_SAMPLE_RATE):
    logger.info("Saving audio to %s @ %d Hz", path, sample_rate)

    if audio.ndim == 2:
        audio = audio.T

    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2 if audio_int16.ndim == 2 else 1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def mux_video_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    import subprocess

    logger.info("Muxing %s + %s -> %s", video_path, audio_path, output_path)

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
