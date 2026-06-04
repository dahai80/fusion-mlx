"""Audio engine utilities for fusion-mlx."""

import struct
import wave
from io import BytesIO
from typing import Optional

import numpy as np


def wav_header(duration_seconds: float, sample_rate: int = 24000, channels: int = 1) -> bytes:
    """Generate a minimal WAV file header.

    Args:
        duration_seconds: Duration of the audio data in seconds.
        sample_rate: Sample rate in Hz.
        channels: Number of audio channels.

    Returns:
        WAV file header bytes.
    """
    sample_width = 2  # 16-bit PCM
    frames = int(sample_rate * duration_seconds * channels)
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00" * min(frames, 100) * sample_width * channels)
    return buf.getvalue()


def wav_bytes_to_pcm_frames(wav_data: bytes, sample_rate: int = 24000) -> Optional[np.ndarray]:
    """Decode WAV bytes to PCM frames as a numpy array.

    Args:
        wav_data: Raw WAV file bytes.
        sample_rate: Expected sample rate.

    Returns:
        Numpy array of PCM samples, or None on failure.
    """
    try:
        buf = BytesIO(wav_data)
        with wave.open(buf, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
        # Convert to numpy array
        if sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            dtype = np.uint8
        pcm = np.frombuffer(raw, dtype=dtype)
        if n_channels > 1:
            pcm = pcm.reshape(-1, n_channels).mean(axis=1)
        # Normalize to float32 [-1, 1]
        if dtype in (np.int16,):
            pcm = pcm.astype(np.float32) / 32768.0
        elif dtype in (np.int32,):
            pcm = pcm.astype(np.float32) / 2147483648.0
        else:
            pcm = pcm.astype(np.float32) / 255.0
        return pcm
    except Exception:
        return None
