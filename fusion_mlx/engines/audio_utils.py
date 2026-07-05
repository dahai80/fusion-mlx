# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for audio engines (STT, TTS, STS)."""

import io
import struct
import wave

import numpy as np

DEFAULT_SAMPLE_RATE = 24000
_MAX_WAV_CHUNK_SIZE = 0xFFFFFFFF


def wav_header(sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    block_align = channels * sample_width
    byte_rate = sample_rate * block_align
    bits_per_sample = sample_width * 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        _MAX_WAV_CHUNK_SIZE,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        _MAX_WAV_CHUNK_SIZE,
    )


def wav_bytes_to_pcm_frames(wav_bytes: bytes) -> tuple[int, int, int, bytes]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return (
            wf.getframerate(),
            wf.getnchannels(),
            wf.getsampwidth(),
            wf.readframes(wf.getnframes()),
        )


def audio_to_wav_bytes(audio_array, sample_rate: int) -> bytes:
    if not isinstance(audio_array, np.ndarray):
        if hasattr(audio_array, "dtype"):
            import mlx.core as mx

            if audio_array.dtype == mx.bfloat16:
                audio_array = audio_array.astype(mx.float32)
        audio_array = np.array(audio_array)
    audio_array = audio_array.flatten()
    audio_int16 = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()
