# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2 audio processor (vendored from mlx-video).
# Uses librosa for macOS/MLX compatibility (replaces torchaudio MelSpectrogram).
# Phase 4 Stage E: audio_vae port.
import logging

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


def load_audio(
    path: str,
    target_sr: int = 16000,
    start_time: float = 0.0,
    max_duration: float | None = None,
    mono: bool = False,
) -> tuple[np.ndarray, int]:
    import librosa

    y, sr = librosa.load(
        path,
        sr=target_sr,
        mono=mono,
        offset=start_time,
        duration=max_duration,
    )
    if y.ndim == 1:
        y = y[np.newaxis, :]
    logger.info("load_audio: %s sr=%d shape=%s", path, sr, y.shape)
    return y.astype(np.float32), sr


def ensure_stereo(waveform: np.ndarray) -> np.ndarray:
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    if waveform.shape[0] == 1:
        waveform = np.concatenate([waveform, waveform], axis=0)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    return waveform


def waveform_to_mel(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 160,
    win_length: int = 1024,
    n_mels: int = 64,
    fmin: float = 0.0,
    fmax: float = 8000.0,
) -> mx.array:
    import librosa

    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    channels = waveform.shape[0]
    mels = []
    for ch in range(channels):
        S = np.abs(
            librosa.stft(
                waveform[ch],
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                center=True,
                pad_mode="reflect",
            )
        )
        mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            norm="slaney",
        )
        mel = mel_basis @ S
        mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
        mel = mel.T
        mels.append(mel)
    mel_spec = np.stack(mels, axis=0)
    mel_spec = mel_spec[np.newaxis, ...]
    return mx.array(mel_spec, dtype=mx.float32)
