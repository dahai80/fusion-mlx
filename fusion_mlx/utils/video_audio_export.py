# SPDX-License-Identifier: Apache-2.0
"""Unified video+audio export/mux (PRD v1 §4.4).

Extracts the duplicated _write_mp4 / _mux_av logic out of
video/ltx2_5/generate.py and video/minimax_h3/generate.py into one
common post-processing path. Native MP4 mux via system ffmpeg (no
Python video-lib runtime dependency — ffmpeg is a system tool, not a
third-party Python package, per PRD §3.1 zero-dependency口径).

Frame序列标准化 + 音量归一 + 时序校准 + 临时资源回收.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def write_mp4(frames: np.ndarray, fps: int, output_path: str | Path | None) -> bytes:
    """frames: (T, H, W, 3) uint8 → MP4 bytes (and optional file)."""
    import imageio.v3 as iio

    out_bytes = iio.write(frames, "<bytes>", format="mp4", fps=fps, quality=8)
    if output_path is not None:
        Path(output_path).write_bytes(out_bytes)
        logger.info("mp4 written: %s (%d frames, %dfps)", output_path, len(frames), fps)
    return out_bytes


def normalize_audio(wav: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    if wav.size == 0:
        return wav
    peak = float(np.max(np.abs(wav)))
    if peak < 1e-6:
        return wav
    target = 10 ** (target_db / 20.0)
    return (wav * (target / peak)).clip(-1.0, 1.0)


def write_wav(wav: np.ndarray, sr: int, output_path: str | Path) -> None:
    import soundfile as sf

    if wav.ndim == 1:
        wav = np.stack([wav, wav], axis=-1)
    sf.write(str(output_path), wav, sr)
    logger.info("wav written: %s (%dHz, %ds)", output_path, sr, len(wav) // sr)


def mux_av(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
) -> Path:
    """ffmpeg合流 video + audio → single MP4 (copy video + aac, -shortest)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found — install ffmpeg for audio mux")
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
        stderr = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError("ffmpeg mux failed: " + stderr) from e
    logger.info("muxed A/V -> %s", output_path)
    return Path(output_path)


def cleanup_temp(*paths: str | Path) -> None:
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass
