"""Audio feature pipeline (MLX) — mirrors musetalk/utils/audio_processor.py.

stacked encoder hidden states (B, seq, 5, 384)  -- get_whisper_chunk -->
per-frame windows (num_frames, 50, 384)  -- PositionalEncoding -->  UNet cross-attn states.

Mel extraction is now torch-free MLX (see whisper/log_mel.py) — the whole audio path
(mel + encoder + chunking + PE) runs without torch/transformers.
"""

from __future__ import annotations

import math

import mlx.core as mx


def positional_encoding(seq_len, d_model=384):
    """Sinusoidal PE matching musetalk/models/unet.py PositionalEncoding."""
    pe = mx.zeros((seq_len, d_model))
    pos = mx.arange(0, seq_len, dtype=mx.float32)[:, None]
    div = mx.exp(
        mx.arange(0, d_model, 2, dtype=mx.float32) * (-math.log(10000.0) / d_model)
    )
    angles = pos * div
    pe = mx.zeros((seq_len, d_model))
    # interleave: even idx = sin, odd idx = cos
    sin, cos = mx.sin(angles), mx.cos(angles)
    inter = mx.stack([sin, cos], axis=-1).reshape(seq_len, d_model)
    return inter[None]  # (1, seq, d)


def apply_pe(x):
    """Add sinusoidal PE to (B, seq, d) audio features (PE cast to x's dtype)."""
    return x + positional_encoding(x.shape[1], x.shape[2]).astype(x.dtype)


def get_whisper_chunk(
    stacked,
    librosa_length,
    fps=25,
    audio_fps=50,
    sr=16000,
    pad_left=2,
    pad_right=2,
    prefix=None,
):
    """stacked: (1, seq, n_hidden, 384) -> (num_frames, (10*n_hidden), 384).

    Faithful port of AudioProcessor.get_whisper_chunk.

    #914 prefix-context cache: ``prefix`` is the prior 5s window's tail embedding
    (1, prefix_len, n_hidden, 384) spliced as left context so consecutive windows
    carry boundary state (smooths lip teleport at 5s edges). When given, the tail
    (last ``prefix_len`` frames of this window's stacked states) is returned for
    the next call via ``extract_prefix_tail``.
    """
    feat_len_per_frame = 2 * (pad_left + pad_right + 1)  # 10
    idx_mult = audio_fps / fps
    num_frames = math.floor((librosa_length / sr) * fps)
    actual_length = math.floor((librosa_length / sr) * audio_fps)
    wf = stacked[:, :actual_length, ...]
    # #914: splice prior-window tail as left context before zero-padding.
    if prefix is not None:
        wf = mx.concatenate([prefix, wf], axis=1)
    pad = math.ceil(idx_mult)
    zeros_l = mx.zeros_like(wf[:, : pad * pad_left])
    zeros_r = mx.zeros_like(wf[:, : pad * 3 * pad_right])
    wf = mx.concatenate([zeros_l, wf, zeros_r], axis=1)

    clips = []
    for fi in range(num_frames):
        ai = math.floor(fi * idx_mult)
        clip = wf[:, ai : ai + feat_len_per_frame]  # (1, 10, n_hidden, 384)
        clips.append(clip)
    prompts = mx.concatenate(clips, axis=0)  # (T, 10, n_hidden, 384)
    t, c, h, w = prompts.shape
    return prompts.reshape(t, c * h, w)  # (T, 50, 384)


_PREFIX_TAIL_LEN = 10


def extract_prefix_tail(stacked, tail_len: int = _PREFIX_TAIL_LEN):
    """#914: slice the tail of this window's stacked states for the next call's prefix.

    stacked: (1, seq, n_hidden, 384) -> (1, min(tail_len, seq), n_hidden, 384).
    The tail length is bounded by the window's available frames so a short final
    window does not over-extend the next prefix.
    """
    seq = stacked.shape[1]
    take = min(tail_len, seq)
    return stacked[:, seq - take :, ...]
