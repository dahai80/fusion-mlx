# SPDX-License-Identifier: Apache-2.0
"""#914: encode_audio prefix-context cache tests."""

import math

import mlx.core as mx

from fusion_mlx.video.musetalk_mlx.whisper.audio2feature import (
    extract_prefix_tail,
    get_whisper_chunk,
)


def _fake_stacked(seq=100, n_hidden=5, d=384):
    return mx.random.uniform(0, 1, (1, seq, n_hidden, d))


def test_extract_prefix_tail_default_len():
    stacked = _fake_stacked(seq=100)
    tail = extract_prefix_tail(stacked)
    assert tail.shape == (1, 10, 5, 384)


def test_extract_prefix_tail_custom_len():
    stacked = _fake_stacked(seq=100)
    tail = extract_prefix_tail(stacked, tail_len=20)
    assert tail.shape == (1, 20, 5, 384)


def test_extract_prefix_tail_clamps_short_window():
    stacked = _fake_stacked(seq=3)
    tail = extract_prefix_tail(stacked, tail_len=10)
    assert tail.shape == (1, 3, 5, 384)


def test_get_whisper_chunk_without_prefix_unchanged():
    stacked = _fake_stacked(seq=100)
    librosa_length = 16000 * 2  # 2s
    chunks = get_whisper_chunk(stacked, librosa_length, fps=25)
    num_frames = math.floor((librosa_length / 16000) * 25)
    assert chunks.shape == (num_frames, 50, 384)


def test_get_whisper_chunk_with_prefix_extends_context():
    stacked = _fake_stacked(seq=100)
    prefix = _fake_stacked(seq=10)
    librosa_length = 16000 * 2
    chunks_no_prefix = get_whisper_chunk(_fake_stacked(seq=100), librosa_length, fps=25)
    chunks_prefix = get_whisper_chunk(stacked, librosa_length, fps=25, prefix=prefix)
    # Same number of output frames (prefix only adds left context, not frames).
    assert chunks_prefix.shape[0] == chunks_no_prefix.shape[0]
    assert chunks_prefix.shape[1:] == (50, 384)


def test_prefix_carries_boundary_state():
    # Two windows: window2's prefix = window1's tail. The first frame of
    # window2 should differ from a no-prefix window2 (context changed).
    w1 = _fake_stacked(seq=100)
    w2 = _fake_stacked(seq=100)
    tail = extract_prefix_tail(w1)
    librosa_length = 16000 * 2
    c2_prefix = get_whisper_chunk(w2, librosa_length, fps=25, prefix=tail)
    c2_noprefix = get_whisper_chunk(w2, librosa_length, fps=25, prefix=None)
    # First-frame chunk should differ (prefix spliced into left context).
    diff = mx.abs(c2_prefix[0] - c2_noprefix[0]).sum()
    mx.eval(diff)
    assert diff > 0.0, "prefix did not alter boundary context"


def test_prefix_tail_is_actual_window_tail():
    stacked = _fake_stacked(seq=100)
    tail = extract_prefix_tail(stacked, tail_len=10)
    mx.eval(stacked, tail)
    # Tail == last 10 frames of stacked.
    assert mx.allclose(tail, stacked[:, 90:, ...]).item()
