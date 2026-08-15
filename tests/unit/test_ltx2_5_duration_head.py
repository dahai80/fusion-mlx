# SPDX-License-Identifier: Apache-2.0
# P5a checkpoint: LTX-2.5 duration-head + frame inference.
# 真实结构：attention-pooler multimodal head（video_tokens/audio_tokens）。
from __future__ import annotations

import mlx.core as mx
import pytest

from fusion_mlx.video.ltx2_5.duration_head import (
    DurationHead,
    duration_to_num_frames,
    infer_num_frames,
    load_duration_head,
)


class TestDurationHead:
    def test_video_only_output_shape(self):
        head = DurationHead(video_cross_attention_dim=32, audio_cross_attention_dim=16, pooler_hidden_dim=8)
        out = head(video_tokens=mx.random.normal((1, 4, 32)))
        assert out.shape == (1,)

    def test_audio_only_output_shape(self):
        head = DurationHead(video_cross_attention_dim=32, audio_cross_attention_dim=16, pooler_hidden_dim=8)
        out = head(audio_tokens=mx.random.normal((1, 6, 16)))
        assert out.shape == (1,)

    def test_both_modalities_batch(self):
        head = DurationHead(video_cross_attention_dim=32, audio_cross_attention_dim=16, pooler_hidden_dim=8)
        out = head(
            video_tokens=mx.random.normal((3, 4, 32)),
            audio_tokens=mx.random.normal((3, 6, 16)),
        )
        assert out.shape == (3,)

    def test_requires_at_least_one_modality(self):
        head = DurationHead(video_cross_attention_dim=32, audio_cross_attention_dim=16, pooler_hidden_dim=8)
        with pytest.raises(ValueError, match="at least one"):
            head()

    def test_predict_duration_clamps(self):
        head = DurationHead(video_cross_attention_dim=8, audio_cross_attention_dim=8, pooler_hidden_dim=8)
        out = head.predict_duration(mx.random.normal((1, 2, 8)))
        assert 0.5 <= float(out[0]) <= 60.0

    def test_predict_duration_no_clamp_runs(self):
        head = DurationHead(video_cross_attention_dim=8, audio_cross_attention_dim=8, pooler_hidden_dim=8)
        out = head.predict_duration(mx.random.normal((1, 2, 8)), clamp=False)
        assert out.shape == (1,)


class TestDurationToNumFrames:
    def test_constraint_mod8_plus1(self):
        for d in [1.0, 2.0, 3.7, 5.5, 10.0]:
            frames = duration_to_num_frames(d, fps=24.0)
            assert frames % 8 == 1

    def test_known_values(self):
        # 2s @ 24fps: round(48/8)*8+1 = 6*8+1 = 49
        assert duration_to_num_frames(2.0, 24.0) == 49
        # 1s @ 24fps: round(24/8)*8+1 = 3*8+1 = 25
        assert duration_to_num_frames(1.0, 24.0) == 25

    def test_min_frames(self):
        # very short duration -> at least 1 frame (round(0/8)*8+1 = 1)
        assert duration_to_num_frames(0.1, 24.0) == 1

    def test_fps_affects_result(self):
        assert duration_to_num_frames(2.0, 24.0) == 49
        # 2s @ 30fps: round(60/8)*8+1 = 8*8+1 = 65
        assert duration_to_num_frames(2.0, 30.0) == 65


class TestInferNumFrames:
    def test_end_to_end_video(self):
        head = DurationHead(video_cross_attention_dim=16, audio_cross_attention_dim=8, pooler_hidden_dim=8)
        frames = infer_num_frames(head, mx.random.normal((1, 5, 16)), None, fps=24.0)
        assert isinstance(frames, int)
        assert frames % 8 == 1
        assert frames >= 1

    def test_end_to_end_both(self):
        head = DurationHead(video_cross_attention_dim=16, audio_cross_attention_dim=8, pooler_hidden_dim=8)
        frames = infer_num_frames(
            head,
            mx.random.normal((1, 5, 16)),
            mx.random.normal((1, 3, 8)),
            fps=24.0,
        )
        assert frames % 8 == 1


class TestLoadDurationHead:
    def test_missing_weights_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="duration-head weights not found"):
            load_duration_head(tmp_path / "nope.safetensors")
