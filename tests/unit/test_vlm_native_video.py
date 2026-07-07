# SPDX-License-Identifier: Apache-2.0
# Regression: native video load_video must be called with a dict ele.
# Previously load_video(path, fps, max_frames) always raised TypeError
# (mlx_vlm.video_generate.load_video expects a single dict) and was
# silently swallowed by `except (ImportError, Exception)`, so native
# Qwen-VL video always fell back to cv2 frame-as-image and lost temporal
# tokens — root cause of "poor video support".
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("mlx.utils")
pytest.importorskip("cv2")

import mlx.core as mx

from fusion_mlx.engines.vlm import VLMBatchedEngine


def _make_native_engine():
    eng = VLMBatchedEngine.__new__(VLMBatchedEngine)
    eng._processor = MagicMock()
    eng._processor.apply_chat_template = MagicMock(return_value="<prompt>")
    eng._enable_thinking = None
    eng._vlm_model = MagicMock()
    eng._vlm_model.config = SimpleNamespace(video_token_id=1)
    eng._vlm_model.language_model = None
    eng._vision_cache = None
    eng._vision_cache_enabled = False
    eng._model_name = "test-vlm"
    eng._compute_vision_features = MagicMock(return_value=None)
    return eng


class TestNativeVideoLoadVideoCall:
    def test_load_video_receives_dict_not_positional_args(self, monkeypatch):
        captured = {}

        def fake_load_video(ele):
            captured["ele"] = ele
            return np.zeros((4, 3, 8, 8), dtype=np.uint8), 2.0

        embed_mock = MagicMock()
        embed_mock.inputs_embeds = MagicMock()
        embed_mock.to_dict = MagicMock(return_value={})

        def fake_prepare_inputs(
            processor, images=None, videos=None, prompts=None, **kw
        ):
            return {
                "input_ids": mx.array([[1, 2, 3]]),
                "attention_mask": mx.array([[1, 1, 1]]),
                "pixel_values": mx.zeros((1, 3, 8, 8)),
            }

        eng = _make_native_engine()
        eng._vlm_model.get_input_embeddings = MagicMock(return_value=embed_mock)
        monkeypatch.setattr(
            "fusion_mlx.utils.video.process_video_input", lambda v: "/tmp/x.mp4"
        )
        monkeypatch.setattr("mlx_vlm.video_generate.load_video", fake_load_video)
        monkeypatch.setattr("mlx_vlm.utils.prepare_inputs", fake_prepare_inputs)
        monkeypatch.setattr(
            "fusion_mlx.engines.vlm.compute_image_hash", lambda imgs: "h"
        )
        monkeypatch.setattr("mlx.core.eval", lambda *a, **k: None)

        result = eng._prepare_native_video_inputs(
            [{"role": "user", "content": "describe"}],
            ["/tmp/x.mp4"],
            video_fps=2.0,
            video_max_frames=8,
        )

        assert isinstance(captured["ele"], dict)
        assert captured["ele"]["video"] == "/tmp/x.mp4"
        assert captured["ele"]["fps"] == 2.0
        assert captured["ele"]["max_frames"] == 8
        assert isinstance(result, tuple) and len(result) == 6

    def test_load_video_failure_falls_back_with_warning(self, monkeypatch, caplog):
        import logging

        eng = _make_native_engine()
        eng._prepare_vision_inputs = MagicMock(
            return_value=([1], None, None, None, 0, [])
        )
        monkeypatch.setattr(
            "fusion_mlx.utils.video.process_video_input", lambda v: "/tmp/x.mp4"
        )
        monkeypatch.setattr(
            "mlx_vlm.video_generate.load_video",
            lambda ele: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        monkeypatch.setattr(
            "fusion_mlx.engines.vlm.extract_video_frames_smart",
            lambda path, fps=2.0, max_frames=8: [],
        )
        monkeypatch.setattr(
            "fusion_mlx.engines.vlm.save_frames_to_temp", lambda frames: []
        )

        with caplog.at_level(logging.WARNING):
            eng._prepare_native_video_inputs(
                [{"role": "user", "content": "describe"}],
                ["/tmp/x.mp4"],
                video_fps=2.0,
                video_max_frames=8,
            )

        assert any("Native load_video failed" in r.message for r in caplog.records)
