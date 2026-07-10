# SPDX-License-Identifier: Apache-2.0
# Regression: the native VLM video path previously SKIPPED the vision cache
# (comment: "ndarray frames have no stable PIL hash"), so every multi-turn
# video conversation recomputed vision features from scratch. compute_video_hash
# now gives frames a content-stable key, so video features flow through the
# same VisionFeatureSSDCache as images.
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("mlx.utils")
pytest.importorskip("cv2")

import mlx.core as mx

from fusion_mlx.engines.vlm import VLMBatchedEngine


def _make_native_engine_with_cache():
    eng = VLMBatchedEngine.__new__(VLMBatchedEngine)
    eng._processor = MagicMock()
    eng._processor.apply_chat_template = MagicMock(return_value="<prompt>")
    eng._enable_thinking = None
    eng._vlm_model = MagicMock()
    eng._vlm_model.config = SimpleNamespace(video_token_id=1)
    eng._vlm_model.language_model = None
    eng._vision_cache = MagicMock()
    eng._vision_cache.get = MagicMock(return_value=None)
    eng._vision_cache.put = MagicMock()
    eng._vision_cache_enabled = True
    eng._model_name = "test-vlm"
    eng._compute_vision_features = MagicMock(return_value="FEATURES_TOKEN")
    return eng


def _patch_native_path(monkeypatch, embed_mock, frames_value=None):
    if frames_value is None:
        frames_value = np.zeros((4, 3, 8, 8), dtype=np.uint8)

    monkeypatch.setattr(
        "fusion_mlx.utils.video.process_video_input", lambda v: "/tmp/x.mp4"
    )
    monkeypatch.setattr(
        "mlx_vlm.utils.load_video",
        lambda video_path, fps=2.0, max_frames=768, **kw: (frames_value, 2.0),
    )

    def fake_prepare_inputs(processor, images=None, videos=None, prompts=None, **kw):
        return {
            "input_ids": mx.array([[1, 2, 3]]),
            "attention_mask": mx.array([[1, 1, 1]]),
            "pixel_values": mx.zeros((1, 3, 8, 8)),
        }

    monkeypatch.setattr("mlx_vlm.utils.prepare_inputs", fake_prepare_inputs)
    monkeypatch.setattr("mlx.core.eval", lambda *a, **k: None)


class TestNativeVideoCaching:
    def test_video_features_cached_under_compute_video_hash(self, monkeypatch):
        eng = _make_native_engine_with_cache()
        embed_mock = MagicMock()
        embed_mock.inputs_embeds = MagicMock()
        embed_mock.to_dict = MagicMock(return_value={})
        eng._vlm_model.get_input_embeddings = MagicMock(return_value=embed_mock)

        _patch_native_path(monkeypatch, embed_mock)

        eng._prepare_native_video_inputs(
            [{"role": "user", "content": "describe"}],
            ["/tmp/x.mp4"],
            video_fps=2.0,
            video_max_frames=8,
        )

        # cache.get was queried (miss on first call)...
        assert eng._vision_cache.get.call_count == 1
        got_hash = eng._vision_cache.get.call_args.args[0]
        got_model = eng._vision_cache.get.call_args.args[1]
        assert got_model == "test-vlm"
        # ...and compute_vision_features ran + result was stored under the SAME hash
        assert eng._compute_vision_features.call_count == 1
        assert eng._vision_cache.put.call_count == 1
        put_hash = eng._vision_cache.put.call_args.args[0]
        put_model = eng._vision_cache.put.call_args.args[1]
        put_features = eng._vision_cache.put.call_args.args[2]
        assert put_hash == got_hash and put_hash != ""
        assert put_model == "test-vlm"
        assert put_features == "FEATURES_TOKEN"

    def test_cache_hit_skips_feature_computation(self, monkeypatch):
        eng = _make_native_engine_with_cache()
        # Simulate a prior cache hit.
        eng._vision_cache.get = MagicMock(return_value="CACHED_FEATURES")
        embed_mock = MagicMock()
        embed_mock.inputs_embeds = MagicMock()
        embed_mock.to_dict = MagicMock(return_value={})
        eng._vlm_model.get_input_embeddings = MagicMock(return_value=embed_mock)

        _patch_native_path(monkeypatch, embed_mock)

        eng._prepare_native_video_inputs(
            [{"role": "user", "content": "describe"}],
            ["/tmp/x.mp4"],
            video_fps=2.0,
            video_max_frames=8,
        )

        # On a hit we must NOT recompute features and must NOT overwrite the cache.
        assert eng._compute_vision_features.call_count == 0
        assert eng._vision_cache.put.call_count == 0

    def test_cache_disabled_skips_cache(self, monkeypatch):
        eng = _make_native_engine_with_cache()
        eng._vision_cache_enabled = False
        embed_mock = MagicMock()
        embed_mock.inputs_embeds = MagicMock()
        embed_mock.to_dict = MagicMock(return_value={})
        eng._vlm_model.get_input_embeddings = MagicMock(return_value=embed_mock)

        _patch_native_path(monkeypatch, embed_mock)

        eng._prepare_native_video_inputs(
            [{"role": "user", "content": "describe"}],
            ["/tmp/x.mp4"],
            video_fps=2.0,
            video_max_frames=8,
        )

        assert eng._vision_cache.get.call_count == 0
        assert eng._vision_cache.put.call_count == 0

    def test_compute_video_hash_stability(self):
        # Sanity: same frames -> same hash; different frames -> different hash.
        from fusion_mlx.utils.video import compute_video_hash

        v1 = np.zeros((16, 3, 8, 8), dtype=np.uint8)
        v1b = np.zeros((16, 3, 8, 8), dtype=np.uint8)
        v2 = np.ones((16, 3, 8, 8), dtype=np.uint8)
        assert compute_video_hash([v1]) == compute_video_hash([v1b])
        assert compute_video_hash([v1]) != compute_video_hash([v2])
        assert compute_video_hash([]) is None
