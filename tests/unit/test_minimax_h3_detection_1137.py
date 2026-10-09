# SPDX-License-Identifier: Apache-2.0
"""#1137: MiniMax-H3 quantized fork repos misdetected as LLM.

Covers:
- DiT-only H3 fork (config.json _class_name=MiniMaxH3DiTModel, no model_type,
  no model_index.json, no diffusers subdirs) → _is_video_model True.
- Official MiniMaxH3ModularPipeline in DIFFUSERS_PIPELINE_TASKS.
- FL2VA/ subdir satisfies the diffusers-subdir gate.
- _apply_decode_weights handles fused weight_norm (weight only, no weight_g/
  weight_v) for ddalcu fork repos.
- from_pretrained globs for *.safetensors when model.safetensors absent.
"""

import json
import os
from unittest.mock import MagicMock, patch

from fusion_mlx.pool.model_discovery import (
    DIFFUSERS_PIPELINE_TASKS,
    _is_video_model,
)


class TestH3DiTOnlyForkDetection:
    def _make_dit_only_fork(self, path):
        # pipenetwork--MiniMax-H3-MLX-4bit layout: config.json with
        # _class_name=MiniMaxH3DiTModel, NO model_type, root safetensors,
        # no model_index.json, no vae/transformer/ subdirs.
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(
            json.dumps({"_class_name": "MiniMaxH3DiTModel"})
        )
        (path / "model-00001-of-00005.safetensors").write_bytes(b"0" * 100)
        (path / "model.safetensors.index.json").write_text("{}")

    def test_dit_only_fork_detected_as_video(self, tmp_path):
        self._make_dit_only_fork(tmp_path)
        assert _is_video_model(tmp_path) is True

    def test_dit_only_fork_without_safetensors_not_video(self, tmp_path):
        # config.json _class_name=MiniMaxH3DiTModel but NO root safetensors
        # → must NOT be misdetected (could be a partial download).
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "config.json").write_text(
            json.dumps({"_class_name": "MiniMaxH3DiTModel"})
        )
        assert _is_video_model(tmp_path) is False

    def test_dit_only_fork_with_model_type_still_works(self, tmp_path):
        # If a fork also has model_type (unlikely but defensive), the existing
        # VIDEO_CONFIG_MODEL_TYPES branch handles it — the _class_name branch
        # is a fallback for when model_type is absent.
        self._make_dit_only_fork(tmp_path)
        data = json.loads((tmp_path / "config.json").read_text())
        data["model_type"] = "minimax_h3_dit"
        (tmp_path / "config.json").write_text(json.dumps(data))
        assert _is_video_model(tmp_path) is True

    def test_non_h3_class_name_with_safetensors_not_video(self, tmp_path):
        # A random repo with _class_name=SomethingElse + safetensors must
        # NOT be misdetected as video.
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "config.json").write_text(
            json.dumps({"_class_name": "SomeOtherModel"})
        )
        (tmp_path / "model.safetensors").write_bytes(b"0" * 100)
        assert _is_video_model(tmp_path) is False

    def test_corrupt_config_json_does_not_crash(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "config.json").write_text("{not valid json")
        (tmp_path / "model.safetensors").write_bytes(b"0" * 100)
        # Must not raise — returns False (falls through to LLM).
        assert _is_video_model(tmp_path) is False


class TestH3ModularPipelineInTaskMap:
    def test_modular_pipeline_registered_as_text_to_video(self):
        assert "MiniMaxH3ModularPipeline" in DIFFUSERS_PIPELINE_TASKS
        assert DIFFUSERS_PIPELINE_TASKS["MiniMaxH3ModularPipeline"] == "text-to-video"


class TestFL2VASubdirGate:
    def _make_fl2va_layout(self, path):
        # Official MiniMaxAI/MiniMax-H3: model_index.json with
        # MiniMaxH3ModularPipeline, diffusers subdirs nested under FL2VA/.
        path.mkdir(parents=True, exist_ok=True)
        (path / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "MiniMaxH3ModularPipeline",
                    "transformer": ["diffusers", "MiniMaxH3DiTModel"],
                }
            )
        )
        fl2va = path / "FL2VA"
        fl2va.mkdir(exist_ok=True)
        (fl2va / "transformer").mkdir(exist_ok=True)
        (fl2va / "video_vae").mkdir(exist_ok=True)
        (fl2va / "audio_vae").mkdir(exist_ok=True)
        (fl2va / "transformer" / "model.safetensors").write_bytes(b"0" * 100)

    def test_fl2va_subdir_satisfies_diffusers_gate(self, tmp_path):
        self._make_fl2va_layout(tmp_path)
        assert _is_video_model(tmp_path) is True

    def test_plain_llm_repo_not_misdetected_by_fl2va_gate(self, tmp_path):
        # An LLM repo must not be caught by the FL2VA gate.
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (tmp_path / "model.safetensors").write_bytes(b"0" * 100)
        assert _is_video_model(tmp_path) is False


class _TensorDict:
    """dict-like that returns MagicMock for any key, with controlled
    __contains__ to steer the conv_pre fused-vs-split branch."""

    _EXCLUDED = frozenset()

    def __init__(self, keys, excluded=()):
        self._keys = list(keys)
        self._vals = {k: MagicMock() for k in keys}
        self._excluded = frozenset(excluded)

    def __contains__(self, k):
        if k in self._excluded:
            return False
        return k in self._vals

    def __getitem__(self, k):
        return self._vals.get(k, MagicMock())

    def keys(self):
        return self._keys


class TestApplyDecodeWeightsFusedWeightNorm:
    _BASE_KEYS = (
        "dec_in_proj.weight",
        "dec_in_proj.bias",
        "decoder.conv_pre.bias",
        "decoder.conv_post.weight_g",
        "decoder.conv_post.weight_v",
    )
    _OUT_KEYS = (
        "dec_in_proj.weight",
        "dec_in_proj.bias",
        "decoder.conv_pre.weight",
        "decoder.conv_pre.bias",
        "decoder.conv_post.weight",
    )

    def test_conv_pre_fused_weight_used_when_no_weight_g(self):
        # ddalcu fork: weight_norm pre-fused into flat weight (no weight_g/
        # weight_v pair). _apply_decode_weights must use weight directly.
        from fusion_mlx.video.minimax_h3.audio_vae import _apply_decode_weights

        keys = self._BASE_KEYS + ("decoder.conv_pre.weight",)
        tensors = _TensorDict(keys, excluded=("decoder.conv_pre.weight_g",))
        model = MagicMock()
        with patch("fusion_mlx.video.minimax_h3.audio_vae._conv1d_to_mlx") as mock_conv:
            with patch("fusion_mlx.video.minimax_h3.audio_vae._convt1d_to_mlx"):
                with patch(
                    "fusion_mlx.video.minimax_h3.audio_vae.reconstruct_weight_norm"
                ) as mock_recon:
                    with patch("fusion_mlx.video.minimax_h3.audio_vae.mx"):
                        import mlx.utils as _mlx_utils

                        with patch.object(
                            _mlx_utils,
                            "tree_flatten",
                            return_value=list(zip(self._OUT_KEYS, [None] * 5)),
                        ):
                            _apply_decode_weights(model, tensors)
        # conv_pre: _conv1d_to_mlx called with flat weight (NOT reconstruct).
        conv_pre_calls = [
            c
            for c in mock_conv.call_args_list
            if c.args and c.args[0] is tensors["decoder.conv_pre.weight"]
        ]
        assert len(conv_pre_calls) == 1
        # reconstruct_weight_norm must NOT have been called for conv_pre.
        recon_args = [a for c in mock_recon.call_args_list for a in c.args]
        assert tensors["decoder.conv_pre.weight"] not in recon_args

    def test_conv_pre_reconstruct_used_when_weight_g_present(self):
        # Official repo: weight_g + weight_v pair → reconstruct_weight_norm.
        from fusion_mlx.video.minimax_h3.audio_vae import _apply_decode_weights

        keys = self._BASE_KEYS + (
            "decoder.conv_pre.weight_g",
            "decoder.conv_pre.weight_v",
        )
        tensors = _TensorDict(keys)
        model = MagicMock()
        with patch("fusion_mlx.video.minimax_h3.audio_vae._conv1d_to_mlx") as mock_conv:
            with patch("fusion_mlx.video.minimax_h3.audio_vae._convt1d_to_mlx"):
                with patch(
                    "fusion_mlx.video.minimax_h3.audio_vae.reconstruct_weight_norm"
                ) as mock_recon:
                    with patch("fusion_mlx.video.minimax_h3.audio_vae.mx"):
                        import mlx.utils as _mlx_utils

                        with patch.object(
                            _mlx_utils,
                            "tree_flatten",
                            return_value=list(zip(self._OUT_KEYS, [None] * 5)),
                        ):
                            _apply_decode_weights(model, tensors)
        # reconstruct_weight_norm called with conv_pre weight_g + weight_v.
        recon_calls_for_conv_pre = [
            c
            for c in mock_recon.call_args_list
            if c.args and c.args[0] is tensors["decoder.conv_pre.weight_g"]
        ]
        assert len(recon_calls_for_conv_pre) == 1
        assert mock_conv.call_count >= 1


class TestFromPretrainedFilenameGlob:
    def test_globs_for_any_safetensors_when_model_dot_absent(self, tmp_path):
        # ddalcu fork: audio_vae.safetensors instead of model.safetensors.
        from fusion_mlx.video.minimax_h3.audio_vae import MiniMaxH3AudioVAE

        tmp_path.mkdir(parents=True, exist_ok=True)
        fake_file = tmp_path / "audio_vae.safetensors"
        fake_file.write_bytes(b"0" * 100)
        mock_safe_open = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_safe_open)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_safe_open.keys.return_value = []
        with patch("safetensors.safe_open", return_value=mock_ctx):
            with patch("fusion_mlx.video.minimax_h3.audio_vae._apply_decode_weights"):
                with patch("fusion_mlx.video.minimax_h3.audio_vae.mx"):
                    result = MiniMaxH3AudioVAE.from_pretrained(str(tmp_path))
        assert result is not None

    def test_prefers_model_dot_safetensors_when_present(self, tmp_path):
        from fusion_mlx.video.minimax_h3.audio_vae import MiniMaxH3AudioVAE

        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "model.safetensors").write_bytes(b"0" * 100)
        (tmp_path / "audio_vae.safetensors").write_bytes(b"0" * 100)
        captured = {}

        class FakeSafeOpen:
            def __init__(self, path, framework):
                captured["path"] = path

            def __enter__(self):
                m = MagicMock()
                m.keys.return_value = []
                return m

            def __exit__(self, *a):
                return False

        with patch("safetensors.safe_open", FakeSafeOpen):
            with patch("fusion_mlx.video.minimax_h3.audio_vae._apply_decode_weights"):
                with patch("fusion_mlx.video.minimax_h3.audio_vae.mx"):
                    MiniMaxH3AudioVAE.from_pretrained(str(tmp_path))
        assert os.path.basename(captured["path"]) == "model.safetensors"

    def test_raises_when_no_safetensors_in_dir(self, tmp_path):
        from fusion_mlx.video.minimax_h3.audio_vae import MiniMaxH3AudioVAE

        tmp_path.mkdir(parents=True, exist_ok=True)
        import pytest

        with pytest.raises(FileNotFoundError):
            MiniMaxH3AudioVAE.from_pretrained(str(tmp_path))
