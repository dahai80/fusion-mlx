# SPDX-License-Identifier: Apache-2.0
"""Tests for pytorch_model.bin weight loading in embedding + convert."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest


class TestLoadPytorchWeights:
    def test_load_pytorch_weights_converts_tensors(self):
        from fusion_mlx.engines.embedding import MLXEmbeddingModel

        with tempfile.TemporaryDirectory() as tmpdir:
            import torch

            state_dict = {
                "embeddings.word_embeddings.weight": torch.randn(100, 32),
                "encoder.layer.0.attention.self.query.weight": torch.randn(32, 32),
            }
            pt_path = Path(tmpdir) / "pytorch_model.bin"
            torch.save(state_dict, str(pt_path))

            weights = MLXEmbeddingModel._load_pytorch_weights([pt_path])
            assert "embeddings.word_embeddings.weight" in weights
            assert "encoder.layer.0.attention.self.query.weight" in weights
            assert isinstance(weights["embeddings.word_embeddings.weight"], mx.array)
            assert weights["embeddings.word_embeddings.weight"].shape == (100, 32)

    def test_load_pytorch_weights_multiple_files(self):
        from fusion_mlx.engines.embedding import MLXEmbeddingModel

        with tempfile.TemporaryDirectory() as tmpdir:
            import torch

            state_dict_1 = {"layer.0.weight": torch.randn(10, 10)}
            state_dict_2 = {"layer.1.weight": torch.randn(10, 10)}
            pt_path_1 = Path(tmpdir) / "pytorch_model-00001-of-00002.bin"
            pt_path_2 = Path(tmpdir) / "pytorch_model-00002-of-00002.bin"
            torch.save(state_dict_1, str(pt_path_1))
            torch.save(state_dict_2, str(pt_path_2))

            weights = MLXEmbeddingModel._load_pytorch_weights(
                sorted([pt_path_1, pt_path_2])
            )
            assert "layer.0.weight" in weights
            assert "layer.1.weight" in weights


class TestLoadNativePytorchFallback:
    def test_load_native_falls_back_to_pytorch_bin(self):
        from fusion_mlx.engines.embedding import MLXEmbeddingModel

        with tempfile.TemporaryDirectory() as tmpdir:
            import torch

            model_path = Path(tmpdir)
            config = {
                "architectures": ["XLMRobertaModel"],
                "model_type": "xlm-roberta",
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "max_position_embeddings": 514,
                "layer_norm_eps": 1e-05,
                "vocab_size": 250002,
                "type_vocab_size": 1,
                "pad_token_id": 1,
                "attention_probs_dropout_prob": 0.1,
                "hidden_dropout_prob": 0.1,
                "position_embedding_type": "absolute",
                "output_past": True,
            }
            (model_path / "config.json").write_text(json.dumps(config))

            from fusion_mlx.models.xlm_roberta import Model, ModelArgs

            model_instance = Model(
                ModelArgs(
                    **{
                        k: v
                        for k, v in config.items()
                        if k in ModelArgs.__dataclass_fields__
                    }
                )
            )
            from mlx.utils import tree_flatten

            params = dict(tree_flatten(model_instance.parameters()))
            import numpy as np

            state_dict = {}
            for k, v in params.items():
                state_dict[k] = torch.from_numpy(np.array(v.astype(mx.float32)))
            pt_path = model_path / "pytorch_model.bin"
            torch.save(state_dict, str(pt_path))

            model = MLXEmbeddingModel(str(model_path))
            with patch.object(MLXEmbeddingModel, "_validate_native_weights"):
                with patch("transformers.AutoTokenizer") as mock_tok:
                    mock_tok.from_pretrained.return_value = MagicMock()
                    result = model._load_native()

            assert result is True
            assert model._loaded is True
            assert model._using_native is True

    def test_load_native_no_weights_returns_false(self):
        from fusion_mlx.engines.embedding import MLXEmbeddingModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            config = {
                "architectures": ["XLMRobertaModel"],
                "model_type": "xlm-roberta",
            }
            (model_path / "config.json").write_text(json.dumps(config))

            model = MLXEmbeddingModel(str(model_path))
            result = model._load_native()
            assert result is False


class TestConvertPytorchToSafetensors:
    def test_convert_creates_safetensors_from_pytorch(self):
        from fusion_mlx.cli_convert import _convert_pytorch_to_safetensors

        with tempfile.TemporaryDirectory() as tmpdir:
            import shutil

            import torch

            model_path = Path(tmpdir)
            state_dict = {"weight": torch.randn(10, 10)}
            pt_path = model_path / "pytorch_model.bin"
            torch.save(state_dict, str(pt_path))
            (model_path / "config.json").write_text("{}")

            result_path = _convert_pytorch_to_safetensors(model_path)
            assert (result_path / "model.safetensors").exists()
            assert (result_path / "config.json").exists()
            assert not (result_path / "pytorch_model.bin").exists()

            loaded = mx.load(str(result_path / "model.safetensors"))
            assert "weight" in loaded

            shutil.rmtree(str(result_path), ignore_errors=True)

    def test_convert_raises_if_no_pytorch_files(self):
        from fusion_mlx.cli_convert import _convert_pytorch_to_safetensors

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No pytorch_model.bin"):
                _convert_pytorch_to_safetensors(Path(tmpdir))


class TestRunConvertPytorchFallback:
    def test_run_convert_retries_with_pytorch_conversion(self):
        from fusion_mlx.cli_convert import _run_convert

        call_count = 0

        def fake_mlx_convert(path, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise FileNotFoundError("No safetensors found in /some/path")

        with tempfile.TemporaryDirectory() as tmpdir:
            import torch

            model_path = Path(tmpdir)
            state_dict = {"weight": torch.randn(4, 4)}
            pt_path = model_path / "pytorch_model.bin"
            torch.save(state_dict, str(pt_path))
            (model_path / "config.json").write_text("{}")

            with (
                patch("mlx_lm.convert") as mock_convert,
                patch(
                    "fusion_mlx.cli_convert._convert_pytorch_to_safetensors"
                ) as mock_pt,
            ):
                mock_convert.side_effect = fake_mlx_convert
                mock_pt.return_value = Path(tmpdir)

                _run_convert(
                    str(model_path),
                    mlx_path="/tmp/out",
                    quantize=False,
                    q_group_size=None,
                    q_bits=None,
                    q_mode="affine",
                    dtype=None,
                    upload_repo=None,
                    dequantize=False,
                    trust_remote_code=False,
                )

            assert call_count == 2
