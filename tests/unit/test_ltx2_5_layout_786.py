# SPDX-License-Identifier: Apache-2.0
# #786: mlx-community flat-layout detection + split VAE/connector resolution.
# Synthetic temp-dir tests — no model download.
import json
from pathlib import Path

import pytest

from fusion_mlx.video.ltx2_5.text_encoder import (
    _load_connector_projection,
    _load_sharded_weights,
)
from fusion_mlx.video.ltx2_5.utils import (
    detect_layout,
    is_mlx_community_layout,
    is_split_layout,
    resolve_component,
)


def _make_mlxcomm_root(tmp_path: Path) -> Path:
    root = tmp_path / "mlxcomm"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "AudioVideo", "model_version": "2.5.0"})
    )
    (root / "connector.safetensors").write_text("")
    (root / "vae_encoder.safetensors").write_text("")
    (root / "vae_decoder.safetensors").write_text("")
    (root / "duration_head.safetensors").write_text("")
    (root / "spatial_upscaler_x2_v1_1.safetensors").write_text("")
    (root / "temporal_upscaler_x2_v1_0.safetensors").write_text("")
    te_dir = root / "gemma4-12b-ltx-v1"
    te_dir.mkdir()
    (te_dir / "config.json").write_text("{}")
    (te_dir / "tokenizer.json").write_text("{}")
    return root


def _make_flat_root(tmp_path: Path) -> Path:
    root = tmp_path / "flat"
    root.mkdir()
    (root / "split_model.json").write_text(json.dumps({"recipe": "ltx-2.5"}))
    (root / "transformer-distilled.safetensors").write_text("")
    (root / "connector.safetensors").write_text("")
    (root / "vae_encoder_conv.safetensors").write_text("")
    (root / "vae_decoder_conv.safetensors").write_text("")
    return root


def _make_comfy_root(tmp_path: Path) -> Path:
    root = tmp_path / "comfy"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "video"}))
    dm = root / "diffusion_models"
    dm.mkdir()
    (dm / "ltx-2.5-22b-distilled-transformer-bf16.safetensors").write_text("")
    return root


class TestMlxCommunityDetection:
    def test_is_mlx_community_layout_true(self, tmp_path):
        root = _make_mlxcomm_root(tmp_path)
        assert is_mlx_community_layout(root) is True

    def test_is_mlx_community_layout_false_for_comfy(self, tmp_path):
        root = _make_comfy_root(tmp_path)
        assert is_mlx_community_layout(root) is False

    def test_is_mlx_community_layout_false_for_flat(self, tmp_path):
        root = _make_flat_root(tmp_path)
        assert is_mlx_community_layout(root) is False

    def test_is_mlx_community_layout_false_when_diffusion_models_present(
        self, tmp_path
    ):
        root = _make_mlxcomm_root(tmp_path)
        (root / "diffusion_models").mkdir()
        assert is_mlx_community_layout(root) is False

    def test_is_mlx_community_layout_false_wrong_model_type(self, tmp_path):
        root = tmp_path / "wrong"
        root.mkdir()
        (root / "config.json").write_text(
            json.dumps({"model_type": "Video", "model_version": "2.5.0"})
        )
        assert is_mlx_community_layout(root) is False

    def test_is_mlx_community_layout_false_wrong_version(self, tmp_path):
        root = tmp_path / "wrong"
        root.mkdir()
        (root / "config.json").write_text(
            json.dumps({"model_type": "AudioVideo", "model_version": "2.0.0"})
        )
        assert is_mlx_community_layout(root) is False

    def test_is_mlx_community_layout_false_no_config(self, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()
        assert is_mlx_community_layout(root) is False


class TestDetectLayout:
    def test_detect_mlxcomm(self, tmp_path):
        assert detect_layout(_make_mlxcomm_root(tmp_path)) == "mlxcomm"

    def test_detect_flat(self, tmp_path):
        assert detect_layout(_make_flat_root(tmp_path)) == "flat"

    def test_detect_comfy_default(self, tmp_path):
        assert detect_layout(_make_comfy_root(tmp_path)) == "comfy"

    def test_detect_flat_priority_over_mlxcomm(self, tmp_path):
        root = _make_mlxcomm_root(tmp_path)
        (root / "split_model.json").write_text(json.dumps({"recipe": "ltx-2.5"}))
        (root / "transformer-distilled.safetensors").write_text("")
        assert detect_layout(root) == "flat"


class TestIsSplitLayout:
    def test_split_true_for_mlxcomm(self, tmp_path):
        assert is_split_layout(_make_mlxcomm_root(tmp_path)) is True

    def test_split_true_for_flat(self, tmp_path):
        assert is_split_layout(_make_flat_root(tmp_path)) is True

    def test_split_false_for_comfy(self, tmp_path):
        assert is_split_layout(_make_comfy_root(tmp_path)) is False


class TestResolveComponentMlxcomm:
    def test_connector(self, tmp_path):
        root = _make_mlxcomm_root(tmp_path)
        p = resolve_component(root, "connector")
        assert p.name == "connector.safetensors"

    def test_vae_encoder_no_conv_suffix(self, tmp_path):
        root = _make_mlxcomm_root(tmp_path)
        p = resolve_component(root, "video_vae_conv_encoder")
        assert p.name == "vae_encoder.safetensors"
        assert "_conv" not in p.name

    def test_vae_decoder_no_conv_suffix(self, tmp_path):
        root = _make_mlxcomm_root(tmp_path)
        p = resolve_component(root, "video_vae_conv_decoder")
        assert p.name == "vae_decoder.safetensors"

    def test_spatial_upscaler_v1_1(self, tmp_path):
        root = _make_mlxcomm_root(tmp_path)
        p = resolve_component(root, "spatial_upscaler")
        assert "v1_1" in p.name

    def test_temporal_upscaler_v1_0(self, tmp_path):
        root = _make_mlxcomm_root(tmp_path)
        p = resolve_component(root, "temporal_upscaler")
        assert "v1_0" in p.name

    def test_text_encoder_is_subdir(self, tmp_path):
        root = _make_mlxcomm_root(tmp_path)
        p = resolve_component(root, "text_encoder")
        assert p.name == "gemma4-12b-ltx-v1"
        assert p.is_dir()

    def test_transformer_raises_valueerror(self, tmp_path):
        root = _make_mlxcomm_root(tmp_path)
        with pytest.raises(ValueError, match="mlx-community layout has no transformer"):
            resolve_component(root, "transformer", variant="distilled")

    def test_transformer_dev_raises_valueerror(self, tmp_path):
        root = _make_mlxcomm_root(tmp_path)
        with pytest.raises(ValueError, match="mlx-community layout has no transformer"):
            resolve_component(root, "transformer", variant="dev")


class TestResolveComponentFlatRegression:
    def test_flat_connector(self, tmp_path):
        root = _make_flat_root(tmp_path)
        p = resolve_component(root, "connector")
        assert p.name == "connector.safetensors"

    def test_flat_vae_encoder_conv_suffix(self, tmp_path):
        root = _make_flat_root(tmp_path)
        p = resolve_component(root, "video_vae_conv_encoder")
        assert p.name == "vae_encoder_conv.safetensors"


class TestResolveComponentComfyRegression:
    def test_comfy_transformer_distilled(self, tmp_path):
        root = _make_comfy_root(tmp_path)
        p = resolve_component(root, "transformer", variant="distilled")
        assert p.parent.name == "diffusion_models"


class TestLoadShardedWeights:
    def test_merges_multiple_shards_via_index(self, tmp_path):
        shard_dir = tmp_path / "te"
        shard_dir.mkdir()
        import mlx.core as mx

        w1 = {"layer.0.weight": mx.zeros((2, 2))}
        w2 = {"layer.1.weight": mx.ones((3, 3))}
        mx.save_safetensors(str(shard_dir / "model-00001-of-00002"), w1)
        mx.save_safetensors(str(shard_dir / "model-00002-of-00002"), w2)
        index = {
            "weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors",
            }
        }
        (shard_dir / "model.safetensors.index.json").write_text(json.dumps(index))
        merged = _load_sharded_weights(shard_dir)
        assert set(merged.keys()) == {"layer.0.weight", "layer.1.weight"}
        assert merged["layer.1.weight"].shape == (3, 3)

    def test_empty_shard_dir_raises(self, tmp_path):
        shard_dir = tmp_path / "empty"
        shard_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="no text-encoder shards"):
            _load_sharded_weights(shard_dir)


class TestLoadConnectorProjection:
    def test_extracts_projection_strips_prefix(self, tmp_path):
        import mlx.core as mx

        connector = {
            "connector.text_embedding_projection.video_aggregate_embed.weight": mx.zeros(
                (4, 4)
            ),
            "connector.text_embedding_projection.video_aggregate_embed.bias": mx.zeros(
                (4,)
            ),
            "connector.transformer.other.weight": mx.zeros((2, 2)),
        }
        conn_prefix = tmp_path / "connector"
        mx.save_safetensors(str(conn_prefix), connector)
        conn_path = tmp_path / "connector.safetensors"
        proj = _load_connector_projection(conn_path)
        assert "video_aggregate_embed.weight" in proj
        assert "video_aggregate_embed.bias" in proj
        assert all(not k.startswith("connector.") for k in proj)
        assert len(proj) == 2

    def test_no_projection_keys_returns_empty(self, tmp_path):
        import mlx.core as mx

        connector = {"connector.transformer.weight": mx.zeros((2, 2))}
        conn_prefix = tmp_path / "connector"
        mx.save_safetensors(str(conn_prefix), connector)
        conn_path = tmp_path / "connector.safetensors"
        proj = _load_connector_projection(conn_path)
        assert proj == {}
