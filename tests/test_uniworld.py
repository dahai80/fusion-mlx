# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def _skip_if_no_mlx():
    try:
        import mlx.core as mx  # noqa: F401
    except ImportError:
        pytest.skip("mlx not available")


# --- Config tests ---


class TestUniWorldConfig:
    def test_defaults(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.uniworld.config import UniWorldConfig

        cfg = UniWorldConfig()
        assert cfg.vlm_hidden_size == 3584
        assert cfg.siglip_hidden_size == 1152
        assert cfg.flux_hidden_size == 3072
        assert cfg.denoise_steps == 50
        assert cfg.guidance_scale == 3.5
        assert cfg.shortcut_scale == 0.5
        assert cfg.vlm_residual_image_factor == 0.3
        assert cfg.no_joint_with_t5 is False

    def test_properties(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.uniworld.config import UniWorldConfig

        cfg = UniWorldConfig(model_path="/tmp/test-uniworld")
        assert cfg.model_dir == Path("/tmp/test-uniworld")
        assert cfg.vlm_dir == Path("/tmp/test-uniworld/vlm")
        assert cfg.siglip_dir == Path("/tmp/test-uniworld/siglip")
        assert cfg.flux_dir == Path("/tmp/test-uniworld/flux")
        assert cfg.projectors_dir == Path("/tmp/test-uniworld/projectors")

    def test_mx_dtype(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.config import UniWorldConfig

        cfg = UniWorldConfig(dtype="float16")
        assert cfg.mx_dtype == mx.float16
        cfg_bf16 = UniWorldConfig(dtype="bfloat16")
        assert cfg_bf16.mx_dtype == mx.bfloat16

    def test_from_pretrained_no_config(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.uniworld.config import UniWorldConfig

        with tempfile.TemporaryDirectory() as tmp:
            cfg = UniWorldConfig.from_pretrained(tmp)
            assert cfg.model_path == tmp
            assert cfg.vlm_hidden_size == 3584

    def test_from_pretrained_with_config(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.uniworld.config import UniWorldConfig

        with tempfile.TemporaryDirectory() as tmp:
            config_data = {"vlm_hidden_size": 2048, "denoise_steps": 25}
            with open(Path(tmp) / "config.json", "w") as f:
                json.dump(config_data, f)
            cfg = UniWorldConfig.from_pretrained(tmp)
            assert cfg.vlm_hidden_size == 2048
            assert cfg.denoise_steps == 25

    def test_from_pretrained_ignores_unknown_keys(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.uniworld.config import UniWorldConfig

        with tempfile.TemporaryDirectory() as tmp:
            config_data = {"vlm_hidden_size": 3584, "unknown_key": 999}
            with open(Path(tmp) / "config.json", "w") as f:
                json.dump(config_data, f)
            cfg = UniWorldConfig.from_pretrained(tmp)
            assert not hasattr(cfg, "unknown_key")

    def test_projector_dims(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.uniworld.config import UniWorldConfig

        cfg = UniWorldConfig()
        assert cfg.denoise_projector_input == 3584
        assert cfg.denoise_projector_hidden == 9216
        assert cfg.denoise_projector_output == 3072
        assert cfg.vae_projector_input == 64
        assert cfg.vae_projector_output == 4096
        assert cfg.siglip_projector_input == 1152
        assert cfg.siglip_projector_output == 4096

    def test_task_head_dims(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.uniworld.config import UniWorldConfig

        cfg = UniWorldConfig()
        assert cfg.task_head_input == 3584
        assert cfg.task_head_hidden == 10240
        assert cfg.task_head_output == 2
        assert cfg.task_head_dropout == 0.3


# --- SigLIP2 tests ---


class TestSigLIP2PatchEmbed:
    def test_output_shape(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.siglip2 import SigLIP2PatchEmbed

        pe = SigLIP2PatchEmbed(image_size=512, patch_size=16, embed_dim=1152)
        x = mx.random.normal((1, 3, 512, 512))
        out = pe(x)
        mx.eval(out)
        assert out.shape == (1, 1024, 1152)

    def test_num_patches(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.uniworld.siglip2 import SigLIP2PatchEmbed

        pe = SigLIP2PatchEmbed(image_size=512, patch_size=16)
        assert pe.num_patches == 1024


class TestSigLIP2Attention:
    def test_output_shape(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.siglip2 import SigLIP2Attention

        attn = SigLIP2Attention(dim=1152, num_heads=16)
        x = mx.random.normal((1, 10, 1152))
        out = attn(x)
        mx.eval(out)
        assert out.shape == (1, 10, 1152)


class TestSigLIP2MLP:
    def test_output_shape(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.siglip2 import SigLIP2MLP

        mlp = SigLIP2MLP(dim=1152, hidden_dim=4304)
        x = mx.random.normal((1, 10, 1152))
        out = mlp(x)
        mx.eval(out)
        assert out.shape == (1, 10, 1152)


class TestSigLIP2EncoderLayer:
    def test_output_shape(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.siglip2 import SigLIP2EncoderLayer

        layer = SigLIP2EncoderLayer(dim=1152, num_heads=16, mlp_hidden=4304)
        x = mx.random.normal((1, 10, 1152))
        out = layer(x)
        mx.eval(out)
        assert out.shape == (1, 10, 1152)


class TestSigLIP2VisionTransformer:
    def test_instantiation(self):
        _skip_if_no_mlx()
        from fusion_mlx.video.uniworld.siglip2 import SigLIP2VisionTransformer

        vit = SigLIP2VisionTransformer(
            image_size=512,
            patch_size=16,
            embed_dim=1152,
            num_heads=16,
            num_layers=27,
            mlp_hidden=4304,
        )
        assert len(vit.layers) == 27

    def test_forward_shape(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.siglip2 import SigLIP2VisionTransformer

        vit = SigLIP2VisionTransformer(
            image_size=64,
            patch_size=16,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            mlp_hidden=256,
        )
        x = mx.random.normal((1, 3, 64, 64))
        out = vit(x)
        mx.eval(out)
        assert out.shape == (1, 16, 128)


class TestSigLIP2VisionEncoder:
    def test_encode_image_shape(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.siglip2 import (
            SigLIP2VisionEncoder,
            SigLIP2VisionTransformer,
        )

        vit = SigLIP2VisionTransformer(
            image_size=64,
            patch_size=16,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            mlp_hidden=256,
        )
        encoder = SigLIP2VisionEncoder(vit, dtype=mx.float16)
        x = mx.random.normal((1, 3, 64, 64))
        out = encoder.encode_image(x)
        mx.eval(out)
        assert out.shape == (1, 16, 128)

    def test_callable(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.siglip2 import (
            SigLIP2VisionEncoder,
            SigLIP2VisionTransformer,
        )

        vit = SigLIP2VisionTransformer(
            image_size=64,
            patch_size=16,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            mlp_hidden=256,
        )
        encoder = SigLIP2VisionEncoder(vit, dtype=mx.float16)
        x = mx.random.normal((1, 3, 64, 64))
        out = encoder(x)
        mx.eval(out)
        assert out.shape[0] == 1


class TestSigLIP2WeightRemap:
    def test_remap_vision_model_prefix(self):
        from fusion_mlx.video.uniworld.siglip2 import _remap_siglip_weights

        weights = {"vision_model.patch_embedding.weight": np.zeros((1,))}
        result = _remap_siglip_weights(weights)
        assert "patch_embed.proj.weight" in result

    def test_remap_model_vision_prefix(self):
        from fusion_mlx.video.uniworld.siglip2 import _remap_siglip_weights

        weights = {
            "model.vision_model.encoder.layers.0.self_attn.qkv.weight": np.zeros((1,))
        }
        result = _remap_siglip_weights(weights)
        assert "layers.0.attn.qkv.weight" in result

    def test_remap_norm_names(self):
        from fusion_mlx.video.uniworld.siglip2 import _remap_siglip_weights

        weights = {
            "vision_model.encoder.layers.0.layer_norm1.weight": np.zeros((1,)),
            "vision_model.encoder.layers.0.layer_norm2.weight": np.zeros((1,)),
        }
        result = _remap_siglip_weights(weights)
        assert "layers.0.norm1.weight" in result
        assert "layers.0.norm2.weight" in result

    def test_skip_text_model(self):
        from fusion_mlx.video.uniworld.siglip2 import _remap_siglip_weights

        weights = {"text_model.encoder.layers.0.weight": np.zeros((1,))}
        result = _remap_siglip_weights(weights)
        assert len(result) == 0

    def test_skip_logit_scale(self):
        from fusion_mlx.video.uniworld.siglip2 import _remap_siglip_weights

        weights = {"logit_scale": np.zeros((1,))}
        result = _remap_siglip_weights(weights)
        assert len(result) == 0

    def test_remap_post_layernorm(self):
        from fusion_mlx.video.uniworld.siglip2 import _remap_siglip_weights

        weights = {
            "vision_model.encoder.layers.0.post_layernorm.weight": np.zeros((1,))
        }
        result = _remap_siglip_weights(weights)
        assert "layers.0.norm.weight" in result


# --- Projector tests ---


class TestDenoiseProjector:
    def test_output_shape(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.projectors import DenoiseProjector

        proj = DenoiseProjector(input_dim=3584, hidden_dim=9216, output_dim=3072)
        x = mx.random.normal((1, 10, 3584))
        out = proj(x)
        mx.eval(out)
        assert out.shape == (1, 10, 3072)


class TestVAEProjector:
    def test_output_shape(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.projectors import VAEProjector

        proj = VAEProjector(input_dim=64, hidden_dim=3072, output_dim=4096)
        x = mx.random.normal((1, 10, 64))
        out = proj(x)
        mx.eval(out)
        assert out.shape == (1, 10, 4096)


class TestSigLIPProjector:
    def test_output_shape(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.projectors import SigLIPProjector

        proj = SigLIPProjector(input_dim=1152, hidden_dim=12288, output_dim=4096)
        x = mx.random.normal((1, 10, 1152))
        out = proj(x)
        mx.eval(out)
        assert out.shape == (1, 10, 4096)


class TestTaskHead:
    def test_output_shape(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.projectors import TaskHead

        head = TaskHead(input_dim=3584, hidden_dim=10240, output_dim=2)
        x = mx.random.normal((1, 1, 3584))
        out = head(x)
        mx.eval(out)
        assert out.shape == (1, 1, 2)

    def test_output_2_classes(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.projectors import TaskHead

        head = TaskHead(input_dim=3584, hidden_dim=10240, output_dim=2)
        x = mx.random.normal((1, 1, 3584))
        out = head(x)
        mx.eval(out)
        assert out.shape[-1] == 2


class TestUniWorldProjectors:
    def test_all_three_projectors(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.projectors import UniWorldProjectors

        proj = UniWorldProjectors()
        vlm_h = mx.random.normal((1, 10, 3584))
        vae_h = mx.random.normal((1, 10, 64))
        siglip_h = mx.random.normal((1, 10, 1152))
        d, v, s = proj(vlm_h, vae_h, siglip_h)
        mx.eval(d, v, s)
        assert d.shape == (1, 10, 3072)
        assert v.shape == (1, 10, 4096)
        assert s.shape == (1, 10, 4096)


class TestProjectorWeightRemap:
    def test_remap_denoise_prefix(self):
        from fusion_mlx.video.uniworld.projectors import _remap_projector_weights

        weights = {
            "model.denoise_tower.denoise_projector.0.weight": np.zeros((1,)),
            "model.denoise_tower.denoise_projector.2.weight": np.zeros((1,)),
        }
        result = _remap_projector_weights(weights)
        assert "denoise_projector.fc1.weight" in result
        assert "denoise_projector.fc2.weight" in result

    def test_remap_vae_prefix(self):
        from fusion_mlx.video.uniworld.projectors import _remap_projector_weights

        weights = {
            "model.denoise_tower.vae_projector.0.weight": np.zeros((1,)),
            "model.denoise_tower.vae_projector.2.bias": np.zeros((1,)),
        }
        result = _remap_projector_weights(weights)
        assert "vae_projector.fc1.weight" in result
        assert "vae_projector.fc2.bias" in result

    def test_remap_siglip_prefix(self):
        from fusion_mlx.video.uniworld.projectors import _remap_projector_weights

        weights = {
            "model.denoise_tower.siglip_projector.0.weight": np.zeros((1,)),
        }
        result = _remap_projector_weights(weights)
        assert "siglip_projector.fc1.weight" in result

    def test_already_prefixed(self):
        from fusion_mlx.video.uniworld.projectors import _remap_projector_weights

        weights = {"denoise_projector.0.weight": np.zeros((1,))}
        result = _remap_projector_weights(weights)
        assert "denoise_projector.fc1.weight" in result


# --- Feature merge tests ---


class TestFindTrueBlocks:
    def test_single_block(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.feature_merge import find_true_blocks

        mask = mx.array([[False, True, True, False, False]])
        blocks = find_true_blocks(mask)
        assert len(blocks) == 1
        assert blocks[0] == [(1, 3)]

    def test_multiple_blocks(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.feature_merge import find_true_blocks

        mask = mx.array([[True, False, True, True, False]])
        blocks = find_true_blocks(mask)
        assert blocks[0] == [(0, 1), (2, 4)]

    def test_empty_mask(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.feature_merge import find_true_blocks

        mask = mx.array([[False, False, False]])
        blocks = find_true_blocks(mask)
        assert blocks[0] == []

    def test_all_true(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.feature_merge import find_true_blocks

        mask = mx.array([[True, True, True]])
        blocks = find_true_blocks(mask)
        assert blocks[0] == [(0, 3)]

    def test_1d_input(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.feature_merge import find_true_blocks

        mask = mx.array([True, True, False])
        blocks = find_true_blocks(mask)
        assert blocks[0] == [(0, 2)]


class TestFindAllTokenPositions:
    def test_basic(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.feature_merge import find_all_token_positions

        ids = mx.array([[1, 2, 3, 2, 5]])
        positions = find_all_token_positions(ids, 2)
        assert positions[0] == [1, 3]

    def test_not_found(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.feature_merge import find_all_token_positions

        ids = mx.array([[1, 3, 5]])
        positions = find_all_token_positions(ids, 99)
        assert positions[0] == []


class TestInsertImgToVlm:
    def test_no_image_end_token(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.feature_merge import insert_img_to_vlm

        vlm_h = mx.random.normal((1, 10, 64))
        siglip_h = mx.random.normal((1, 5, 64))
        ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        out = insert_img_to_vlm(vlm_h, siglip_h, ids, image_end_token_id=999)
        mx.eval(out)
        assert out.shape == vlm_h.shape

    def test_with_image_end_token(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.feature_merge import insert_img_to_vlm

        IMAGE_END = 151646
        vlm_h = mx.zeros((1, 10, 64))
        siglip_h = mx.ones((1, 3, 64))
        ids = mx.array([[1, 2, IMAGE_END, 4, 5, IMAGE_END, 7, 8, 9, 10]])
        out = insert_img_to_vlm(vlm_h, siglip_h, ids, image_end_token_id=IMAGE_END)
        mx.eval(out)
        assert out.shape == (1, 10, 64)


class TestApplyShortcutBlend:
    def test_blend_at_mask(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.feature_merge import apply_shortcut_blend

        vlm_h = mx.zeros((1, 5, 8))
        shortcut = mx.ones((1, 5, 8))
        mask = mx.array([[False, True, False, True, False]])
        out = apply_shortcut_blend(vlm_h, shortcut, mask, scale=0.5)
        mx.eval(out)
        assert out.shape == (1, 5, 8)

    def test_none_shortcut(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.feature_merge import apply_shortcut_blend

        vlm_h = mx.zeros((1, 5, 8))
        out = apply_shortcut_blend(vlm_h, None, mx.array([[True]]), scale=0.5)
        mx.eval(out)
        assert out.shape == vlm_h.shape

    def test_zero_scale(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.feature_merge import apply_shortcut_blend

        vlm_h = mx.ones((1, 5, 8))
        shortcut = mx.zeros((1, 5, 8))
        mask = mx.array([[True] * 5])
        out = apply_shortcut_blend(vlm_h, shortcut, mask, scale=0.0)
        mx.eval(out)
        assert out.shape == (1, 5, 8)


class TestApplyResidualImageFactor:
    def test_residual_at_mask(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.feature_merge import apply_residual_image_factor

        vlm_h = mx.ones((1, 5, 8))
        original = mx.ones((1, 5, 8))
        mask = mx.array([[True, False, True, False, True]])
        out = apply_residual_image_factor(vlm_h, original, mask, factor=0.3)
        mx.eval(out)
        assert out.shape == (1, 5, 8)

    def test_none_original(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.feature_merge import apply_residual_image_factor

        vlm_h = mx.ones((1, 5, 8))
        out = apply_residual_image_factor(vlm_h, None, mx.array([[True]]), factor=0.3)
        mx.eval(out)
        assert out.shape == vlm_h.shape

    def test_zero_factor(self):
        _skip_if_no_mlx()
        import mlx.core as mx

        from fusion_mlx.video.uniworld.feature_merge import apply_residual_image_factor

        vlm_h = mx.ones((1, 5, 8))
        original = mx.ones((1, 5, 8))
        mask = mx.array([[True] * 5])
        out = apply_residual_image_factor(vlm_h, original, mask, factor=0.0)
        mx.eval(out)
        assert out.shape == (1, 5, 8)


# --- Backend tests ---


class TestUniWorldBackend:
    def test_detect_positive(self):
        from fusion_mlx.engines.video_backends.uniworld import UniWorldBackend

        assert UniWorldBackend.detect("uniworld-v1")
        assert UniWorldBackend.detect("UniWorld-V1-7B")
        assert UniWorldBackend.detect("univa-model")

    def test_detect_negative(self):
        from fusion_mlx.engines.video_backends.uniworld import UniWorldBackend

        assert not UniWorldBackend.detect("wan2.1-14b")
        assert not UniWorldBackend.detect("ltx-video")
        assert not UniWorldBackend.detect("cogvideox-2b")

    def test_name(self):
        from fusion_mlx.engines.video_backends.uniworld import UniWorldBackend

        assert UniWorldBackend.name == "uniworld"

    def test_constraints(self):
        from fusion_mlx.engines.video_backends.uniworld import UniWorldBackend

        backend = UniWorldBackend("uniworld-v1")
        c = backend.constraints()
        assert c.supports_i2v is True
        assert c.max_n == 1
        assert c.dim_divisibility == 16

    def test_constraints_num_frames(self):
        from fusion_mlx.engines.video_backends.uniworld import UniWorldBackend

        backend = UniWorldBackend("uniworld-v1")
        c = backend.constraints()
        assert c.num_frames_validator(1) is True
        assert c.num_frames_validator(2) is False

    def test_registry_resolve(self):
        from fusion_mlx.engines.video_backends import UniWorldBackend, resolve_backend

        backend = resolve_backend("uniworld-v1")
        assert isinstance(backend, UniWorldBackend)

    def test_registry_alias(self):
        from fusion_mlx.engines.video_backends import UniWorldBackend, resolve_backend

        backend = resolve_backend("test-model", explicit="uniworld")
        assert isinstance(backend, UniWorldBackend)

    def test_registry_alias_univa(self):
        from fusion_mlx.engines.video_backends import UniWorldBackend, resolve_backend

        backend = resolve_backend("test-model", explicit="univa")
        assert isinstance(backend, UniWorldBackend)

    def test_get_stats(self):
        from fusion_mlx.engines.video_backends.uniworld import UniWorldBackend

        backend = UniWorldBackend("uniworld-v1")
        stats = backend.get_stats()
        assert stats["backend"] == "uniworld"
        assert stats["loaded"] is False
        assert stats["vlm_loaded"] is False

    def test_classify_defaults_to_generation(self):
        _skip_if_no_mlx()
        from fusion_mlx.engines.video_backends.uniworld import UniWorldBackend

        backend = UniWorldBackend("uniworld-v1")
        result = backend._classify_task("generate an image")
        assert result is not None


# --- Weight converter tests ---


class TestWeightConverter:
    def test_remap_identity(self):
        from fusion_mlx.video.uniworld.weight_converter import convert_uniworld_weights

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source"
            out = Path(tmp) / "output"
            src.mkdir()
            result = convert_uniworld_weights(str(src), str(out))
            assert isinstance(result, dict)

    def test_empty_source(self):
        from fusion_mlx.video.uniworld.weight_converter import convert_uniworld_weights

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source"
            out = Path(tmp) / "output"
            src.mkdir()
            result = convert_uniworld_weights(str(src), str(out))
            assert "errors" in result

    def test_creates_subdirs(self):
        from fusion_mlx.video.uniworld.weight_converter import convert_uniworld_weights

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source"
            out = Path(tmp) / "output"
            src.mkdir()
            convert_uniworld_weights(str(src), str(out))
            assert (out / "vlm").exists()
            assert (out / "siglip").exists()
            assert (out / "flux").exists()
            assert (out / "projectors").exists()

    def test_meta_json_created(self):
        from fusion_mlx.video.uniworld.weight_converter import convert_uniworld_weights

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source"
            out = Path(tmp) / "output"
            src.mkdir()
            convert_uniworld_weights(str(src), str(out))
            meta_path = out / "conversion_meta.json"
            assert meta_path.exists()
            with open(meta_path) as f:
                meta = json.load(f)
            assert "dtype" in meta


# --- Package import test ---


class TestPackageImport:
    def test_import_uniworld(self):
        from fusion_mlx.video import uniworld

        assert uniworld is not None

    def test_submodules(self):
        pass

    def test_public_api(self):
        pass
