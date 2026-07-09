"""Test _patch_vlm_sanitize correctly guards RMSNorm +1.0 shift.

Upstream bug: mlx_vlm qwen3_5/qwen3_5_moe/minicpmv4_6 sanitize()
unconditionally adds 1.0 to RMSNorm weights. The patch calls orig first
(original +1.0 is applied), then subtracts 1.0 when should_shift=False.
Net effect: identity for no-MTP/no-unsanitized-conv1d checkpoints.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _reset_patch_state():
    import fusion_mlx.engines.vlm as vlm_mod
    for cls, orig in vlm_mod._VLM_ORIGINAL_SANITIZES.items():
        cls.sanitize = orig
    vlm_mod._VLM_ORIGINAL_SANITIZES.clear()
    vlm_mod._VLM_SANITIZE_PATCHED = False


class TestPatchVlmSanitize:

    def setup_method(self):
        _reset_patch_state()

    def teardown_method(self):
        _reset_patch_state()

    def test_patch_applies_to_qwen3_5(self):
        from fusion_mlx.engines.vlm import _patch_vlm_sanitize
        _patch_vlm_sanitize()
        import fusion_mlx.engines.vlm as vlm_mod
        assert vlm_mod._VLM_SANITIZE_PATCHED is True

    def test_no_double_patch(self):
        from fusion_mlx.engines.vlm import _patch_vlm_sanitize
        _patch_vlm_sanitize()
        _patch_vlm_sanitize()
        import fusion_mlx.engines.vlm as vlm_mod
        assert vlm_mod._VLM_SANITIZE_PATCHED is True

    def test_sanitize_identity_when_no_mtp_no_conv1d(self):
        import mlx.core as mx
        from fusion_mlx.engines.vlm import _patch_vlm_sanitize
        _patch_vlm_sanitize()

        try:
            from mlx_vlm.models.qwen3_5.qwen3_5 import Model
        except ImportError:
            pytest.skip("mlx_vlm qwen3_5 not installed")

        model = Model.__new__(Model)
        model["config"] = MagicMock()
        model["config"].text_config = MagicMock()
        model["config"].text_config.tie_word_embeddings = False

        weights = {
            "language_model.model.layers.0.input_layernorm.weight": mx.ones(8) * 1.5,
            "language_model.model.layers.0.post_attention_layernorm.weight": mx.ones(8) * 1.3,
            "language_model.model.norm.weight": mx.ones(8) * 1.2,
            "language_model.model.layers.0.self_attn.q_norm.weight": mx.ones(8) * 1.1,
            "language_model.model.layers.0.self_attn.k_norm.weight": mx.ones(8) * 1.1,
            "some_other.weight": mx.ones(8) * 5.0,
        }

        result = model.sanitize(weights)
        assert mx.allclose(
            result["language_model.model.layers.0.input_layernorm.weight"],
            mx.ones(8) * 1.5,
            atol=1e-5,
        ).item()
        assert mx.allclose(
            result["language_model.model.layers.0.post_attention_layernorm.weight"],
            mx.ones(8) * 1.3,
            atol=1e-5,
        ).item()
        assert mx.allclose(
            result["language_model.model.norm.weight"],
            mx.ones(8) * 1.2,
            atol=1e-5,
        ).item()
        assert mx.allclose(
            result["some_other.weight"],
            mx.ones(8) * 5.0,
            atol=1e-5,
        ).item()

    def test_sanitize_keeps_plus1_when_mtp_present(self):
        import mlx.core as mx
        from fusion_mlx.engines.vlm import _patch_vlm_sanitize
        _patch_vlm_sanitize()

        try:
            from mlx_vlm.models.qwen3_5.qwen3_5 import Model
        except ImportError:
            pytest.skip("mlx_vlm qwen3_5 not installed")

        model = Model.__new__(Model)
        model["config"] = MagicMock()
        model["config"].text_config = MagicMock()
        model["config"].text_config.tie_word_embeddings = False

        weights = {
            "mtp.0.input_layernorm.weight": mx.ones(8),
            "language_model.model.layers.0.input_layernorm.weight": mx.ones(8) * 0.5,
            "some_other.weight": mx.ones(8) * 3.0,
        }

        result = model.sanitize(weights)
        assert "mtp.0.input_layernorm.weight" not in result
        assert mx.allclose(
            result["language_model.model.layers.0.input_layernorm.weight"],
            mx.ones(8) * 1.5,
            atol=1e-5,
        ).item()

    def test_sanitize_keeps_plus1_when_unsanitized_conv1d(self):
        import mlx.core as mx
        from fusion_mlx.engines.vlm import _patch_vlm_sanitize
        _patch_vlm_sanitize()

        try:
            from mlx_vlm.models.qwen3_5.qwen3_5 import Model
        except ImportError:
            pytest.skip("mlx_vlm qwen3_5 not installed")

        model = Model.__new__(Model)
        model["config"] = MagicMock()
        model["config"].text_config = MagicMock()
        model["config"].text_config.tie_word_embeddings = False

        weights = {
            "language_model.model.layers.0.linear_attn.conv1d.weight": mx.ones((4, 8, 3)),
            "language_model.model.layers.0.input_layernorm.weight": mx.ones(8) * 0.5,
            "some_other.weight": mx.ones(8) * 3.0,
        }

        result = model.sanitize(weights)
        assert mx.allclose(
            result["language_model.model.layers.0.input_layernorm.weight"],
            mx.ones(8) * 1.5,
            atol=1e-5,
        ).item()
