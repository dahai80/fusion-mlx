# SPDX-License-Identifier: Apache-2.0
"""#1260: draft-model checkpoints are annotated, never standalone-servable.

`Qwen3.8-27B-DFlash2` (architecture DFlash2DraftModel, dflash_config
pointing at its teacher) was discovered as type "llm" and listed in
/v1/models. A chat request reached BatchedEngine.start, mlx_lm blew up on
the checkpoint's nested `rope_parameters` config (transformers 5.x
convention vs mlx-lm flat signature), and the client saw an opaque 500.

Fix: discovery classifies draft checkpoints as model_type "draft"
(architecture name contains "draft", or a dflash_config/draft_config
sub-config); get_engine refuses them with an explicit ModelNotFoundError
(→ 404); _mlx_compat flattens nested rope_parameters → flat rope_theta so
legitimate z-lab teacher checkpoints with the same config convention load.
"""

import json

import pytest

from fusion_mlx.pool.model_discovery import (
    _is_draft_model,
    detect_model_type,
    discover_models,
)


def _write_model(root, name, config, with_safetensors=False):
    model_dir = root / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(config))
    if with_safetensors:
        (model_dir / "model.safetensors").write_bytes(b"")
    return model_dir


_DFLASH_CONFIG = {
    "architectures": ["DFlash2DraftModel"],
    "model_type": "dflash2_draft",
    "num_hidden_layers": 5,
    "dflash_config": {"target_layer_ids": [10, 20, 30, 40, 47]},
}

_TEACHER_NESTED_ROPE = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
}


class TestDraftDetection:
    def test_dflash_config_marks_draft(self, tmp_path):
        model_dir = _write_model(tmp_path, "Qwen3.8-27B-DFlash2", _DFLASH_CONFIG)
        assert _is_draft_model(model_dir) is True
        assert detect_model_type(model_dir) == "draft"

    def test_architecture_substring_marks_draft(self, tmp_path):
        config = {"architectures": ["SomeFutureDraftModel"], "model_type": "qwen3"}
        model_dir = _write_model(tmp_path, "future-draft", config)
        assert _is_draft_model(model_dir) is True

    def test_plain_chat_model_not_draft(self, tmp_path):
        config = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}
        model_dir = _write_model(tmp_path, "qwen3-plain", config)
        assert _is_draft_model(model_dir) is False
        assert detect_model_type(model_dir) == "llm"

    def test_missing_config_not_draft(self, tmp_path):
        assert _is_draft_model(tmp_path) is False

    def test_discovery_annotates_draft(self, tmp_path):
        _write_model(
            tmp_path, "Qwen3.8-27B-DFlash2", _DFLASH_CONFIG, with_safetensors=True
        )
        _write_model(
            tmp_path,
            "qwen3-plain",
            {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"},
            with_safetensors=True,
        )
        discovered = discover_models(tmp_path)
        assert "Qwen3.8-27B-DFlash2" in discovered
        draft = discovered["Qwen3.8-27B-DFlash2"]
        assert draft.model_type == "draft"
        assert draft.engine_type == "draft"
        assert discovered["qwen3-plain"].model_type == "llm"


class TestGetEngineRefusesDraft:
    def test_refuses_with_explicit_404(self, tmp_path):
        import asyncio

        from fusion_mlx.exceptions import ModelNotFoundError
        from fusion_mlx.pool.engine_pool import EnginePool

        _write_model(
            tmp_path, "Qwen3.8-27B-DFlash2", _DFLASH_CONFIG, with_safetensors=True
        )
        discovered = discover_models(tmp_path)
        pool = EnginePool()
        pool._merge_discovered(discovered)
        with pytest.raises(ModelNotFoundError) as exc:
            asyncio.run(pool.get_engine("Qwen3.8-27B-DFlash2"))
        assert "draft" in str(exc.value).lower()
        assert "#1260" in str(exc.value)

    def test_chat_route_maps_to_404(self):
        # The chat handler maps ModelNotFoundError → HTTP 404; the draft
        # refusal rides the same mapping (see api/openai/chat.py).
        import inspect

        from fusion_mlx.api.openai import chat as chat_mod

        src = inspect.getsource(chat_mod)
        assert "ModelNotFoundError" in src
        assert "status_code=404" in src


class TestRopeParametersShim:
    def test_nested_rope_flattened(self, tmp_path):
        pytest.importorskip("mlx_lm.utils")
        from mlx_lm.utils import load_config

        from fusion_mlx._mlx_compat import _install_rope_parameters_shim

        assert _install_rope_parameters_shim() is True
        model_dir = _write_model(tmp_path, "zlab-teacher", _TEACHER_NESTED_ROPE)
        config = load_config(model_dir)
        assert config["rope_theta"] == 10000000

    def test_flat_rope_untouched(self, tmp_path):
        pytest.importorskip("mlx_lm.utils")
        from mlx_lm.utils import load_config

        from fusion_mlx._mlx_compat import _install_rope_parameters_shim

        assert _install_rope_parameters_shim() is True
        config_body = {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "rope_theta": 1000000.0,
        }
        model_dir = _write_model(tmp_path, "mlx-community-model", config_body)
        config = load_config(model_dir)
        assert config["rope_theta"] == 1000000.0
        assert "rope_parameters" not in config


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
