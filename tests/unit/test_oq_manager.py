# SPDX-License-Identifier: Apache-2.0
"""Tests for the OQManager admin component."""

import json

import pytest

from fusion_mlx.admin.oq_manager import OQManager


@pytest.fixture
def fp_model_dir(tmp_path):
    """One directory with a full-precision (quantizable) source model."""
    d = tmp_path / "models1"
    d.mkdir()
    model = d / "Llama-3B"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "num_hidden_layers": 32,
            }
        )
    )
    (model / "model.safetensors").write_bytes(b"\x00" * 4096)
    return d


@pytest.fixture
def second_fp_model_dir(tmp_path):
    """A second directory holding a different full-precision model."""
    d = tmp_path / "models2"
    d.mkdir()
    model = d / "Qwen-7B"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "num_hidden_layers": 28,
            }
        )
    )
    (model / "model.safetensors").write_bytes(b"\x00" * 4096)
    return d


class TestOQManagerUpdateModelDirs:
    @pytest.mark.asyncio
    async def test_picks_up_added_dir(self, fp_model_dir, second_fp_model_dir):
        # Mirrors the real Settings UI flow: server starts with one model
        # directory, the user adds a second one at runtime via Settings, and
        # _apply_model_dirs_runtime calls update_model_dirs(). Without that
        # call, models in the newly added directory never show up in the oQ
        # Quantization "Source Model" dropdown.
        manager = OQManager(model_dirs=[str(fp_model_dir)])
        source_before, _ = await manager.list_quantizable_models()
        names_before = {m["name"] for m in source_before}
        assert "Llama-3B" in names_before
        assert "Qwen-7B" not in names_before

        manager.update_model_dirs([str(fp_model_dir), str(second_fp_model_dir)])

        source_after, _ = await manager.list_quantizable_models()
        names_after = {m["name"] for m in source_after}
        assert "Llama-3B" in names_after
        assert "Qwen-7B" in names_after

    def test_output_dir_tracks_primary_dir(self, fp_model_dir, second_fp_model_dir):
        # Output is always written to the primary (first) directory.
        manager = OQManager(model_dirs=[str(fp_model_dir)])
        assert manager._output_dir == fp_model_dir

        manager.update_model_dirs([str(second_fp_model_dir), str(fp_model_dir)])
        assert manager._output_dir == second_fp_model_dir


class TestOQManagerMtpDetection:
    def _write_model(self, root, name, *, index_weight_map=None):
        model = root / name
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5",
                    "text_config": {
                        "model_type": "qwen3_5_text",
                        "num_hidden_layers": 32,
                        "mtp_num_hidden_layers": 1,
                    },
                }
            )
        )
        (model / "model.safetensors").write_bytes(b"\x00" * 4096)
        if index_weight_map is not None:
            (model / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "weight_map": index_weight_map,
                    }
                )
            )
        return model

    @pytest.mark.asyncio
    async def test_config_only_mtp_is_not_reported_as_preservable(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        self._write_model(root, "QwenPawLike")

        manager = OQManager(model_dirs=[str(root)])
        source_models, _ = await manager.list_quantizable_models()

        [model] = source_models
        assert model["has_mtp_heads"] is False

    @pytest.mark.asyncio
    async def test_mtp_weight_index_is_reported_as_preservable(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        self._write_model(
            root,
            "QwenMtp",
            index_weight_map={
                "language_model.mtp.fc.weight": "model.safetensors",
            },
        )

        manager = OQManager(model_dirs=[str(root)])
        source_models, _ = await manager.list_quantizable_models()

        [model] = source_models
        assert model["has_mtp_heads"] is True

    @pytest.mark.asyncio
    async def test_start_quantization_disables_preserve_mtp_without_weights(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "models"
        root.mkdir()
        self._write_model(root, "QwenPawLike")

        manager = OQManager(model_dirs=[str(root)])

        async def _noop_run(task_id):
            return None

        monkeypatch.setattr(manager, "_run_quantization", _noop_run)

        task = await manager.start_quantization(
            str(root / "QwenPawLike"),
            4,
            preserve_mtp=True,
        )
        await manager._active_tasks[task.task_id]

        assert task.preserve_mtp is False
        assert task.output_name == "QwenPawLike-oQ4"


class TestOQManagerDtypeSupport:
    @pytest.mark.asyncio
    async def test_start_quantization_rejects_deepseek_v4_float16(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        model = root / "DeepSeek-V4-Flash"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps({"model_type": "deepseek_v4"}),
            encoding="utf-8",
        )

        manager = OQManager(model_dirs=[str(root)])

        with pytest.raises(ValueError, match="dtype=float16.*deepseek_v4"):
            await manager.start_quantization(str(model), 4, dtype="float16")

        assert manager._tasks == {}
        assert not (root / "DeepSeek-V4-Flash-oQ4-fp16").exists()


class TestOQManagerHfCacheScan:
    """#1337: HF cache layout (models--org--name/snapshots/<rev>) must be scanned."""

    @pytest.fixture
    def hf_cache_model_dir(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        repo = d / "models--mlx-community--Qwen3.8-27B-8bit"
        snap = repo / "snapshots" / "815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5",
                    "num_hidden_layers": 48,
                }
            )
        )
        (snap / "model.safetensors").write_bytes(b"\x00" * 4096)
        return d

    async def test_hf_cache_model_listed_with_repo_display_name(
        self, hf_cache_model_dir
    ):
        manager = OQManager(model_dirs=[str(hf_cache_model_dir)])
        source_models, all_models = await manager.list_quantizable_models()
        names = [m["name"] for m in all_models]
        assert "models--mlx-community--Qwen3.8-27B-8bit" in names
        paths = [m["path"] for m in all_models]
        assert any("snapshots" in p for p in paths)

    async def test_hf_cache_model_in_source_models_if_quantizable(
        self, hf_cache_model_dir
    ):
        manager = OQManager(model_dirs=[str(hf_cache_model_dir)])
        source_models, _ = await manager.list_quantizable_models()
        assert any(
            m["name"] == "models--mlx-community--Qwen3.8-27B-8bit"
            for m in source_models
        )


class TestOQValidateModelPath:
    """#1337: _validate_model_path must accept configured model dirs, not a
    hardcoded ~/.cache/fusion-mlx/models root."""

    def _manager(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        return OQManager(model_dirs=[str(d)]), d

    def test_accepts_path_inside_configured_model_dir(self, tmp_path, monkeypatch):
        from fusion_mlx.admin import helpers as helpers_module
        from fusion_mlx.admin import oq as oq_module

        manager, d = self._manager(tmp_path)
        monkeypatch.setattr(helpers_module, "_oq_manager", manager)
        model = d / "Qwen-27B"
        model.mkdir()
        resolved = oq_module._validate_model_path(str(model))
        assert resolved == str(model.resolve())

    def test_rejects_path_outside_model_dirs(self, tmp_path, monkeypatch):
        from fusion_mlx.admin import helpers as helpers_module
        from fusion_mlx.admin import oq as oq_module

        manager, _ = self._manager(tmp_path)
        monkeypatch.setattr(helpers_module, "_oq_manager", manager)
        outside = tmp_path / "elsewhere" / "model"
        outside.mkdir(parents=True)
        with pytest.raises(Exception) as exc_info:
            oq_module._validate_model_path(str(outside))
        assert "403" in str(exc_info.value) or "model_path" in str(exc_info.value)


class TestOQManagerLiveBinding:
    """#1337: oq routes must see a manager set AFTER the route module import.

    The admin routers are imported before the server initializes the
    manager, so a by-value ``from .helpers import _oq_manager`` binds None
    forever and every oq route silently degrades to the empty path.
    """

    def test_set_after_import_is_visible_to_routes(self, tmp_path, monkeypatch):
        from fusion_mlx.admin import helpers as helpers_module
        from fusion_mlx.admin import oq as oq_module

        monkeypatch.setattr(helpers_module, "_oq_manager", None)
        assert oq_module._get_oq_manager() is None
        manager, _ = TestOQValidateModelPath()._manager(tmp_path)
        monkeypatch.setattr(helpers_module, "_oq_manager", manager)
        assert oq_module._get_oq_manager() is manager


class TestOQRecipeValidation:
    """#1337: unsupported recipes must fail fast; supported ones pass through."""

    async def test_unimplemented_recipe_rejected(self, fp_model_dir):
        manager = OQManager(model_dirs=[str(fp_model_dir)])
        source = fp_model_dir / "Llama-3B"
        with pytest.raises(ValueError, match="not implemented"):
            await manager.start_quantization(
                model_path=str(source), recipe="quant2_all"
            )

    async def test_supported_recipe_accepted(self, fp_model_dir, monkeypatch):
        manager = OQManager(model_dirs=[str(fp_model_dir)])
        monkeypatch.setattr(manager, "_run_mlx_convert", lambda *a, **k: None)
        source = fp_model_dir / "Llama-3B"
        task = await manager.start_quantization(
            model_path=str(source), recipe="mixed_2_6"
        )
        assert task.recipe == "mixed_2_6"
        assert task.status.value in ("pending", "queued", "loading", "running")
