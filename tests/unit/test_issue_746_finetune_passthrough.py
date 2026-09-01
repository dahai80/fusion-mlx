# Tests for #746: FineTuneConfig passthrough of three SFT hyperparameters
# (weight_decay, max_grad_norm, lora_target_modules) to mlx-lm 0.31.3.
#
# Importers/callers: FineTuneConfig re-exported by fusion_mlx.training.__init__,
# constructed by fine_tune_route.create_fine_tune_job(**config_body).
# Affected API: POST /admin/api/fine-tune/jobs config body.
# Data schemas: FineTuneConfig dataclass (weight_decay: float,
# max_grad_norm: float | None, lora_target_modules: list[str] | None).
#
# Strategy mirrors test_issue_402_qlora_config.py: static config + validate()
# guards only (no real-model load). Real-model end-to-end is gated behind
# FUSION_MLX_REAL_MODEL_TESTS and run separately; these unit tests exercise
# the production guards and the to_mlx_args wiring deterministically.

import pytest

from fusion_mlx.training.service import FineTuneConfig


class TestFineTuneConfig746Fields:
    def test_defaults_backward_compatible(self):
        cfg = FineTuneConfig()
        assert cfg.weight_decay == 0.0
        assert cfg.max_grad_norm is None
        assert cfg.lora_target_modules is None

    def test_weight_decay_settable(self):
        cfg = FineTuneConfig(weight_decay=0.01)
        assert cfg.weight_decay == 0.01

    def test_max_grad_norm_settable(self):
        cfg = FineTuneConfig(max_grad_norm=1.0)
        assert cfg.max_grad_norm == 1.0

    def test_lora_target_modules_settable(self):
        cfg = FineTuneConfig(lora_target_modules=["q_proj", "v_proj"])
        assert cfg.lora_target_modules == ["q_proj", "v_proj"]


class TestToMlxArgs746Wiring:
    def test_weight_decay_threaded_into_optimizer_config(self):
        cfg = FineTuneConfig(optimizer="adamw", weight_decay=0.05)
        args = cfg.to_mlx_args("/tmp/adt", "/tmp/data", "/tmp/model")
        assert args.optimizer_config["adamw"]["weight_decay"] == 0.05

    def test_weight_decay_threaded_for_all_supporting_optimizers(self):
        cfg = FineTuneConfig(weight_decay=0.02)
        args = cfg.to_mlx_args("/tmp/adt", "/tmp/data", "/tmp/model")
        for name in ("adamw", "muon", "sgd", "adafactor"):
            assert args.optimizer_config[name]["weight_decay"] == 0.02

    def test_adam_optimizer_config_has_no_weight_decay_key(self):
        # Adam has no weight_decay arg; the adam entry stays empty so the
        # service-layer optimizer build never passes it (issue #746).
        cfg = FineTuneConfig(optimizer="adam", weight_decay=0.0)
        args = cfg.to_mlx_args("/tmp/adt", "/tmp/data", "/tmp/model")
        assert args.optimizer_config["adam"] == {}

    def test_default_weight_decay_zero_keeps_prior_behavior(self):
        args = FineTuneConfig().to_mlx_args("/tmp/adt", "/tmp/data", "/tmp/model")
        for name in ("adamw", "muon", "sgd", "adafactor"):
            assert args.optimizer_config[name]["weight_decay"] == 0.0

    def test_lora_target_modules_carried_as_raw_names(self):
        # to_mlx_args only carries raw target-module names so the saved
        # adapter_config.json records intent; keys are resolved post-load in
        # _execute_training from the loaded model's module tree.
        cfg = FineTuneConfig(lora_target_modules=["q_proj", "v_proj"])
        args = cfg.to_mlx_args("/tmp/adt", "/tmp/data", "/tmp/model")
        assert args.lora_parameters["target_modules"] == ["q_proj", "v_proj"]

    def test_no_lora_target_modules_omits_key(self):
        args = FineTuneConfig().to_mlx_args("/tmp/adt", "/tmp/data", "/tmp/model")
        assert "target_modules" not in args.lora_parameters


class TestExecuteTraining746Validation:
    # Exercise the REAL production guard via FineTuneConfig.validate() — the
    # single source of truth that _execute_training calls before model load.
    # Rule 9: tests must fail if the production raise is deleted or its
    # message drifts, unlike inline-replicated tests which pass regardless.

    def test_weight_decay_with_adam_raises(self):
        # mlx.optimizers.Adam has no weight_decay arg; a non-zero value with
        # the plain Adam optimizer is a config error — fail visibly (Rule 12).
        cfg = FineTuneConfig(optimizer="adam", weight_decay=0.01)
        with pytest.raises(ValueError, match="weight_decay") as exc:
            cfg.validate()
        assert "adam" in str(exc.value).lower()
        assert "#746" in str(exc.value)

    def test_weight_decay_zero_with_adam_ok(self):
        # weight_decay=0.0 with Adam is fine (default, prior behavior).
        FineTuneConfig(optimizer="adam", weight_decay=0.0).validate()

    def test_weight_decay_with_adamw_ok(self):
        FineTuneConfig(optimizer="adamw", weight_decay=0.01).validate()

    def test_weight_decay_with_sgd_ok(self):
        FineTuneConfig(optimizer="sgd", weight_decay=0.01).validate()

    def test_max_grad_norm_non_positive_raises(self):
        for bad in (0.0, -1.0):
            cfg = FineTuneConfig(max_grad_norm=bad)
            with pytest.raises(ValueError, match="max_grad_norm"):
                cfg.validate()

    def test_max_grad_norm_positive_ok(self):
        FineTuneConfig(max_grad_norm=1.0).validate()

    def test_lora_target_modules_with_full_raises(self):
        # full fine-tuning unfreezes layers; no LoRA adapters, so targeting
        # has no effect — fail visibly rather than silently ignore.
        cfg = FineTuneConfig(fine_tune_type="full", lora_target_modules=["q_proj"])
        with pytest.raises(ValueError, match="lora_target_modules") as exc:
            cfg.validate()
        assert "full" in str(exc.value)

    def test_lora_target_modules_with_lora_ok(self):
        FineTuneConfig(
            fine_tune_type="lora", lora_target_modules=["q_proj", "v_proj"]
        ).validate()

    def test_lora_target_modules_with_qlora_ok(self):
        FineTuneConfig(
            fine_tune_type="qlora",
            quantize_base=True,
            lora_target_modules=["q_proj"],
        ).validate()

    def test_all_three_passthrough_fields_valid_together(self):
        FineTuneConfig(
            optimizer="adamw",
            weight_decay=0.01,
            max_grad_norm=1.0,
            lora_target_modules=["q_proj", "v_proj"],
        ).validate()
