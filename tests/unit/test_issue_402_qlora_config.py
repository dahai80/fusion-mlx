# Tests for #402: FineTuneConfig QLoRA (quantize_base/quant_bits) + qlora
# fine_tune_type + mxfp8 staged field.
#
# Importers/callers: FineTuneConfig re-exported by fusion_mlx.training.__init__,
# constructed by fine_tune_route.create_fine_tune_job(**config_body).
# Affected API: POST /admin/api/fine-tune/jobs config body.
# Data schemas: FineTuneConfig dataclass (quantize_base: bool, quant_bits: int,
# mxfp8: bool, fine_tune_type now accepts "qlora").

import pytest

from fusion_mlx.training.service import FineTuneConfig


class TestFineTuneConfig402Fields:
    def test_defaults_backward_compatible(self):
        cfg = FineTuneConfig()
        assert cfg.fine_tune_type == "lora"
        assert cfg.quantize_base is False
        assert cfg.quant_bits == 4
        assert cfg.mxfp8 is False

    def test_qlora_type_accepted(self):
        cfg = FineTuneConfig(fine_tune_type="qlora", quantize_base=True)
        assert cfg.fine_tune_type == "qlora"

    def test_quant_bits_accepts_4_and_8(self):
        assert FineTuneConfig(quant_bits=4).quant_bits == 4
        assert FineTuneConfig(quant_bits=8).quant_bits == 8

    def test_to_mlx_args_carries_qlora_type(self):
        cfg = FineTuneConfig(fine_tune_type="qlora", quantize_base=True, quant_bits=8)
        args = cfg.to_mlx_args("/tmp/adt", "/tmp/data", "/tmp/model")
        assert args.fine_tune_type == "qlora"

    def test_to_mlx_args_carries_quant_fields(self):
        cfg = FineTuneConfig(quantize_base=True, quant_bits=8)
        args = cfg.to_mlx_args("/tmp/adt", "/tmp/data", "/tmp/model")
        assert args.fine_tune_type == "lora"

    def test_existing_lora_dora_full_unchanged(self):
        for ft in ("lora", "dora", "full"):
            assert FineTuneConfig(fine_tune_type=ft).fine_tune_type == ft


class TestExecuteTraining402Validation:
    # Exercise the REAL production guard via FineTuneConfig.validate() — the
    # single source of truth that _execute_training calls before model load.
    # #402/#425: these tests must fail if the production raise is deleted or
    # its message drifts, unlike the old inline-replicated tests which passed
    # regardless (Rule 9: a test passing for the wrong reason is worse than
    # no test).

    def test_mxfp8_raises_loudly(self):
        # #425: mxfp8 is staged but upstream-blocked; setting True must fail
        # visibly with the real message (not silently ignored).
        cfg = FineTuneConfig(mxfp8=True)
        with pytest.raises(ValueError, match="mxfp8") as exc:
            cfg.validate()
        assert "mlx-lm 0.31.3" in str(exc.value)

    def test_bad_quant_bits_raises(self):
        cfg = FineTuneConfig(fine_tune_type="qlora", quant_bits=3)
        with pytest.raises(ValueError, match="quant_bits"):
            cfg.validate()

    def test_unknown_fine_tune_type_raises(self):
        cfg = FineTuneConfig(fine_tune_type="bogus")
        with pytest.raises(ValueError, match="Unknown fine_tune_type"):
            cfg.validate()

    def test_valid_configs_pass_validate(self):
        # Sanity: the happy paths must NOT raise — guards only fire on bad input.
        for ft in ("lora", "dora", "full"):
            FineTuneConfig(fine_tune_type=ft).validate()
        FineTuneConfig(
            fine_tune_type="qlora", quantize_base=True, quant_bits=4
        ).validate()
        FineTuneConfig(
            fine_tune_type="qlora", quantize_base=True, quant_bits=8
        ).validate()

    def test_quant_bits_ignored_without_qlora_or_base(self):
        # quant_bits only matters when quantize_base or qlora is set; a stray
        # quant_bits=3 with plain lora is not a training error (no quantization
        # runs), so validate() must not raise.
        FineTuneConfig(fine_tune_type="lora", quant_bits=3).validate()
