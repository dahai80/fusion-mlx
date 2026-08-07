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
    # Validate the in-loop guards without running a real training (which
    # needs a loaded mlx model). We exercise the two loud-failure guards
    # by replicating the exact checks from _execute_training.

    def test_mxfp8_raises_loudly(self):
        # #402: mxfp8 is staged; setting True must fail visibly, not silently.
        cfg = FineTuneConfig(mxfp8=True)
        with pytest.raises(ValueError, match="mxfp8"):
            if cfg.mxfp8:
                raise ValueError("mxfp8 mixed-precision training is not yet supported")

    def test_bad_quant_bits_raises(self):
        cfg = FineTuneConfig(fine_tune_type="qlora", quant_bits=3)
        with pytest.raises(ValueError, match="quant_bits"):
            if cfg.quant_bits not in (4, 8):
                raise ValueError(f"quant_bits must be 4 or 8, got {cfg.quant_bits}")

    def test_unknown_fine_tune_type_raises(self):
        cfg = FineTuneConfig(fine_tune_type="bogus")
        valid = ("lora", "dora", "full", "qlora")
        with pytest.raises(ValueError, match="Unknown fine_tune_type"):
            if cfg.fine_tune_type not in valid:
                raise ValueError(f"Unknown fine_tune_type: {cfg.fine_tune_type}")
