import pytest

from fusion_mlx.training.vlm_service import VLMFineTuneConfig, _quantize_vlm_base


class TestVLMQLoRAConfig1567:
    def test_fields_default_off(self):
        cfg = VLMFineTuneConfig()
        assert cfg.quantize_base is False
        assert cfg.quant_bits == 4

    def test_qlora_accepts_fields(self):
        cfg = VLMFineTuneConfig(quantize_base=True, quant_bits=8)
        cfg.validate()

    def test_rejects_full_finetune_conflict(self):
        with pytest.raises(ValueError, match="incompatible with full_finetune"):
            VLMFineTuneConfig(quantize_base=True, full_finetune=True).validate()

    @pytest.mark.parametrize("bits", [2, 3, 5, 16])
    def test_rejects_bad_bits(self, bits):
        with pytest.raises(ValueError, match="quant_bits must be 4 or 8"):
            VLMFineTuneConfig(quantize_base=True, quant_bits=bits).validate()

    def test_bad_bits_ignored_when_off(self):
        # quant_bits is meaningless without quantize_base — no error.
        VLMFineTuneConfig(quant_bits=5).validate()


class TestQuantizeVLMBase:
    def _fake_vlm(self):
        import mlx.nn as nn

        class LM(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(256, 256)
                self.fc2 = nn.Linear(256, 128)

        class Fake:
            pass

        f = Fake()
        f.language_model = LM()
        return f

    def test_applies_quantize(self):
        import mlx.nn as nn

        model = self._fake_vlm()
        assert _quantize_vlm_base(model, 4) is True
        assert isinstance(model.language_model.fc1, nn.QuantizedLinear)
        assert model.language_model.fc1.bits == 4
        assert isinstance(model.language_model.fc2, nn.QuantizedLinear)

    def test_noop_on_already_quantized(self):
        model = self._fake_vlm()
        assert _quantize_vlm_base(model, 8) is True
        # Second call: already quantized — returns False, keeps 4-bit param
        # count (no double quantize).
        assert _quantize_vlm_base(model, 8) is False
        assert model.language_model.fc1.bits == 8

    def test_route_config_body_parses(self):
        # Admin route builds VLMFineTuneConfig(**config_body); the new keys
        # must survive a round-trip without a 400 (unknown-key behavior on
        # older versions is acceptable, but here they exist).
        cfg = VLMFineTuneConfig(**{"quantize_base": True, "quant_bits": 4, "iters": 5})
        cfg.validate()
        assert cfg.quantize_base is True
        assert cfg.quant_bits == 4
