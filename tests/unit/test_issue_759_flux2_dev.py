import mlx.core as mx
import pytest

from fusion_mlx.engines.flux2_dev.text_encoder import Mistral3TextEncoder
from fusion_mlx.engines.flux2_dev.variant import (
    _DEV_TRANSFORMER_OVERRIDES,
    Flux2Dev,
    _DevModelConfig,
)
from fusion_mlx.engines.image_gen import VARIANT_MAP, _infer_variant


def test_flux2_dev_variant_detected():
    assert _infer_variant("AITRADER/FLUX2-dev-mlx-8bit") == "flux2_dev"
    assert _infer_variant("FLUX.2-dev") == "flux2_dev"
    assert _infer_variant("flux2-dev") == "flux2_dev"
    assert _infer_variant("black-forest-labs/FLUX.2-dev") == "flux2_dev"


def test_flux2_dev_variant_not_misclassified_as_klein():
    assert _infer_variant("black-forest-labs/FLUX.2-klein") == "txt2img"
    assert _infer_variant("flux2-klein-9b") == "txt2img"


def test_flux2_dev_variant_map_native():
    entry = VARIANT_MAP["flux2_dev"]
    assert entry[0].startswith("fusion_mlx."), entry
    assert entry[1] == "Flux2Dev"
    assert entry[2] == "flux2_dev"
    assert entry[3] == 3.5


def test_flux2_dev_start_no_fail_visible_raise():
    import inspect

    from fusion_mlx.engines.image_gen import ImageGenEngine

    src = inspect.getsource(ImageGenEngine.start)
    assert "flux2_dev variant and no all-MLX Mistral3" not in src
    assert "mflux-community/mflux#707" not in src


def test_flux2_dev_transformer_overrides():
    assert _DEV_TRANSFORMER_OVERRIDES["num_layers"] == 8
    assert _DEV_TRANSFORMER_OVERRIDES["num_single_layers"] == 48
    assert _DEV_TRANSFORMER_OVERRIDES["num_attention_heads"] == 48
    assert _DEV_TRANSFORMER_OVERRIDES["joint_attention_dim"] == 15360
    assert _DEV_TRANSFORMER_OVERRIDES["attention_head_dim"] == 128
    assert _DEV_TRANSFORMER_OVERRIDES["guidance_embeds"] is True


def test_flux2_dev_transformer_built_with_overrides():
    pytest.importorskip("mflux")
    from mflux.models.flux2.model.flux2_transformer.transformer import (
        Flux2Transformer,
    )

    t = Flux2Transformer(**_DEV_TRANSFORMER_OVERRIDES)
    assert t.inner_dim == 6144
    assert len(t.transformer_blocks) == 8
    assert len(t.single_transformer_blocks) == 48
    assert t.time_guidance_embed.guidance_linear_1 is not None
    assert t.time_guidance_embed.guidance_linear_2 is not None


def test_dev_model_config_attrs():
    cfg = _DevModelConfig()
    assert cfg.num_train_steps == 1000
    assert cfg.max_sequence_length == 512
    assert cfg.supports_guidance is True
    assert cfg.requires_sigma_shift is True
    assert cfg.precision == mx.bfloat16


def test_mistral3_text_encoder_shape():
    enc = Mistral3TextEncoder(
        hidden_size=5120,
        num_hidden_layers=30,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=32768,
        vocab_size=131072,
    )
    input_ids = mx.array([[1, 2, 3, 4, 5]])
    attention_mask = mx.ones((1, 5), dtype=mx.int32)
    embeds = enc.get_prompt_embeds(
        input_ids=input_ids,
        attention_mask=attention_mask,
        hidden_state_layers=(9, 18, 27),
    )
    mx.eval(embeds)
    assert embeds.shape[-1] == 15360
    assert embeds.ndim == 3


def test_text_encoder_sanitize_strips_vision():
    enc = Mistral3TextEncoder(hidden_size=5120, num_hidden_layers=4)
    raw = {
        "model.embed_tokens.weight": mx.zeros((4, 5120)),
        "model.layers.0.self_attn.q_proj.weight": mx.zeros((4096, 5120)),
        "model.norm.weight": mx.ones((5120,)),
        "vision_tower.weight": mx.zeros((1,)),
        "multi_modal_projector.weight": mx.zeros((1,)),
        "tekken_model.dummy": mx.zeros((1,)),
    }
    sanitized = enc.sanitize(raw)
    assert "embed_tokens.weight" in sanitized
    assert "layers.0.self_attn.q_proj.weight" in sanitized
    assert "norm.weight" in sanitized
    assert not any(k.startswith("vision_tower") for k in sanitized)
    assert not any(k.startswith("multi_modal_projector") for k in sanitized)
    assert not any(k.startswith("tekken_model") for k in sanitized)


def test_flux2_dev_init_signature_accepts_native_args():
    import inspect

    sig = inspect.signature(Flux2Dev.__init__)
    params = set(sig.parameters)
    assert "model_config" in params
    assert "model_path" in params
    assert "quantize" in params


def test_mistral_tokenizer_forced_right_padding():
    # Regression guard (#759): the Mistral chat template defaults to
    # left-padding, which produces fully-masked pad rows under causal
    # attention -> NaN that leaks into real positions via causal keys.
    # Right-padding keeps real tokens at positions 0..N so the text encoder
    # stays NaN-free. The static check always runs; the live check runs when
    # the tokenizer cache is reachable.
    import inspect

    from fusion_mlx.engines.flux2_dev import tokenizer as tok_mod

    src = inspect.getsource(tok_mod.load_mistral_tokenizer)
    assert 'padding_side = "right"' in src, (
        "load_mistral_tokenizer must force padding_side=right "
        "(left-pad causes NaN under causal attention, #759)"
    )

    try:
        tok = tok_mod.load_mistral_tokenizer(max_length=32)
    except Exception as exc:
        pytest.skip(f"tokenizer cache unavailable offline: {exc}")
    assert tok.tokenizer.padding_side == "right"
    out = tok.tokenize(prompt="a cat", max_length=32)
    mx.eval(out.input_ids)
    mx.eval(out.attention_mask)
    amask = mx.array(out.attention_mask)
    import numpy as np

    arr = np.array(amask, copy=False)[0]
    first_zero = int(np.argmin(arr))
    real_count = int(arr.sum())
    assert (
        first_zero == real_count
    ), f"expected right-pad (first_zero==real_count), got first_zero={first_zero} real_count={real_count}"
