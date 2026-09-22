# SPDX-License-Identifier: Apache-2.0
"""#948: ddalcu/MiniMax-H3-FL2VA-MLX-Serve-4bit text_encoder ships a
pre-quantized language_model-only checkpoint with the ``language_model.``
prefix STRIPPED (keys are ``model.*`` / ``visual.*``) and NO
``quantization_config`` in config.json. mlx-vlm ``load_model`` expects
``language_model.model.*`` / ``vision_tower.*`` + a quantization config
-> 1780 unmatched parameters + no nn.quantize -> weight shape mismatch.

Fix: detect stripped layout, remap keys, inject quantization config,
trim num_hidden_layers to the checkpoint's actual layer range (ddalcu
only stores layers 0..49 since H3 reads layer 49), load strict=False
for the unused lm_head / final_norm.
"""

import mlx.core as mx

from fusion_mlx.video.minimax_h3.text_encoder import (
    _is_ddalcu_stripped,
    _remap_ddalcu_keys,
)


def _synth_ddalcu():
    return {
        "model.embed_tokens.weight": mx.zeros((10, 8)),
        "model.layers.0.self_attn.q_proj.weight": mx.zeros((4, 4), mx.uint32),
        "model.layers.0.self_attn.q_proj.scales": mx.zeros((4, 1)),
        "model.layers.0.self_attn.q_proj.biases": mx.zeros((4, 1)),
        "model.layers.49.mlp.down_proj.weight": mx.zeros((4, 4), mx.uint32),
        "visual.pos_embed.weight": mx.zeros((1, 8)),
        "visual.merger.linear_fc1.scales": mx.zeros((4, 1)),
    }


def test_is_ddalcu_stripped_detects_model_prefix_no_language_model():
    assert _is_ddalcu_stripped(_synth_ddalcu()) is True


def test_is_ddalcu_stripped_false_for_standard_layout():
    standard = {
        "language_model.model.embed_tokens.weight": mx.zeros((10, 8)),
        "vision_tower.patch_embed.proj.weight": mx.zeros((1, 8)),
    }
    assert _is_ddalcu_stripped(standard) is False


def test_is_ddalcu_stripped_false_when_no_model_prefix():
    assert _is_ddalcu_stripped({"visual.pos_embed.weight": mx.zeros((1, 8))}) is False


def test_remap_model_prefix_to_language_model_model():
    out = _remap_ddalcu_keys(_synth_ddalcu())
    assert "model.embed_tokens.weight" not in out
    assert "language_model.model.embed_tokens.weight" in out
    assert "language_model.model.layers.0.self_attn.q_proj.weight" in out
    assert "language_model.model.layers.0.self_attn.q_proj.scales" in out
    assert "language_model.model.layers.49.mlp.down_proj.weight" in out


def test_remap_visual_prefix_to_vision_tower():
    out = _remap_ddalcu_keys(_synth_ddalcu())
    assert "visual.pos_embed.weight" not in out
    assert "vision_tower.pos_embed.weight" in out
    assert "vision_tower.merger.linear_fc1.scales" in out


def test_remap_preserves_unrecognized_keys():
    w = {"lm_head.weight": mx.zeros((10, 8)), "model.norm.weight": mx.zeros((8,))}
    out = _remap_ddalcu_keys(w)
    assert out["lm_head.weight"] is not None
    # model.norm -> language_model.model.norm (model. prefix remap applies).
    assert "language_model.model.norm.weight" in out


def test_remap_is_total_no_key_lost_or_duplicated():
    src = _synth_ddalcu()
    out = _remap_ddalcu_keys(src)
    assert len(out) == len(src)
