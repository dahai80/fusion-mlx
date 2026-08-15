# SPDX-License-Identifier: Apache-2.0
# LTX-2.5 independent transformer port: key-tree + ff-bias + connector + smoke.
# 不依赖真实权重的结构断言；真实权重 strict-load 单独标记 skip（需 68GB 下载）。
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from fusion_mlx.video.ltx2_5.config import LTX2_5ModelConfig
from fusion_mlx.video.ltx2_5.embeddings_connector import Embeddings1DConnector
from fusion_mlx.video.ltx2_5.ltx2_5_model import LTX2_5Model, LTX2_5X0Model
from fusion_mlx.video.ltx2_5.transformer import BasicAVTransformerBlock

CFG = LTX2_5ModelConfig()


def _block_keys(block):
    return sorted(k for k, _ in nn.utils.tree_flatten(block.parameters()))


def _module_keys(mod):
    return sorted(k for k, _ in nn.utils.tree_flatten(mod.parameters()))


class TestBlockKeyTree:
    def _make_block(self):
        return BasicAVTransformerBlock(
            idx=0,
            video=CFG.get_video_config(),
            audio=CFG.get_audio_config(),
            has_prompt_adaln=CFG.has_prompt_adaln,
            ff_bias=CFG.ff_bias,
            audio_ff_bias=CFG.audio_ff_bias,
        )

    def test_block_has_84_keys(self):
        assert len(_block_keys(self._make_block())) == 84

    def test_video_ff_has_no_bias(self):
        bk = _block_keys(self._make_block())
        ff = [k for k in bk if k.startswith("ff.")]
        assert ff == ["ff.proj_in.weight", "ff.proj_out.weight"]

    def test_audio_ff_has_bias(self):
        bk = _block_keys(self._make_block())
        aff = [k for k in bk if k.startswith("audio_ff.")]
        assert "audio_ff.proj_in.bias" in aff
        assert "audio_ff.proj_out.bias" in aff

    def test_six_gate_logits_modules(self):
        bk = _block_keys(self._make_block())
        gl = [k for k in bk if "to_gate_logits" in k]
        assert len(gl) == 12  # 6 attn modules × (weight + bias)

    def test_prompt_adaln_tables_present(self):
        bk = _block_keys(self._make_block())
        assert "prompt_scale_shift_table" in bk
        assert "audio_prompt_scale_shift_table" in bk

    def test_av_ca_tables_present(self):
        bk = _block_keys(self._make_block())
        assert "scale_shift_table_a2v_ca_audio" in bk
        assert "scale_shift_table_a2v_ca_video" in bk


class TestConnectorKeyTree:
    def test_video_connector_129_keys(self):
        vc = Embeddings1DConnector(
            attention_head_dim=CFG.attention_head_dim,
            num_attention_heads=CFG.num_attention_heads,
            num_layers=CFG.connector_num_layers,
            positional_embedding_max_pos=CFG.connector_positional_embedding_max_pos,
            num_learnable_registers=CFG.connector_num_learnable_registers,
            apply_gated_attention=CFG.connector_apply_gated_attention,
            ff_bias=True,
        )
        assert len(_module_keys(vc)) == 129
        assert "learnable_registers" in _module_keys(vc)

    def test_audio_connector_129_keys(self):
        ac = Embeddings1DConnector(
            attention_head_dim=CFG.audio_attention_head_dim,
            num_attention_heads=CFG.audio_num_attention_heads,
            num_layers=CFG.connector_num_layers,
            positional_embedding_max_pos=CFG.connector_positional_embedding_max_pos,
            num_learnable_registers=CFG.connector_num_learnable_registers,
            apply_gated_attention=CFG.connector_apply_gated_attention,
            ff_bias=True,
        )
        assert len(_module_keys(ac)) == 129

    def test_connector_eight_blocks(self):
        vc = Embeddings1DConnector(
            attention_head_dim=CFG.attention_head_dim,
            num_attention_heads=CFG.num_attention_heads,
            num_layers=CFG.connector_num_layers,
            positional_embedding_max_pos=CFG.connector_positional_embedding_max_pos,
            num_learnable_registers=CFG.connector_num_learnable_registers,
            apply_gated_attention=CFG.connector_apply_gated_attention,
            ff_bias=True,
        )
        vk = _module_keys(vc)
        for i in range(8):
            assert any(f"transformer_1d_blocks.{i}." in k for k in vk)


class TestModelKeyTree:
    def test_model_param_count_4349(self):
        # 48 blocks × 84 + 59 head/tail + 258 connectors + 1 keyframes = 4349.
        m = LTX2_5Model(CFG)
        assert len(_module_keys(m)) == 4349

    def test_model_has_connectors(self):
        m = LTX2_5Model(CFG)
        mk = _module_keys(m)
        assert any("video_embeddings_connector" in k for k in mk)
        assert any("audio_embeddings_connector" in k for k in mk)

    def test_model_has_keyframes_embedding(self):
        m = LTX2_5Model(CFG)
        mk = _module_keys(m)
        assert "keyframes_abs_pos_embedding" in mk

    def test_model_has_prompt_adaln(self):
        m = LTX2_5Model(CFG)
        mk = _module_keys(m)
        assert "prompt_adaln_single.linear.weight" in mk
        assert "audio_prompt_adaln_single.linear.weight" in mk

    def test_model_no_caption_projection(self):
        # has_prompt_adaln=True → prompt_adaln_single, not caption_projection.
        m = LTX2_5Model(CFG)
        mk = _module_keys(m)
        assert not any("caption_projection" in k for k in mk)

    def test_sanitize_strips_prefix_and_remaps(self):
        m = LTX2_5Model(CFG)
        raw = {
            "model.diffusion_model.transformer_blocks.0.attn1.to_out.0.weight": mx.zeros(
                (1, 1)
            ),
            "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight": mx.zeros(
                (1, 1)
            ),
            "model.diffusion_model.transformer_blocks.0.ff.net.2.weight": mx.zeros(
                (1, 1)
            ),
            "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.weight": mx.zeros(
                (1, 1)
            ),
            "model.diffusion_model.video_embeddings_connector.learnable_registers": mx.zeros(
                (1, 1)
            ),
            "model.diffusion_model.keyframes_abs_pos_embedding": mx.zeros((1, 1)),
            "unrelated.other.key": mx.zeros((1, 1)),
        }
        s = m.sanitize(raw)
        assert "transformer_blocks.0.attn1.to_out.weight" in s
        assert "transformer_blocks.0.ff.proj_in.weight" in s
        assert "transformer_blocks.0.ff.proj_out.weight" in s
        assert "adaln_single.emb.timestep_embedder.linear1.weight" in s
        # connector keys are NOT skipped (2.5 delta).
        assert "video_embeddings_connector.learnable_registers" in s
        assert "keyframes_abs_pos_embedding" in s
        # unrelated keys dropped.
        assert "unrelated.other.key" not in s

    def test_sanitize_passthrough_no_prefix(self):
        m = LTX2_5Model(CFG)
        raw = {"already.flat.weight": mx.zeros((1, 1))}
        assert m.sanitize(raw) is raw


class TestX0Model:
    def test_x0_wraps_velocity_model(self):
        m = LTX2_5Model(CFG)
        x0 = LTX2_5X0Model(m)
        assert x0.velocity_model is m


REAL_CHECKPOINT = (
    "/Users/dahai/.fusion-mlx/models/models--Lightricks--LTX-2.5/snapshots/"
    "ce298b1259d61ce6c87e05154b9ad339b16f32a0/"
    "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
)


@pytest.mark.realmodel
class TestRealWeightStrictLoad:
    def test_strict_load_zero_mismatch(self):
        import os

        if not os.path.exists(REAL_CHECKPOINT):
            pytest.skip("real 22B checkpoint not downloaded")
        m = LTX2_5Model.from_pretrained(REAL_CHECKPOINT, strict=True)
        mk = _module_keys(m)
        assert len(mk) == 4349
