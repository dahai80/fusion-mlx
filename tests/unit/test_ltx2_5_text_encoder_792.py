# SPDX-License-Identifier: Apache-2.0
# #792: Gemma4-12b text-encoder k_proj reshape crash — config field guard.
# Root cause: missing attention_k_eq_v=True. With default False, full_attention
# layers use num_key_value_heads=8 -> expect k_proj=4096, but checkpoint ships
# 512 (1*512) -> reshape ValueError. Also need explicit layer_types (5 sliding +
# 1 full * 8 = 48) + rope_parameters.
from fusion_mlx.video.ltx2_5.text_encoder import _build_default_text_config


class TestBuildDefaultTextConfig792:
    def test_attention_k_eq_v_true(self):
        cfg = _build_default_text_config()
        assert cfg.attention_k_eq_v is True

    def test_num_hidden_layers_48(self):
        cfg = _build_default_text_config()
        assert cfg.num_hidden_layers == 48

    def test_layer_types_length_48(self):
        cfg = _build_default_text_config()
        assert len(cfg.layer_types) == 48

    def test_layer_types_pattern_5_sliding_1_full(self):
        cfg = _build_default_text_config()
        for block_idx in range(8):
            base = block_idx * 6
            for i in range(5):
                assert cfg.layer_types[base + i] == "sliding_attention"
            assert cfg.layer_types[base + 5] == "full_attention"

    def test_full_attention_count_8(self):
        cfg = _build_default_text_config()
        assert cfg.layer_types.count("full_attention") == 8
        assert cfg.layer_types.count("sliding_attention") == 40

    def test_rope_parameters_present(self):
        cfg = _build_default_text_config()
        assert "full_attention" in cfg.rope_parameters
        assert "sliding_attention" in cfg.rope_parameters
        assert cfg.rope_parameters["full_attention"]["rope_theta"] == 1000000.0
        assert cfg.rope_parameters["sliding_attention"]["rope_theta"] == 10000.0

    def test_global_kv_heads_1_matches_checkpoint(self):
        # full_attention layers k_proj=512 = 1 * 512 -> num_global_key_value_heads=1
        cfg = _build_default_text_config()
        assert cfg.num_global_key_value_heads == 1
        assert cfg.global_head_dim == 512

    def test_sliding_head_dim_256(self):
        cfg = _build_default_text_config()
        assert cfg.head_dim == 256
