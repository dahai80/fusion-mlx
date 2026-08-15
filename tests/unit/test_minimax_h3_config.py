# SPDX-License-Identifier: Apache-2.0
# P0 checkpoint：验证 H3Config / H3VAEConfig / H3AudioVAEConfig 的超参与
# partition classmethod 正确性。纯结构断言，不加载权重。

from fusion_mlx.video.minimax_h3 import (
    H3AudioVAEConfig,
    H3Config,
    H3Partition,
    H3VAEConfig,
)
from fusion_mlx.video.minimax_h3.config import H3Partition as P


class TestH3TransformerConfig:
    def test_defaults_match_upstream_config(self):
        c = H3Config()
        # 逐字段对齐 FL2VA/transformer/config.json
        assert c.dim == 5376
        assert c.num_layers == 50
        assert c.token_refiner_layers == 2
        assert c.num_heads == 56
        assert c.head_dim == 128
        assert c.ffn_dim == 14336
        assert c.latents_dim == 24
        assert c.audio_latents_dim == 32
        assert c.patch_size == (1, 2, 2)
        assert c.text_dim == 5120
        assert c.timestep_input_dim == 256
        assert c.time_embed_hidden == 5376
        assert c.time_embed_dim == 2688
        assert c.adaln_out == 96768
        assert c.final_adaln_out == 10752
        assert c.rope_inv_freq_len == 16
        assert c.norm_eps == 1e-5

    def test_head_dim_real(self):
        c = H3Config()
        assert c.head_dim_real == 5376 // 56

    def test_fl2va_partition(self):
        c = H3Config.fl2va()
        assert c.partition == H3Partition.FL2VA
        assert c.tasks == ("t2va", "i2va", "l2va", "fl2va")
        assert c.video_shift == 12.0
        assert c.audio_shift == 3.0

    def test_ref2va_partition(self):
        c = H3Config.ref2va()
        assert c.partition == H3Partition.REF2VA
        assert c.tasks == ("ref2va",)
        assert c.video_shift == 12.0


class TestH3VAEConfig:
    def test_visual_vae_defaults(self):
        v = H3VAEConfig()
        # 源自 FL2VA/video_vae/source/config.json
        assert v.ch == 128
        assert v.ch_mult == (1, 2, 2, 4, 4, 8)
        assert v.embed_dim == 24
        assert v.z_channels == 24
        assert v.in_channels == 3
        assert v.num_res_blocks == 2
        assert v.space_down == (2, 2, 2, 2, 1, 1)
        assert v.time_down == (1, 2, 2, 1, 1, 1)
        assert v.use_3d_conv is True
        assert v.use_t_isolated_gn is True
        assert v.use_vit_decoder is True
        assert v.causal_encoder is True
        assert v.causal_decoder is False
        assert v.vae_ratio == 16
        assert v.vae_ratio_t == 4
        # ViT3D
        assert v.vit_num_layers == 36
        assert v.vit_heads == 32
        assert v.vit_dim_head == 64
        assert v.vit_rope_dim_ratio == 0.75
        assert v.vit_rope_theta == 100.0
        assert v.vit_norm_type == "rms_norm"
        assert v.vit_ffn_use_gated is True
        # 分块默认
        assert v.vae_clip_length == 17
        assert v.vae_token_drop == 3
        assert v.vae_tile_size == 256
        assert v.vae_tile_overlap_min == 64

    def test_nesting_in_h3config(self):
        c = H3Config()
        assert isinstance(c.vae, H3VAEConfig)
        assert c.vae.embed_dim == 24


class TestH3AudioVAEConfig:
    def test_audio_vae_defaults(self):
        a = H3AudioVAEConfig()
        # 源自 FL2VA/audio_vae/metadata.json
        assert a.encoder_dim == 64
        assert a.decoder_dim == 1024
        assert a.latent_dim == 2048
        assert a.vae_latent_channels == 32
        assert a.sample_rate == 32000
        assert a.encoder_rates == (2, 4, 4, 5, 5)
        assert a.decoder_rates == (5, 5, 2, 2, 2, 2, 2)
        assert a.decoder_type == "bigvgan"
        assert a.attn_proj is True
        assert a.attn_heads == 8
        assert a.upsample_kernel_sizes == (9, 9, 4, 4, 4, 4, 4)
        assert a.resblock_kernel_sizes == (3, 7, 11)

    def test_nesting_in_h3config(self):
        c = H3Config()
        assert isinstance(c.audio_vae, H3AudioVAEConfig)
        assert c.audio_vae.sample_rate == 32000


class TestH3PartitionEnum:
    def test_partition_values(self):
        assert H3Partition.FL2VA.value == "fl2va"
        assert H3Partition.REF2VA.value == "ref2va"

    def test_import_alias(self):
        assert P is H3Partition

    def test_to_dict_serializes_enum(self):
        c = H3Config()
        d = c.to_dict()
        assert d["partition"] == "fl2va"
