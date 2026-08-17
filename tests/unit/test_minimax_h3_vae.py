# SPDX-License-Identifier: Apache-2.0
# P1 checkpoint：验证 VisualVAE 编码器/解码器前向形状正确性（随机权重，
# 不加载真实 checkpoint，不测精度）。逐层验证维度变换。
import mlx.core as mx
import pytest

from fusion_mlx.video.minimax_h3.config import H3VAEConfig
from fusion_mlx.video.minimax_h3.vae import (
    CausalConv3d,
    DiagonalGaussianDistribution,
    Downsample3D,
    EncoderFCN3D,
    GroupNorm3D,
    MiniMaxH3VideoVAE,
    RotaryEmbeddingND,
    ViT3DDecoder,
    _remap_vae_weights,
    create_token_ids,
)


@pytest.fixture(autouse=True)
def _fp32():
    mx.set_default_device(mx.cpu)
    yield


class TestCausalConv3d:
    def test_preserves_shape_causal_pad(self):
        c = CausalConv3d(128, 128, kernel_size=3, padding=1, causal=True)
        x = mx.zeros((1, 128, 5, 16, 16), dtype=mx.float32)
        out = c(x)
        assert out.shape == (1, 128, 5, 16, 16)

    def test_single_frame_expands_time(self):
        c = CausalConv3d(128, 128, kernel_size=3, padding=1, causal=True)
        x = mx.zeros((1, 128, 1, 16, 16), dtype=mx.float32)
        out = c(x)
        # 单帧 repeat 到 kt=3 后卷积 -> (3-3)//1+1=1
        assert out.shape == (1, 128, 1, 16, 16)

    def test_stride2_downsample(self):
        c = CausalConv3d(
            128, 256, kernel_size=3, stride=(1, 2, 2), padding=(1, 0, 0), causal=True
        )
        x = mx.zeros((1, 128, 5, 16, 16), dtype=mx.float32)
        # 无空间 padding, stride2: (16-3)//2+1=7
        out = c(x)
        assert out.shape == (1, 256, 5, 7, 7)


class TestGroupNorm3D:
    def test_t_isolated_preserves_shape(self):
        gn = GroupNorm3D(128, use_t_isolated_gn=True)
        x = mx.zeros((1, 128, 5, 16, 16), dtype=mx.float32)
        assert gn(x).shape == x.shape

    def test_plain_preserves_shape(self):
        gn = GroupNorm3D(128, use_t_isolated_gn=False)
        x = mx.zeros((1, 128, 5, 16, 16), dtype=mx.float32)
        assert gn(x).shape == x.shape


class TestDownsample3D:
    def test_space_down(self):
        ds = Downsample3D(128, 256, time_stride=1, space_stride=2, causal=True)
        x = mx.zeros((1, 128, 5, 16, 16), dtype=mx.float32)
        out = ds(x)
        assert out.shape[1] == 256  # 通道翻倍
        assert out.shape[3] == 8  # (16-3)//2+1 round


class TestRotaryEmbedding:
    def test_create_token_ids_shape(self):
        ids = create_token_ids((2, 8, 8), mx.float32)
        assert ids.shape == (1, 2 * 8 * 8, 3)

    def test_rope_output_shape(self):
        rope = RotaryEmbeddingND(48, rotary_base=100.0, n_dim=3, use_angle=True)
        ids = create_token_ids((2, 8, 8), mx.float32)
        ids = mx.broadcast_to(ids, (1, ids.shape[1], ids.shape[2]))
        cos, sin = rope(ids)
        assert cos.shape == (1, ids.shape[1], 1, 48)
        assert sin.shape == (1, ids.shape[1], 1, 48)


class TestEncoderFCN3D:
    def test_output_compression_ratio(self):
        cfg = H3VAEConfig()
        enc = EncoderFCN3D(
            ch=cfg.ch,
            ch_mult=cfg.ch_mult,
            space_down=cfg.space_down,
            time_down=cfg.time_down,
            num_res_blocks=1,
            in_channels=3,
            z_channels=cfg.z_channels,
            use_t_isolated_gn=cfg.use_t_isolated_gn,
            causal=True,
        )
        x = mx.zeros((1, 3, 5, 128, 128), dtype=mx.float32)
        out = enc(x)
        # double_z -> 2*z_channels=48; 空间 128//16=8
        assert out.shape[1] == 48
        assert out.shape[3] == 8
        assert out.shape[4] == 8


class TestViT3DDecoder:
    def test_output_upsample_ratio(self):
        cfg = H3VAEConfig()
        dec = ViT3DDecoder(
            patch_size=cfg.vae_ratio,
            patch_size_t=cfg.vae_ratio_t,
            t_causal=cfg.causal_decoder,
            in_channels=cfg.z_channels,
            out_channels=cfg.in_channels,
            num_layers=2,
            heads=cfg.vit_heads,
            dim_head=cfg.vit_dim_head,
            norm_type=cfg.vit_norm_type,
            ffn_use_gated=cfg.vit_ffn_use_gated,
            rope_theta=cfg.vit_rope_theta,
            rope_dim_ratio=cfg.vit_rope_dim_ratio,
        )
        z = mx.zeros((1, cfg.z_channels, 2, 8, 8), dtype=mx.float32)
        out = dec(z)
        # 2*patch_size_t=8, 8*patch_size=128
        assert out.shape == (1, 3, 8, 128, 128)


class TestDiagonalGaussian:
    def test_sample_shape(self):
        moments = mx.zeros((1, 48, 3, 4, 4), dtype=mx.float32)
        d = DiagonalGaussianDistribution(moments)
        z = d.sample()
        assert z.shape == (1, 24, 3, 4, 4)


class TestMiniMaxH3VideoVAE:
    def test_param_tree_has_encoder_decoder_quant(self):
        from mlx.utils import tree_flatten

        vae = MiniMaxH3VideoVAE()
        keys = {k for k, _ in tree_flatten(vae.parameters())}
        assert "encoder.conv_in.weight" in keys
        assert "quant_conv.weight" in keys
        assert "post_quant_conv.weight" in keys
        assert "decoder.x_embedder.weight" in keys
        assert "decoder.proj_out.weight" in keys

    def test_quant_conv_is_2d(self):
        vae = MiniMaxH3VideoVAE()
        assert vae.quant_conv.weight.ndim == 2

    def test_encode_shape(self):
        cfg = H3VAEConfig()
        vae = MiniMaxH3VideoVAE(config=cfg)
        x = mx.zeros((1, 3, 9, 64, 64), dtype=mx.float32)
        moments = vae.encode(x)
        assert moments.shape[1] == 48  # 2*z_channels

    def test_encode_base_shape(self):
        cfg = H3VAEConfig()
        vae = MiniMaxH3VideoVAE(config=cfg)
        x = mx.zeros((1, 3, 9, 64, 64), dtype=mx.float32)
        z = vae.encode_base(x)
        assert z.shape[1] == 24  # z_channels

    def test_split_tiles_small_passthrough(self):
        # input_len <= tile_size -> 单块全覆盖，无 overlap。
        cfg = H3VAEConfig()
        vae = MiniMaxH3VideoVAE(config=cfg)
        start, length, overlap = vae._split_tiles(256)
        assert start == [0]
        assert length == [256]
        assert overlap == []

    def test_split_tiles_covers_full_extent(self):
        # 分块 start+length 必须覆盖整段，相邻块 overlap>=overlap_min。
        cfg = H3VAEConfig()
        vae = MiniMaxH3VideoVAE(config=cfg)
        start, length, overlap = vae._split_tiles(768)
        assert start[0] == 0
        assert start[-1] + length[-1] >= 768
        for o in overlap:
            assert o >= cfg.vae_tile_overlap_min
        for ln in length:
            assert ln == cfg.vae_tile_size

    def test_blend_linear_ramp(self):
        # blend_extent=2：w_a=[1,0.5], w_b=[0,0.5]。
        # a 尾 [10,10] 与 b 头 [20,20] -> [10*1+20*0, 10*0.5+20*0.5]=[10,15]。
        a = mx.array([[0.0], [0.0], [10.0], [10.0]], dtype=mx.float32)  # (4,1)
        b = mx.array([[20.0], [20.0], [30.0], [30.0]], dtype=mx.float32)
        out = MiniMaxH3VideoVAE._blend(a, b, 2, dim=0)
        mx.eval(out)
        assert float(out[0, 0]) == 10.0
        assert float(out[1, 0]) == 15.0
        assert float(out[2, 0]) == 30.0

    def test_tiled_decode_shape_matches_single_pass(self):
        # latent 48×84（>tile 16）触发分块；decoder 36 层太重，缩到 2 层验证形状。
        cfg = H3VAEConfig()
        vae = MiniMaxH3VideoVAE(config=cfg)
        vae.decoder = ViT3DDecoder(
            patch_size=cfg.vae_ratio,
            patch_size_t=cfg.vae_ratio_t,
            t_causal=cfg.causal_decoder,
            in_channels=cfg.z_channels,
            out_channels=cfg.in_channels,
            num_layers=2,
            heads=cfg.vit_heads,
            dim_head=cfg.vit_dim_head,
            norm_type=cfg.vit_norm_type,
            ffn_use_gated=cfg.vit_ffn_use_gated,
            rope_theta=cfg.vit_rope_theta,
            rope_dim_ratio=cfg.vit_rope_dim_ratio,
        )
        z = mx.zeros((1, cfg.z_channels, 5, 48, 84), dtype=mx.float32)
        dec = vae.decode(z)
        mx.eval(dec)
        assert dec.shape == (1, 3, 20, 768, 1344)


class TestRemapWeights:
    def test_pointwise_5d_squeezed_to_2d(self):
        import numpy as np

        params = {
            "quant_conv.weight": mx.array(
                np.zeros((48, 48, 1, 1, 1), dtype=np.float32)
            ),
            "post_quant_conv.weight": mx.array(
                np.zeros((24, 24, 1, 1, 1), dtype=np.float32)
            ),
        }
        out = _remap_vae_weights(params)
        assert out["quant_conv.weight"].shape == (48, 48)
        assert out["post_quant_conv.weight"].shape == (24, 24)

    def test_causal_conv_5d_transposed(self):
        import numpy as np

        # PyTorch [O,I,D,H,W] = [2,3,3,3,3]
        params = {
            "encoder.conv_in.weight": mx.array(
                np.zeros((2, 3, 3, 3, 3), dtype=np.float32)
            ),
        }
        out = _remap_vae_weights(params)
        # MLX [O,D,H,W,I] = [2,3,3,3,3]
        assert out["encoder.conv_in.weight"].shape == (2, 3, 3, 3, 3)
