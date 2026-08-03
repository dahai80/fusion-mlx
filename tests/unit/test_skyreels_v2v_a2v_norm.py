"""V2V/A2V norm 命名对齐 diffusers 单元测试 (issue #164 后续).

验证 V2V/A2V AttentionBlock 与 SkyReelsV2VDiT/SkyReelsA2VDiT 复刻 #168 R2V 修复:
  - norm1 (self-attn 前): elementwise_affine=False (对齐 diffusers block.norm1)
  - norm2 (cross-attn 前): elementwise_affine=True (对齐 diffusers block.norm2)
  - norm3 (ffn 前): elementwise_affine=False (对齐 diffusers block.norm3)
  - cross_attn_type 默认 t2v; added_kv_proj_dim 非 None 或 config 显式 -> i2v
"""

import mlx.core as mx

from fusion_mlx.video.skyreels_v3.transformer_a2v import (
    A2VAttentionBlock,
    SkyReelsA2VDiT,
)
from fusion_mlx.video.skyreels_v3.transformer_v2v import (
    SkyReelsV2VDiT,
    V2VAttentionBlock,
)

_TINY_CFG = dict(
    num_layers=4,
    dim=128,
    ffn_dim=256,
    num_heads=4,
    num_kv_heads=4,
    patch_size=(1, 2, 2),
    in_dim=16,
    out_dim=16,
    text_dim=128,
    text_len=8,
    freq_dim=128,
)


def _block_norm_affine(block):
    return {
        "norm1": block.norm1.elementwise_affine,
        "norm2": block.norm2.elementwise_affine if block.norm2 is not None else None,
        "norm3": block.norm3.elementwise_affine,
    }


def test_v2v_block_norm_naming_affine():
    block = V2VAttentionBlock(dim=128, ffn_dim=256, num_heads=4, cross_attn_norm=True)
    aff = _block_norm_affine(block)
    assert aff["norm1"] is False, f"norm1 应 affine=False (self-attn 前): {aff}"
    assert aff["norm2"] is True, f"norm2 应 affine=True (cross-attn 前): {aff}"
    assert aff["norm3"] is False, f"norm3 应 affine=False (ffn 前): {aff}"


def test_a2v_block_norm_naming_affine():
    block = A2VAttentionBlock(dim=128, ffn_dim=256, num_heads=4, cross_attn_norm=True)
    aff = _block_norm_affine(block)
    assert aff["norm1"] is False, f"norm1 应 affine=False (self-attn 前): {aff}"
    assert aff["norm2"] is True, f"norm2 应 affine=True (cross-attn 前): {aff}"
    assert aff["norm3"] is False, f"norm3 应 affine=False (ffn 前): {aff}"


def test_v2v_cross_attn_type_default_t2v():
    dit = SkyReelsV2VDiT(_TINY_CFG)
    assert (
        dit.cross_attn_type == "t2v_cross_attn"
    ), f"默认应 t2v (无 added_kv_proj_dim): {dit.cross_attn_type}"


def test_v2v_cross_attn_type_i2v_when_added_kv():
    cfg = dict(_TINY_CFG, added_kv_proj_dim=128)
    dit = SkyReelsV2VDiT(cfg)
    assert (
        dit.cross_attn_type == "i2v_cross_attn"
    ), f"added_kv_proj_dim 非 None 应 i2v: {dit.cross_attn_type}"


def test_v2v_cross_attn_type_explicit_config_wins():
    cfg = dict(_TINY_CFG, cross_attn_type="i2v_cross_attn")
    dit = SkyReelsV2VDiT(cfg)
    assert (
        dit.cross_attn_type == "i2v_cross_attn"
    ), f"显式 cross_attn_type 应优先: {dit.cross_attn_type}"


def test_a2v_cross_attn_type_default_t2v():
    dit = SkyReelsA2VDiT(_TINY_CFG)
    assert (
        dit.cross_attn_type == "t2v_cross_attn"
    ), f"默认应 t2v (无 added_kv_proj_dim): {dit.cross_attn_type}"


def test_a2v_cross_attn_type_i2v_when_added_kv():
    cfg = dict(_TINY_CFG, added_kv_proj_dim=128)
    dit = SkyReelsA2VDiT(cfg)
    assert (
        dit.cross_attn_type == "i2v_cross_attn"
    ), f"added_kv_proj_dim 非 None 应 i2v: {dit.cross_attn_type}"


def test_v2v_dit_forward_smoke():
    dit = SkyReelsV2VDiT(_TINY_CFG)
    x = mx.random.normal((1, 16, 2, 4, 4))
    t = mx.array([0.5])
    ctx = mx.random.normal((1, 8, 128))
    out = dit(x, t, ctx, [8], [(2, 2, 2)])
    mx.eval(out)
    assert out.shape == (1, 16, 2, 4, 4), f"V2V 前向 shape 错: {out.shape}"
