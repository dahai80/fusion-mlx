# SPDX-License-Identifier: Apache-2.0
"""Tier-2 DiT 主干统一回归测试 (issue #186).

验证 SkyReelsBaseDiT 抽取后三变体 (R2V/V2V/A2V) 行为:
  1. forward_partial(n_blocks=num_layers) 与 __call__ bit-identical (CP3)
  2. V2V/A2V lazy mx.compile 数值容差 (CP4, 编译融合噪声 < 1e-2, pre-existing)
  3. 三变体均继承 SkyReelsBaseDiT; A2V 独有 audio_embedding
  4. 输出 shape 正确
"""

import mlx.core as mx

from fusion_mlx.video.skyreels_v3.dit_base import SkyReelsBaseDiT
from fusion_mlx.video.skyreels_v3.transformer_a2v import SkyReelsA2VDiT
from fusion_mlx.video.skyreels_v3.transformer_r2v import SkyReelsR2VDiT
from fusion_mlx.video.skyreels_v3.transformer_v2v import SkyReelsV2VDiT

TINY = dict(
    dim=80,
    ffn_dim=160,
    num_heads=4,
    num_kv_heads=4,
    num_layers=4,
    patch_size=(1, 2, 2),
    in_dim=16,
    out_dim=16,
    text_dim=80,
    text_len=32,
    freq_dim=32,
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
)
B = 1
F, H, W = 2, 4, 4
SEQ = F * H * W


def _mk():
    return mx.random.normal((B, 16, F, H * 2, W * 2)), mx.random.uniform(0, 1, (B,))


def test_inheritance_and_hooks():
    assert issubclass(SkyReelsR2VDiT, SkyReelsBaseDiT)
    assert issubclass(SkyReelsV2VDiT, SkyReelsBaseDiT)
    assert issubclass(SkyReelsA2VDiT, SkyReelsBaseDiT)
    mx.random.seed(7)
    r2v = SkyReelsR2VDiT(dict(TINY, cross_attn_type="t2v_cross_attn"))
    v2v = SkyReelsV2VDiT(
        dict(TINY, cross_attn_type="t2v_cross_attn", temporal_window=96)
    )
    a2v = SkyReelsA2VDiT(
        dict(TINY, cross_attn_type="t2v_cross_attn", temporal_window=32, audio_dim=768)
    )
    # A2V 独有 audio_embedding (构造顺序 text -> audio -> time); R2V/V2V 无
    assert not hasattr(r2v, "audio_embedding")
    assert not hasattr(v2v, "audio_embedding")
    assert hasattr(a2v, "audio_embedding")
    # lazy_compile 标志: R2V 关, V2V/A2V 开
    assert r2v._lazy_compile is False
    assert v2v._lazy_compile is True
    assert a2v._lazy_compile is True


def test_output_shapes():
    mx.random.seed(7)
    r2v = SkyReelsR2VDiT(dict(TINY, cross_attn_type="t2v_cross_attn"))
    x, t = _mk()
    ctx = mx.random.normal((B, 32, 80))
    out = r2v(x, t, ctx, [SEQ], [(F, H, W)])
    mx.eval(out)
    assert out.shape == (B, 16, F, H * 2, W * 2)


def test_r2v_forward_partial_parity():
    mx.random.seed(7)
    m = SkyReelsR2VDiT(dict(TINY, cross_attn_type="t2v_cross_attn"))
    x, t = _mk()
    ctx = mx.random.normal((B, 32, 80))
    full = m(x, t, ctx, [SEQ], [(F, H, W)])
    mx.eval(full)
    p_none = m.forward_partial(x, t, ctx, [SEQ], [(F, H, W)], n_blocks=None)
    mx.eval(p_none)
    p_full = m.forward_partial(x, t, ctx, [SEQ], [(F, H, W)], n_blocks=m.num_layers)
    mx.eval(p_full)
    assert bool(mx.all(p_none == full))
    assert bool(mx.all(p_full == full))
    # 更少 block 必须不同 (验证 n_blocks 真的截断)
    p_half = m.forward_partial(x, t, ctx, [SEQ], [(F, H, W)], n_blocks=2)
    mx.eval(p_half)
    assert not bool(mx.all(p_half == full))


def test_v2v_forward_partial_parity():
    mx.random.seed(7)
    m = SkyReelsV2VDiT(dict(TINY, cross_attn_type="t2v_cross_attn", temporal_window=96))
    x, t = _mk()
    ctx = mx.random.normal((B, 32, 80))
    full = m(x, t, ctx, [SEQ], [(F, H, W)], temporal_len=F)
    mx.eval(full)
    p = m.forward_partial(
        x, t, ctx, [SEQ], [(F, H, W)], n_blocks=m.num_layers, temporal_len=F
    )
    mx.eval(p)
    assert bool(mx.all(p == full)), float(mx.max(mx.abs(p - full)))


def test_a2v_forward_partial_parity():
    mx.random.seed(7)
    m = SkyReelsA2VDiT(
        dict(TINY, cross_attn_type="t2v_cross_attn", temporal_window=32, audio_dim=768)
    )
    x, t = _mk()
    aud = mx.random.normal((B, 10, 768))
    txt = mx.random.normal((B, 32, 80))
    full = m(x, t, aud, txt, [SEQ], [(F, H, W)], temporal_len=F)
    mx.eval(full)
    p = m.forward_partial(
        x, t, aud, txt, [SEQ], [(F, H, W)], n_blocks=m.num_layers, temporal_len=F
    )
    mx.eval(p)
    assert bool(mx.all(p == full)), float(mx.max(mx.abs(p - full)))


def test_v2v_lazy_compile_tolerance():
    # CP4: mx.compile 融合噪声 (pre-existing, OLD inline 与 NEW _run_blocks 一致),
    # compiled(2nd) vs uncompiled(1st) maxdiff < 1e-2
    mx.random.seed(7)
    m = SkyReelsV2VDiT(dict(TINY, cross_attn_type="t2v_cross_attn", temporal_window=96))
    x, t = _mk()
    ctx = mx.random.normal((B, 32, 80))
    o1 = m(x, t, ctx, [SEQ], [(F, H, W)], temporal_len=F)
    mx.eval(o1)  # 1st: uncompiled _call_raw
    assert (
        m._compiled_call is not None and m._compiled_call is not False
    )  # compile 已触发
    o2 = m(x, t, ctx, [SEQ], [(F, H, W)], temporal_len=F)
    mx.eval(o2)  # 2nd: compiled
    md = float(mx.max(mx.abs(o1 - o2)))
    assert md < 1e-2, f"V2V compile maxdiff {md} >= 1e-2"


def test_a2v_lazy_compile_tolerance():
    mx.random.seed(7)
    m = SkyReelsA2VDiT(
        dict(TINY, cross_attn_type="t2v_cross_attn", temporal_window=32, audio_dim=768)
    )
    x, t = _mk()
    aud = mx.random.normal((B, 10, 768))
    txt = mx.random.normal((B, 32, 80))
    o1 = m(x, t, aud, txt, [SEQ], [(F, H, W)], temporal_len=F)
    mx.eval(o1)
    assert m._compiled_call is not None and m._compiled_call is not False
    o2 = m(x, t, aud, txt, [SEQ], [(F, H, W)], temporal_len=F)
    mx.eval(o2)
    md = float(mx.max(mx.abs(o1 - o2)))
    assert md < 1e-2, f"A2V compile maxdiff {md} >= 1e-2"


def test_r2v_config_t2v_cross_attn_no_image_proj():
    # issue #188: R2V-14B 源权重 (SkyReelsA2WanI2v3DModel) 无 k_img/v_img/norm_k_img,
    # 参考图+文本 769 token 经同一 k/v 投影 -> r2v_14b MODEL_TYPES 必须用 t2v_cross_attn.
    # i2v_cross_attn 会创建 40 层 × 5 = 200 个无源权重的随机初始化 k_img/v_img/norm_k_img.
    from mlx.utils import tree_flatten

    from fusion_mlx.video.skyreels_v3.convert_skyreels_v3 import MODEL_TYPES

    assert (
        MODEL_TYPES["r2v_14b"]["cross_attn_type"] == "t2v_cross_attn"
    ), "r2v_14b MODEL_TYPES 必须用 t2v_cross_attn (源权重无 k_img/v_img/norm_k_img)"

    def _leaves(model):
        # tree_flatten 已返回点分路径字符串, 直接取 str(path)
        return [str(p) for p, _ in tree_flatten(model.parameters())]

    # t2v: 参考图 k_img/v_img/norm_k_img 不存在 (源权重无对应键, 不应随机初始化)
    mx.random.seed(7)
    t2v = SkyReelsR2VDiT(dict(TINY, cross_attn_type="t2v_cross_attn"))
    lt = _leaves(t2v)
    assert not any("k_img" in n for n in lt), "t2v 不应有 k_img"
    assert not any("v_img" in n for n in lt), "t2v 不应有 v_img"
    assert not any("norm_k_img" in n for n in lt), "t2v 不应有 norm_k_img"

    # 负对照: i2v 必须创建 k_img/v_img/norm_k_img (证明本断言能抓到 #188 回归)
    mx.random.seed(7)
    i2v = SkyReelsR2VDiT(dict(TINY, cross_attn_type="i2v_cross_attn"))
    li = _leaves(i2v)
    assert any("k_img" in n for n in li), "i2v 负对照必须有 k_img"
    assert any("v_img" in n for n in li), "i2v 负对照必须有 v_img"
    assert any("norm_k_img" in n for n in li), "i2v 负对照必须有 norm_k_img"
