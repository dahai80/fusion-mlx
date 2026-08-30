import math

import mlx.core as mx

from fusion_mlx.video.minimax_h3.vae import Attention


def _rms_norm_noaffine(x, eps=1e-5):
    # official qk_norm_type="rms_norm", qk_norm_affine=False: x / sqrt(mean(x^2)+eps)
    rms = mx.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return x / rms


def _manual_attn(q, k, v, heads, dim_head, qk_norm=False, eps=1e-5):
    # reference SDPA with optional qk RMSNorm (no affine), mirroring official attention.py
    if qk_norm:
        q = _rms_norm_noaffine(q, eps)
        k = _rms_norm_noaffine(k, eps)
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    scale = 1.0 / math.sqrt(dim_head)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    return out.transpose(0, 2, 1, 3)


def _make_attn(heads, dim_head, embed_dim, qk_norm):
    attn = Attention(
        heads,
        dim_head,
        embed_dim,
        bias=False,
        eps=1e-5,
        qk_norm_type=qk_norm,
        qk_norm_affine=False,
    )
    # deterministic weights: to_qkv identity-like, to_out identity
    mx.random.seed(0)
    attn.to_qkv.weight = mx.random.normal(attn.to_qkv.weight.shape) * 0.5
    attn.to_out.weight = mx.random.normal(attn.to_out.weight.shape) * 0.02
    return attn


def test_qk_norm_rms_applied_to_query_and_key():
    # qk_norm="rms_norm" MUST normalize q,k before SDPA. Output must match a
    # reference that applies RMSNorm(no affine) to q,k. If the Attention impl
    # skips qk_norm, this comparison diverges.
    heads, dim_head, embed_dim = 2, 4, 8
    b, n = 1, 6

    mx.random.seed(1)
    hidden = mx.random.normal((b, n, embed_dim)) * 3.0

    attn = _make_attn(heads, dim_head, embed_dim, qk_norm="rms_norm")
    out = attn(hidden)
    mx.eval(out)

    # reference: replicate to_qkv split + qk_norm + sdpa + to_out
    qkv = attn.to_qkv(hidden).reshape(b, n, -1, 3 * dim_head)
    q, k, v = mx.split(qkv, 3, axis=-1)
    q = q.reshape(b, n, heads, dim_head)
    k = k.reshape(b, n, heads, dim_head)
    v = v.reshape(b, n, heads, dim_head)
    ref = _manual_attn(q, k, v, heads, dim_head, qk_norm=True).reshape(
        b, n, heads * dim_head
    )
    ref_out = attn.to_out(ref)
    mx.eval(ref_out)

    diff = float(mx.max(mx.abs(out - ref_out)))
    assert (
        diff < 1e-4
    ), f"qk_norm output diverges from RMSNorm reference: max abs diff={diff}"


def test_qk_norm_none_skips_normalization():
    # qk_norm="none" must NOT normalize: output matches a reference with qk_norm=False.
    heads, dim_head, embed_dim = 2, 4, 8
    b, n = 1, 6

    mx.random.seed(1)
    hidden = mx.random.normal((b, n, embed_dim)) * 3.0

    attn = _make_attn(heads, dim_head, embed_dim, qk_norm="none")
    out = attn(hidden)
    mx.eval(out)

    qkv = attn.to_qkv(hidden).reshape(b, n, -1, 3 * dim_head)
    q, k, v = mx.split(qkv, 3, axis=-1)
    q = q.reshape(b, n, heads, dim_head)
    k = k.reshape(b, n, heads, dim_head)
    v = v.reshape(b, n, heads, dim_head)
    ref = _manual_attn(q, k, v, heads, dim_head, qk_norm=False).reshape(
        b, n, heads * dim_head
    )
    ref_out = attn.to_out(ref)
    mx.eval(ref_out)

    diff = float(mx.max(mx.abs(out - ref_out)))
    assert (
        diff < 1e-4
    ), f"qk_norm=none output diverges from no-norm reference: max abs diff={diff}"


def test_qk_norm_changes_output_when_norm_matters():
    # Sanity: rms_norm vs none must produce different outputs when q,k have
    # large-norm variation (proves qk_norm actually does something, not a no-op).
    heads, dim_head, embed_dim = 2, 4, 8
    b, n = 1, 6

    mx.random.seed(1)
    hidden = mx.random.normal((b, n, embed_dim)) * 3.0

    attn_norm = _make_attn(heads, dim_head, embed_dim, qk_norm="rms_norm")
    attn_none = _make_attn(heads, dim_head, embed_dim, qk_norm="none")
    # share weights
    attn_none.to_qkv.weight = attn_norm.to_qkv.weight
    attn_none.to_out.weight = attn_norm.to_out.weight

    out_norm = attn_norm(hidden)
    out_none = attn_none(hidden)
    mx.eval(out_norm, out_none)

    diff = float(mx.max(mx.abs(out_norm - out_none)))
    assert (
        diff > 1e-3
    ), f"qk_norm rms vs none produced identical output (diff={diff}); qk_norm is a no-op"
