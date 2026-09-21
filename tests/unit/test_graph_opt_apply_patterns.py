# SPDX-License-Identifier: Apache-2.0
"""#918: apply_patterns structural rewrite + honest compile_with_custom_pass.

MLX uses channels-last (NHWC) layout. nn.Module vars() exposes only internal
flags — children come from children() (name -> Module | list | dict), which is
exactly what apply_patterns walks.
"""

import logging

import mlx.core as mx
import mlx.nn as nn

from fusion_mlx.graph_opt import apply_patterns, compile_with_custom_pass
from fusion_mlx.graph_opt.passes import _PATTERN_REGISTRY
from fusion_mlx.nn_ext import SafeGroupNorm


def _make_triples():
    return [
        nn.Conv2d(4, 4, 3, padding=1),
        nn.GroupNorm(2, 4),
        nn.SiLU(),
    ] * 2


class TripleHolder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = _make_triples()


def test_apply_patterns_rewrites_list_container():
    m = TripleHolder()
    n = apply_patterns(m)
    assert n == 2
    assert [type(b).__name__ for b in m.blocks] == [
        "ConvGroupNormSiLU",
        "ConvGroupNormSiLU",
    ]


def test_apply_patterns_rewrites_sequential():
    m = TripleHolder()
    m.seq = nn.Sequential(*_make_triples())
    n = apply_patterns(m)
    assert n == 4
    assert len(m.seq.layers) == 2
    assert all(isinstance(l, nn.Module) for l in m.seq.layers)


def test_apply_patterns_rewrites_dict_container():
    m = TripleHolder()
    m.d = {"a": nn.Conv2d(4, 4, 3, padding=1), "b": nn.GroupNorm(2, 4), "c": nn.SiLU()}
    n = apply_patterns(m)
    assert n == 3  # 2 in blocks + 1 in dict
    assert list(m.d.keys()) == ["a"]
    assert type(m.d["a"]).__name__ == "ConvGroupNormSiLU"


def test_apply_patterns_skips_non_triples():
    m = TripleHolder()
    m.partial = [nn.Conv2d(4, 4, 3, padding=1), nn.GroupNorm(2, 4)]
    n = apply_patterns(m)
    assert n == 2
    assert len(m.partial) == 2


def test_apply_patterns_output_parity_and_weight_sharing(monkeypatch):
    # Parity target is the 3-op chain path; pin env so an externally-set
    # FUSION_FUSED_CONV_GN_SILU=1 (fp16 MSL kernel) cannot flip the path.
    monkeypatch.delenv("FUSION_FUSED_CONV_GN_SILU", raising=False)
    m = TripleHolder()
    x = mx.random.normal((1, 8, 8, 4))
    mx.eval(m.parameters())
    c1, g1, s1, c2, g2, s2 = m.blocks
    ref = s2(g2(c2(s1(g1(c1(x))))))
    mx.eval(ref)
    apply_patterns(m)
    out = m.blocks[1](m.blocks[0](x))
    mx.eval(out)
    assert mx.allclose(out, ref, atol=1e-5).item()
    assert m.blocks[0].conv is c1
    assert m.blocks[1].conv is c2


def test_apply_patterns_preserves_groupnorm_semantics():
    # MLX GroupNorm(pytorch_compatible=False) groups differently from the
    # torch layout — the fused module must keep the source setting (#918).
    m = TripleHolder()
    apply_patterns(m)
    assert isinstance(m.blocks[0].groupnorm, SafeGroupNorm)
    assert m.blocks[0].groupnorm.pytorch_compatible is False
    assert m.blocks[0].groupnorm.eps == 1e-5


def test_apply_patterns_idempotent():
    m = TripleHolder()
    assert apply_patterns(m) == 2
    assert apply_patterns(m) == 0
    assert [type(b).__name__ for b in m.blocks] == [
        "ConvGroupNormSiLU",
        "ConvGroupNormSiLU",
    ]


def test_apply_patterns_nested_module_descends():
    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = TripleHolder()

    o = Outer()
    assert apply_patterns(o) == 2
    assert type(o.inner.blocks[0]).__name__ == "ConvGroupNormSiLU"


def test_compile_with_custom_pass_warns_patterns_not_applied(caplog):
    assert len(_PATTERN_REGISTRY) > 0

    def fn(x):
        return x * 2

    with caplog.at_level(logging.INFO, logger="fusion_mlx.graph_opt.passes"):
        compile_with_custom_pass(fn)
    assert any("not applied" in r.message for r in caplog.records)


def test_compile_with_custom_pass_silent_without_patterns(caplog):
    saved = dict(_PATTERN_REGISTRY)
    _PATTERN_REGISTRY.clear()
    try:

        def fn(x):
            return x * 2

        with caplog.at_level(logging.INFO, logger="fusion_mlx.graph_opt.passes"):
            compile_with_custom_pass(fn)
        assert not any("not applied" in r.message for r in caplog.records)
    finally:
        _PATTERN_REGISTRY.update(saved)


def test_compile_with_custom_pass_tensor_pass_applied():
    def fn(x):
        return x * 2

    compiled = compile_with_custom_pass(fn, pass_fn=lambda outs: [o + 1 for o in outs])
    x = mx.array([1.0, 2.0])
    out = compiled(x)
    mx.eval(out)
    # pass runs first: (x+1)*2
    assert mx.allclose(out, mx.array([4.0, 6.0])).item()
