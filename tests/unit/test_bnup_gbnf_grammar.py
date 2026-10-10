"""Unit tests for BNUP Socratic DSL v3.0 GBNF grammar (issue #1142).

Tests grammar compilation (llguidance), alias resolution, and constraint
behavior (valid/invalid visual_type, tag structure, CJK thinking text).
"""

import pathlib

import pytest

_GRAMMAR_DIR = (
    pathlib.Path(__file__).resolve().parent.parent.parent / "fusion_mlx" / "grammars"
)
_GBNF_FILE = _GRAMMAR_DIR / "socratic_dsl_v3.gbnf"


# ─── file exists ──────────────────────────────────────────────────────────────


def test_grammar_file_exists():
    assert _GBNF_FILE.exists(), f"grammar file not found: {_GBNF_FILE}"
    content = _GBNF_FILE.read_text(encoding="utf-8")
    assert "root ::=" in content
    assert "thinking-block" in content
    assert "dsl-block" in content


def test_grammar_file_has_visual_type_enum():
    content = _GBNF_FILE.read_text(encoding="utf-8")
    expected = [
        "array_grid",
        "tape_diagram",
        "geometry_2d",
        "isometric_3d",
        "track_timeline",
        "bucket_divider",
        "data_chart",
        "flow_card",
    ]
    for v in expected:
        assert v in content, f"visual_type enum missing: {v}"


# ─── alias registry ──────────────────────────────────────────────────────────


def test_alias_registered():
    from fusion_mlx.api.grammar_aliases import (
        list_grammar_aliases,
        resolve_grammar_alias,
    )

    aliases = list_grammar_aliases()
    names = [a["alias"] for a in aliases]
    assert "bnup-socratic" in names

    resolved = resolve_grammar_alias("bnup-socratic")
    assert resolved is not None
    assert "grammar" in resolved
    assert resolved["format"] == "gbnf"
    assert "root ::=" in resolved["grammar"]


def test_alias_not_found_returns_none():
    from fusion_mlx.api.grammar_aliases import resolve_grammar_alias

    assert resolve_grammar_alias("nonexistent-alias") is None


def test_is_grammar_alias():
    from fusion_mlx.api.grammar_aliases import is_grammar_alias

    assert is_grammar_alias("bnup-socratic") is True
    assert is_grammar_alias("not-an-alias") is False


# ─── grammar compilation (llguidance) ─────────────────────────────────────────

llguidance = pytest.importorskip("llguidance")


def _load_grammar_str():
    gbnf = _GBNF_FILE.read_text(encoding="utf-8")
    return llguidance.grammar_from("gbnf", gbnf)


def test_grammar_compiles_llguidance():
    grammar_str = _load_grammar_str()
    assert len(grammar_str) > 0


def _make_matcher(grammar_str):
    from fusion_mlx.api.grammar import _build_ll_tokenizer

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B",
        )
    except Exception:
        pytest.skip("no tokenizer available for matcher test")

    ll_tok = _build_ll_tokenizer(tok, vocab_size=None)
    return ll_tok, llguidance.LLMatcher(ll_tok, grammar_str)


def _test_accepts(text, should_accept=True):
    grammar_str = _load_grammar_str()
    ll_tok, _ = _make_matcher(grammar_str)
    m = llguidance.LLMatcher(ll_tok, grammar_str)

    from fusion_mlx.api.grammar import _build_ll_tokenizer

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception:
        pytest.skip("no tokenizer")

    ll_tok = _build_ll_tokenizer(tok, vocab_size=None)
    m = llguidance.LLMatcher(ll_tok, grammar_str)
    tokens = tok.encode(text, add_special_tokens=False)
    consumed = m.try_consume_tokens(tokens)
    stopped = m.is_stopped()

    if should_accept:
        assert (
            stopped
        ), f"expected acceptance but rejected: consumed={consumed} stopped={stopped} err={m.get_error()}"
    else:
        assert (
            not stopped
        ), f"expected rejection but accepted: consumed={consumed} stopped={stopped}"


_VALID_DSL = (
    "<thinking>some thinking</thinking>"
    "<dsl>"
    '{"meta":{"id":"t1","grade":3,"topic":"math","visual_type":"tape_diagram","fallback_type":"flow_card"},'
    '"parameters":{},"entities":[],'
    '"pipeline":[{"step":1,"title":"s1","formula":"1+1","eval":"1+1","result_unit":"","visual_state":{}}]}'
    "</dsl>"
)


def test_valid_dsl_accepted():
    _test_accepts(_VALID_DSL, should_accept=True)


def test_invalid_visual_type_rejected():
    invalid = _VALID_DSL.replace('"tape_diagram"', '"INVALID_TYPE"')
    _test_accepts(invalid, should_accept=False)


def test_missing_closing_thinking_rejected():
    bad = "<thinking>text<dsl>{}</dsl>"
    _test_accepts(bad, should_accept=False)


def test_missing_pipeline_rejected():
    invalid = (
        "<thinking>x</thinking>"
        "<dsl>"
        '{"meta":{"id":"t1","grade":3,"topic":"m","visual_type":"flow_card","fallback_type":"flow_card"},'
        '"parameters":{},"entities":[]}'
        "</dsl>"
    )
    _test_accepts(invalid, should_accept=False)


def test_chinese_thinking_accepted():
    chinese = (
        "<thinking>识别知识点：圆柱与圆锥。错因预判：学生易混淆。</thinking>"
        "<dsl>"
        '{"meta":{"id":"t1","grade":6,"topic":"geo","visual_type":"isometric_3d","fallback_type":"flow_card"},'
        '"parameters":{},"entities":[],'
        '"pipeline":[{"step":1,"title":"计算","formula":"V","eval":"V","result_unit":"cm3","visual_state":{}}]}'
        "</dsl>"
    )
    _test_accepts(chinese, should_accept=True)


def test_all_visual_types_accepted():
    visual_types = [
        "array_grid",
        "tape_diagram",
        "geometry_2d",
        "isometric_3d",
        "track_timeline",
        "bucket_divider",
        "data_chart",
        "flow_card",
    ]
    for vt in visual_types:
        dsl = (
            "<thinking>x</thinking>"
            "<dsl>"
            f'{{"meta":{{"id":"t","grade":1,"topic":"t","visual_type":"{vt}","fallback_type":"flow_card"}},'
            '"parameters":{},"entities":[],'
            '"pipeline":[{"step":1,"title":"s","formula":"x","eval":"x","result_unit":"u","visual_state":{}}]}'
            "</dsl>"
        )
        _test_accepts(dsl, should_accept=True)


# ─── fallback_type enum (spec line 1016: tape_diagram | flow_card only) ──────


def test_grammar_file_has_fallback_type_enum():
    content = _GBNF_FILE.read_text(encoding="utf-8")
    assert "fallback-type-enum" in content, "fallback-type-enum rule missing"
    assert (
        'fallback-type-enum ::= "\\"tape_diagram\\"" | "\\"flow_card\\""' in content
    ), "fallback-type-enum must be exactly tape_diagram | flow_card (spec line 1016)"
    assert "fallback_type" in content


def test_valid_fallback_types_accepted():
    for ft in ("tape_diagram", "flow_card"):
        dsl = _VALID_DSL.replace(
            '"fallback_type":"flow_card"', f'"fallback_type":"{ft}"'
        )
        _test_accepts(dsl, should_accept=True)


def test_invalid_fallback_type_rejected():
    invalid = _VALID_DSL.replace(
        '"fallback_type":"flow_card"', '"fallback_type":"array_grid"'
    )
    _test_accepts(invalid, should_accept=False)


def test_invalid_fallback_geometry_rejected():
    invalid = _VALID_DSL.replace(
        '"fallback_type":"flow_card"', '"fallback_type":"geometry_2d"'
    )
    _test_accepts(invalid, should_accept=False)


# ─── xgrammar compilation (gap 2: both backends must compile) ────────────────


def test_grammar_compiles_xgrammar():
    xgrammar = pytest.importorskip("xgrammar")
    from fusion_mlx._torch_stub import install as _install_torch_stub

    _install_torch_stub()

    gbnf_text = _GBNF_FILE.read_text(encoding="utf-8")
    grammar = xgrammar.Grammar.from_ebnf(gbnf_text)
    assert grammar is not None
    compiler = xgrammar.GrammarCompiler(xgrammar.TokenizerInfo(["<pad>"], vocab_size=1))
    compiled = compiler.compile_grammar(grammar)
    assert compiled is not None


def test_grammar_compiles_xgrammar_matcher():
    xgrammar = pytest.importorskip("xgrammar")
    from fusion_mlx._torch_stub import install as _install_torch_stub

    _install_torch_stub()

    gbnf_text = _GBNF_FILE.read_text(encoding="utf-8")
    grammar = xgrammar.Grammar.from_ebnf(gbnf_text)
    compiler = xgrammar.GrammarCompiler(xgrammar.TokenizerInfo(["<pad>"], vocab_size=1))
    compiled = compiler.compile_grammar(grammar)
    matcher = xgrammar.GrammarMatcher(compiled)
    assert matcher is not None
