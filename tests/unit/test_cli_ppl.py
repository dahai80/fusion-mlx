# SPDX-License-Identifier: Apache-2.0
"""D2.7: ppl CLI corpus loader + parser wiring tests.

Real-model ppl is gated to FUSION_MLX_REAL_MODEL_TESTS — these cover the
deterministic pieces: calibration corpus loading/filtering and argparse
wiring. ppl_command itself loads MLX weights (no mock path by design).
"""

import argparse

from fusion_mlx.cli_ppl import (
    _CALIBRATION_PATH,
    _load_corpus,
    add_ppl_parser,
    ppl_command,
)


def test_calibration_data_exists():
    assert (
        _CALIBRATION_PATH.exists()
    ), "oq_calibration_data.json must ship with the package"


def test_load_corpus_filters_non_strings_and_caps_samples():
    corpus = _load_corpus(samples_per_category=3)
    assert isinstance(corpus, dict)
    assert len(corpus) >= 1
    for cat, texts in corpus.items():
        assert isinstance(cat, str)
        assert len(texts) <= 3
        for t in texts:
            assert isinstance(t, str)
            assert t.strip()


def test_load_corpus_zero_samples_yields_empty_lists():
    corpus = _load_corpus(samples_per_category=0)
    assert all(len(v) == 0 for v in corpus.values())


def test_add_ppl_parser_registers_subcommand():
    parser = argparse.ArgumentParser(prog="fusion-mlx")
    sub = parser.add_subparsers(dest="command")
    add_ppl_parser(sub)
    args = parser.parse_args(["ppl", "qwen3.5-9b-4bit", "--quant", "int4"])
    assert args.command == "ppl"
    assert args.model == "qwen3.5-9b-4bit"
    assert args.quant == "int4"
    assert args.samples_per_category == 5
    assert args.func is ppl_command


def test_add_ppl_parser_default_quant_is_none():
    parser = argparse.ArgumentParser(prog="fusion-mlx")
    sub = parser.add_subparsers(dest="command")
    add_ppl_parser(sub)
    args = parser.parse_args(["ppl", "some-model"])
    assert args.quant is None
    assert args.samples_per_category == 5
