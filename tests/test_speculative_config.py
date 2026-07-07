# SPDX-License-Identifier: Apache-2.0
# Migrated from Rapid-MLX test_speculative_config.py
# Adapted to fusion-mlx config/registry data model

from __future__ import annotations

import subprocess
import sys

import pytest

try:
    from fusion_mlx.speculative.config import (
        SpeculativeConfigError,
        parse_speculative_config,
        require_migrated_speculative_config,
    )
    from fusion_mlx.speculative.registry import get_spec_decoder, iter_spec_decoders

    _HAS_SPEC_CONFIG = True
except ImportError:
    _HAS_SPEC_CONFIG = False


def _require_spec_config():
    if not _HAS_SPEC_CONFIG:
        pytest.skip("fusion_mlx.speculative.config/registry not migrated yet")


@pytest.fixture(autouse=True)
def _guard_spec_config():
    _require_spec_config()


def test_parse_speculative_config_accepts_vllm_common_keys() -> None:
    cfg = parse_speculative_config(
        '{"method":"mtp","model":"local/draft","num_speculative_tokens":4}'
    )

    assert cfg is not None
    assert cfg.method == "mtp"
    assert cfg.model == "local/draft"
    assert cfg.num_speculative_tokens == 4
    assert cfg.tree_budget == 0


def test_parse_ddtree_speculative_config_accepts_method_keys() -> None:
    cfg = parse_speculative_config(
        '{"method":"ddtree","model":"local/draft",'
        '"num_speculative_tokens":8,"tree_budget":24}'
    )

    assert cfg is not None
    assert cfg.method == "ddtree"
    assert cfg.model == "local/draft"
    assert cfg.num_speculative_tokens == 8
    assert cfg.tree_budget == 24


def test_parse_dflash_speculative_config_accepts_drafter_model() -> None:
    cfg = parse_speculative_config(
        '{"method":"dflash","model":"z-lab/Qwen3.5-27B-DFlash"}'
    )

    assert cfg is not None
    assert cfg.method == "ddtree"
    assert cfg.model == "z-lab/Qwen3.5-27B-DFlash"
    assert cfg.num_speculative_tokens == 5
    assert cfg.tree_budget == 0


def test_parse_speculative_config_normalizes_registered_alias() -> None:
    cfg = parse_speculative_config('{"method":"ngram"}')

    assert cfg is not None
    assert cfg.method == "suffix"


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("", "empty speculative config"),
        ("[]", "JSON object"),
        ("{bad", "invalid JSON"),
        ('{"model":"x"}', "'method' key is required"),
        ('{"method":"mtp","num_speculative_tokens":0}', "positive int"),
        ('{"method":"unknown"}', "unknown method"),
    ],
)
def test_parse_speculative_config_rejects_bad_payloads(raw: str, match: str) -> None:
    with pytest.raises(SpeculativeConfigError, match=match):
        parse_speculative_config(raw)


def test_require_migrated_speculative_config_accepts_mtp() -> None:
    cfg = require_migrated_speculative_config('{"method":"mtp"}')
    assert cfg is not None
    assert cfg.method == "mtp"


def test_require_migrated_speculative_config_accepts_ddtree() -> None:
    cfg = require_migrated_speculative_config('{"method":"ddtree"}')
    assert cfg is not None


def test_require_migrated_speculative_config_accepts_dflash() -> None:
    cfg = require_migrated_speculative_config('{"method":"dflash"}')
    assert cfg is not None
    assert cfg.method == "ddtree"


def test_spec_decoder_registry_lists_existing_backends() -> None:
    methods = {plugin.method for plugin in iter_spec_decoders()}

    assert {"ddtree", "mtp", "suffix"}.issubset(methods)
    assert get_spec_decoder("ddtree").config_enabled is True
    assert get_spec_decoder("dflash") == get_spec_decoder("ddtree")
    assert get_spec_decoder("mtp").config_enabled is True
    assert get_spec_decoder("ngram") == get_spec_decoder("suffix")


def test_serve_help_exposes_spec_decode_flag() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "fusion_mlx.cli", "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    assert "--spec-decode" in proc.stdout
