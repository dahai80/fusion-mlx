# SPDX-License-Identifier: Apache-2.0
"""Integration-style tests for the DFlash2 Phase-2 plumbing.

These exercise the CLI surface and alias registry only -- they do NOT load
a real model (that is Phase 3, real-model validation). They run under the
pure-python stub shims installed by tests/conftest.py, so no mlx runtime is
required.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def test_serve_parser_exposes_enable_dflash2() -> None:
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "fusion_mlx.cli", "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, out.stderr
    assert "--enable-dflash2" in out.stdout
    assert "--dflash2-drafter-path" in out.stdout
    assert "--dflash2-block-size" in out.stdout


def test_serve_parser_spec_decode_choices_include_dflash2() -> None:
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "fusion_mlx.cli", "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, out.stderr
    # --spec-decode choices render as {none,mtp,dflash,dflash2,...}
    assert "dflash2" in out.stdout


def test_alias_registry_has_qwen38_alias() -> None:
    from fusion_mlx.model_aliases import resolve_profile

    profile = resolve_profile("qwen3.8-27b-4bit")
    assert profile is not None
    assert profile.hf_path == "mlx-community/Qwen3.8-27B-4bit"
    assert profile.supports_dflash2 is True
    assert profile.is_moe is False
    assert profile.model_family == "qwen3_8"


def test_family_detection_qwen38() -> None:
    from fusion_mlx.model_auto_config import _detect_family_from_path

    assert _detect_family_from_path("mlx-community/Qwen3.8-27B-4bit") == "qwen3_8"
    assert _detect_family_from_path("Qwen3.8-27B-DFlash2") == "qwen3_8"
    # qwen3.5 must NOT match the qwen3_8 regex, and vice versa
    assert _detect_family_from_path("Qwen3.5-27B-4bit") == "qwen3_5"
    # plain qwen3 must still resolve (negative lookahead intact)
    assert _detect_family_from_path("Qwen3-8B-4bit") == "qwen3"


def test_routing_table_qwen38_prefers_dflash2() -> None:
    from fusion_mlx.speculative.auto_router import (
        METHOD_DFLASH2,
        METHOD_NGRAM,
        routing_table,
    )

    entries = [e for e in routing_table() if e.family == "qwen3_8"]
    assert entries, "qwen3_8 family has no routing entries"
    # first listed method for qwen3_8 dense must be dflash2
    assert entries[0].methods[0] == METHOD_DFLASH2
    # not_moe constraint present on the dflash2 entry
    assert "not_moe" in entries[0].constraints
    # ngram fallback entry exists for the same family
    ngram_entry = [e for e in entries if METHOD_NGRAM in e.methods]
    assert ngram_entry, "qwen3_8 missing ngram fallback route"


def test_registry_lists_dflash2_plugin() -> None:
    from fusion_mlx.speculative.registry import iter_spec_decoders

    names = [p.method for p in iter_spec_decoders()]
    assert "dflash2" in names


def test_scheduler_config_carries_dflash2_fields() -> None:
    from fusion_mlx.config import SchedulerConfig

    cfg = SchedulerConfig(dflash2_drafter_path="/tmp/draft", dflash2_block_size=5)
    assert cfg.dflash2_drafter_path == "/tmp/draft"
    assert cfg.dflash2_block_size == 5
    # defaults
    cfg_default = SchedulerConfig()
    assert cfg_default.dflash2_drafter_path == ""
    assert cfg_default.dflash2_block_size == 5


def test_per_request_route_reports_dflash2() -> None:
    from fusion_mlx.speculative.auto_router import METHOD_DFLASH, METHOD_DFLASH2
    from fusion_mlx.speculative.per_request_route import loaded_methods

    info = loaded_methods(
        mtp=False,
        dflash=False,
        dflash2=True,
        dspark=False,
        suffix=False,
    )
    assert info[METHOD_DFLASH2] is True
    assert info[METHOD_DFLASH] is False


def test_spec_routes_describe_dflash2() -> None:
    from fusion_mlx.api.spec_routes import _METHOD_DESCRIPTIONS
    from fusion_mlx.speculative.auto_router import METHOD_DFLASH2

    assert METHOD_DFLASH2 in _METHOD_DESCRIPTIONS
    assert "dflash2" in _METHOD_DESCRIPTIONS[METHOD_DFLASH2].lower()


def test_qwen38_alias_exposes_dflash2_capability() -> None:
    from fusion_mlx.model_aliases import resolve_profile

    profile = resolve_profile("qwen3.8-27b-4bit")
    assert profile is not None
    assert profile.supports_dflash2 is True
    # supports_dflash2 must flow into the capabilities set reported by
    # GET /v1/models and the CLI models command.
    assert "dflash2" in profile.capabilities
