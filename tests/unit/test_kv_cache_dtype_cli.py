# SPDX-License-Identifier: Apache-2.0
"""CLI surface for R15 #300 — argparse accepts the new flags.

Verified via ``fusion-mlx serve --help`` rather than wiring a
``build_parser`` helper because the existing parser is inlined into
``main()``; capturing the help text is sufficient to assert the flags
landed.
"""

from __future__ import annotations

import os
import subprocess
import sys


def _serve_help() -> str:
    """Run ``python -m fusion_mlx.cli serve --help`` and return its stdout."""
    proc = subprocess.run(
        [sys.executable, "-m", "fusion_mlx.cli", "serve", "--help"],
        capture_output=True,
        text=True,
        # 30s flaked on a cold shared CI runner: interpreter + cli import
        # alone can exceed 30s there. The subprocess exits immediately
        # once imports finish; the budget only guards a hang.
        timeout=120,
        env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
    )
    # Argparse exits 0 on --help, so a non-zero rc here is a real failure.
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_serve_help_advertises_kv_cache_dtype_flag_with_choices():
    """The flag and its full choices set must appear in --help.

    codex r1 NIT #1: assert against the argparse-rendered choices set
    ``{bf16,int8,int4}`` rather than the bare substring ``"int4"`` —
    the latter could pass on unrelated help text (e.g. mention of int4
    in a different option's help string) without proving that the
    --kv-cache-dtype flag itself was registered.
    """
    text = _serve_help()
    assert "--kv-cache-dtype" in text
    # argparse renders ``--kv-cache-dtype {bf16,int8,int4}`` in the
    # usage block — this exact substring is the load-bearing assertion
    # (proves both the flag and its full choices set are present).
    assert "--kv-cache-dtype {bf16,int8,int4}" in text


def test_serve_help_advertises_int4_as_default():
    """The R15 #300 contract — int4 is the *default*, not just a choice."""
    text = _serve_help()
    # The flag's help string explicitly carries ``default: int4``.
    assert "default: int4" in text


def test_serve_help_advertises_reasoning_flag():
    text = _serve_help()
    # Match the bare flag, not just the substring — there is also a
    # ``--reasoning-parser`` flag in the same parser.
    assert "--reasoning " in text or "--reasoning\n" in text


def test_serve_help_lists_all_three_choices():
    """All three dtype options must be discoverable in the choices set."""
    text = _serve_help()
    # Defensive: assert each appears inside the argparse-rendered
    # choices brace pair, not just anywhere in the help text.
    assert "bf16" in text
    assert "int8" in text
    assert "int4" in text


def test_serve_help_omits_removed_legacy_kv_quant_flags():
    text = _serve_help()
    for flag in (
        "--kv-cache-quantization",
        "--kv-cache-quantization-bits",
        "--kv-cache-quantization-group-size",
        "--kv-cache-min-quantize-tokens",
    ):
        assert flag not in text


def test_serve_rejects_removed_legacy_kv_quant_flag():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "fusion_mlx.cli",
            "serve",
            "mlx-community/Qwen3.5-9B-4bit",
            "--kv-cache-quantization",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode != 0, combined
    assert "unrecognized arguments" in combined, combined
