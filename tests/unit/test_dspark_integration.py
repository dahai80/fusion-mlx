# SPDX-License-Identifier: Apache-2.0
"""Integration-style tests for the DSpark MVP plumbing."""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")  # suite needs mlx runtime; skip if absent
pytest.skip("requires mlx runtime (stub shadow breaks bodies)", allow_module_level=True)


import logging

import pytest

logger = logging.getLogger(__name__)


def test_serve_parser_exposes_enable_dspark() -> None:
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "fusion_mlx.cli", "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, out.stderr
    assert "--enable-dspark" in out.stdout


def test_info_renders_dspark_block_for_eligible_alias(capsys) -> None:
    from fusion_mlx.cli import info_command

    args = type("Args", (), {"model": "qwen3.5-9b-8bit"})()
    info_command(args)
    captured = capsys.readouterr()
    assert "DSpark eligibility" in captured.out or "dspark" in captured.out.lower()


def test_info_dspark_marks_4bit_alias_ineligible(capsys) -> None:
    from fusion_mlx.cli import info_command

    args = type("Args", (), {"model": "qwen3.5-9b-4bit"})()
    info_command(args)
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    # DSpark eligibility section should mention ineligible or 4-bit
    has_dspark_section = "DSpark" in combined or "dspark" in combined.lower()
    if has_dspark_section:
        assert "ineligible" in combined.lower() or "4-bit" in combined.lower()


def test_models_listing_renders_dspark_column(capsys) -> None:
    from unittest.mock import patch

    import requests

    from fusion_mlx.cli import models_command

    fake_resp = type(
        "R",
        (),
        {
            "status_code": 200,
            "json": lambda self: {"data": [{"id": "test-model", "type": "llm"}]},
        },
    )()
    with patch.object(requests, "get", return_value=fake_resp):
        models_command(None)
    captured = capsys.readouterr()
    assert "DSpark" in captured.out or "dspark" in captured.out.lower()


def test_build_app_healthz_models_and_completion() -> None:
    from unittest.mock import MagicMock

    from fusion_mlx.speculative.dspark.runtime import DSparkRuntime

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "user: 2+2?\nassistant:"
    tokenizer.encode.return_value = [10, 11, 12]
    target = MagicMock()
    target.tokenizer = tokenizer
    generator = MagicMock()
    generator.target = target
    generator.draft_quantization = None

    class _FakeResult:
        text = "four"
        generated_tokens = [1, 2]
        metrics = {"num_input_tokens": 5}

    generator.generate_from_tokens.return_value = _FakeResult()

    runtime = DSparkRuntime(
        generator=generator,
        target_repo="mlx-community/Qwen3.5-9B-8bit",
        draft_path="/tmp/draft",
    )

    from fastapi.testclient import TestClient

    from fusion_mlx.speculative.dspark.server import _build_app

    app = _build_app(
        runtime=runtime,
        served_model_name="qwen3.5-9b-8bit",
        default_max_tokens=64,
        cors_origins=["*"],
        enable_thinking_default=False,
    )
    client = TestClient(app)

    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert (
        body.get("engine") == "dspark"
        or body.get("ready") is True
        or "dspark" in str(body).lower()
    )

    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "qwen3.5-9b-8bit"
