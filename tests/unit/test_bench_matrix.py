# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

import benchmarks.generate_matrix as gm


def _fake_report(
    model: str, tps: float, ttft: float | None, error: str | None = None
) -> dict:
    r = {
        "model": model,
        "prompt_tokens_requested": 512,
        "gen_tokens_requested": 128,
        "prompt_tokens": 700,
        "completion_tokens": 128,
        "wall_seconds": 128 / tps if tps else 1.0,
        "tokens_per_second": tps,
        "ttft_seconds": ttft,
        "timestamp": "20260101-000000",
        "base_url": "http://127.0.0.1:11434",
    }
    if error:
        r = {"model": model, "error": error, "timestamp": "20260101-000000"}
    return r


@pytest.fixture()
def tmp_reports(tmp_path: Path) -> Path:
    rep_a = _fake_report("Qwen3-0.6B-4bit", 91.72, 0.462)
    rep_b = _fake_report("Meta-Llama-3.1-8B-Instruct-4bit", 59.07, 1.159)
    rep_err = _fake_report("phi-4-4bit", 0, None, error="HTTP 404 not found")
    (tmp_path / "Qwen3-0.6B-4bit_20260101-000000.json").write_text(
        json.dumps(rep_a), encoding="utf-8"
    )
    (tmp_path / "Meta-Llama-3.1-8B-Instruct-4bit_20260101-000000.json").write_text(
        json.dumps(rep_b), encoding="utf-8"
    )
    (tmp_path / "phi-4-4bit_20260101-000000.json").write_text(
        json.dumps(rep_err), encoding="utf-8"
    )
    summary = {"timestamp": "20260101-000000", "results": [rep_a, rep_b, rep_err]}
    (tmp_path / "SUMMARY_20260101-000000.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    return tmp_path


def test_derive_quant():
    assert gm.derive_quant("Qwen3-0.6B-4bit") == "4bit"
    assert gm.derive_quant("Qwen3-0.6B-8bit") == "8bit"
    assert gm.derive_quant("Qwen3.5-4B-bf16") == "bf16"
    assert gm.derive_quant("Qwen3.6-27B-mxfp8") == "mxfp8"
    assert gm.derive_quant("gemma-4-26b-a4b-it-4bit") == "a4b"
    assert gm.derive_quant("foo") == "unknown"


def test_display_name_strips_prefix():
    assert (
        gm.display_name("mlx-community--Qwen2.5-Coder-32B-Instruct-4bit")
        == "Qwen2.5-Coder-32B-Instruct-4bit"
    )
    assert (
        gm.display_name("mlx-community-Llama-3.2-1B-Instruct-4bit")
        == "Llama-3.2-1B-Instruct-4bit"
    )
    assert gm.display_name("Qwen3-0.6B-4bit") == "Qwen3-0.6B-4bit"


def test_load_reports_dedupes(tmp_reports: Path):
    latest = gm.load_reports(tmp_reports)
    assert set(latest.keys()) == {
        "Qwen3-0.6B-4bit",
        "Meta-Llama-3.1-8B-Instruct-4bit",
        "phi-4-4bit",
    }


def test_build_rows_and_matrix(tmp_reports: Path, tmp_path: Path, monkeypatch):
    latest = gm.load_reports(tmp_reports)
    rows = gm.build_rows(latest, "Apple M5 Max")
    by_model = {r["model_id"]: r for r in rows}
    assert by_model["Qwen3-0.6B-4bit"]["quant"] == "4bit"
    assert by_model["Qwen3-0.6B-4bit"]["tok_per_sec_decode"] == 91.72
    assert by_model["Qwen3-0.6B-4bit"]["tok_per_sec_prefill"] == round(700 / 0.462, 2)
    assert by_model["Meta-Llama-3.1-8B-Instruct-4bit"]["quant"] == "4bit"
    assert by_model["phi-4-4bit"]["error"] is not None

    out_md = tmp_path / "MATRIX.md"
    out_json = tmp_path / "matrix.json"
    monkeypatch.setattr(gm, "MATRIX_MD", out_md)
    monkeypatch.setattr(gm, "MATRIX_JSON", out_json)
    gm.write_outputs(rows, "Apple M5 Max")

    assert out_md.exists()
    assert out_json.exists()
    md_text = out_md.read_text(encoding="utf-8")
    assert "Apple M5 Max" in md_text
    assert "Qwen3-0.6B-4bit" in md_text
    assert "4bit" in md_text
    assert "Errored runs" in md_text
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["row_count"] == 3
    assert any(r["model_id"] == "phi-4-4bit" for r in payload["rows"])
