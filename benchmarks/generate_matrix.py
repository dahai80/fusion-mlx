#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Public benchmark matrix generator for fusion-mlx.
# Reads ALL benchmarks/reports/*.json (individual + SUMMARY), dedupes to the
# latest run per model, derives quant from the model id, detects the host chip
# (reports do not record it), and emits MATRIX.md + matrix.json.
# Gaps (missing TTFT, mem GB, spec-decode, errored runs) are logged loudly, not
# skipped silently (Rule 12).

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [matrix] %(message)s",
)
logger = logging.getLogger("fusion_bench_matrix")

REPORTS_DIR = Path(__file__).parent / "reports"
MATRIX_MD = Path(__file__).parent / "MATRIX.md"
MATRIX_JSON = Path(__file__).parent / "matrix.json"

_QUANT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("a4b", re.compile(r"(?:^|[-_])a4b(?:$|[-_])", re.I)),
    ("bf16", re.compile(r"(?:^|[-_])bf16(?:$|[-_])", re.I)),
    ("mxfp8", re.compile(r"(?:^|[-_])mxfp8(?:$|[-_])", re.I)),
    ("fp8", re.compile(r"(?:^|[-_])fp8(?:$|[-_])", re.I)),
    ("8bit", re.compile(r"(?:^|[-_])8bit(?:$|[-_])", re.I)),
    ("8-bit", re.compile(r"(?:^|[-_])8-?bit(?:$|[-_])", re.I)),
    ("4bit", re.compile(r"(?:^|[-_])4bit(?:$|[-_])", re.I)),
    ("4-bit", re.compile(r"(?:^|[-_])4-?bit(?:$|[-_])", re.I)),
]


def derive_quant(model_id: str) -> str:
    for label, pat in _QUANT_PATTERNS:
        if pat.search(model_id):
            return label
    return "unknown"


def display_name(model_id: str) -> str:
    name = model_id
    for prefix in ("mlx-community--", "mlx-community-"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name


def detect_chip() -> str:
    try:
        import subprocess

        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        if out:
            return out
    except Exception as exc:
        logger.warning("chip detect via sysctl failed: %s", exc)
    return "unknown"


def _iter_result_dicts(raw: Any) -> list[dict]:
    if isinstance(raw, dict):
        if "results" in raw and isinstance(raw["results"], list):
            return [r for r in raw["results"] if isinstance(r, dict)]
        if "model" in raw:
            return [raw]
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    return []


def load_reports(reports_dir: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    files = sorted(reports_dir.glob("*.json")) if reports_dir.exists() else []
    if not files:
        logger.warning("no report JSON found in %s", reports_dir)
        return latest
    for f in files:
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("skip unparseable %s: %s", f.name, exc)
            continue
        for r in _iter_result_dicts(raw):
            mid = r.get("model")
            if not mid:
                continue
            ts = r.get("timestamp", "")
            prev = latest.get(mid)
            if prev is None or str(ts) >= str(prev.get("timestamp", "")):
                latest[mid] = r
    logger.info("loaded %d unique models from %d files", len(latest), len(files))
    return latest


def build_rows(latest: dict[str, dict], chip: str) -> list[dict]:
    rows: list[dict] = []
    for mid, r in latest.items():
        quant = derive_quant(mid)
        toks = r.get("tokens_per_second")
        ttft = r.get("ttft_seconds")
        prompt_tok = r.get("prompt_tokens")
        comp_tok = r.get("completion_tokens")
        wall = r.get("wall_seconds")
        prefill_tps = None
        if ttft and prompt_tok and ttft > 0:
            prefill_tps = round(prompt_tok / ttft, 2)
        row = {
            "model_id": mid,
            "model": display_name(mid),
            "chip": chip,
            "quant": quant,
            "tok_per_sec_decode": toks if toks is not None else None,
            "tok_per_sec_prefill": prefill_tps,
            "ttft_seconds": ttft,
            "mem_gb": None,
            "spec_decode": None,
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp_tok,
            "wall_seconds": wall,
            "timestamp": r.get("timestamp", ""),
            "error": r.get("error"),
        }
        rows.append(row)
    rows.sort(key=lambda x: (x["chip"], x["quant"], -(x["tok_per_sec_decode"] or 0)))
    return rows


def log_gaps(rows: list[dict]) -> int:
    gaps = 0
    for row in rows:
        if row["error"]:
            logger.warning(
                "GAP model=%s errored run: %s", row["model_id"], row["error"][:120]
            )
            gaps += 1
        if row["tok_per_sec_decode"] is None and not row["error"]:
            logger.warning("GAP model=%s missing decode tok/s", row["model_id"])
            gaps += 1
        if row["ttft_seconds"] is None and not row["error"]:
            logger.warning(
                "GAP model=%s missing TTFT (prefill tok/s unavailable)", row["model_id"]
            )
            gaps += 1
        if row["mem_gb"] is None:
            logger.info(
                "GAP model=%s mem GB not recorded by run_bench.py", row["model_id"]
            )
        if row["spec_decode"] is None:
            logger.info(
                "GAP model=%s spec-decode flag not recorded by run_bench.py",
                row["model_id"],
            )
    return gaps


def _fmt(val: Any, fmt: str = "") -> str:
    if val is None:
        return "n/a"
    if fmt:
        return format(val, fmt)
    return str(val)


def render_matrix_md(rows: list[dict], chip: str) -> str:
    lines: list[str] = []
    lines.append("# Benchmark Matrix")
    lines.append("")
    lines.append("Auto-generated by `generate_matrix.py` — do not edit by hand.")
    lines.append("Source: `benchmarks/reports/*.json`. Regenerate after a fresh")
    lines.append("`run_bench.py` run: `python benchmarks/generate_matrix.py`.")
    lines.append("")
    chips: dict[str, list[dict]] = {}
    for row in rows:
        chips.setdefault(row["chip"], []).append(row)
    for c in sorted(chips):
        lines.append(f"## {c}")
        lines.append("")
        lines.append(
            "| Model | Quant | tok/s decode | tok/s prefill | TTFT (s) | "
            "mem GB | spec-decode |"
        )
        lines.append(
            "|-------|-------|--------------|---------------|----------|--------|-------------|"
        )
        for row in chips[c]:
            lines.append(
                f"| {row['model']} | {row['quant']} | "
                f"{_fmt(row['tok_per_sec_decode'], '.1f')} | "
                f"{_fmt(row['tok_per_sec_prefill'], '.1f')} | "
                f"{_fmt(row['ttft_seconds'], '.3f')} | "
                f"{_fmt(row['mem_gb'])} | {_fmt(row['spec_decode'])} |"
            )
        lines.append("")
    err_rows = [r for r in rows if r["error"]]
    if err_rows:
        lines.append("## Errored runs (reported, not skipped)")
        lines.append("")
        for r in err_rows:
            lines.append(f"- `{r['model_id']}`: {r['error'][:160]}")
        lines.append("")
    lines.append("## Gaps")
    lines.append("")
    lines.append("- `mem GB`: `run_bench.py` does not record resident memory. Gap.")
    lines.append("- `spec-decode`: spec-decode mode is a server config, not in the")
    lines.append(
        "  report JSON. Gap. Re-run with `--spec-decode` flag recorded to fill."
    )
    lines.append("- `tok/s prefill`: approximated as `prompt_tokens / ttft_seconds`.")
    lines.append("  Unavailable when TTFT was not captured.")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_outputs(rows: list[dict], chip: str) -> None:
    md = render_matrix_md(rows, chip)
    MATRIX_MD.write_text(md, encoding="utf-8")
    logger.info("wrote %s (%d rows)", MATRIX_MD, len(rows))
    payload = {
        "generated_for_chip": chip,
        "row_count": len(rows),
        "rows": rows,
    }
    MATRIX_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("wrote %s", MATRIX_JSON)


def main() -> int:
    ap = argparse.ArgumentParser(description="fusion-mlx public bench matrix generator")
    ap.add_argument("--reports-dir", default=str(REPORTS_DIR))
    ap.add_argument("--chip", default=None, help="override detected chip name")
    args = ap.parse_args()

    chip = args.chip or detect_chip()
    logger.info("chip=%s (reports do not record chip; inferred from host)", chip)

    latest = load_reports(Path(args.reports_dir))
    if not latest:
        logger.error("no benchmark data — run benchmarks/run_bench.py first")
        return 2

    rows = build_rows(latest, chip)
    gaps = log_gaps(rows)
    write_outputs(rows, chip)
    logger.info("matrix built: %d rows, %d logged gaps", len(rows), gaps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
