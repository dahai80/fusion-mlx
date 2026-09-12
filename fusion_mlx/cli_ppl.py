# SPDX-License-Identifier: Apache-2.0
"""D2.7/G12/L17: perplexity cost CLI.

``fusion-mlx ppl <model> [--quant <mode>]`` computes mean cross-entropy
perplexity on the oq calibration corpus (code/en/zh/ja/ko/tool_calling/
reasoning). Quantization cost is measurable: a 4-bit model should show
higher ppl than its bf16 parent. The ``--quant`` flag labels the output
so an operator can build the README quant-cost table from real runs.

Real-model only: loads weights via mlx_lm.load. No mock path — ppl on a
mocked model is meaningless. Gate via the model path argument.
"""

import argparse
import json
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

_CALIBRATION_PATH = Path(__file__).resolve().parent / "oq_calibration_data.json"


def _load_corpus(samples_per_category: int) -> dict[str, list[str]]:
    if not _CALIBRATION_PATH.exists():
        raise FileNotFoundError(f"calibration data not found: {_CALIBRATION_PATH}")
    raw = json.loads(_CALIBRATION_PATH.read_text())
    corpus: dict[str, list[str]] = {}
    for cat, texts in raw.items():
        if not isinstance(texts, list):
            continue
        corpus[cat] = [t for t in texts if isinstance(t, str) and t.strip()][
            :samples_per_category
        ]
    logger.info(
        "ppl: loaded %d categories, %d total samples",
        len(corpus),
        sum(len(v) for v in corpus.values()),
    )
    return corpus


def _compute_loss(model, tokenizer, text: str) -> tuple[float, int]:
    import mlx.core as mx

    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return 0.0, 0
    input_ids = mx.array(tokens[:-1])
    target_ids = mx.array(tokens[1:])
    logits = model(input_ids[None])
    log_probs = mx.nn.losses.cross_entropy(logits, target_ids[None], reduction="mean")
    return float(log_probs), len(target_ids)


def ppl_command(args: argparse.Namespace) -> int:
    import mlx_lm

    model_path = args.model
    quant_label = args.quant or "as-loaded"
    samples_per_cat = args.samples_per_category
    logger.info(
        "ppl: model=%s quant=%s samples/cat=%d",
        model_path,
        quant_label,
        samples_per_cat,
    )
    corpus = _load_corpus(samples_per_cat)
    model, tokenizer = mlx_lm.load(model_path)
    cat_results: dict[str, dict] = {}
    total_loss = 0.0
    total_tokens = 0
    for cat, texts in corpus.items():
        cat_loss = 0.0
        cat_tokens = 0
        for text in texts:
            loss, ntok = _compute_loss(model, tokenizer, text)
            cat_loss += loss * ntok
            cat_tokens += ntok
        if cat_tokens == 0:
            continue
        mean_loss = cat_loss / cat_tokens
        cat_ppl = math.exp(mean_loss) if mean_loss < 50 else float("inf")
        cat_results[cat] = {
            "mean_nll": round(mean_loss, 4),
            "ppl": round(cat_ppl, 2) if cat_ppl != float("inf") else None,
            "tokens": cat_tokens,
        }
        total_loss += cat_loss
        total_tokens += cat_tokens
        logger.info(
            "ppl[%s]: nll=%.4f ppl=%.2f tokens=%d", cat, mean_loss, cat_ppl, cat_tokens
        )
    overall_nll = total_loss / total_tokens if total_tokens else 0.0
    overall_ppl = math.exp(overall_nll) if overall_nll < 50 else float("inf")
    result = {
        "model": model_path,
        "quant": quant_label,
        "overall_nll": round(overall_nll, 4),
        "overall_ppl": round(overall_ppl, 2) if overall_ppl != float("inf") else None,
        "total_tokens": total_tokens,
        "per_category": cat_results,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info(
        "ppl: overall nll=%.4f ppl=%.2f (%d tokens, quant=%s)",
        overall_nll,
        overall_ppl,
        total_tokens,
        quant_label,
    )
    return 0


def add_ppl_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "ppl",
        help="Compute perplexity on calibration corpus (quant cost measurement)",
    )
    parser.add_argument("model", help="Model alias or HF repo id")
    parser.add_argument(
        "--quant",
        default=None,
        help="Quantization mode label for the output (e.g. mixed_2_4, int4). Does not re-quantize — labels the loaded model's mode.",
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=5,
        help="Calibration samples per category (default 5; full corpus is slow)",
    )
    parser.set_defaults(func=ppl_command)
