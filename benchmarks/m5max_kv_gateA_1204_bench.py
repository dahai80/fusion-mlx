# SPDX-License-Identifier: Apache-2.0
# Gate A baseline bench (#1204, branch exp/m5max-kv-baseline-11645).
#
# Measures real-model load time, prefill throughput / TTFT and decode
# throughput of a hybrid linear+full-attention model (Qwen3.8-27B family)
# at several context lengths, optionally with MLX-LM KV quantization.
#
# Answers report question 1: how much decode time grows with context, to
# decide whether KV traffic is the dominant bottleneck on M5 Max 128G.
#
# Usage:
#   python benchmarks/m5max_kv_gateA_1204_bench.py \
#       --model mlx-community/Qwen3.8-27B-4bit \
#       --ctx-lengths 2048 8192 32768 --decode-tokens 128 --repeats 1
#   python benchmarks/m5max_kv_gateA_1204_bench.py --kv-bits 8   # KV8 pass
#
# Output: prints a table and writes JSON to
#   benchmarks/reports/m5max_kv_gateA_1204_<suffix>.json

import argparse
import json
import logging
import resource
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gateA_1204")

FILLER_LINE = (
    "def helper_{i}(values):\n"
    "    total = 0\n"
    "    for value in values:\n"
    "        if value > 0:\n"
    "            total += value * {i}\n"
    "    return total\n\n"
)

TASK_PROMPT = (
    "Review the helper functions above and write a single function "
    "merge_all(values) that returns the sum of helper_0 through helper_N "
    "outputs for the given values. Return code only."
)


def build_prompt_ids(tokenizer, ctx_tokens, eos_id=None):
    filler = "".join(FILLER_LINE.format(i=i % 64) for i in range(ctx_tokens // 20 + 2))
    filler_ids = tokenizer.encode(filler)
    task_ids = tokenizer.encode(TASK_PROMPT)
    budget = max(ctx_tokens - len(task_ids), 16)
    if len(filler_ids) > budget:
        filler_ids = filler_ids[:budget]
    ids = filler_ids + task_ids
    if eos_id is not None:
        ids = [eos_id] + ids
    return ids[: max(ctx_tokens, 32)]


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def run_once(model, tokenizer, prompt_ids, decode_tokens, kv_bits):
    mx.reset_peak_memory()
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    first_token = None
    token_times = []
    n_generated = 0
    kwargs = {}
    if kv_bits:
        kwargs = {"kv_bits": kv_bits, "kv_group_size": 64, "quantized_kv_start": 0}
    for response in stream_generate(
        model,
        tokenizer,
        prompt_ids,
        max_tokens=decode_tokens,
        prompt_cache=cache,
        **kwargs,
    ):
        now = time.perf_counter()
        if first_token is None:
            first_token = now - t0
        else:
            token_times.append(now - t0)
        t0 = now
        n_generated += 1
        text = response.text
    peak = mx.get_peak_memory()
    return {
        "n_generated": n_generated,
        "ttft_s": first_token,
        "prefill_tokens": len(prompt_ids),
        "prefill_tok_s": len(prompt_ids) / first_token if first_token else None,
        "decode_tok_s": (n_generated - 1) / sum(token_times) if token_times else None,
        "inter_token_ms_p50": (
            statistics.median(token_times) * 1000 if token_times else None
        ),
        "inter_token_ms_p95": (
            statistics.quantiles(token_times, n=20)[18] * 1000
            if len(token_times) >= 20
            else None
        ),
        "peak_memory_gib": peak / (1024**3),
        "rss_gib": rss_gib(),
        "last_text_len": len(text),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3.8-27B-4bit")
    parser.add_argument(
        "--ctx-lengths", type=int, nargs="+", default=[2048, 8192, 32768]
    )
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--kv-bits", type=int, default=None, choices=[None, 4, 8])
    parser.add_argument("--suffix", default=None)
    args = parser.parse_args()

    logger.info("loading model %s", args.model)
    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    load_s = time.perf_counter() - t0
    logger.info("model loaded in %.1fs", load_s)

    eos_id = None
    if getattr(tokenizer, "bos_token_id", None) is not None:
        eos_id = tokenizer.bos_token_id

    results = {
        "bench": "m5max_kv_gateA_1204",
        "model": args.model,
        "kv_bits": args.kv_bits,
        "load_s": round(load_s, 3),
        "runs": [],
    }
    for ctx in args.ctx_lengths:
        prompt_ids = build_prompt_ids(tokenizer, ctx, eos_id=eos_id)
        logger.info("ctx=%d prompt_tokens=%d", ctx, len(prompt_ids))
        for repeat in range(args.repeats):
            metrics = run_once(
                model, tokenizer, prompt_ids, args.decode_tokens, args.kv_bits
            )
            metrics.update(
                {"ctx_target": ctx, "prompt_tokens": len(prompt_ids), "repeat": repeat}
            )
            logger.info(
                "ctx=%d repeat=%d ttft=%.3fs prefill=%.1f tok/s decode=%.2f tok/s peak=%.2fGiB",
                ctx,
                repeat,
                metrics["ttft_s"],
                metrics["prefill_tok_s"],
                metrics["decode_tok_s"] or 0.0,
                metrics["peak_memory_gib"],
            )
            results["runs"].append(metrics)

    suffix = args.suffix or (f"kv{args.kv_bits}" if args.kv_bits else "kv16")
    out_path = Path(__file__).parent / "reports" / f"m5max_kv_gateA_1204_{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    logger.info("wrote %s", out_path)

    print(
        f"{'ctx':>7} {'rep':>3} {'ttft_s':>8} {'prefill':>9} {'decode':>8} {'p50_ms':>8} {'peak_GiB':>9}"
    )
    for run in results["runs"]:
        print(
            f"{run['ctx_target']:>7} {run['repeat']:>3} {run['ttft_s']:>8.3f} "
            f"{run['prefill_tok_s']:>9.1f} {run['decode_tok_s'] or 0:>8.2f} "
            f"{run['inter_token_ms_p50'] or 0:>8.1f} {run['peak_memory_gib']:>9.2f}"
        )


if __name__ == "__main__":
    main()
