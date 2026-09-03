#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import psutil

logger = logging.getLogger("benchmark_fusion_takeover")

_MODELS = {
    "qwen3-0.6b-8bit": "mlx-community/Qwen3-0.6B-8bit",
    "llama-3.2-1b-4bit": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "llama-3.1-8b-4bit": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
}

PROMPT_SHORT = "Explain quantum entanglement in two sentences."
PROMPT_LONG = (
    "Write a detailed essay about the history of computing, covering "
    "the abacus, mechanical calculators, vacuum tubes, transistors, "
    "integrated circuits, microprocessors, personal computers, the "
    "internet, and artificial intelligence. Include at least one key "
    "inventor or invention for each era and explain its significance. "
    "Be thorough and precise."
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def mem_info() -> dict:
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return {
        "used_gb": round(vm.used / (1024**3), 2),
        "avail_gb": round(vm.available / (1024**3), 2),
        "swap_gb": round(sw.used / (1024**3), 2),
    }


def resolve_path(repo: str) -> str:
    cache = Path.home() / ".fusion-mlx" / "models"
    slug = "models--" + repo.replace("/", "--")
    snap_dir = cache / slug / "snapshots"
    if snap_dir.is_dir():
        snaps = sorted(snap_dir.iterdir())
        for s in snaps:
            if any(s.glob("*.safetensors")):
                return str(s)
    return repo


def _gen(model, tokenizer, prompt_ids, max_tokens):
    from mlx_lm import stream_generate

    last = None
    for resp in stream_generate(model, tokenizer, prompt_ids, max_tokens=max_tokens):
        last = resp
    return last


def run_config(repo, prompt_ids, max_tokens, warmup, repeats, takeover=None):
    import mlx_lm

    from fusion_mlx import fusion_mlx_lm

    settings = None
    if takeover == "off":
        from fusion_mlx.model_settings import ModelSettings

        settings = ModelSettings(fusion_takeover_enabled=False)
    elif takeover == "on":
        from fusion_mlx.model_settings import ModelSettings

        settings = ModelSettings(
            fusion_takeover_enabled=True,
            fusion_quant="nvfp4",
            fusion_target_model_types=(),
        )
    fusion_mlx_lm.set_fusion_model_settings(settings)

    if takeover is None:
        model, tokenizer = mlx_lm.load(resolve_path(repo))
    else:
        model, tokenizer = fusion_mlx_lm.load(resolve_path(repo))

    mx.eval(mx.zeros((1,)))

    for _ in range(warmup):
        _gen(model, tokenizer, prompt_ids, max_tokens)
        mx.eval(mx.zeros((1,)))

    results = []
    for i in range(repeats):
        t0 = time.perf_counter()
        last = _gen(model, tokenizer, prompt_ids, max_tokens)
        mx.eval(last.token if hasattr(last, "token") else mx.zeros((1,)))
        dt = time.perf_counter() - t0
        results.append(
            {
                "wall_s": round(dt, 4),
                "prefill_tps": round(last.prompt_tps, 3),
                "decode_tps": round(last.generation_tps, 3),
                "gen_tokens": int(last.generation_tokens),
                "prompt_tokens": int(last.prompt_tokens),
                "peak_mem_gb": round(last.peak_memory, 3),
            }
        )
        logger.info("  rep %d/%d: %s", i + 1, repeats, results[-1])
    post = mem_info()
    del model, tokenizer
    try:
        mx.clear_cache()
    except Exception:
        pass
    return results, post


def median(runs, key):
    return round(statistics.median(r[key] for r in runs), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(_MODELS.keys()))
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt", choices=["short", "long"], default="short")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(Path.home() / "fusion" / "audit"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    prompt = PROMPT_SHORT if args.prompt == "short" else PROMPT_LONG

    rows = []
    for key in args.models:
        repo = _MODELS[key]
        logger.info("==== %s (%s) ====", key, repo)
        from mlx_lm import load as _up_load

        _, tok = _up_load(resolve_path(repo))
        prompt_ids = tok.encode(prompt)
        for takeover, label in [
            (None, "upstream"),
            ("off", "fusion_off"),
            ("on", "fusion_on"),
        ]:
            logger.info("-- config: %s --", label)
            try:
                runs, post_mem = run_config(
                    repo,
                    prompt_ids,
                    args.max_tokens,
                    args.warmup,
                    args.repeats,
                    takeover,
                )
            except Exception as e:
                logger.error("config %s failed: %s", label, e)
                rows.append(
                    {
                        "model": key,
                        "config": label,
                        "error": str(e)[:80],
                    }
                )
                continue
            row = {
                "model": key,
                "config": label,
                "prompt_tokens": runs[0]["prompt_tokens"],
                "gen_tokens": median(runs, "gen_tokens"),
                "prefill_tps": median(runs, "prefill_tps"),
                "decode_tps": median(runs, "decode_tps"),
                "wall_s": median(runs, "wall_s"),
                "peak_mem_gb": median(runs, "peak_mem_gb"),
                "post_used_gb": post_mem["used_gb"],
                "post_swap_gb": post_mem["swap_gb"],
            }
            rows.append(row)
            logger.info("  median: %s", row)

    _write_report(args, prompt, rows)
    print("benchmark complete, rows:", len(rows))


def _write_report(args, prompt, rows):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    path = out_dir / f"perf-baseline-fusion-takeover-{ts}.md"
    lines = []
    lines.append("# Fusion Takeover Performance Baseline")
    lines.append("")
    lines.append(f"- Date: {ts}")
    lines.append("- Hardware: M5 Max 128GB (40-core GPU)")
    lines.append(
        "- mlx_lm: see installed version; fusion-mlx branch feat/enhance-arch-0826"
    )
    lines.append(
        f"- max_tokens={args.max_tokens} warmup={args.warmup} repeats={args.repeats} median-of-N"
    )
    lines.append(f"- prompt={args.prompt} ({len(prompt)} chars)")
    lines.append(
        "- greedy (argmax) deterministic; configs: upstream=mlx_lm.load | fusion_off=shim takeover disabled | fusion_on=shim takeover enabled (nvfp4 tag, Phase 0 metadata only)"
    )
    lines.append("")
    ok = [r for r in rows if "error" not in r]
    if ok:
        lines.append(
            "| model | config | prompt_tok | gen_tok | prefill tok/s | decode tok/s | wall s | peak mem GB | post used GB | post swap GB |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in ok:
            lines.append(
                f"| {r['model']} | {r['config']} | {r['prompt_tokens']} | {r['gen_tokens']} "
                f"| {r['prefill_tps']} | {r['decode_tps']} | {r['wall_s']} "
                f"| {r['peak_mem_gb']} | {r['post_used_gb']} | {r['post_swap_gb']} |"
            )
    errs = [r for r in rows if "error" in r]
    if errs:
        lines.append("")
        lines.append("## Errors")
        lines.append("| model | config | error |")
        lines.append("|---|---|---|")
        for r in errs:
            lines.append(f"| {r['model']} | {r['config']} | {r['error']} |")
    lines.append("")
    lines.append("## Shim overhead (fusion_off vs upstream)")
    lines.append("")
    by_model = {}
    for r in ok:
        by_model.setdefault(r["model"], {})[r["config"]] = r
    if by_model:
        lines.append(
            "| model | upstream decode | fusion_off decode | delta% | upstream prefill | fusion_off prefill | delta% |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for m, cfgs in by_model.items():
            up = cfgs.get("upstream")
            off = cfgs.get("fusion_off")
            if up and off:
                d_dec = (
                    round(
                        (off["decode_tps"] - up["decode_tps"]) / up["decode_tps"] * 100,
                        2,
                    )
                    if up["decode_tps"]
                    else 0
                )
                d_pre = (
                    round(
                        (off["prefill_tps"] - up["prefill_tps"])
                        / up["prefill_tps"]
                        * 100,
                        2,
                    )
                    if up["prefill_tps"]
                    else 0
                )
                lines.append(
                    f"| {m} | {up['decode_tps']} | {off['decode_tps']} | {d_dec}% | {up['prefill_tps']} | {off['prefill_tps']} | {d_pre}% |"
                )
    path.write_text("\n".join(lines))
    print("report:", path)


if __name__ == "__main__":
    main()
