#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Raw mlx_lm decode ceiling benchmark (no server).

Measures pure decode tok/s for a local model path. Useful for fast step-by-step
quant comparison before wiring through the fusion-mlx server.

Usage: python scripts/bench_raw_mlx.py <model_path> [max_tokens]
"""
import logging
import sys
import time

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bench_raw")
log.setLevel(logging.INFO)


def bench(model_path: str, max_tokens: int = 200, warmup: int = 1, runs: int = 3):
    import mlx_lm

    log.info("loading %s ...", model_path)
    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(model_path)
    log.info("loaded in %.1fs", time.perf_counter() - t0)

    # Qwen3.6 reasons first; disable thinking so content tokens are emitted.
    apply_kwargs = {}
    try:
        chat_tk = getattr(tokenizer, "_tokenizer", None)
        if chat_tk is not None and hasattr(chat_tk, "apply_chat_template"):
            pass
    except Exception:
        pass

    prompt_msgs = [{"role": "user", "content": "Write a short essay about the ocean."}]

    def gen_once():
        try:
            prompt = tokenizer.apply_chat_template(
                prompt_msgs,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                prompt_msgs, add_generation_prompt=True, tokenize=False
            )
        tok_s = 0
        ntok = 0
        t_first = None
        t0 = time.perf_counter()
        for _ in mlx_lm.stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens
        ):
            if t_first is None:
                t_first = time.perf_counter()
            ntok += 1
        t_end = time.perf_counter()
        prefill = (t_first - t0) if t_first else 0.0
        decode = (t_end - t_first) if t_first else 0.0
        tok_s = (ntok / decode) if decode > 0 else 0.0
        return {
            "ntok": ntok,
            "prefill_s": round(prefill, 3),
            "decode_s": round(decode, 3),
            "decode_tok_s": round(tok_s, 2),
        }

    for i in range(warmup):
        r = gen_once()
        log.info("warmup[%d]: %s", i, r)

    results = []
    for i in range(runs):
        r = gen_once()
        results.append(r)
        log.info("run[%d]: %s", i, r)

    speeds = [r["decode_tok_s"] for r in results]
    mean = sum(speeds) / len(speeds) if speeds else 0.0
    log.info("RESULT model=%s mean_decode_tok_s=%.2f baseline=29.24 ratio=%.3fx",
             model_path, mean, mean / 29.24)
    print(f"MODEL={model_path}")
    print(f"MEAN_DECODE_TOK_S={mean:.2f}")
    print(f"BASELINE=29.24")
    print(f"RATIO={mean/29.24:.3f}")
    print(f"PASS={'yes' if mean >= 29.24 else 'no'}")
    return mean


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: bench_raw_mlx.py <model_path> [max_tokens]", file=sys.stderr)
        sys.exit(2)
    mp = sys.argv[1]
    mt = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    m = bench(mp, max_tokens=mt)
    sys.exit(0 if m >= 29.24 else 1)
