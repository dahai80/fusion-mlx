#!/usr/bin/env python3
# fusion-mlx n-gram spec A/B benchmark worker.
#
# Run in a FRESH subprocess per mode. The caller sets FUSION_NGRAM_SPEC_ENABLED
# (and FUSION_NGRAM_SPEC_ORDER for the ON mode) in the env BEFORE launch --
# the gate is read at import time, so toggling after import is a no-op.
#
# Measures TTFT (prefill -> pp_tps) and post-TTFT decode rate (-> tg_tps) on a
# cyclic-counting prompt that forces a repeating token sequence (n-gram spec
# fires with high acceptance once cycle 1 is in history). Prints one JSON line
# on stdout.
#
# Methodology (prior sessions): consecutive 27B generations in one process
# halve throughput via MLX memory/thermal pressure, so spec ON vs OFF MUST
# be measured in separate subprocesses with a cooldown between. Within each
# process: 1 short warmup gen -> cooldown -> 1 measured (streaming) gen.
# Both modes use the identical schedule, so the A/B ratio is fair even if
# absolute tok/s runs below the 18.46 board baseline (row 1) -- this is a
# spec A/B entry, not a representative-perf claim.
import asyncio
import json
import logging
import os
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bench_spec_ab")

MODEL = os.environ.get(
    "BENCH_MODEL",
    "/Users/dahai/.fusion-mlx/models/Qwen3.6-27B-mxfp8",
)
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "400"))
WARMUP_TOKENS = int(os.environ.get("BENCH_WARMUP_TOKENS", "40"))
COOLDOWN_S = float(os.environ.get("BENCH_COOLDOWN_S", "30"))

# 10-token cycle seeded 8x. The model continues "1, 2, ..., 10, 1, 2, ..."
# giving a perfect repeating n-gram; order=3 hits after cycle 1.
PROMPT = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, " * 8


async def main() -> None:
    from fusion_mlx import BatchedEngine

    spec_on = os.environ.get("FUSION_NGRAM_SPEC_ENABLED", "1") != "0"
    mode = "spec-on" if spec_on else "spec-off"
    logger.info("mode=%s model=%s max_tokens=%d warmup=%d cooldown=%.0fs",
                mode, MODEL, MAX_TOKENS, WARMUP_TOKENS, COOLDOWN_S)

    # enable_thinking=False: the raw completion prompt carries no chat template,
    # but be explicit so the Qwen3 thinking head never fires (it would emit
    # <think prose where the cyclic n-gram can't match -> spec never fires).
    eng = BatchedEngine(MODEL, enable_thinking=False)
    await eng.start()
    # engine._engine = AsyncEngineCore; .engine = EngineCore; .scheduler = Scheduler
    sched = eng._engine.engine.scheduler

    # Warmup: load weights, prime MLX caches, compile kernels.
    t0 = time.perf_counter()
    w = await eng.generate(PROMPT, max_tokens=WARMUP_TOKENS, temperature=0.0)
    logger.info("warmup done: ct=%d dt=%.2fs", w.completion_tokens, time.perf_counter() - t0)

    # Cooldown: let thermal/memory settle so the measured run isn't squeezed
    # by the warmup generation (the known 27B double-gen pressure).
    if COOLDOWN_S > 0:
        logger.info("cooldown %.0fs...", COOLDOWN_S)
        await asyncio.sleep(COOLDOWN_S)

    # Measured run -- streaming so we can split TTFT (prefill) from decode.
    t0 = time.perf_counter()
    ttft: float | None = None
    last_ct = 0
    final = None
    async for chunk in eng.stream_generate(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0):
        if ttft is None and chunk.completion_tokens > 0:
            ttft = time.perf_counter() - t0
        last_ct = max(last_ct, chunk.completion_tokens)
        final = chunk
    t_total = time.perf_counter() - t0
    if ttft is None:
        ttft = t_total
    decode_dt = max(1e-6, t_total - ttft)
    ct = last_ct
    tg_tps = (ct - 1) / decode_dt if ct > 1 else (ct / t_total if t_total > 0 else 0.0)
    prompt_tokens = (final.prompt_tokens if final and final.prompt_tokens else 0)
    pp_tps = (prompt_tokens / ttft) if ttft > 0 and prompt_tokens else 0.0
    logger.info("measured: ct=%d ttft=%.0fms pp_tps=%.1f tg_tps=%.2f (engine: pp=%.1f tg=%.1f)",
                ct, ttft * 1000, pp_tps, tg_tps,
                getattr(final, "prompt_tps", 0.0), getattr(final, "generation_tps", 0.0))

    stats: dict = {}
    st = getattr(sched, "_ngram_spec_state", None)
    if st is not None:
        try:
            stats = st.get_stats() or {}
        except Exception as e:
            stats = {"error": str(e)}

    result = {
        "mode": mode,
        "model": os.path.basename(MODEL),
        "prompt": PROMPT[:60] + "...",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": ct,
        "ttft_ms": round(ttft * 1000, 1),
        "wall_s": round(t_total, 3),
        "pp_tps": round(pp_tps, 1),
        "tg_tps": round(tg_tps, 2),
        "engine_pp_tps": round(getattr(final, "prompt_tps", 0.0), 1) if final else 0,
        "engine_tg_tps": round(getattr(final, "generation_tps", 0.0), 1) if final else 0,
        "spec_stats": stats,
        "env": {
            "FUSION_NGRAM_SPEC_ENABLED": os.environ.get("FUSION_NGRAM_SPEC_ENABLED"),
            "FUSION_NGRAM_SPEC_ORDER": os.environ.get("FUSION_NGRAM_SPEC_ORDER", "5"),
            "FUSION_NGRAM_SPEC_NUM_DRAFT": os.environ.get("FUSION_NGRAM_SPEC_NUM_DRAFT", "3"),
        },
    }
    print(json.dumps(result, default=str))
    try:
        await eng.stop()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
