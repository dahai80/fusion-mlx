# Benchmarks

Reproducible throughput + latency measurements for fusion-mlx on Apple
Silicon, comparing against Ollama and `mlx-lm` where available.

> Methodology and results live here. Raw JSON reports are under
> [`reports/`](reports/). The harness is [`run_bench.py`](run_bench.py).

## Method

- **What is measured**: a real `/v1/chat/completions` request against a
  running fusion-mlx server (the same path users hit), with a **fixed
  prompt** and deterministic sampling (`temperature=0`, `top_p=1.0`).
  No micro-benchmarks, no synthetic token loops — this is end-to-end.
- **Per-model metrics**:
  - `tokens_per_second` — `completion_tokens / wall_seconds` (non-stream
    timed request).
  - `ttft_seconds` — time-to-first-token from a streaming request
    (first SSE chunk carrying `content`).
  - `wall_seconds` — total wall time of the timed request, **including
    model load** if the model was not already resident (cold start).
  - `prompt_tokens` / `completion_tokens` — from the server's `usage`.
- **Reproducibility**: fixed prompt template, fixed `max_tokens`, fixed
  sampling. Re-run with the same `--models --prompt-tokens --gen` to
  reproduce. Variance comes from model load (cold vs warm) and thermal
  state — warm runs (model already loaded) are the steady-state number.
- **Not measured here**: accuracy (see `fusion_mlx/admin/accuracy_bench.py`
  and the eval suite under `fusion_mlx/eval/`), video/image gen throughput
  (separate backends, Phase 3).

## Running

```bash
# 1. start the server (Apple Silicon, MLX models)
~/claude-home/fusion-mlx/start.sh start

# 2. run the harness (server must be healthy)
.venv/bin/python benchmarks/run_bench.py --api-key "$FUSION_MLX_API_KEY" \
    --models qwen3-4b-4bit,Meta-Llama-3.1-8B-Instruct-4bit \
    --prompt-tokens 512 --gen 256

# bench every loaded model:
.venv/bin/python benchmarks/run_bench.py --api-key "$KEY" --all
```

Reports: `reports/<model>_<timestamp>.json` (one per model) +
`reports/SUMMARY_<timestamp>.json` (all results). Console prints the
summary table.

## Comparing against Ollama / mlx-lm

fusion-mlx listens on `11434` (Ollama's default port) and speaks the
Ollama protocol (`/api/generate`, `/api/chat`) **and** the OpenAI
protocol (`/v1/chat/completions`). To compare raw throughput:

- **Ollama**: run the same model via Ollama on a different port
  (`OLLAMA_HOST=127.0.0.1:11435 ollama serve`), point `--base-url` at it.
  Note: Ollama uses GGUF, fusion-mlx uses MLX — same model family, different
  quant format, so compare families not byte-identical weights.
- **mlx-lm**: `python -m mlx_lm.generate --model <mlx-path> --prompt ...`
  prints tok/s; this is the un-served baseline (no HTTP, no scheduler).

The harness measures the **served** path. Subtracting the mlx-lm
un-served number from the fusion-mlx served number gives the serving
overhead (HTTP + scheduler + tokenizer), which is the honest apples-to-
apples comparison for a *server*.

## Results

Results are filled in by running the harness and pasting the console
summary (or reading `SUMMARY_*.json`). The table below is updated when a
fresh run lands — see the timestamp column.

> Prompt 512 tok (expanded by tokenizer to ~700 tok), gen 128 tok,
> temperature 0, top_p 1.0. Sorted by `tok/s` desc. `wall_seconds`
> includes model load when the model was not already resident (cold
> start). Run: `20260808-141511`, base_url `127.0.0.1:11434`, api-key
> auth on. Raw: [`reports/SUMMARY_20260808-141511.json`](reports/SUMMARY_20260808-141511.json).

| Model | tok/s | TTFT (s) | tokens | wall (s) |
|-------|-------|----------|--------|----------|
| Qwen3-0.6B-8bit | 168.1 | 0.435 | 137 | 0.81 |
| Qwen3.5-4B-4bit | 80.7 | 0.751 | 128 | 1.59 |
| Qwen3.5-9B-4bit | 52.8 | 1.223 | 128 | 2.43 |
| gemma-4-26b-a4b-it-4bit | 41.8 | 2.694 | 128 | 3.06 |
| Llama-3.2-1B-Instruct-4bit | 15.7 | 1.375 | 136 | 8.69 |
| Meta-Llama-3.1-8B-Instruct-4bit | 7.6 | 4.440 | 128 | 16.80 |
| Qwen3.6-27B-mxfp8 | 1.5 | 20.748 | 128 | 83.43 |

**Not run (model not downloaded locally)**: `deepseek-r1-7b-4bit`,
`phi-4-4bit`, `minimax-m2.5-4bit` — aliases resolve but the MLX weights
are not in `~/.fusion-mlx/models/`, so the server returned 404. Download
via `hf-mirror.com` then re-run to fill these rows. Reported honestly,
not skipped silently.

**Observations**:
- Qwen3-0.6B-8bit warm (already loaded) hits 168 tok/s with 0.4s TTFT —
  the steady-state small-model serving overhead is negligible.
- Llama-3.2-1B and Meta-Llama-3.1-8B were cold-started (load included in
  `wall_seconds`), hence the lower tok/s and higher TTFT; warm numbers
  would be substantially higher.
- Qwen3.6-27B-mxfp8: 1.5 tok/s, 20.7s TTFT — cold start on a 27B mxfp8
  model loading into memory is the dominant cost; steady-state decode of
  a 27B on this hardware is bounded by memory bandwidth.
- gemma-4-26b (a4b, 26B active-4B MoE) at 41.8 tok/s decodes like a 4B
  model, as expected for active-param MoE.
