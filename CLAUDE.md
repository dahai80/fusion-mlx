# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

fusion-mlx — unified local model serving for Apple Silicon (Metal/MLX-native).
OpenAI/Anthropic/Responses/Ollama-compatible API server on `127.0.0.1:11434`.
Drop-in replacement for Ollama/vLLM. macOS / Apple Silicon only. Beta, single-maintainer.

## Environment

```bash
cd /Users/dahai/fusion/fusion-mlx
source .venv/bin/activate        # REQUIRED before any Python work
```

Python >=3.11 (CI runs 3.11/3.12/3.13). Build: `pip install -e ".[dev]"`.
Optional extras (install only what use): `[audio]` `[image]` `[video]` `[mcp]` `[dflash2]` `[mfa]` `[grammar]` `[vision]` `[chat]`.
Image gen needs vendored wheel: `pip install -e ".[image]" --find-links packaging/_wheels`.

## Lifecycle (start.sh)

```bash
./start.sh start     # start server, default 127.0.0.1:11434
./start.sh stop
./start.sh status    # PID, port, memory, loaded models
./start.sh log [-f]
./start.sh doctor    # health check
./start.sh tune
```
From monorepo root the same script is `~/claude-home/fusion-mlx/start.sh`.
API key + port + HF mirror resolved from `~/.fusion-mlx/settings.json` (env vars `FUSION_MLX_API_KEY`, `FUSION_HOST`, `HF_MIRROR` override). Default mirror `https://hf-mirror.com`. Model cache `~/.fusion-mlx/models`.

## CLI entry points

`fusion-mlx` and `fm` both = `fusion_mlx.cli:main`.

```bash
fusion-mlx serve <model> --port 11434          # OpenAI-compatible server
fusion-mlx serve qwen3.5-9b-4bit               # short alias or full HF repo id both work
fusion-mlx bench <model> --num-prompts 10
fusion-mlx chat <model>                        # interactive REPL
fusion-mlx pull <model>                        # download via mirror
fusion-mlx models | ps | rm | info | doctor | agents | telemetry | upgrade
```
Serve flags of note: `--enable-dspark` (DSpark spec decode), `--enable-dflash2` (DFlash2 block-diffusion spec decode), `--enable-mtp` (multi-token-prediction).

## Tests

```bash
pytest tests/unit -q                                   # active suite (skips quarantined + real-model)
pytest tests/unit/test_<name>.py -q                    # single file
pytest tests/unit/test_x.py::test_y -v                 # single test
FUSION_MLX_REAL_MODEL_TESTS=1 pytest tests/unit -q     # loads real MLX weights — slow, needs models on disk
```

- Config: `asyncio_mode=auto`, `testpaths=["tests"]`, marker `real_model` gated by env var above.
- `tests/conftest.py` auto-mocks `mlx.core` when real MLX absent (Linux CI) — unit tests run headless.
- ~301 test files are **quarantined** in `tests/unit/debt_modules.txt` (`collect_ignore`). Active ~377 files. Do not claim a higher count than `pytest --collect-only -q | tail -1` reports.
- Real-model tests need the server running (`./start.sh start`).
- **Failing-test rule**: a failing test — even unrelated to your change — must be located and fixed (or filed as issue). Do not leave the suite redder than you found it.

## Lint & format

```bash
ruff check fusion_mlx/ tests/        # CI gate; --fix to autofix
black --check fusion_mlx/ tests/     # CI gate
ruff-format                          # pre-commit
```
- `fusion_mlx/patches/` excluded from lint (upstream-derived vendor code — merge churn).
- MLX-family pkgs pinned `known-third-party` in `[tool.ruff.lint.isort]` (CI on ubuntu can't install macOS-only `mlx` wheel → isort misclassifies).
- Many rules deliberately ignored (`pyproject.toml [tool.ruff.lint] ignore`) to match the permissive mature-codebase style; do not tighten without reason.

## Code style

- Indentation multiples of 4. **No docstrings** in new code.
- Fail visibly, not silently — loud errors over silent fallbacks.
- Surgical changes — touch only what your change requires; don't reformat adjacent code.
- Logging by default — enough to locate problems.
- Convention beats novelty — match the surrounding file even if you prefer another.

## Commit & PR flow

```bash
git checkout -b <type>/<short-desc>      # feat/fix/docs/chore/refactor
git commit -m "<type>(#issue): <subject>"
git push -u fusion-mlx <branch>          # NOT origin (that's the homebrew tap)
gh pr create --repo dahai80/fusion-mlx --title "<type>(#issue): <subject>"
```
PR checklist: `ruff` + `black --check` pass; `pytest tests/unit -q` no redder than before; CHANGELOG.md entry if user-facing; README/docs updated if behavior changed.
Upstream-blocking issues (mlx-lm/mlx-vlm limits): do not fabricate a path — file upstream issue, link it, keep a `raise` that fails visibly.

## Architecture

Request flow: **FastAPI app (`server.py`) → route module (`api/*`) → `EnginePool` → engine (`engines/*`) → `AsyncEngineCore` → `Scheduler` → MLX**.

### Top-level layout (`fusion_mlx/`)
- `cli.py` / `cli_serve.py` / `cli_commands.py` — argparse CLI (`serve`/`bench`/`chat`/`pull`/`models`/`ps`). `serve_command` is the main boot path; has a separate audio-only serve mode (`_serve_audio_mode`) that skips the text-LM loader.
- `server.py` — FastAPI `app` + `create_app` + `Server`. Wires all routers and middleware. The lifespan/router docstring at top lists every mounted route group.
- `config.py` — `ServerConfig`, `MemoryConfig`/`MemoryTier` (safe/balanced/aggressive/custom), `SchedulerConfig`, `SchedulingPolicy` (FCFS/PRIORITY), `get_config()`.
- `engine_core.py` — `AsyncEngineCore` + `EngineConfig`: continuous-batching loop driving the Scheduler, owns the request/output collector.
- `public_api.py` — **stable public surface**. Downstream code (fusion-comfyui etc.) MUST `from fusion_mlx.public_api import X`. Direct imports of internal submodules (`engines.*`, `model_registry`, `config`, `video.*.pipeline`) are internal API, not stable. (`public_api.py` ≠ `api/` package — latter is server route pydantic models.)
- `request.py` — `Request`, `RequestOutput`, `SamplingParams`, `RequestStatus`.

### Engines (`engines/`)
One engine type per modality, all under `EnginePool` LRU management:
`BatchedEngine` (LLM, continuous batching), `VLMBatchedEngine` (vision-language), `EmbeddingEngine`, `RerankerEngine`, `ImageGenEngine`, `VideoGenEngine`, `TTSEngine`/`STTEngine`/`STSEngine` (audio), `NEREngine`. `base.py` = `BaseEngine`/`BaseNonStreamingEngine`.

### Pool (`pool/`)
`EnginePool` — multi-model serving with LRU eviction, pinning, TTL auto-unload, pre-load memory check. `ProcessMemoryEnforcer` = 4-tier memory guard. `ModelDiscovery` = alias→HF-id resolution.

### Scheduler (`scheduler/`)
Submodule split (re-exports at package `__init__` for back-compat). `core.py` = `Scheduler` (re-assembles split via star-imports, F403/F405 intentional). `sched_admission.py` (admission/KV headroom), `sched_batch.py` (prefill/completion batching), `config.py` re-exports from `config.py`. Installs `_mlx_compat` shim (M5 single-stream) at package import — must run before any `mlx_lm` import. Monkeypatches (`monkeypatches.py`) vendor MTP.

### Cache (`cache/`)
Multiple coexisting cache layers (read the right one before touching):
- `prefix_cache.py` (125K — block-aware prefix cache, copy-on-write)
- `paged_cache.py` / `paged_ssd_cache.py` (paged KV cache + SSD cold layer)
- `radix_diffusion_cache.py`, `mllm_cache.py`, `vision_embedding_cache.py` / `vision_feature_cache.py`
- `latent_cache.py` — UMA radix *latent* cache (VAE-encoded frame reuse, multi-shot `session_id` continuity). Env `FUSION_SESSION_TAIL_CACHE=1` (default OFF). See `cache/LATENT_CACHE.md`.
- `tiered_cache.py`, `response_cache.py`, `boundary_snapshot_store.py`, `type_handlers.py`/`type_registry.py`.

### Speculative decode (`speculative/`)
Multiple strategies, opt-in per serve flag: `dspark/` (vendored DSpark for MLX), `dflash/`+`dflash2/` (block-diffusion), `eagle3/`, `mtp/` (multi-token-prediction, vendored `_mtp_vendored.py`), `ngram_spec.py`, `vlm_mtp.py`. See `docs/speculative-decoding.md`. Note: speculative *denoise* was falsified (0% acceptance) — env-gated OFF, documented in `SPECULATIVE_DENOISE.md`.

### Video backends (`video/`)
Pure-MLX ports replacing mlx-video: `ltx2/`, `ltx2_5/`, `wan2/`, `skyreels_v3/`, `cogvideox/`, `cosmos/`, `hunyuanvideo/`, `svd/`, `uniworld/`, `opensora/`, `minimax_h3/`, `mage/`, `latentsync_mlx/`, `musetalk_mlx/`, `pulid_mlx/`. Routes at `/v1/videos/generate` (`api/videos_routes.py`).

### API routes (`api/`)
One router per surface, mounted in `server.py`:
- OpenAI: `openai_routes.py` (82K, `/v1/chat/completions`, `/v1/completions`), `responses_adapter.py` (Responses API)
- Anthropic: `anthropic_routes.py` (`/v1/messages`, `/v1/count_tokens`), `anthropic_adapter.py`/`anthropic_utils.py`
- Ollama-compat: `ollama_routes.py`
- Modality: `audio_routes.py`, `images.py`/`images_sr.py`, `videos_routes.py`, `embeddings_routes.py`, `rerank_routes.py`, `ner_routes.py`, `ocr_routes.py`
- Tooling: `tool_calling.py` (60K — parser dispatch), `grammar.py`/`guided.py` (constrained decode), `thinking.py` (reasoning), `strict_json_schema.py`
- `mcp_routes.py`, `agent_routes.py` (OpenClaw), `watermark_routes.py`, `convert_routes.py`, `layered_quantize_routes.py`, `distributed_routes.py`, `session_routes.py`

### Other subsystems
- `dispatch/` — `RequestRouter` + `CloudRouter` (cloud fallback routing).
- `middleware/` — auth/CORS/rate-limit/body-limit/route-guard/request-id, exception handlers, probe fastpath.
- `patches/` — upstream-derived model/patch code (deepseek_v4, glm_moe_dsa, mlx_lm_mtp, mlx_vlm_*mtp, step3p7). **Not authored here, lint-excluded.**
- `migrate/` — cross-format weight conversion (analyzer/architectures/codegen/converter/validator/weight_mapper).
- `admin/` — admin dashboard routes + templates + i18n + auth (`require_admin`).
- `agents/` — agent profiles + OpenClaw adapter.
- `cluster/` + `distributed/` — multi-node / mDNS platform routing (windows-cuda node in `backends/cuda_node/`).
- `telemetry/`, `memory_monitor.py`, `memory_cache.py`, `server_metrics.py` — observability + memory enforcement.
- `model_auto_config.py` (60K) — auto-detect model arch/quant from weights; `model_aliases.py` (15K) + `aliases.json` (24K) — alias registry.
- `_torch_stub.py` — torch API stub so torch-dependent code paths import without torch installed.
- `custom_kernels/` + `kernels/` — Metal kernels (Metal Flash Attention vendored, `pflash.py`).

### Quantization
`turboquant.py`/`turboquant_kv.py` (KV cache quant), `oq.py` + `oq_calibration_data.json` (1.2M, offline quant calibration), `mxfp4_moe_guardrail.py`, `kv_cache_dtype.py`. Quant knobs drive the benchmark table in README (mixed_2_4, quant2-flat etc.).

## Conventions to respect
- `__init__.py` files are re-export surfaces — `F401` ignored there on purpose.
- `scheduler/core.py` uses intentional star-imports (`F403`/`F405` ignored) to re-assemble the split Scheduler.
- `B023` loop-capture ignored in `sched_schedule.py`/`prefix_cache.py` — sync progress callbacks invoked same-iteration, false positive.
- When adding a route: mount it in `server.py`, set any context setter (`set_X_context`) the router needs.
- New public symbols downstream will depend on → add to `public_api.py` `__all__`, not just the internal module.
- Generated code: write skeleton first, fill with edits (Rule 13). Long files (`cli_serve.py` 127K, `openai_routes.py` 82K) — read the relevant function, don't load whole file blindly.

## Docs
`docs/architecture.md` (deep dive), `docs/api-reference.md`, `docs/cli-reference.md`, `docs/configuration.md`, `docs/speculative-decoding.md`, per-modality `docs/<modality>-image.md` / `docs/video-input.md`, `docs/public-api-boundary.md`, `docs/cuda-node.md`. `ROADMAP.md` = full plan. `CHANGELOG.md` = user-facing changes (add entry on user-facing PRs).
