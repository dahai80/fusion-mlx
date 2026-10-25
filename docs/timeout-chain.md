# Timeout chain (issue #1191)

## Architecture invariant: mx.eval is not interruptible

**mx.eval is NOT interruptible.** Once a compute graph is submitted to Metal,
it cannot be cancelled — `asyncio.wait_for` can only abandon the Python-side
future, the kernel keeps running to completion (or forever, if wedged). A
request-level timeout can therefore only fire **upstream of eval**: the
`wait_for` around the `run_in_executor` dispatch bails out and returns an
error to the client while the underlying work may still occupy the executor
thread until it finishes.

When an eval itself wedges (GPU hang, kernel deadlock), no timeout can fire —
that is what the **executor watchdog** (#1189, `fusion_mlx/executor_watchdog.py`)
is for: it detects the stalled job, marks the executor poisoned, and all
subsequent callers fast-fail with 503 until the process restarts.

**Timeout + watchdog = two layers.** The timeout bounds the client wait; the
watchdog bounds the system. Neither alone is sufficient.

## The chain

```text
request
  → endpoint (body receive idle 15s, body_size middleware)
    → route handler
      → engine await (per-modality timeout, table below)
        → run_in_executor dispatch (bounded per-modality)
          → mx.eval  (NOT interruptible — kernel runs to completion)
            → executor watchdog backstop (#1189)
               windows: llm 300s (scaled, 1800s cap) · image 600s ·
               video 600s · audio 300s · laya 120s
```

Every hop is bounded. The only unbounded step — `mx.eval` — is covered by
the watchdog layer, not by the request timeout.

## Full inventory

Timeouts current as of issue #1191. Every endpoint maps to a bounded wait;
"NEW this PR" entries are the gaps this PR closed.

| Surface | Timeout | Source | Env knob |
|---|---|---|---|
| LLM routes (chat / completions / ollama / anthropic / responses / sessions / reasoning / watermark / layered_quantize / spec) — non-stream | 600s | `engine_core.py:1985` | `FUSION_MLX_GENERATE_TIMEOUT_S` |
| LLM routes — streaming | 5s heartbeat, no overall cap | `engine_core.py:1893-1907` | — |
| LLM watchdog | 300s base, scaled to 1800s cap | `engine_core.py:273-278` | `FUSION_LLM_WATCHDOG_TIMEOUT` / `FUSION_LLM_WATCHDOG_MAX_TIMEOUT` |
| embeddings | exec 30s, load 120s | `engines/embedding.py:794,744` | — |
| rerank | exec 30s, load 120s | `engines/reranker.py:1087,1052` | — |
| ner | exec 60s, load 120s | `engines/ner.py:185,140` (via `base.py:416-424`) | — |
| ocr | rides the VLM scheduler path (LLM watchdog) | `api/ocr_routes.py:22,76` | — |
| TTS | 180s | `engines/tts.py:269` | `FUSION_TTS_TIMEOUT` |
| TTS stream chunk | 5s per chunk | `engines/tts.py:377` | — |
| STT | 60s | `engines/stt.py:189` | `FUSION_STT_TIMEOUT` |
| STS | 60s | `engines/sts.py:305` | `FUSION_STS_TIMEOUT` |
| music (ACE-Step) | **600s (NEW this PR — was NONE)** | `engines/music.py` | `FUSION_MUSIC_TIMEOUT` |
| audio engine load | 120s | `api/audio_routes.py:246` | `FUSION_AUDIO_LOAD_TIMEOUT` |
| images | 600s × n INACTIVITY budget (restarts on every worker event, #1407), cap 3600s; absolute 7200s total cap; cancel grace 30s; first-event load window 120s (#1332, <= 0 disables) | `image_gen.py:245-300`, `media/job_manager.py` | `FUSION_IMAGE_TIMEOUT`, `FUSION_IMAGE_TOTAL_TIMEOUT`, `FUSION_IMAGE_LOAD_TIMEOUT` |
| videos | 7200s | `engines/video.py` (`get_video_gen_timeout`) | `FUSION_VIDEO_GEN_TIMEOUT` |
| video backends (wan2 / ltx2 / skyreels / svd / hunyuanvideo / cosmos / minimax_h3 / ltx2_5 / uniworld) | same knob; uniworld was hardcoded 600s (fixed this PR) | `video/uniworld/backend.py` | `FUSION_VIDEO_GEN_TIMEOUT` |
| 3d | image timeout class + httpx 30s URL fetch | `api/threed_routes.py:111` | — |
| laya decide | **dispatch 120s (NEW this PR — was NONE)** | `api/decisions_routes.py` | `FUSION_LAYA_DISPATCH_TIMEOUT` |
| laya watchdog | 120s | `executor_watchdog.py` (#1189) | `FUSION_LAYA_WATCHDOG_TIMEOUT` |
| MCP tool calls | 30s per server config | `mcp/client.py:342` | — |
| /metrics render | 10s → 503 | `routes_internal/metrics.py:31` | — |
| body receive idle | 15s | `middleware/body_size.py`, `config.py:389` | `FUSION_MLX_BODY_RECEIVE_TIMEOUT_SECONDS` |
| model load (LLM) | 300s | `engines/batched.py:339` | `FUSION_MLX_MODEL_LOAD_TIMEOUT` |
| model load (VLM) | 120s | `engines/vlm.py:397` | — |
| unload / evict clear_cache | 30s | `pool/engine_pool.py:2525` | `FUSION_MLX_SYNC_TIMEOUT` |
| stream drain | 30s | `engines/base.py:370` | — |
| gui_compat inference | 300s per call | `gui_compat/queued_inference.py:75` | — |
| cloud fallback | 30s × 3 attempts | `dispatch/cloud_router.py:360` | — |
| shutdown teardown | 5s | `engine_core.py:1087,2316` | — |
| admission / agent / convert / distributed / session routes | bounded by the engine path they call — no separate route-level timer (by design, documented here) | — | — |

### On TimeoutError semantics

When a bounded dispatch times out, the handler converts `TimeoutError` into
the surface-appropriate failure: route layer → 503 (retryable), engine layer
→ `RuntimeError` with the knob name in the message. No poisoning is performed
by these timeouts — poisoning belongs exclusively to the executor watchdog
(#1189), which is the layer that can actually observe a wedged `mx.eval`.
The image subprocess hang deadline and load window (#1332) likewise kill and
reap the worker without poisoning; the next image request spawns a fresh
worker.

## ServerConfig.default_timeout (removed)

`ServerConfig.default_timeout` and the serve `--timeout` flag that mirrored
it were **removed** in #1354 (config audit D1). The field never had a
production reader — only write sites in `cli_serve/config_resolve.py` and
`cli_serve/audio_mode.py` — so leaving it in place was a pure decoy for
operators who expected a global timeout to apply.

**Per-modality knobs are the contract** — each modality owns its bounded
wait and its env override (table above). There is no global request timeout.

## Streaming semantics

Streaming LLM endpoints do **not** enforce an overall wall-clock cap. Instead
they bound **per-chunk progress**: a 5s heartbeat timeout fires whenever no
token arrives within the window (`engine_core.py:1893-1907`). A slow-but-
progressing stream never times out; a stalled stream trips the executor
watchdog, which poisons the executor and fails the stream fast.
