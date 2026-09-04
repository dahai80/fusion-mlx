# Telemetry & Observability Alignment with Rapid-MLX — Design Spec

**Date:** 2026-09-04
**Status:** Approved (design sections 1-3), pending spec review
**Reference:** `raullenchai/Rapid-MLX` (`vllm_mlx/telemetry/`, `vllm_mlx/routes/metrics.py`)

## Goal

Supplement complete observability/telemetry capability in fusion-mlx to align with Rapid-MLX. Close three gaps: (1) activation/engagement funnel, (2) per-request telemetry wiring, (3) Prometheus metric breadth. Fusion is already ahead on response-cache series richness, `/v1/health` (OOM risk + MLX memory), `/v1/metrics/json`, admin stats, drain — those stay unchanged.

## Tech Stack

**Python.** Telemetry is not on a performance-critical path — fire-and-forget into a bounded lossy queue, daemon-flushed, redacted; `/metrics` renders at scrape time from in-memory counters. Microsecond overhead either way, so Rust's performance advantage does not apply here. The alignment target (Rapid-MLX telemetry) is pure Python; fusion's existing `fusion_mlx/telemetry/` shares that ancestry. Introducing Rust would mean diverging from the target and adding a cargo/PyO3 toolchain to an otherwise pure-Python MLX stack — violates simplicity and convention. Performance concern resolved: Python is sufficient.

## Delivery Approach — C (Layered by risk)

Two PRs split at the prod-behavior-change boundary:

- **PR1 — Telemetry package + activation (dark, consent-gated, zero prod change).** All library additions, no call sites in prod request path. Ships dark.
- **PR2 — Wiring + Prometheus metrics (prod-visible).** Lights up PR1's emit helpers in live paths; expands `/metrics`.

## Scope — full alignment

Wire every fusion subsystem that exists; skip only missing modules (no fabrication):

- EXISTS (wireable): `radix_index`, `disk_kv_checkpoint`, `spec_decode`, `pflash`, `ubc_evict`, `turboquant`
- MISSING (skip, no module): `suffix_decode`, `spec_decode_mtp`, `mxfp4_moe_guardrail`

---

## Section 1 — Telemetry package + activation funnel (PR1, dark)

Everything consent-gated, zero prod call sites. Ships dark.

### New module — `fusion_mlx/telemetry/activation_spec.py` (~110L)

Port from Rapid-MLX verbatim with renames. Pure constants — no consent/network deps.

- `ACTIVATION_SPEC_VERSION = 2`
- 7 milestone kinds (each fires at most once per install/client_id): `first_inference`, `model_pull`, `agent_setup`, `first_chat_reply`, `first_vision_reply`, `first_dictation`, `first_image`
- 3 surfaces: `cli`, `api`, `desktop`
- `ACTIVATION_KIND_SURFACE_PAIRS` frozenset — closed set of valid (kind, surface) combinations
- `is_allowed_activation(activation_kind, surface)` — validates pairs so no caller-controlled free-form text reaches payload
- `CHAT_SPAWN_ENV = "FUSION_MLX_CHAT_SPAWN"` (renamed from RAPID_MLX_CHAT_SPAWN) — env var chat front-end sets on its spawned server so chat-driven first inference attributes to `cli` surface, not `api`
- `INFERENCE_ENDPOINTS = frozenset({"/v1/chat/completions"})` — single endpoint instrumented for activation
- `is_successful_inference(status, completion_tokens)` — single success predicate: HTTP 2xx AND non-empty completion. Shared by call sites and tests.

### `schema.py` additions

- `ActivationPayload` dataclass: `activation_kind`, `surface`, `client_id`, `spec_version`, `occurred_at_epoch`
- `RequestPayload` field additions: `caller_agent`, `output_degenerate`, `completion_empty`, `completion_abnormally_short`
- `TelemetryPayload` gains `activation` slot (alongside session/request/error)
- `sample_request_preview_payload()` builder — for CLI `telemetry preview` (currently missing; preview shows only session sample)

### `redact.py` addition

- `normalize_caller_agent(user_agent)` + `_CALLER_AGENT_MARKERS` table (claude-code, cursor, aider, continue, cline, roo, other). Buckets free-form UA string into fixed allowlist — no raw UA on the wire.

### `state.py` additions

- `activation_marker_path(kind)` — path to once-per-install marker file
- `claim_activation_marker(kind)` — atomic file-create + in-process latch, returns True only first time per install
- `_validate_activation_kind(kind)` — validates against `ACTIVATION_KINDS`
- `reset_state()` extended to glob+delete `activation_seen_*` marker files + clear in-process latch (currently clears only consent + client-id)
- `CURRENT_CONSENT_SCHEMA_VERSION` bump 1 → 2 (forces re-consent since schema gained activation slot)

### `emit.py` additions

- `activation(activation_kind, surface, **extra)` — consent-gated, validates kind/surface via `is_allowed_activation`, claims marker (fires once per install), builds `ActivationPayload`, enqueues
- `server_surface()` — reads `FUSION_MLX_CHAT_SPAWN` env → `cli` if set else `api`
- `request()` signature widens: `+caller_agent=None`, `+output_degenerate=False`, `+completion_empty=False`, `+completion_abnormally_short=False` (threaded into RequestPayload via redact)
- Request sampling gate: `_request_sample_rate()` reads `FUSION_MLX_TELEMETRY_REQUEST_SAMPLE` (default 0.1), `_should_sample_request()` gate inside `request()` before enqueue

### CLI

`telemetry preview` shows both session sample AND request sample (the missing `sample_request_preview_payload`).

### Doc

`docs/telemetry-activation.md` — human-facing activation funnel spec. English (README-English rule). Doc-only.

### `__init__.py`

No new public exports — activation is emit-internal; spec module imported by emit/state directly.

### Tests (PR1)

- `tests/unit/test_telemetry_activation_spec.py` — pair validation, success predicate, spec version
- `tests/unit/test_telemetry_activation_marker.py` — once-per-install claim, reset wipes markers
- `tests/unit/test_telemetry_request_preview_payload.py` — request sample builder shape

### No prod call sites in PR1

Zero request-path change. All dark until PR2 wires it.

---

## Section 2 — Per-request wiring + error wiring (PR2, prod-visible)

Lights up PR1's dark emit helpers in live paths.

### Per-request wiring (non-streaming)

In `routes_internal/chat.py` completion path (after token counts aggregated, before response return), call:

```
emit.request(
    endpoint="/v1/chat/completions",
    model_alias=<alias>,
    stream=False,
    tool_call_used=<bool>,
    prompt_tokens=<n>,
    completion_tokens=<n>,
    ttft_ms=<ms>,
    tps=<float>,
    status=<int>,
    caller_agent=<UA header>,
    output_degenerate=<coherence>,
    completion_empty=<bool>,
    completion_abnormally_short=<bool>,
)
```

Sampling gate (default 0.1) inside `emit.request` — every request evaluates, fraction enqueued. `is_enabled()` guard short-circuits consent-declined installs.

### Per-request wiring (streaming)

Same `emit.request(...)` in the streaming generator finalize path, once final token counts + TPS + TTFT known. One call per stream finalize.

### Activation wiring

At chat completion where `is_successful_inference(status, completion_tokens)` matches, call `emit.activation(activation_kind=ACTIVATION_FIRST_INFERENCE, surface=emit.server_surface())`. Marker claim ensures once-per-install. Also `model_pull` milestone on CLI `pull` path (`cli.py` model-download complete).

### Error wiring

- `server.py` lifespan startup — model-load failure → `emit.error(category="model_load_failure", phase="startup", fingerprint=<fingerprint>)`
- `server.py` lifespan shutdown — unhandled shutdown traceback → `emit.error(category="shutdown_traceback", phase="shutdown", fingerprint=<fingerprint>)`
- `cli.py` model-load path — load failure → `emit.error(category="model_load_failure", phase="chat", ...)`

### Output degenerate coherence detector

Lightweight heuristic, no model (Rule 5 — decide with code not tokens). `coherence.looks_like_garbage(text)`: True if `completion_tokens > 0` but output empty OR hits repetition-character threshold. Pure deterministic. Wired in chat route, passed to `output_degenerate` / `completion_empty` / `completion_abnormally_short`.

### TTFT/TPS collection

Fusion already tracks `prefill_duration` / `generation_duration` in `ServerMetrics.record_request_complete`. Capture TTFT (time-to-first-byte) and TPS (tokens/sec) alongside, pass to emit. No new timing infrastructure — reuse existing timers.

### Tests (PR2 wiring)

- `tests/unit/test_telemetry_request_wiring.py` — non-streaming emit fires with correct bucket fields
- `tests/unit/test_telemetry_streaming_request_wiring.py` — stream finalize fires once
- `tests/unit/test_telemetry_error_wiring.py` — lifespan load failure fires error event
- `tests/unit/test_telemetry_route_attribution.py` — surface=cli when `FUSION_MLX_CHAT_SPAWN` set, else api

All consent-gated — assert no fire when `is_enabled()==False`.

---

## Section 3 — Prometheus metric expansion (PR2, prod-visible)

Closes 91-vs-20 gap. Wire only existing subsystems; skip missing.

### Histogram strategy

Fusion `/metrics` currently has counters only — no `_bucket`/`_sum`/`_count`/`_max`. Add two true histograms (manual bucket rendering, no prom_client dep — match existing `_fmt_metric` pattern in `routes_internal/metrics.py`):

- `fusion_mlx_model_ttft_seconds{model}` — buckets `[0.01,0.05,0.1,0.25,0.5,1,2.5,5,10,30,+Inf]` + `_sum` + `_count` + `_max`
- `fusion_mlx_model_decode_tokens_per_second{model}` — buckets `[1,5,10,20,40,80,160,+Inf]` + `_sum` + `_count` + `_max` + `_last`

Per-request observe into per-model `Histogram` helper (in `server_metrics.py` — `record_request_complete` already called per request, extend to feed histogram). Render at scrape time.

### Metric families to add (wire to existing subsystems)

| Family | Source module | Series |
|---|---|---|
| prefix cache | `radix_index` (`to_dict`: hits/misses/depth/nodes/inserts/removes) | `fusion_mlx_prefix_cache_radix_{entries,nodes,max_depth,hits,misses,inserts,removes,deduped_bytes}` |
| KV checkpoint | `disk_kv_checkpoint` (stats: bytes/evictions/loads/writes/hook_errors) | `fusion_mlx_kv_checkpoint_{bytes,evictions_total,loads_total,writes_total,hook_errors_total}` |
| spec decode | `spec_decode` (`total_draft_accepted` counter) | `fusion_mlx_spec_decode_{accepted_total,drafts_proposed_total,accept_ratio}` |
| UBC eviction | `ubc_evict` | `fusion_mlx_ubc_evicted_bytes` |
| turboquant | `turboquant` | `fusion_mlx_turboquant_{applied_total,skipped_total}` |
| pflash | `pflash` | `fusion_mlx_pflash_{ops_total,bytes}` |
| embedding truncations | embedding path | `fusion_mlx_embedding_truncations_total` |

### Metric families SKIP (no module — fabrication would violate fail-visible)

- `suffix_decode`, `spec_decode_mtp`, `mxfp4_moe_guardrail`

Reason: no corresponding fusion subsystem exists. Logged in spec + commit.

### Prefix-cache bytes family

`fusion_mlx_prefix_cache_{current_bytes,cap_bytes,pressure_evictions_total}` — wire to scheduler/cache stats if exposed; if bytes not tracked, emit `_current_bytes` as gauge from available stat, log if absent.

### Queue gauges (missing)

- `fusion_mlx_requests_running` (gauge) — from scheduler running count
- `fusion_mlx_requests_waiting` (gauge) — from scheduler waiting count

### Uptime (missing)

- `fusion_mlx_uptime_seconds` (gauge) — already in `/v1/health`, surface to `/metrics`

### Metal memory (missing)

- `fusion_mlx_metal_active_bytes`, `fusion_mlx_metal_cache_bytes`, `fusion_mlx_metal_peak_bytes` — from `mx.get_active_memory()` etc. (reuse `health._mlx_memory_stats`)

### Fix stray prefix

Rename 5 `rapid_mlx_response_format_strict_*` → `fusion_mlx_response_format_strict_*` in `routes_internal/metrics.py`. Pure rename, values preserved.

### Tests (PR2 metrics)

- `tests/unit/test_prometheus_metric_expansion.py` — new families render at zero without engine, histograms render bucket/count/sum/max shape, `rapid_mlx_` prefix gone, skipped families absent
- `tests/unit/test_prometheus_histogram_render.py` — histogram bucket math

### Render dispatch

All new render functions added to `routes_internal/metrics.py` `render_prometheus_metrics()` dispatch. Each wrapped in try/except → emit nothing if subsystem absent (fail-visible in log, not crash on `/metrics`).

---

## Section 4 — Lead capabilities (beyond Rapid-MLX)

Sections 1-3 reach parity with what Rapid-MLX has. Section 4 adds observability dimensions Rapid-MLX structurally lacks — exploiting fusion's multimodal, Paged-KV, and distributed-decode advantages. Target: after landing + absorbing the 3 feature-gap losses (#787/#788/#789), fusion still leads because these dimensions have no Rapid-MLX counterpart.

### Multimodal inference metrics (Rapid is text-only)

- `fusion_mlx_vision_requests_total` — image/video inference request count
- `fusion_mlx_audio_requests_total` — STT/TTS request count
- `fusion_mlx_video_requests_total` — video generation request count
- `fusion_mlx_image_generation_requests_total` — image generation count
- `fusion_mlx_vae_encode_seconds{model}` — VAE encode time histogram
- `fusion_mlx_vae_decode_seconds{model}` — VAE decode time histogram
- `fusion_mlx_video_generation_seconds{model}` — total video generation time histogram (per model)

Per-modality request counts + VAE/video timing — Rapid has no multimodal dimension.

### Multimodal activation milestones (extend funnel beyond Rapid)

Rapid's activation spec covers text + 4 desktop milestones. fusion extends with API-surface multimodal milestones:

- `first_image_generation` — fusion has image gen, Rapid has no such milestone → NEW
- `first_video_generation` — fusion has video gen, Rapid has no such milestone → NEW
- `first_audio_transcription` — Rapid has `first_dictation` (desktop only); fusion adds API-surface variant

`ACTIVATION_KIND_SURFACE_PAIRS` extended: `+ (first_image_generation, api)`, `+ (first_video_generation, api)`. `ACTIVATION_SPEC_VERSION` → 3 (beyond Rapid v2).

### KV cache advanced metrics (fusion has Paged-KV, Rapid does not)

fusion shipped Paged-KV (v0.8.74 LRU + sliding-window). Rapid has no paged-KV subsystem → expose its internal state as a lead:

- `fusion_mlx_kv_cache_pages_total{model}` — paged-KV page count gauge
- `fusion_mlx_kv_cache_evictions_total{model}` — LRU eviction count
- `fusion_mlx_kv_cache_block_utilization{model}` — block utilization gauge

### Prefix-cache advanced metrics (beyond Rapid's 9 radix series)

Rapid exposes radix hits/misses/depth/nodes/inserts/removes. fusion adds derived gauges:

- `fusion_mlx_prefix_cache_radix_hit_rate` — hit-rate gauge (Rapid has hits/misses counts, no ratio gauge)
- `fusion_mlx_prefix_cache_radix_avg_depth` — average-depth gauge (Rapid has max_depth only)

### Distributed decode metrics (fusion has #630 distributed decode, Rapid does not)

- `fusion_mlx_distributed_decode_tokens_total{node}` — distributed decode token count
- `fusion_mlx_distributed_decode_rtt_seconds{node}` — node RTT histogram

### Daemon/serve lifecycle metrics (Rapid has none)

- `fusion_mlx_lifespan_startup_seconds` — startup time histogram (model load to ready)
- `fusion_mlx_lifespan_shutdown_seconds` — shutdown time histogram

### Tests (Section 4)

- `tests/unit/test_prometheus_multimodal_metrics.py` — modality counters + VAE/video histograms render
- `tests/unit/test_telemetry_activation_multimodal.py` — new milestones fire once-per-install, surface=api, spec version 3
- `tests/unit/test_prometheus_kv_cache_advanced.py` — paged-KV gauges render from pool state
- `tests/unit/test_prometheus_lifespan_metrics.py` — startup/shutdown histograms

### Lead accounting

| Dimension | Rapid-MLX | fusion after Section 4 |
|---|---|---|
| Telemetry package (usage events) | full | full (parity) |
| Activation funnel milestones | 7 (text + desktop) | 9 (+ image/video gen, API multimodal) |
| Prometheus modality metrics | none | 7 series |
| Paged-KV metrics | none (no subsystem) | 3 series |
| Distributed decode metrics | none | 2 series |
| Lifespan metrics | none | 2 series |
| Prefix-cache derived gauges | raw counts only | + hit_rate, avg_depth |
| Observability endpoints | basic probes + /metrics + /v1/status | + /v1/health, /v1/metrics/json, admin stats, drain |

Net: Sections 1-3 = parity on shared surface; Section 4 = lead on fusion-only surface. Feature gaps #787/#788/#789 filed as issues (not telemetry responsibility); their metric families wire when the features land. Landing losses absorbed, fusion still ahead.

---

## Global Constraints

- Only modify `/Users/dahai/claude-home/fusion-mlx` code
- 4-multiple indentation, NO docstrings, code must log
- No `mx.clear_streams()` in tests (#630 GOTCHA); `mx.clear_cache()` allowed
- Lint: `black --check --target-version py313` + `ruff`; never touch `debt_modules.txt`
- Real model loads gated `FUSION_*_REAL_MODEL=on`; default small model; no OOM
- README.md English-only; README_CN.md Chinese; GitHub ops English
- PyPI SKIPPED (no token); squash-merge PRs to main; push via SSH
- Clean process data after tests, keep only final artifacts + logs
- Tests must assert something meaningful (Rule 9)

## Open Items / Decisions Logged

- Tech stack: Python (not Rust) — performance not on critical path, target is Python, Rust = toolchain overhead with no benefit. User confirmed performance was only concern.
- Skipped metric families (suffix_decode, spec_decode_mtp, mxfp4_moe_guardrail): no fusion module. Documented, not fabricated. Issues filed: #787 (suffix-decode), #788 (MTP), #789 (mxfp4 guardrail) — feature gaps, not telemetry responsibility.
- `output_degenerate` coherence detector: deterministic heuristic, no model routing (Rule 5).
- Section 4 (lead capabilities): fusion leads after landing via multimodal / Paged-KV / distributed-decode / lifespan dimensions Rapid structurally lacks. `ACTIVATION_SPEC_VERSION` → 3 (beyond Rapid v2). Target is lead, not parity — parity absorbs landing losses into a loss.
