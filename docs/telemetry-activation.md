# Telemetry Activation Funnel

> **Status:** Dark — activation emit calls ship behind the consent gate
> (`is_enabled()`). No activation event fires unless the user has run
> `fusion-mlx telemetry enable`. This document is the human-facing spec
> for the growth/engagement funnel; the code contract lives in
> `fusion_mlx/telemetry/activation_spec.py`.

## Purpose

The activation funnel measures how users move from installing fusion-mlx
to deriving value from it, so we can prioritise the features and
surfaces that actually drive adoption. Each milestone fires at most
**once per install** — a file marker on disk records that the milestone
was seen, preventing duplicate counts on every subsequent run.

## Spec version

`ACTIVATION_SPEC_VERSION = 3`.

Rapid-MLX's activation spec sits at v2 (7 milestones). fusion-mlx
extends to v3 (9 milestones) by adding **multimodal** milestones that
Rapid does not track: `first_image`, `first_image_generation`, and
`first_video_generation`. The version bump also forces a consent
re-prompt under the new disclosure copy (`CURRENT_CONSENT_SCHEMA_VERSION`
rose to 3 in `state.py`).

## The 9 milestones

| Kind | Surface | Meaning |
|---|---|---|
| `first_inference` | cli, api | First completed inference (any endpoint in `INFERENCE_ENDPOINTS`). |
| `model_pull` | cli | First model downloaded/pulled to the local cache. |
| `agent_setup` | cli | First agent/tool-calling configuration observed. |
| `first_chat_reply` | desktop | First text chat reply delivered to the user. |
| `first_vision_reply` | desktop | First multimodal (vision) reply delivered. |
| `first_dictation` | desktop | First speech-to-text dictation completed. |
| `first_image` | desktop | First image consumed as an input. |
| `first_image_generation` | api | First image generated. |
| `first_video_generation` | api | First video generated. |

The last three are the multimodal additions over Rapid's v2.

## The 3 surfaces

- **cli** — `fusion-mlx` invoked directly in a terminal.
- **api** — requests arriving over HTTP to the server.
- **desktop** — the native macOS app.

Not every milestone can fire on every surface. The
`ACTIVATION_KIND_SURFACE_PAIRS` frozenset is the single source of truth
for which `(kind, surface)` combinations are valid; `is_allowed_activation`
checks it before any event is enqueued. An invalid pair is logged at WARN
and dropped — never sent.

## Surface attribution — `FUSION_MLX_CHAT_SPAWN`

When a CLI session is spawned *by* another tool (an agent harness, an
IDE plugin), the env var `FUSION_MLX_CHAT_SPAWN` (constant
`CHAT_SPAWN_ENV`) attributes that run to the **cli** surface rather than
letting it blur into the api surface. `emit.server_surface()` reads it:
set → `SURFACE_CLI`, unset → `SURFACE_API`.

## Inference scope — `INFERENCE_ENDPOINTS`

`INFERENCE_ENDPOINTS` is the frozenset of HTTP paths that count as an
"inference" call for the `first_inference` milestone. Today it is
`{"/v1/chat/completions"}`. A request outside this set does not advance
the funnel even if it returns 200.

## Success predicate — `is_successful_inference`

`is_successful_inference(status, completion_tokens)` decides whether a
request counts as a *successful* inference for funnel purposes:

- HTTP status in `[200, 300)`, **and**
- `completion_tokens > 0` (the model produced output).

Bad inputs (non-numeric status or tokens) are logged at WARN and return
`False` rather than raising — a malformed request must never crash the
telemetry path.

## Once-per-install markers

Each milestone writes a marker file under `~/.fusion-mlx/` on first
claim (`activation_seen_<kind>`). `claim_activation_marker(kind)` uses
`O_CREAT | O_EXCL` so the first caller wins atomically; subsequent
callers see `False` and the event is never re-sent, even across
restarts. An in-process latch (`_activation_latch`) short-circuits the
filesystem check for repeat calls within the same run.

`reset_state()` (the `fusion-mlx telemetry reset` command) wipes all
marker files and clears the latch — the next run re-claims every
milestone as if freshly installed.

## Privacy

Activation payloads carry only: the `activation_kind`, the `surface`,
the persistent `client_id` (a random UUID4 — see `state.py`), the
`spec_version`, and an `occurred_at_epoch` timestamp. No prompt text,
no model identifiers beyond what the kind already implies, no user
content. The same consent gate (`is_enabled()`) that governs all
telemetry governs activation — disable with `fusion-mlx telemetry
disable` or `--no-telemetry`, and no activation event is ever sent.
