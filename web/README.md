# fusion-mlx web monitor (union-web shell module, #1220)

React + Vite monitor SPA for the fusion-mlx engine, packaged as a
union-web shell module. Three pages: **Models** (discover/load/unload,
per-model settings), **Metrics** (tiles, charts, MLX memory, OOM badge),
**Logs** (tail + live SSE stream).

## Build

```bash
cd web
bun install
bun run build          # typecheck + vite build -> web/dist
```

`web/dist` is a build artifact and is not committed. The engine serves it
when present: `GET /` returns `index.html`, `/assets/*` is mounted behind
every API router. Without a dist, `/` keeps the legacy `{"status": "ok"}`
JSON probe (`/health`, `/healthz`, `/livez` always stay JSON).

## Run standalone (dev)

```bash
cd web
VITE_API_KEY=<engine api key> bun run dev     # http://127.0.0.1:11471
```

The dev server proxies `/v1`, `/admin`, `/metrics`, `/health`, `/stats` to
`FUSION_DEV_BACKEND` (default `http://127.0.0.1:11434`). Standalone builds
send `Authorization: Bearer <VITE_API_KEY>`; embedded builds send no auth
header and rely on the gateway's injected credentials.

## Serve (production, #1276)

```bash
~/fusion/fusion-mlx/start.sh web start    # builds assumed; serves :11471
~/fusion/fusion-mlx/start.sh web status
~/fusion/fusion-mlx/start.sh web stop
# or directly: fusion-mlx web-serve [--port 11471]
```

`fusion_mlx/admin/webserve.py` serves `web/dist` at `/` and reverse-proxies
`/v1` `/admin` `/metrics` `/stats` to the engine (default
`http://127.0.0.1:11434`, override `FUSION_MLX_WEB_ENGINE_URL`) — this is
the `upstream` the manifest declares. `FUSION_MLX_WEB_API_KEY` (when set)
gates those API paths on the bridge; forwarded requests carry the engine
key (env > settings.json), so the UI bundle holds no secret.
`FUSION_MLX_WEB_UPSTREAM` overrides the manifest upstream at registration
time.

## Embed in the union-web gateway

`manifest.json` describes the module (`entry: /shell/fusion-mlx/`). Register
it one of two ways:

- Static: gateway config / shell registry.
- Automatic: set `FUSION_SHELL_GATEWAY_URL` + `FUSION_GATEWAY_ADMIN_KEY`
  (or a `shell` block in `~/.fusion-mlx/settings.json`) and
  `fusion_mlx/shell_registry.py` registers the manifest and heartbeats.

Because the SPA is built with a relative base and a hash router, the same
dist works at `/` and under `/shell/fusion-mlx/`.

## Features (#1278)

- **Metrics**: capacity gauge first screen — memory-pressure bar
  (unified used/total), KV utilization / Metal / footprint tiles read from
  `/metrics`; usage retrospection with 5m/30m/2h/24h windows aggregated
  client-side from the localStorage sample ring (`src/components/usageStore.ts`).
- **Models**: cards grouped by modality, collapsible groups, one-click
  load/unload.
- **Logs**: level floor / keyword / "from HH:MM" filters over the live tail.
- **Embedded** (#1277): minimal top tab bar (Models/Metrics/Logs) when served
  inside the shell; standalone keeps the bottom nav.

## Layout

```
src/api.ts                  API client (auth header, log-line parser)
src/sdk.ts                  shell host bridge (standalone vs embedded)
src/components/usageStore.ts  localStorage usage sample ring (#1278)
src/pages/ModelsPage.tsx
src/pages/MetricsPage.tsx
src/pages/LogsPage.tsx
vendor/@fusion/web-ui    vendored design-system preset (shared with other shells)
vendor/@fusion/web-sdk   vendored shell host SDK
```

## Endpoints used

`GET /admin/api/models`, `POST /admin/api/models/{id}/{load,unload}`,
`GET /admin/api/global-settings`, `GET /admin/api/stats`, `GET /metrics`,
`GET /admin/api/logs`, `GET /admin/api/logs/stream` (SSE).
The stream is consumed with `fetch` + `ReadableStream` because `EventSource`
cannot set the auth header.
