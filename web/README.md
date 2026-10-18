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

## Embed in the union-web gateway

`manifest.json` describes the module (`entry: /shell/fusion-mlx/`). Register
it one of two ways:

- Static: gateway config / shell registry.
- Automatic: set `FUSION_SHELL_GATEWAY_URL` + `FUSION_GATEWAY_ADMIN_KEY`
  (or a `shell` block in `~/.fusion-mlx/settings.json`) and
  `fusion_mlx/shell_registry.py` registers the manifest and heartbeats.

Because the SPA is built with a relative base and a hash router, the same
dist works at `/` and under `/shell/fusion-mlx/`.

## Layout

```
src/api.ts               API client (auth header, log-line parser)
src/sdk.ts               shell host bridge (standalone vs embedded)
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
