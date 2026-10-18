# fusion-mlx web 监控台（union-web shell 模块，#1220）

fusion-mlx 引擎的 React + Vite 监控单页应用，以 union-web shell 模块形式发布。
共三个页面：**Models**（发现/加载/卸载、单模型设置）、**Metrics**（指标卡、
图表、MLX 显存、OOM 徽标）、**Logs**（尾部查看 + SSE 实时流）。

## 构建

```bash
cd web
bun install
bun run build          # typecheck + vite build -> web/dist
```

`web/dist` 是构建产物，不入库。存在时由引擎托管：`GET /` 返回
`index.html`，`/assets/*` 挂载在所有 API 路由之后。没有 dist 时 `/` 保持
旧版 `{"status": "ok"}` JSON 探测（`/health`、`/healthz`、`/livez` 始终是 JSON）。

## 独立运行（开发）

```bash
cd web
VITE_API_KEY=<引擎 api key> bun run dev     # http://127.0.0.1:11471
```

开发服务器把 `/v1`、`/admin`、`/metrics`、`/health`、`/stats` 代理到
`FUSION_DEV_BACKEND`（默认 `http://127.0.0.1:11434`）。独立构建发送
`Authorization: Bearer <VITE_API_KEY>`；嵌入构建不发送认证头，由网关注入。

## 嵌入 union-web 网关

`manifest.json` 描述模块（`entry: /shell/fusion-mlx/`）。两种注册方式：

- 静态：网关配置 / shell 注册表。
- 自动：设置 `FUSION_SHELL_GATEWAY_URL` + `FUSION_GATEWAY_ADMIN_KEY`
  （或 `~/.fusion-mlx/settings.json` 中的 `shell` 段），
  `fusion_mlx/shell_registry.py` 完成注册与心跳。

SPA 使用相对 base + hash 路由，同一份 dist 在 `/` 与 `/shell/fusion-mlx/`
下都能运行。

## 目录结构

```
src/api.ts               API 客户端（认证头、日志行解析）
src/sdk.ts               shell host 桥接（独立 vs 嵌入）
src/pages/ModelsPage.tsx
src/pages/MetricsPage.tsx
src/pages/LogsPage.tsx
vendor/@fusion/web-ui    内置设计系统 preset（与其他 shell 共用）
vendor/@fusion/web-sdk   内置 shell host SDK
```

## 使用的接口

`GET /admin/api/models`、`POST /admin/api/models/{id}/{load,unload}`、
`GET /admin/api/global-settings`、`GET /admin/api/stats`、`GET /metrics`、
`GET /admin/api/logs`、`GET /admin/api/logs/stream`（SSE）。
日志流用 `fetch` + `ReadableStream` 消费，因为 `EventSource` 无法设置认证头。
