# Gateway as Sole Entry

Deployment recipe for running fusion-mlx as a gateway-internal backend: every
client talks to `fusion-gateway` (`:11432`), and `:11434` rejects anything that
does not present the shared API key.

## Why

By default fusion-mlx binds loopback and, when no API key is configured, serves
anonymous requests with a startup WARNING. That is the right default for a
laptop. It is the wrong posture for a machine whose only intended entry point
is the gateway: a client left pointing at `:11434` keeps working, bypassing
gateway logging, metering and routing.

## 1. Set the key, and turn on strict mode

Pick one source for the key (`--api-key` > `FUSION_MLX_API_KEY` >
`settings.json` `auth.api_key`), then enable strict mode by any of:

```bash
fusion-mlx serve <model> --strict-auth          # flag
FUSION_MLX_STRICT_AUTH=1 fusion-mlx serve ...   # env
```

```json
{ "auth": { "api_key": "...", "strict": true } }
```

In strict mode the server **refuses to start** when:

- no API key resolves from any source, or
- `FUSION_ALLOW_ANONYMOUS=true` is set (the anonymous override disables auth
  on every route, which is exactly what strict mode exists to prevent).

Failing at boot is deliberate. A warning line is not a control: the failure
mode being prevented is a production box quietly serving anonymous inference.

## 2. Share the key with the gateway only

Give the key to `fusion-gateway` via `backends.fusion-mlx.api_key_env`, and do
not put `FUSION_MLX_API_KEY` in any client environment. Clients authenticate to
the gateway, not to fusion-mlx.

## 3. Point clients at the gateway

All client base URLs become `http://127.0.0.1:11432` rather than
`http://127.0.0.1:11434`. Anything still pointing at `:11434` shows up in the
server log as a rate-limited warning, one line per host+path per 60s:

```
auth: rejected (invalid API key) host=203.0.113.7 path=/v1/chat/completions —
a caller still using :11434 directly? (one line per 60s per host+path)
```

That log is the migration checklist: it names the hosts still to move.

## 4. Verify

```bash
curl -s http://127.0.0.1:11434/v1/models                    # 401
curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:11434/v1/models  # 200
curl -s http://127.0.0.1:11434/health                       # 200, unauthenticated
```

## Route coverage

Every mounted route enforces the key except the probe and docs endpoints:

| Exempt path | Why |
| --- | --- |
| `/` | root probe / web shell index |
| `/health`, `/health/ready`, `/healthz`, `/livez`, `/readyz` | supervisor and load-balancer probes |
| `/docs`, `/docs/oauth2-redirect`, `/redoc`, `/openapi.json` | API documentation |
| `/favicon.ico` | static asset |

`/metrics`, `/v1/status`, `/v1/health`, the admin API and all model management
routes require the key.

This is enforced, not just documented: `fusion_mlx/middleware/auth_audit.py`
walks the live route table and `tests/unit/test_auth_strict_1381.py` fails if a
route is ever reachable without an auth dependency. The server also logs the
audit result at startup.
