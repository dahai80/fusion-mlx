# Replay provider (`replay-dev`)

Deterministic, zero-GPU model for pipeline E2E structural runs (#1518).

## What it is

`replay-dev` is a built-in pseudo-model: requests never reach the engine
pool, the executor, or the GPU. Every reply is deterministic:

- **Echo mode** — the assistant message is the verbatim last user message.
- **JSON mode** — when the last user message contains the marker
  `[REPLAY_JSON]`, the reply is a canned JSON object:
  `{"echo": <user text minus marker>, "message_count": <n>, "replay": true}`.

Both streaming (SSE, single delta) and non-streaming shapes are supported.
Token usage is always zero.

## Usage

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"replay-dev","messages":[{"role":"user","content":"ping"}]}'
```

Reachable through fusion-gateway (`:11432`) with no extra configuration —
the gateway proxies `/v1/*` to fusion-mlx.

## Catalog semantics

`/v1/models` always lists `replay-dev` with `capabilities: {"tier":
"test"}` and `owned_by: "system"`. Chat-default resolution
(`model: "default"`) can never select it: the default resolver reads the
alias/registry config, where `replay-dev` is not registered.

## Pinning a seat to replay mode

In a delivery-pipeline (fusion-cowork) template, set the seat's model to
`replay-dev` for structural E2E runs — the DAG, tool plumbing, and
response handling are exercised end-to-end while no tokens are billed to
real models. Switch the seat back to the real model id for
quality-gated runs.
