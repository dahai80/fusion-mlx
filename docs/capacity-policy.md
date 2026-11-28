---
doc_id: fusion-mlx/capacity-policy
product: fusion-mlx
class: core
entry: service
doc_type: guide
title: { zh: "多模型容量策略与角色团队承载", en: "Multi-model capacity policy and role-team hosting" }
audience: engineer
status: draft
version: 1.0
updated: 2026-11-28
owner: @dahai
source_of_truth:
  - docs/capacity-policy.md
---

# Multi-model capacity policy and role-team hosting

How to host a multi-agent role team (e.g. a 5-role business pipeline: req / prd /
rnd / valid / launch) on one local MLX host without falling over to cloud, and
which capacity levers fusion-mlx already exposes. Answers fusion-console#1743.

## 1. Capacity levers (already shipped)

| Lever | Surface | Notes |
|-------|---------|-------|
| Manual unload | `POST /admin/api/models/{model_id}/unload` | Scoped keys: `fsb_*` / `model_mgr_*`. Returns 404 if unknown, 400 if not loaded. |
| Manual load | `POST /admin/api/models/{model_id}/load` | Same key scope. |
| Load state | `GET /admin/api/models/{model_id}/status` | `loaded`, `is_loading`, `pinned`. |
| Idle TTL unload | per-model `ttl_seconds` in settings; global idle timeout; env `FUSION_MLX_DEFAULT_TTL_SECONDS` (default `600`) | Non-pinned models always auto-evict after idle TTL (#1172). Runtime override via the ollama shim `keep_alive` path. |
| Pinning | `entry.pinned` | Pinned models never TTL-evict. Pin the one heavy model you always want resident. |
| LRU eviction | automatic under memory pressure | `_claim_lru_victim()` picks the least-recently-used engine when a new load needs room. |
| Admission preflight | heavy admission gate | `resident + projected_budget <= phys * 0.85`; rejected loads fail fast instead of wedging Metal. |

There is no "unload everything" endpoint by design; iterate the per-model
unload call over the pool listing (`GET /admin/api/models` or the `fusion-mlx
ps` CLI output).

## 2. Scriptable swap sequence (unload-before-load)

The gateway falls back to cloud only when the local model is unavailable or
saturated. To guarantee capacity for a team role, free memory *before* the
load, from the console scheduler job:

```bash
BASE=http://127.0.0.1:11434
KEY=model_mgr_...   # scoped admin key

# 1. List what is resident (CLI works too: fusion-mlx ps)
curl -s -H "Authorization: Bearer $KEY" $BASE/admin/api/models | jq '.[] | {id, loaded}'

# 2. Unload everything except the intended working model
for m in $(curl -s -H "Authorization: Bearer $KEY" $BASE/admin/api/models \
           | jq -r '.[] | select(.loaded and .id != "mlx-community/Qwen3.8-27B-4bit") | .id'); do
  curl -s -X POST -H "Authorization: Bearer $KEY" "$BASE/admin/api/models/$m/unload"
done

# 3. Load the working model (admission preflight will reject if still too big)
curl -s -X POST -H "Authorization: Bearer $KEY" \
     -H 'Content-Type: application/json' \
     -d '{"model_id": "mlx-community/Qwen3.8-27B-4bit"}' \
     $BASE/admin/api/models/load
```

Prefer TTL over manual unload for roles that run on a cadence: set the role
model's `ttl_seconds` in settings to its expected idle gap; the pool unloads
it between rounds and reloads on demand.

## 3. Lightweight role models

Only `rnd` (heavy coding) needs a large resident model. The other four roles
are stage review / verdict / summarization workloads — a 4-bit small model is
sufficient:

| Role | Model class | Approx resident weight (4-bit) |
|------|-------------|-------------------------------|
| rnd (coding) | 8B–27B coder, 4-bit | 4.5–17 GB |
| req / prd / valid / launch | 4B-class instruct, 4-bit | ~2–3 GB |

Place the small model in its own pool slot with a short `ttl_seconds` so the
four light roles share one slot instead of four. Verify the measured size of
your exact checkpoint via the pool status/admin surfaces — the numbers above
are checkpoints-class approximations, not per-repo facts.

If a role only needs structured verdicts (pass/fail + reason), combine this
policy with structured outputs (response_format json_schema) on the small
model rather than prompt-only discipline.

## 4. Concurrent-role budget

Memory, not compute, is the binding constraint for the mix above: the light
roles serialize on one small model through the continuous-batching scheduler,
and the coder model serves the single `rnd` stream.

- Start with **1 concurrent stream per small model** and **1 for rnd**; raise
  the small-model concurrency to 2–3 only if the queue-wait metric stays flat
  (`/metrics` exposes scheduler queue wait, #1717).
- Watch two signals before adding concurrency: rising scheduler queue-wait
  and falling per-request tok/s. Both indicate the batch is saturated.
- For a 5-agent burst pattern (all roles fire, then idle), the TTL policy from
  §2 removes the residency problem entirely: only one model is resident at a
  time, and the light roles queue behind each other on the shared slot.
- Budget check before any load: `sum(resident weights) + projected peak <= 85%
  of physical memory` — the admission gate enforces this; a rejected load
  means some other model must be unloaded first (see §2 loop).

## 5. Keeping the gateway on local

The gateway's cloud fallback defeats the privacy requirement only if it can
route team traffic to cloud models. Fusion-gateway (separate repo) owns the
cloud-breaker/model-routing configuration — configure the team route as
local-only there; fusion-mlx side, a 404/503 from a missing model is the
signal the gateway would react to. Do not "fix" saturation by adding cloud
routes; fix it with §2 (swap) or §3 (smaller role model).
