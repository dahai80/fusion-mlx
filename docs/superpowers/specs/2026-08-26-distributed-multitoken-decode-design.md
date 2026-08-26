# Distributed Multi-Token Decode — KV-Cache Loop Landing (#630 follow-up)

Status: DRAFT — awaiting human approval (brainstorming HARD-GATE).
Date: 2026-08-26
Related: shipped `/distributed/decode` PR #638 (commit 58c6f449), issue #630,
`docs/distributed-pipeline.md`.

## Context / What shipped

PR #638 shipped `/distributed/decode` (`fusion_mlx/distributed/shard.py:290`,
`fusion_mlx/api/distributed_routes.py:174`). It closes the gap that
`pipeline_step` ran only the layer loop and returned **un-normed** post-layer
hidden states, so distributed PIPELINE mode could slice layers across nodes
but never produced a token. `decode()` applies the final `inner.norm` +
`lm_head` (tied → `embed_tokens.as_linear`) and samples one token id per
position batch via `make_sampler`.

The shipped surface is a **single forward pass over one position batch**. The
`docs/distributed-pipeline.md` Out-of-scope list (L312-319) names exactly two
items this work lands:

- KV-cache / attention-mask threading across shards.
- Multi-token autoregressive loop — the scheduler composes
  `pipeline_step`+`decode` across nodes; `decode` is a single forward pass.

The docstrings repeat this: `shard.py:316-317` "the scheduler loops
pipeline_step+decode across nodes for multi-token output", `shard.py:48-49`
"No KV cache, no mask — first version is a pure feed-forward step. KV-cache
and attention-mask threading arrive with the streaming scheduler."

## Gap (what this lands)

Today a multi-token generation requires the scheduler to call
`pipeline_step`+`decode` **once per output token**, re-running the full layer
loop over the **entire prompt + generated-so-far** each time. Cost is
O(seq_len²) — token N re-processes all N-1 prior tokens. No KV is cached, so
each step pays the full prefill cost again. Correct but wasteful; the docstring
even says "the scheduler loops ... for multi-token output" as if that were
free.

This landing adds **per-shard in-process KV-cache** + a **cache-aware step
endpoint** (`decode_step`) so each shard appends the input positions' K/V to
its own cache and the scheduler loops a cheap decode step. **Prefill** (one
pass over the full prompt, `decode_step` with `input_ids=[P tokens]`) costs
the same as today's `pipeline_step`+`decode` — each prompt position traverses
all shards, populating KV. **The generation tail** (each subsequent token,
`decode_step` with a single-token input) drops to O(1) per token per shard:
one cheap step appending one K/V row, instead of re-processing the growing
context. The activation that crosses the wire is `[P, hidden]` on prefill
(same as today) and `[1, hidden]` per decode token (constant, tiny); no KV
ever crosses the wire. The win is on the autoregressive tail, which is the
case that matters for long generations.

## Scope decisions (confirmed with user)

1. **Full multi-token KV-cache loop** (not minimal, not scheduler-only). Lands
   the cache + step endpoint + loop composition in fusion-mlx.
2. **Per-shard in-process KV-cache, no transport.** Each shard holds its own
   KV-cache list in the `ShardManager` registry, grown from the activations
   each shard receives. NO KV crosses the wire — only **activations** cross
   (full `[P, hidden]` on prefill, `[1, hidden]` per decode token). KV is
   computed locally from those activations; each shard owns its KV privately.
   **Tradeoff accepted: KV is lost on shard process restart** (a restarted
   shard must re-prefill). Snapshot/restore endpoint deferred to a follow-up
   issue if needed.

These two decisions define the architecture. Everything below follows from
"in-process KV, no transport".

## Architecture

Generation splits into two phases, mirroring standard autoregressive LLMs:

**Prefill** (one call, full prompt):
- shard 0: `decode_step(input_ids=[prompt...], is_last_shard=False)` embeds +
  runs layers [start:end) **with cache** — each layer's `KVCache` populates
  with the prompt's K/V rows (P rows). Returns the **full `[P, hidden]`**
  boundary activation (all positions) to the next shard.
- shard 1..N-1: `decode_step(hidden_states=<[P, hidden]>, is_last_shard=False)`
  runs layers with cache, populating P rows. Returns `[P, hidden]`.
- last shard: `decode_step(hidden_states=<[P, hidden]>, is_last_shard=True)`
  → norm + lm_head on the last position + sample → token id #1. KV now holds P
  rows on every shard.

**Decode loop** (one token per iteration, N-1 more tokens):
- The sampled token becomes the single-token input. shard 0:
  `decode_step(input_ids=[token], is_last_shard=False)` — embeds 1 token, runs
  layers with cache (appends 1 K/V row; `cache.offset` now = P+1). Returns the
  single-position `[1, hidden]`.
- shard 1..N-1: `decode_step(hidden_states=<[1, hidden]>, is_last_shard=False)`
  — append 1 row each.
- last shard: `decode_step(..., is_last_shard=True)` → token id #2. Repeat.

Cross-wire payload: **`(P, hidden)` on prefill** (same as today's
`pipeline_step`), shrinking to **`(1, hidden)` per decode token** (constant
regardless of sequence length). KV stays in-process on each shard, grown from
the activations each shard receives — never transported.

### Why this works without transporting KV

`mlx_lm` layers are cache-aware: `LlamaDecoderLayer.__call__(x, mask, cache)`
passes `cache` to `self_attn`, which calls `cache.update_and_fetch(keys,
values)` (`mlx_lm/models/cache.py:333`) — appends this step's K/V and returns
the full accumulated K/V for the attention dot-product. RoPE uses
`cache.offset` for position. So a shard that holds its `cache` list across
calls and feeds `cache=cache[i]` to `layers[i]` recomputes attention exactly
as the un-split model would — the K/V of prior tokens is in the shard's own
memory, never sent. The split-at-layer-boundary bit-exactness invariant from
#638 holds: only the **activation** at the boundary hops the wire. On prefill
that's the full `[P, hidden]`; on decode it's `[1, hidden]`. KV is always
local, derived from whatever activation a shard receives.

### KV-cache storage

`ShardManager._shards[shard_id]` gains a `"kv_cache": list[KVCache] | None`
field. `None` until first `decode_step`; lazily initialized as
`[KVCache() for _ in range(num_layers)]` (one per decoder layer, matching
`LlamaModel.__call__`'s `cache=[None]*len(layers)` convention at
`llama.py:184`). Each layer index in `[start:end)` uses `cache[i]` (the cache
list is full-model-length; a shard only touches its own slice — `cache[j]` for
`j not in [start,end)` stays empty/unused). `drop_shard` discards it;
`sync_weights` clears it (stale K/V under new weights is wrong — see Error
handling / Risks #3).

## API surface

Add **one** new endpoint; keep `pipeline_step` and `decode` unchanged (they
remain the cache-less single-pass path for callers that want a stateless
forward, and for the existing tests).

### `POST /distributed/decode_step` (NEW)

Cache-aware forward + sample. The workhorse of the multi-token loop. Serves
both prefill (multi-token `input_ids`, populates KV with P rows) and decode
(single-token input, appends 1 row). One call = run the shard's layer range
with cache, appending `len(input)` K/V rows per layer + (if last shard) sample
the next token from the last position.

Request:
```json
{
  "shard_id": "shard-...",
  "hidden_states": "<base64 .npy of incoming hidden, [P,hidden] prefill or [1,hidden] decode>",
  "input_ids": [128001, 264, ...],
  "temperature": null,
  "top_p": null,
  "return_logits": false,
  "is_last_shard": true
}
```

- `hidden_states` / `input_ids` — exactly one present, same rule as
  `pipeline_step`. First shard (`input_ids`) embeds; later shards
  (`hidden_states`) consume the boundary activation. **Any length** — prefill
  sends the full prompt (length P, shape `[P, hidden]` / `[P]` ids); decode
  sends one token (length 1, shape `[1, hidden]` / `[1]` ids). The cache
  appends `len(input)` rows either way via `update_and_fetch`. No `prefill`
  flag — input length is the signal, and `mlx_lm` treats prefill and decode as
  the same code path (differing only in sequence length). Multi-position is
  the prefill call; single-position is the decode step. (Corrected from the
  earlier single-position draft — see KV-cache lifecycle §"Core insight".)
- `is_last_shard` — `true` on the shard that owns the final layers; it applies
  norm+lm_head and samples (from the **last position** of its input — prefill
  samples token #1 from position P-1; decode samples from the single position),
  populating `token_ids` in the response. `false` on intermediate shards; they
  return only `hidden_states` for the next shard. This lets one endpoint serve
  both roles (the scheduler knows which shard is last from the `layer_range`).
- `temperature` / `top_p` / `return_logits` — only read when
  `is_last_shard=true`; ignored otherwise.

Response (intermediate shard, `is_last_shard=false`):
```json
{
  "hidden_states": "<base64 .npy, [P,hidden] prefill or [1,hidden] decode>",
  "shape": [1, 2048],
  "dtype": "mlx.core.bfloat16",
  "token_ids": null,
  "kv_offset": 5
}
```

Response (last shard, `is_last_shard=true`):
```json
{
  "hidden_states": null,
  "token_ids": [912],
  "shape": [1],
  "dtype": "mlx.core.int32",
  "logits": null,
  "kv_offset": 5
}
```

- `kv_offset` — the shard's cache length after this step (read from
  `cache[start].offset`; debug/introspection, lets the scheduler confirm all
  shards are in lockstep without transporting KV).
- `token_ids` — single-element list (one sampled token, from the last
  position) on the last shard; `null` on intermediate.
- `shape` — the outgoing tensor shape: `[P, hidden]` or `[1, hidden]` for an
  intermediate shard's hidden; `[1]` for the last shard's sampled token.

### Existing endpoints — unchanged

`pipeline_step` and `decode` keep their current cache-less behavior. They are
the "stateless single forward" path. `decode_step` is the "stateful
autoregressive" path. A generation either uses `decode_step` throughout
(recommended) or stays on the cache-less pair (correct but O(seq²)).

### `POST /distributed/reset_cache` (NEW, small)

Clear a shard's KV-cache so a new generation can prefill from scratch without
dropping/reloading the shard (weights stay on GPU).

```json
{ "shard_id": "shard-..." }
```
→ `{ "shard_id": "shard-...", "kv_cleared": true, "prev_offset": 12 }`

Needed because KV lives in the registry; without this, a second prompt on the
same shard would append to the first prompt's cache (wrong). The scheduler
calls `reset_cache` at the start of each new generation, or after EOS.

## shard.py changes

### `ShardManager._shards` registry

Add `"kv_cache": None` to the shard dict in `load_shard` (L204-211). Lazily
filled on first `decode_step`. Full-model-length list once filled
(`[KVCache() for _ in range(total)]`); a shard only touches indices in its
`layer_range`.

### `decode_step(...)` method (NEW)

Signature:
```python
def decode_step(
    self,
    shard_id: str,
    hidden_states_b64: str | None,
    input_ids: list[int] | None,
    is_last_shard: bool,
    temperature: float | None = None,
    top_p: float | None = None,
    return_logits: bool = False,
) -> dict:
```

Logic:
1. `shard = self._get_shard(shard_id)`, `model`, `inner`, `start, end`,
   `layers`.
2. Validate: exactly one of `hidden_states_b64` / `input_ids` present (else
   `ShardError`). No length restriction — prefill passes `len(input_ids)==P`
   (full prompt) or `hidden_states` shape `[P, hidden]`; decode passes
   `len==1` / `[1, hidden]`. The cache appends `len(input)` rows either way.
   If `hidden_states_b64`: deserialize to `[P, hidden]` (accept `[hidden]` or
   `[1, hidden]` by adding the seq dim → `[1, hidden]`). First shard always
   uses `input_ids`; later shards always use `hidden_states`.
3. Build input hidden: first-shard path embeds via
   `inner.embed_tokens(mx.array(input_ids)[None,:])` → `[1, P, hidden]`
   (prefill: P=prompt len; decode: P=1). later-shard path uses the
   deserialized hidden `[1, P, hidden]`.
4. Lazy-init cache: if `shard["kv_cache"] is None`, set it to
   `[KVCache() for _ in range(total)]`. Import `from mlx_lm.models.cache
   import KVCache`.
5. Run the layer loop with cache: `for i in range(start, end): hidden =
   layers[i](hidden, cache=shard["kv_cache"][i])`. (Pass `mask=None` — the
   cache's `make_mask` / offset handles causal masking; single position needs
   no explicit mask when cache is present. Verify against un-split `generate`
   in tests.)
6. `mx.eval(hidden)`.
7. If `is_last_shard`: apply `inner.norm` + lm_head on the **last position**
   (`hidden[:, -1:, :]`) — reuse the exact block from existing `decode()`
   L335-351, factored into a private `_project_and_sample(hidden_last,
   temperature, top_p, return_logits)` helper shared by `decode` and
   `decode_step`. Sample, build `token_ids` (single element). Return
   `{"hidden_states": None, "token_ids": [...], "shape", "dtype",
   "logits"?, "kv_offset": shard["kv_cache"][start].offset}`.
8. Else: return `{"hidden_states": serialize_activation(hidden), "shape",
   "dtype", "token_ids": None, "kv_offset": shard["kv_cache"][start].offset}`
   — the FULL `[1, P, hidden]` goes to the next shard (prefill: all positions;
   decode: the single position).

### `reset_cache(shard_id)` method (NEW)

```python
def reset_cache(self, shard_id: str) -> dict:
    shard = self._get_shard(shard_id)
    start = shard["layer_range"][0]
    cache = shard["kv_cache"]
    prev = cache[start].offset if cache is not None else 0
    shard["kv_cache"] = None
    logger.info("distributed: reset KV cache shard %s (was offset %d)", shard_id, prev)
    return {"shard_id": shard_id, "kv_cleared": True, "prev_offset": prev}
```

**Note:** reads `cache[start].offset`, NOT `cache[0]` — the cache list is
full-model-length and `cache[0]` is an unused empty cache for a shard whose
range starts > 0 (see Risks #5). The `kv_offset` returned by `decode_step`
and shown in `ShardInfo` likewise reads `cache[start].offset`.

### `drop_shard` / `sync_weights`

`drop_shard` already discards the shard dict → KV dropped with it. No change.
`sync_weights`: clear KV after a weight swap (stale K/V under new weights is
semantically wrong) — set `shard["kv_cache"] = None` at the end of
`sync_weights`, log it. (Open question: confirm — see Risks.)

### `decode()` refactor

Extract the norm+lm_head+sample block (L335-379) into
`_project_and_sample(self, hidden, temperature, top_p, return_logits)` and
have both `decode()` and `decode_step(is_last_shard=True)` call it. Pure
mechanical extraction, no behavior change to `decode()` — its existing tests
must stay green.

## distributed_routes.py changes

### New Pydantic models

```python
class DecodeStepRequest(BaseModel):
    shard_id: str
    hidden_states: str | None = None
    input_ids: list[int] | None = None
    is_last_shard: bool
    temperature: float | None = None
    top_p: float | None = None
    return_logits: bool = False

class DecodeStepResponse(BaseModel):
    hidden_states: str | None = None
    shape: list[int]
    dtype: str
    token_ids: list[int] | None = None
    logits: list[list[float]] | None = None
    kv_offset: int

class ResetCacheRequest(BaseModel):
    shard_id: str

class ResetCacheResponse(BaseModel):
    shard_id: str
    kv_cleared: bool
    prev_offset: int
```

### New routes

```python
@router.post("/decode_step")
async def decode_step(
    req: DecodeStepRequest, _: None = Depends(verify_api_key)
) -> DecodeStepResponse:
    try:
        result = get_manager().decode_step(
            shard_id=req.shard_id,
            hidden_states_b64=req.hidden_states,
            input_ids=req.input_ids,
            is_last_shard=req.is_last_shard,
            temperature=req.temperature,
            top_p=req.top_p,
            return_logits=req.return_logits,
        )
        return DecodeStepResponse(**result)
    except ShardError as exc:
        raise _shard_error_response(exc)

@router.post("/reset_cache")
async def reset_cache(
    req: ResetCacheRequest, _: None = Depends(verify_api_key)
) -> ResetCacheResponse:
    try:
        result = get_manager().reset_cache(req.shard_id)
        return ResetCacheResponse(**result)
    except ShardError as exc:
        raise _shard_error_response(exc)
```

Reuses the existing `_shard_error_response(exc)` (404 unknown shard / 502 load
fail / 400 otherwise) and `Depends(verify_api_key)`. No new error-mapping
code. Response shapes mirror `PipelineStepResponse` / `DecodeResponse` for
consistency (same `hidden_states`/`shape`/`dtype`/`token_ids` keys).

### No change to existing routes

`load_shard`, `pipeline_step`, `decode`, `sync_weights`, `list_shards`,
`drop_shard` untouched. `ShardsListResponse` / `ShardInfo` could optionally
expose a `kv_offset` field (read from `shard["kv_cache"][0].offset if kv_cache
else 0`) — included as a small convenience for debugging lockstep, but not
required for correctness. Decision: **add `kv_offset: int` to `ShardInfo`**
(one line, read-only, harmless if cache is None → 0).

## KV-cache lifecycle + correctness

### Lifecycle

```
load_shard     → _shards[id] = {..., "kv_cache": None}
decode_step #1 → lazy init: kv_cache = [KVCache()]*num_layers
                 prefill: layer loop appends prompt K/V rows
                 kv_cache[i].offset = len(prompt) on every layer
decode_step #k → append 1 K/V row per layer; offset grows +1 each call
reset_cache    → kv_cache = None (new generation re-prefills)
drop_shard     → shard dict gone → KV freed with it
sync_weights   → kv_cache = None (stale K/V under new weights = wrong)
```

### Correctness invariants

1. **Full-model-length cache list.** `kv_cache` has `num_layers` entries (one
   per decoder layer), NOT `end - start`. A shard only *uses* indices
   `[start, end)` but the list spans the whole model. Why: matches
   `LlamaModel.__call__`'s `cache=[None]*len(layers)` convention; if we ever
   split differently the indexing stays stable. Indices outside `[start, end)`
   stay default-empty KVCache objects — never touched, zero cost.

2. **Lockstep offsets across shards.** Every shard appends exactly one row per
   `decode_step` call. After K decode_step calls on shard S, `S.kv_cache[i].offset
   == base + K` for the same `base` on all shards (base = prefill length). The
   scheduler does NOT enforce this — the *call pattern* does: each token flows
   shard 0 → 1 → ... → N-1, one decode_step each. `kv_offset` in the response
   lets the scheduler *verify* lockstep if it wants; it's not required for
   correctness (a skipped shard would just produce wrong attention, fail
   visibly in output).

3. **No transport = no consistency protocol.** Because KV never leaves a shard,
   there's no cache-coherence problem. Each shard's KV is private state, derived
   from the activations it receives. The only cross-shard datum is the boundary
   activation (`[P, hidden]` prefill / `[1, hidden]` decode), bit-exact as in
   #638. This is the key simplification — the whole design exists to avoid a KV
   transport/consistency layer.

4. **`mask=None` is correct with cache.** `mlx_lm` attention with a present
   `KVCache` builds the causal mask internally from `cache.offset`
   (`cache.py:make_mask`). For a single decode position with cached history,
   no explicit mask is needed — the cache knows its length. We pass `mask=None`
   to `layers[i](hidden, mask=None, cache=cache[i])`. **Verify** in tests:
   decode_step output must bit-match (within bf16 tolerance) the equivalent
   position from un-split `mlx_lm.generate` on the same prompt. If it
   diverges, the mask handling is wrong — that's the #1 correctness risk.

5. **Prefill vs decode: activation size, not a separate mode.** The boundary
   activation that crosses the wire is the FULL sequence on prefill
   (`[P, hidden]`, all prompt positions) and a SINGLE position on decode
   (`[1, hidden]`). `decode_step` accepts both — it does NOT enforce
   single-position. Prefill sends the full prompt; decode sends one token.
   This is the same shape `pipeline_step` already uses (full hidden tensor);
   the difference is `decode_step` threads the cache so each shard's KV is
   populated from the activations it receives.

### Core insight: KV is derived from activations, no transport needed

KV is computed locally in each shard from the activations entering that shard's
layers. A shard that receives the full `[P, hidden]` activation sequence for
its input layer computes its own KV for all P positions locally — it never
needs another shard's KV. "No transport" means no KV crosses the wire;
**activations do cross the wire** (that's the whole point of pipeline
parallelism, and `pipeline_step` already does it). So:

- **Prefill (one pass, full prompt):** shard 0 gets `input_ids=[P tokens]`,
  embeds to `[P, hidden]`, runs layers `[0,k)` **with cache** → KV populated
  with P rows, outputs `[P, hidden]` boundary activation. The FULL `[P, hidden]`
  is sent to shard 1 (not just the last position). Shard 1 runs layers `[k,2k)`
  with cache → KV populated P rows, outputs `[P, hidden]`. ... Last shard runs
  with cache, samples from the last position → token #1. **All shards now hold
  P KV rows.** Cost: O(P × depth) — each prompt position traverses all shards,
  same as the current `pipeline_step`+`decode` (no regression).
- **Decode loop (one token each, P+1..end):** shard 0 gets `input_ids=[1 token]`,
  embeds `[1, hidden]`, runs with cache → appends 1 row (offset P+1), outputs
  `[1, hidden]`. Single position sent onward. ... Last shard samples → token #2.
  Repeat. Cost: **O(1) per token per shard** — one cheap step appending one K/V
  row. WITHOUT the cache, each generated token re-traversed the full growing
  context (O(seq²) total). WITH the cache, the generation tail is O(tokens).

**The win is on the generation tail**, which is exactly the autoregressive
case that matters for long generations. Prefill cost is unchanged from the
current cache-less path (and inherent to layer-split pipeline). The boundary
tensor on prefill is `[P, hidden]` (same as today); on decode it shrinks to
`[1, hidden]` (constant per token). No KV ever crosses the wire; each shard
owns its KV privately, grown from the activations it receives.

### `decode_step` accepts multi-position (prefill) and single-position (decode)

Given the insight above, `decode_step` does NOT enforce single-position input.
The single-position rule in the API-surface section is **dropped** — replace
with: "`input_ids` / `hidden_states` may be any length; prefill sends the full
prompt (length P), decode sends one token (length 1). The cache appends
`len(input)` rows either way." No `prefill` flag needed — the length of the
input IS the signal, and the cache handles both uniformly via
`update_and_fetch`. This is simpler than a flag and matches how `mlx_lm`
itself works (prefill and decode are the same code path, differing only in
sequence length).

## Error handling

Reuses the existing `ShardError` + `_shard_error_response(exc)` mapping
(`distributed_routes.py:129`), no new error types:

| Condition | `ShardError` trigger | HTTP |
|-----------|---------------------|------|
| unknown shard_id | `KeyError` lookup miss | 404 |
| both/neither `input_ids`+`hidden_states` | validation `ShardError` | 400 |
| load failure (model missing) | existing load path | 502 |
| cache op on dropped shard | caught as unknown shard | 404 |

- **`decode_step` on a shard whose KV was reset mid-generation:** returns
  wrong output (attention over empty cache) but does NOT crash. The scheduler
  owns the reset→prefill ordering; a stray reset is a scheduler bug, not a
  server crash. Log a `WARNING` if `kv_offset==0` but `input_ids` is a single
  token (likely a decode call with no prefill — suspicious) — fail visibly,
  don't silently produce garbage. Actually: don't warn-and-continue on a likely
  bug; **raise `ShardError("decode_step single-token input but KV empty —
  prefill first")`** (400). Prefill (multi-token) on empty KV is normal and
  allowed.
- **`reset_cache` on a shard with `kv_cache is None`:** no-op, returns
  `prev_offset: 0, kv_cleared: True`. Idempotent, not an error.
- **`sync_weights` clearing KV:** set `kv_cache=None`, log `INFO`. A weight
  swap invalidates all cached K/V (different weights → different K/V). The
  caller must re-prefill after a sync. Documented in the response? No —
  `sync_weights` response unchanged; the KV-clear is a side effect logged
  server-side. Open question: should we surface it? (see Risks) — default no.

## Tests

Per the project rule "涉及到大模型测试，须真实加载模型" — tests load a real
model. Use **Llama-3.2-1B-Instruct-4bit** (small, fast, already in
`~/.fusion-mlx/models` per memory; if absent, download via hf-mirror.com).
Start/stop fusion-mlx via `~/claude-home/fusion-mlx/start.sh start|stop`.

### Unit tests (no model, fast)

`tests/unit/test_distributed_decode_step.py` (NEW):
- `test_decode_step_rejects_both_input_modes` — both `input_ids`+`hidden_states`
  → 400.
- `test_decode_step_rejects_neither_input_mode` — neither → 400.
- `test_decode_step_rejects_single_token_on_empty_cache` — single-token
  `input_ids` with `kv_cache is None` → 400 ("prefill first").
- `test_decode_step_accepts_prefill_on_empty_cache` — multi-token `input_ids`
  on empty cache → 200, `kv_offset == len(input_ids)`.
- `test_reset_cache_idempotent_on_empty` — `reset_cache` on `kv_cache=None` →
  `prev_offset: 0, kv_cleared: True`, no error.
- `test_reset_cache_clears_after_prefill` — prefill, `reset_cache` →
  `kv_offset` back to 0 path; subsequent single-token → 400 (must re-prefill).
- `test_shard_info_exposes_kv_offset` — `list_shards` shows `kv_offset` field.
- `test_decode_step_unknown_shard_404` — bad `shard_id` → 404.
Mock the manager where possible; these are route/validation tests, no model.

### Integration tests (real model, slow — mark `@pytest.mark.realmodel`)

`tests/integration/test_distributed_decode_step_e2e.py` (NEW):
- `test_single_shard_decode_step_matches_generate` — load whole model as ONE
  shard (start=0, end=num_layers). Prefill a short prompt via `decode_step`
  (input_ids, is_last_shard=True) → token #1. Then loop decode_step
  (single-token, is_last_shard=True) for 5 tokens. Compare the token sequence
  to `mlx_lm.generate` on the same prompt/sampler. **Bit-exact (greedy,
  temp=None) is the headline correctness test.** This pins that threading the
  cache through `decode_step` reproduces un-split generation.
- `test_two_shard_decode_step_matches_generate` — split at the midpoint
  (shard A: [0, L/2), shard B: [L/2, L)). Prefill: A (input_ids, not last) →
  B (hidden_states, last) → token #1. Decode loop: A → B → token #2...5.
  Compare to `mlx_lm.generate`. **Bit-exact.** Pins that the boundary
  activation crossing is correct WITH cache (the #638 invariant held
  cache-less; this extends it to cached).
- `test_decode_loop_kv_offset_grows` — after prefill (P tokens), `kv_offset==P`
  on both shards; after K decode steps, `kv_offset==P+K`. Lockstep.
- `test_reset_then_reuse` — generate, `reset_cache` on both shards, generate a
  different prompt → correct (cache didn't bleed across generations).
- `test_kv_survives_within_process` — two generations WITHOUT reset on the
  same shard append (wrong) — document the expected behavior: the test asserts
  the server does NOT auto-reset (scheduler's job), so a missing reset gives
  wrong output. This pins the "reset is the caller's responsibility" contract.

Clean up process data after tests (project rule) — keep only final outputs +
logs.

## Risks / open questions

1. **`mask=None` correctness with cache (TOP RISK).** Unverified that
   `layers[i](hidden, mask=None, cache=cache[i])` reproduces un-split
   attention bit-exactly for the decode step. `mlx_lm`'s `KVCache.make_mask`
   should handle it, but the layer signature must accept `cache` and pass it
   to `self_attn`. **Mitigation:** the `test_*_matches_generate` integration
   tests are the gate — bit-exact vs `mlx_lm.generate`. If they fail, root-cause
   the mask handling before shipping (systematic-debugging, not a workaround).

2. **Tied embeddings + cache interaction.** The shipped `decode()` uses
   `inner.embed_tokens.as_linear` for tied models (shard.py:343). `decode_step`
   reuses the factored `_project_and_sample` → same path, no new risk. But
   confirm `args.tie_word_embeddings` probe still works on the loaded shard
   model (it did in #638).

3. **`sync_weights` clearing KV — confirm desired.** Default: clear (stale K/V
   under new weights is wrong). Alternative: leave KV, let caller reset
   explicitly. **Decision needed:** lean clear-and-log (safer default; a weight
   swap mid-generation is rare and re-prefill is cheap relative to a weight
   load). Open for user review.

4. **Concurrency: two generations on the same shard.** KV is per-shard
   process-singleton state. Two concurrent generations sharing a shard would
   corrupt each other's KV. **Out of scope** — the caller serializes per shard,
   or uses separate shards. Document this limit in the doc. A future
   per-generation KV handle would fix it; not now.

5. **`ShardInfo.kv_offset` / `reset_cache` prev_offset must read `start`, not
   `0`.** The cache list is full-model-length; `kv_cache[0]` is unused (empty)
   for a shard whose range starts > 0. Read `kv_cache[start].offset` instead.
   (Caught in this spec review — the implementation must use `start`, and the
   earlier `reset_cache` snippet reading `kv_cache[0]` is corrected to
   `kv_cache[start]`.)

6. **Prefill sends `[P, hidden]` across the wire — bandwidth.** Same as
   today's `pipeline_step`, so no regression. For very long prompts this is the
   dominant cost. Out of scope (inherent to layer-split pipeline; tensor
   parallelism would shard the activation, not the layers).

7. **Naming: `decode_step` vs a `use_cache` flag on `pipeline_step`.** Chose a
   new endpoint to keep `pipeline_step`/`decode` as the documented cache-less
   stateless path. Explicit endpoint is clearer and matches the doc's "two
   paths" framing. Rejected the flag (conflates stateful and stateless).
