# Distributed Pipeline Parallelism (#621)

First-version surface for fusion-multi-node **Pipeline Parallelism**: split a
transformer's layers across nodes, each node runs forward over its slice, and
activation tensors hop the boundary between shards.

This document covers the layer-range semantics, dtype support, and the HTTP
endpoints. Transport-level framing (FMP binary protocol, AES-GCM, compression)
is the scheduler layer's job — these endpoints expose only the forward step.

## Why

A single Apple Silicon node tops out at the model size it can hold in unified
memory. Pipeline parallelism lets a fleet of nodes cooperate on a model bigger
than any one of them: node A holds layers 0–7, node B holds layers 8–15, and
the activation tensor at the boundary is the only thing that crosses the wire.

The core invariant this surface guarantees: **splitting the forward at a layer
boundary, with a serialize→deserialize hop at the boundary, reproduces the
un-split forward bit-exactly.** Verified against
`mlx-community/Llama-3.2-1B-Instruct-4bit` (16 layers, hidden 2048) in
`tests/unit/test_distributed_pipeline.py`.

## Layer-range semantics

`layer_range` is a half-open interval `[start, end)` over the transformer's
decoder layers:

```
shard 0:  layers [0, 8)    embeds input_ids, runs layers 0..7, emits hidden
shard 1:  layers [8, 16)   receives hidden,  runs layers 8..15, emits hidden
```

Rules:

- `end > start` (a shard owns at least one layer).
- `end <= num_layers` (cannot slice past the model's layer count).
- The **first** shard (`shard_index=0`, or any shard called with no
  `hidden_states`) embeds `input_ids` via `embed_tokens`, then runs its layers.
- **Later** shards receive `hidden_states` (base64 `.npy`) and run their layers
  directly — no embedding.
- `load_shard` is idempotent for the same `(model_id, layer_range)` — repeated
  calls reuse the in-memory model (weights already on GPU) and return the same
  `shard_id`.

The forward over a shard is a plain loop: `for i in range(start, end): hidden =
layers[i](hidden)`. No KV cache, no mask — first version is a pure
feed-forward step (one position batch). KV-cache and attention-mask threading
arrive with the streaming scheduler.

## Activation tensor format

Activation tensors travel as **base64-encoded `.npy`** bytes:

```
mx.array --mx.save--> .npy bytes --base64--> str --HTTP--> 
str --base64decode--> .npy bytes --mx.load--> mx.array
```

This preserves **every mlx dtype bit-exactly**, including `bfloat16` — a numpy
detour would lose bf16 (numpy has no bf16 host type, so `np.array()` breaks on
PEP 3118 buffer protocol). `mx.save`/`mx.load` round-trip the raw mlx dtype
through the `.npy` container with no conversion.

Supported dtypes (verified in `test_activation_roundtrip_preserves_dtype_and_values`):

| dtype     | bit-exact round-trip |
|-----------|----------------------|
| float32   | yes                  |
| bfloat16  | yes                  |
| int32     | yes                  |

Malformed payloads (bad base64, empty, corrupt `.npy`) raise `ShardError`,
which the routes map to HTTP 400 — never a 500.

## Endpoints

All endpoints live under `/distributed` and require the API key
(`Depends(verify_api_key)`).

### `POST /distributed/load_shard`

Load a model and register a layer-range shard.

Request:
```json
{
  "model_id": "mlx-community/Llama-3.2-1B-Instruct-4bit",
  "shard_index": 0,
  "layer_range": [0, 8],
  "dtype": null
}
```

- `model_id` — HF repo id, or an absolute/local path. Bare repo ids resolve
  against `FUSION_MLX_MODEL_DIR` (default `~/.fusion-mlx/models`).
- `shard_index` — informational shard ordinal (for scheduler bookkeeping).
- `layer_range` — `[start, end)` half-open layer slice.
- `dtype` — informational hint (first version does not cast).

Response:
```json
{
  "shard_id": "shard-a1b2c3d4e5f6",
  "model_id": "mlx-community/Llama-3.2-1B-Instruct-4bit",
  "shard_index": 0,
  "layer_range": [0, 8],
  "num_layers": 16,
  "dtype": null
}
```

### `POST /distributed/pipeline_step`

Run forward over the shard's layers.

Request (first shard — embeds):
```json
{
  "shard_id": "shard-a1b2c3d4e5f6",
  "hidden_states": null,
  "input_ids": [1, 2, 3, 4],
  "position_ids": null
}
```

Request (later shard — receives hidden):
```json
{
  "shard_id": "shard-...",
  "hidden_states": "<base64 .npy>",
  "input_ids": null,
  "position_ids": null
}
```

- Exactly one of `hidden_states` / `input_ids` must be present. First shard
  needs `input_ids`; later shards need `hidden_states`.
- `position_ids` is accepted for forward-compat (first version is informational).

Response:
```json
{
  "hidden_states": "<base64 .npy>",
  "shape": [1, 4, 2048],
  "dtype": "mlx.core.bfloat16"
}
```

The returned `hidden_states` feeds the next shard's `pipeline_step`, or — for
the last shard — `decode` (below) applies the final norm + lm_head.

### `POST /distributed/decode` (#630)

Apply the final `norm` + `lm_head` to the **last** shard's hidden states and
return sampled token ids. This closes the gap left by `pipeline_step`, which
runs only the layer loop and returns the **un-normed** post-layer hidden states
— so distributed PIPELINE mode could slice layers but never produced a token.

`decode` is a single forward pass over one position batch; it returns one
sampled token id per position (the last is the next token for autoregressive
generation). The scheduler loops `pipeline_step`+`decode` across nodes for
multi-token output; KV-cache threading is a future streaming-scheduler concern.

Request:
```json
{
  "shard_id": "shard-...",
  "hidden_states": "<base64 .npy of the last shard's hidden states>",
  "temperature": null,
  "top_p": null,
  "return_logits": false
}
```

- `hidden_states` — required; the base64 `.npy` from the last shard's
  `pipeline_step`. Missing → 400.
- `temperature` — sampling temperature. `0` / `null` = greedy `argmax`
  (deterministic, matches a direct `mlx_lm` forward). `>0` routes through
  `make_sampler` with `top_p`.
- `top_p` — nucleus sampling mass (used only with `temperature > 0`).
- `return_logits` — include base64 `.npy` logits in the response. Off by
  default (a `(batch, seq, vocab)` tensor is bandwidth-heavy).

Output projection: tied-embedding models (`args.tie_word_embeddings`) reuse
`inner.embed_tokens.as_linear` (handles quantized matmul); otherwise the
dedicated `model.lm_head` is used. A model with no `lm_head` and
`tie_word_embeddings=False` → 400.

Response:
```json
{
  "token_ids": [128001, 306, 4990, 912],
  "shape": [1, 4],
  "dtype": "mlx.core.int32",
  "logits": null,
  "logits_shape": null,
  "logits_dtype": null
}
```

`token_ids` holds one sampled id per position. With `return_logits=true`, the
three `logits*` fields are populated (base64 `.npy` + shape + dtype of the
`(batch, seq, vocab)` logits tensor).

### `POST /distributed/sync_weights`

Hot-update a shard's weights (LoRA swap, adapter, weight sync).

Request:
```json
{
  "shard_id": "shard-...",
  "weights": "<base64 .npz of {param_path: array}>",
  "manifest": null
}
```

- `weights` — base64 `.npz` of `{param_path: array}`, applied via
  `model.load_weights(..., strict=False)`.
- `manifest` — forward-compat `{path: pull_url}`; accepted but not yet fetched.
  The scheduler must inline weights for now.

Response:
```json
{ "shard_id": "shard-...", "params_updated": 42 }
```

### `GET /distributed/shards`

List registered shards (ops/debug).

Response:
```json
{
  "shards": [
    { "shard_id": "shard-...", "model_id": "...", "shard_index": 0,
      "layer_range": [0, 8], "dtype": null, "num_layers": 16 }
  ]
}
```

### `DELETE /distributed/shards/{shard_id}`

Drop a shard's registration. The loaded model stays cached for other shards of
the same model.

Response:
```json
{ "shard_id": "shard-...", "dropped": true }
```

## Error mapping

| `ShardError` message prefix | HTTP status |
|-----------------------------|-------------|
| `unknown shard_id`          | 404         |
| `failed to load model`      | 502         |
| everything else             | 400         |

## Single-machine round-trip tests

`tests/unit/test_distributed_pipeline.py::test_pipeline_split_matches_unsplit_forward`
is the activation acceptance test:

1. Load `Llama-3.2-1B-Instruct-4bit`.
2. Reference: un-split forward `embed → layers[0:16]`.
3. Split: `embed → layers[0:8] → serialize → deserialize → layers[8:16]`.
4. Assert `mx.array_equal(split, reference)` — bit-exact.

`test_decode_matches_unsplit_lm_head_forward` is the #630 decode acceptance
test:

1. Two shards over the model (`pipeline_step` on each).
2. `decode` (norm + tied `lm_head` + `argmax`) on the last shard's hidden.
3. Reference: un-split `embed → layers[0:16] → norm → embed_tokens.as_linear
   → argmax`.
4. Assert `decode token_ids == reference argmax` — bit-exact.

Plus: greedy determinism (`test_decode_greedy_is_deterministic`),
`return_logits` round-trip (`test_decode_return_logits_round_trips`), and error
paths (missing hidden / unknown shard).

Skipped when the model is absent or mlx unavailable, so CI on non-Metal runners
passes without a model download.

## Python API

The endpoints are thin wrappers over `fusion_mlx.distributed`:

```python
from fusion_mlx.distributed import (
    ShardManager,
    serialize_activation,
    deserialize_activation,
    ShardError,
)

mgr = ShardManager()
a = mgr.load_shard(model_path, 0, [0, 8])
b = mgr.load_shard(model_path, 1, [8, 16])

# shard A embeds + runs layers [0,8)
out_a = mgr.pipeline_step(a["shard_id"], None, [1, 2, 3], None)
# shard B runs layers [8,16)
out_b = mgr.pipeline_step(b["shard_id"], out_a["hidden_states"], None, None)
# decode: norm + lm_head + argmax over B's hidden -> token ids
dec = mgr.decode(b["shard_id"], out_b["hidden_states"], temperature=0.0)
print(dec["token_ids"])  # one id per position, greedy = deterministic
```

## Out of scope (first version)

- KV-cache / attention-mask threading across shards (streaming scheduler).
- Multi-token autoregressive loop inside `decode` — the scheduler composes
  `pipeline_step`+`decode` across nodes; `decode` is a single forward pass.
- `manifest` URL pull (scheduler inlines weights for now).
- Cross-process coordination / failure recovery (scheduler layer).
- Compression / encryption of activation tensors (transport layer).
