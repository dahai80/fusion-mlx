# In-place LoRA Swap (#389)

By default, fusion-mlx serves each LoRA adapter as a **separate derived engine**:
the pool lazily builds a new entry keyed by `model_id::adapter` and reloads the
full base model (weights, tokenizer, KV-cache state) for every adapter. On a
128 GB machine several LoRAs across one base can exhaust memory — each adapter
pays for a second copy of the base weights.

The in-place swap path keeps **one base engine resident** and swaps the LoRA
adapter onto it in place, using mlx_lm's own `LoRALinear` machinery
(`load_adapters` / `remove_lora_layers`). The low-rank `lora_a`/`lora_b` arrays
are added beside the base quantized linear — never fused into the packed
weights — so it is correct for 4-bit / 8-bit quantized bases and allocates no
second base copy.

This mirrors the video DiT in-place inject/remove pattern
(`video/adapters/animatediff.py`).

## Trade-off

Only **one adapter is active on a given base at a time**. Concurrent multi-LoRA
would race on the shared model graph. The pool serializes the
apply → infer → restore window with a per-base `asyncio.Lock`, and a bare-base
request waits for any in-flight swap to restore before reading the weights.
This matches the PRD: "single base + N LoRA co-resident, millisecond switch".

## Enabling

Two environment variables (both default OFF):

| Variable | Default | Purpose |
|---|---|---|
| `FUSION_LORA_INPLACE_SWAP` | `0` | `1` enables the in-place swap path. |
| `FUSION_LORA_ALLOWED_DIRS` | *(empty)* | Comma-separated allow-list of adapter directories. Required for **any** per-request adapter (in-place or derived). Default-deny. |

```bash
export FUSION_LORA_INPLACE_SWAP=1
export FUSION_LORA_ALLOWED_DIRS="$HOME/.fusion-mlx/adapters"
~/claude-home/fusion-mlx/start.sh start
```

With the flag OFF, the existing per-adapter derived-engine behavior is
unchanged.

## Usage

Send the adapter directory path in the `adapters` field of any chat / generate
request:

```bash
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  http://127.0.0.1:11434/v1/chat/completions \
  -d '{
    "model": "Qwen3-0.6B-4bit",
    "adapters": "/Users/dahai/.fusion-mlx/adapters/Qwen3-0.6B-4bit/fixA",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 16
  }'
```

The `adapters` value must resolve to a path inside one of the
`FUSION_LORA_ALLOWED_DIRS` directories (realpath, default-deny).

## How it works

1. First request cold-loads the base engine (no adapter) — the derived path
   still handles the very first load.
2. A subsequent request with `adapters` on an **already-loaded** base takes the
   in-place path: acquire the per-base lock → restore any prior swap →
   `InPlaceLoRASwap.apply()` (wraps base linears as `LoRALinear`, loads the
   adapter) → return the base engine with the adapter active.
3. On release, `InPlaceLoRASwap.restore()` calls `remove_lora_layers`,
   unwrapping back to the original base linears (bit-exact restore), and
   releases the lock.
4. A bare-base request (no `adapters`) on a base with an active swap waits for
   the swap to restore first, so it never reads LoRA-applied weights.

Switch latency is ~7 ms apply / ~1 ms restore on Qwen3-0.6B-4bit (112
`LoRALinear` modules), measured on Apple Silicon.

## Implementation

- `fusion_mlx/adapter/weight_swap.py` — `InPlaceLoRASwap` class.
- `fusion_mlx/pool/engine_pool.py` — `_inplace_swap` flag, `_swap_lock`,
  `_acquire_inplace_adapter` / `_release_inplace_adapter`, and the
  `get_engine` / `release_engine` early-return hooks.
