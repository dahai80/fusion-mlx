# VLM Serving for GUI Agents (#795)

GUI agents (computer-use, browser automation, UI grounding) drive a vision-language model in a tight perception→action loop: every step the agent sends the current screenshot plus a short instruction and expects the next action (click, type, scroll) back within ~100 ms to feel interactive. This doc gives the recommended fusion-mlx setup to hit that target on Apple Silicon.

## Why fusion-mlx fits

Three already-shipped capabilities make sub-100 ms VLM turns achievable:

1. **Prefix KV-cache reuse across requests (#798)** — the system prompt, agent instructions, and recent screenshot-history prefix is identical across turns. `BlockAwarePrefixCache` (default) caches that prefix at block granularity (256 tokens) so only the *new* screenshot + new instruction is prefilled each turn. The shared prefix is paid once, reused every subsequent turn. See [Configuration → Prefix Cache](configuration.md#prefix-cache).
2. **Concurrent Fast+Slow residency (#796)** — keep a small fast VLM resident alongside a larger reasoning model; route GUI-grounding turns to the fast model. Pin both so neither is evicted. See [Configuration → Concurrent Multi-Model Serving](configuration.md#concurrent-multi-model-serving-796).
3. **TurboQuant 4-bit KV cache** — compresses the V-only KV cache ~4×, shrinking per-turn memory traffic. On by default.

## Recommended setup

### Model

Use a small, fast, GUI-tuned VLM as the **fast** model:

- `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` (or the 7B 4-bit if memory allows) — Qwen2.5-VL is trained on GUI/screenshots and grounds UI elements well.
- For pure action-grounding (no reasoning), a 3B 4-bit model is the right size/latency tradeoff on M-series SoCs.

Pin it so it stays resident:

```json
{
     "Qwen2.5-VL-3B-Instruct-4bit": {
          "pinned": true,
          "ttl_seconds": 0,
          "stream_interval": 1,
          "turboquant_kv_enabled": true
     }
}
```

### Serve flags

```bash
fusion-mlx serve Qwen2.5-VL-3B-Instruct-4bit --port 11434
```

Relevant defaults that help latency (no flags needed):
- `chunked_prefill=True` — prefill the new screenshot in chunks so decode can start sooner.
- `kv_cache_quant_enabled=True` — TurboQuant 4-bit KV (lower memory traffic).
- `stream_interval=1` — stream every token so the first action token returns ASAP.

### Request shape

To maximize prefix-cache hits, keep the prefix **byte-identical** across turns:

- Fixed system prompt (agent role + output format).
- Fixed action vocabulary / tool schema.
- Append screenshots + user turn at the *end* of the sequence, so the shared prefix stays a stable prefix and `BlockAwarePrefixCache` matches it.

Do **not** rephrase the system prompt per turn or inject timestamps into the prefix — any change breaks the chain-hash match and forces a full re-prefill.

```json
{
     "model": "Qwen2.5-VL-3B-Instruct-4bit",
     "messages": [
          {"role": "system", "content": "You are a GUI agent. Output the next action as JSON: {\"action\": ..., \"args\": ...}. Actions: click, type, scroll, wait, done."},
          {"role": "user", "content": [
               {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
               {"type": "text", "text": "What is the next action?"}
          ]}
     ],
     "max_tokens": 64,
     "stream": true
}
```

Limit `max_tokens` for the action (e.g. 64) — a single JSON action is short; capping it bounds the decode tail.

## Latency budget (M-series, 3B 4-bit VLM, prefix cached)

After the first turn warms the prefix cache, a typical GUI-agent turn breaks down roughly as:

| Phase | Cached | Notes |
|-------|--------|-------|
| Prefix prefill (system prompt + history) | reused (#798) | ~0 ms — block hit, not recomputed |
| New screenshot prefill | ~tens of ms | chunked prefill, image tokens only |
| Decode action tokens | ~tens of ms | 4-bit KV, `max_tokens` capped |
| Network (localhost) | <5 ms | |

The dominant cost shrinks to *new-screenshot prefill + short decode*, which is what keeps the loop under 100 ms once the prefix is warm. The first turn (cold prefix) is higher; the agent should warm the prefix with one throwaway turn if strict first-action latency matters.

## Tuning if over budget

- **Smaller model** — drop 7B → 3B, or 3B → a 1B/2B VLM if available; grounding quality trades against latency.
- **Lower resolution screenshots** — downscale before sending to cut image-token count (the biggest single prefill cost for VLMs).
- **`chunked_prefill_tokens`** — smaller chunks start decode sooner but add round-trips; tune for your model.
- **Pin the fast model + evict the slow one** (#796) when memory-tight, so the fast VLM is never the LRU victim.

## Verification

- Prefix-cache reuse across turns is unit-tested in `tests/unit/test_kv_cache_reuse_798.py`.
- Concurrent Fast+Slow residency + pin/lease protection is unit-tested in `tests/unit/test_concurrent_multi_model_796.py`.
- For an end-to-end latency number on your hardware, run the server (`./start.sh start`), warm the prefix with one request, then time a second request with the same prefix + a new screenshot via the `/v1/chat/completions` streaming endpoint.
