# fusion-mlx Enhancement Proposal: Grammar Constrained Decoding + Prefix Caching

## Summary

Proposal to add two high-value inference optimizations to the `fusion-mlx` local backend, enabling structured tool-call generation (eliminating client-side JSON parse failures) and multi-turn prompt prefix reuse (slashing TTFT for agentic coding flows).

These enhancements are consumed transparently by `fusion-code` (the local-first CLI agent) and any OpenAI-compatible client. No client-side changes required.

---

## 1. Grammar Constrained Decoding (outlines)

### Motivation

Local 7B/14B coder models (Qwen2.5-Coder, DeepSeek-Coder-V2) frequently emit malformed tool-call JSON when running unconstrained sampling — missing quotes, truncated braces, stray markdown fences. This forces the CLI client to implement fragile regex repair and still surfaces `Invalid tool parameters` errors mid-session.

By constraining token sampling at the logits level ( DFA masking), the model is mathematically guaranteed to emit 100% schema-valid JSON. This eliminates an entire class of client-side parsing failures.

### Proposed API

Add optional `response_format` passthrough on `/v1/chat/completions`, honoring the OpenAI JSON Schema convention:

```json
POST /v1/chat/completions
{
  "model": "Qwen3.6-27B-mxfp8",
  "messages": [...],
  "response_format": {
    "type": "json_schema",
    "json_schema": { ...oneOf tool-call schema... }
  }
}
```

### Implementation sketch

```python
import outlines
from functools import lru_cache

model, tokenizer = mlx_lm.load(...)
outlines_model = outlines.from_mlxlm(model, tokenizer)

@lru_cache(maxsize=16)
def get_cached_json_generator(schema_str: str):
    # DFA compile only on cache miss; ~1-3s for 20+ tool oneOf
    return outlines.generate.json(outlines_model, schema_str)

@app.post("/v1/chat/completions")
async def chat_completion(req):
    prompt = tokenizer.apply_chat_template(req.messages, tokenize=False)
    if req.response_format and req.response_format.get("type") == "json_schema":
        schema_str = json.dumps(req.response_format["json_schema"], sort_keys=True)
        gen = get_cached_json_generator(schema_str)
        content = gen(prompt, max_tokens=req.max_tokens or 1024)
    else:
        content = outlines.generate.text(outlines_model)(prompt, max_tokens=...)
    return {"choices": [{"message": {"content": content}}]}
```

### Key behaviors

- `response_format` absent → unconstrained free text (backward compatible)
- `response_format.type === "json_object"` → constrain to generic JSON object
- `response_format.type === "json_schema"` → constrain to provided schema
- LRU cache keyed by canonical (sort_keys=True) schema string → 0ms replay for unchanged toolsets
- **Prompt layout for cache friendliness**: system prompt → tool schemas → static project context → dynamic conversation tail (prefix-maximal so KV cache hits longest)

### Client contract (fusion-code side, already wired)

`fusion-code` sends Anthropic Messages-format `tools[]`; the adapter translates to OpenAI `functions[]`. When `response_format` is supported by the backend, the adapter additionally packages the tool list as a `oneOf` json_schema so the backend can enforce it. **No client change needed once backend lands this PR.**

---

## 2. Prefix Caching (KV Cache Reuse)

### Motivation

Agentic coding sessions resend ~80% identical prefix (system prompt + tool definitions + early conversation) every turn. Recomputing prefill KV each turn wastes M-series unified memory bandwidth and inflates TTFT from ~50ms (cache hit) to multiple seconds (full recompute).

`mlx-lm` ships `make_prompt_cache(model)` returning an `LRUPromptCache` that trims and reuses KV automatically by longest-prefix match. We just need to thread it through the request loop.

### Proposed implementation

```python
from mlx_lm.models.cache import make_prompt_cache

# Global per-model cache (survives across requests in the same session)
shared_kv_cache = make_prompt_cache(model)

@app.post("/v1/chat/completions")
async def chat_completion(req):
    prompt = tokenizer.apply_chat_template(req.messages, tokenize=False)
    # mlx-lm's generator auto-detects cached prefix tokens and only prefills the delta
    output = generate(model, tokenizer, prompt, max_tokens=..., cache=shared_kv_cache)
    return {"choices": [{"message": {"content": output}}]}
```

### Caveats

- Only works correctly with **full-attention** models (Llama 3.2, Mistral, DeepSeek-Coder full-attention variants). Sliding-window / Mamba/SSM architectures degrade to full recompute — should detect and warn.
- Cache eviction is automatic (LRU); no manual invalidation needed when conversation diverges.
- Concurrent requests on the same model should either share cache (sequential) or use per-request cache (parallel) — recommend a `cache_mode` query param: `shared` (default) vs `isolated`.

---

## 3. Performance targets

| Metric | Current | Target |
|--------|---------|--------|
| Tool-call JSON parse failures (client side) | ~3-8% of turns | 0% |
| Multi-turn TTFT (10k token prefix) | 2-4s | <100ms |
| DFA compile overhead (20 tools) | n/a | One-time ~1-3s, then 0ms via LRU |

---

## 4. Non-goals

- No changes to OpenAI-compatible response shape
- No new endpoints — only optional fields on existing `/v1/chat/completions`
- No client coupling — `fusion-code` already works without this; it just degrades to client-side repair

---

## 5. Acceptance criteria

- [ ] `response_format.type === "json_schema"` returns schema-valid JSON for all tested schemas
- [ ] Repeated identical schema → no recompile (verify via logs)
- [ ] Multi-turn prefix reuse → measurable TTFT reduction on second turn
- [ ] Backward compatible: requests without `response_format` behave unchanged
- [ ] Warning emitted when running on sliding-window architecture

---

## 6. References

- outlines mlx-lm integration: https://outlines-dev.github.io/outlines/
- mlx-lm prompt cache: https://github.com/ml-explore/mlx-lm (see `models/cache.py`)
- Client-side validator (current fallback): `fusion-code/src/services/api/fusion-mlx-tool-validator.ts`

---

Prepared by: fusion-code agent (on behalf of @dahai)
Date: 2026-07-20
