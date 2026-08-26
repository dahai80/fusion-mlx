# Distributed Multi-Token Decode (KV-Cache Loop) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a cache-aware `decode_step` + `reset_cache` endpoint pair to the distributed pipeline surface so multi-token autoregressive generation drops from O(seq²) to O(tokens) on the generation tail, without transporting KV across nodes.

**Architecture:** Each shard holds a full-model-length list of `mlx_lm` `KVCache` objects in the `ShardManager` registry, grown lazily from the activations each shard receives. KV never crosses the wire — only activations do (`[P, hidden]` on prefill, `[1, hidden]` per decode token). `decode_step` serves both prefill (multi-token input, populates P KV rows) and decode (single-token input, appends 1 row) via input length, no `prefill` flag. The scheduler composes shard 0 → 1 → … → N-1 per token; the last shard applies norm + lm_head + samples.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, `mlx_lm` (KVCache, create_attention_mask, make_sampler), pytest. Test model: `mlx-community/Llama-3.2-1B-Instruct-4bit` (real, ~0.7GB, in `~/.fusion-mlx/models`).

**Spec:** `docs/superpowers/specs/2026-08-26-distributed-multitoken-decode-design.md` (committed `30dda9d2`).

## Global Constraints

- **4-space indentation, no docstrings, default logging on every new code path** (project rules).
- **Real model for integration tests** — `@skip_no_model` skipif when the small LM is absent (follow `tests/unit/test_distributed_pipeline.py` convention, NOT a `realmodel` marker). Start/stop fusion-mlx via `~/claude-home/fusion-mlx/start.sh start|stop` if a live server is needed.
- **Bit-exact correctness gate** — every integration test that splits layers MUST assert the decode token sequence equals `mlx_lm.generate_step` (or `mlx_lm.generate`) on the same prompt with a greedy sampler (`make_sampler(temp=0.0)`). This is non-negotiable; a divergence means the mask/cache threading is wrong and must be root-caused (systematic-debugging), not worked around.
- **Mask handling (corrected from spec):** the spec said "pass `mask=None` to `layers[i]`". That is WRONG. `LlamaModel.__call__` builds the mask ONCE before its layer loop via `create_attention_mask(h, cache[fa_idx])` and passes it positionally to each layer. `decode_step` must mirror this: build `mask = create_attention_mask(h, cache[start])` using a cache INSIDE the shard's slice (`start`), then pass `mask` positionally to `layers[i](h, mask, cache=cache[i])`. Validated bit-exact in a pre-plan probe (one-shard AND two-shard splits both match `generate_step`).
- **`pipeline_step` / `decode` stay unchanged** — they remain the cache-less stateless path. `decode_step` is the stateful autoregressive path. Existing `tests/unit/test_distributed_pipeline.py` must stay green.
- **KV is per-shard in-process, no transport.** `kv_cache` is a full-model-length `list[KVCache]`; a shard only touches indices `[start, end)`. `kv_offset` reads `cache[start].offset` (NOT `cache[0]` — `cache[0]` is unused for a shard whose range starts > 0).
- **`sync_weights` clears KV** (stale K/V under new weights is wrong): set `shard["kv_cache"] = None`, log INFO, response unchanged.
- **Concurrency:** one generation per shard at a time (KV is process-singleton state). Out of scope; documented.
- **Git:** branch `feat/630-multitoken-kv-decode` off main. Commit after each task. No pushes until the user says so.

---

## File Structure

- **`fusion_mlx/distributed/shard.py`** (MODIFY) — add `"kv_cache": None` to the shard dict in `load_shard`; add `decode_step()` + `reset_cache()` methods; extract `_project_and_sample()` helper shared by `decode()` and `decode_step()`; clear KV in `sync_weights`.
- **`fusion_mlx/api/distributed_routes.py`** (MODIFY) — add `DecodeStepRequest`/`DecodeStepResponse`/`ResetCacheRequest`/`ResetCacheResponse` Pydantic models; add `kv_offset: int` to `ShardInfo`; add `/decode_step` and `/reset_cache` routes.
- **`tests/unit/test_distributed_decode_step.py`** (CREATE) — fast unit tests: validation, 404, idempotent reset, KV-offset growth, single-token-on-empty-cache rejection. Mock the manager where possible; no model load for validation/route tests.
- **`tests/unit/test_distributed_decode_step_e2e.py`** (CREATE) — real-model integration tests (under the same file, `@skip_no_model`): one-shard + two-shard bit-exact vs `generate_step`, decode-loop KV-offset lockstep, reset-then-reuse, no-auto-reset contract. Bit-exact greedy is the headline gate.
- **`docs/distributed-pipeline.md`** (MODIFY) — document `decode_step`/`reset_cache`, the KV lifecycle, the no-transport model, and the concurrency limit.

---

## Task 1: Extract `_project_and_sample` helper from `decode()`

**Files:**
- Modify: `fusion_mlx/distributed/shard.py:318-379` (the `decode` method body)
- Test: `tests/unit/test_distributed_pipeline.py` (existing — must stay green)

**Interfaces:**
- Produces: `ShardManager._project_and_sample(self, hidden, temperature, top_p, return_logits) -> dict` — the norm + lm_head + sample block, factored out. `decode()` calls it. `decode_step()` (Task 3) will call it too.

- [ ] **Step 1: Write a regression anchor (use the existing bit-exact test)**

The existing `test_decode_matches_unsplit_lm_head_forward` in `tests/unit/test_distributed_pipeline.py:307` already pins that `decode()` reproduces the un-split argmax. Run it to confirm the pre-refactor baseline is green:

Run: `source .venv/bin/activate && pytest tests/unit/test_distributed_pipeline.py -k "decode_matches or decode_greedy or decode_return_logits" -v`
Expected: PASS (4 tests). If any fail, STOP — do not refactor on a red baseline.

- [ ] **Step 2: Add the `_project_and_sample` helper**

In `fusion_mlx/distributed/shard.py`, insert this private method on `ShardManager` immediately BEFORE the `decode` method (currently at line 290). It is the exact norm + lm_head + sample block lifted from `decode`, parameterized with an explicit `model` arg (so the helper does not depend on a shard dict — `decode` and `decode_step` both resolve the model, then delegate). No docstring (project rule):

```python
    def _project_and_sample(
        self,
        model: object,
        hidden: "mx.array",
        temperature: float | None,
        top_p: float | None,
        return_logits: bool,
    ) -> dict:
        import mlx.core as mx

        inner = model.model
        h = inner.norm(hidden)
        tie = bool(getattr(model.args, "tie_word_embeddings", False))
        if tie:
            logits = inner.embed_tokens.as_linear(h)
        else:
            if not hasattr(model, "lm_head"):
                raise ShardError(
                    "model has no lm_head and tie_word_embeddings is False — "
                    "cannot produce logits"
                )
            logits = model.lm_head(h)
        mx.eval(logits)
        temp = float(temperature) if temperature is not None else 0.0
        tp = float(top_p) if top_p is not None else 0.0
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temp, top_p=tp)
        sampled = sampler(logits)
        mx.eval(sampled)
        token_ids = [int(t) for t in sampled.reshape(-1).tolist()]
        out: dict = {
            "token_ids": token_ids,
            "shape": list(sampled.shape),
            "dtype": str(sampled.dtype),
        }
        if return_logits:
            out["logits"] = serialize_activation(logits)
            out["logits_shape"] = list(logits.shape)
            out["logits_dtype"] = str(logits.dtype)
        return out
```

- [ ] **Step 3: Refactor `decode()` to call the helper**

Replace the body of `decode` (lines ~318-379) so it resolves the model + hidden, then delegates. Keep the docstring, the `ShardError` for missing `hidden_states`, and the INFO log. New `decode` body:

```python
        shard = self._get_shard(shard_id)
        model = self._models[shard["model_id"]]
        if not hidden_states_b64:
            raise ShardError("decode needs hidden_states from the last shard")
        hidden = deserialize_activation(hidden_states_b64)
        logger.debug(
            "distributed: decode %s received hidden shape=%s dtype=%s",
            shard_id,
            hidden.shape,
            hidden.dtype,
        )
        out = self._project_and_sample(
            model, hidden, temperature, top_p, return_logits
        )
        logger.info(
            "distributed: decode %s produced %d token ids (temp=%s)",
            shard_id,
            len(out["token_ids"]),
            float(temperature) if temperature is not None else 0.0,
        )
        return out
```

- [ ] **Step 4: Run the existing decode tests — must stay green**

Run: `source .venv/bin/activate && pytest tests/unit/test_distributed_pipeline.py -k "decode" -v`
Expected: PASS (5 tests: matches_unsplit, greedy_is_deterministic, return_logits_round_trips, rejects_missing_hidden_states, unknown_shard_errors). No behavior change — pure extraction.

- [ ] **Step 5: Commit**

```bash
git add fusion_mlx/distributed/shard.py
git commit -m "refactor(distributed): extract _project_and_sample from decode (#630)"
```

---

## Task 2: Add `kv_cache` to the shard registry + `ShardInfo.kv_offset`

**Files:**
- Modify: `fusion_mlx/distributed/shard.py:204-211` (`load_shard` shard dict), `:436-437` (`list_shards`)
- Modify: `fusion_mlx/api/distributed_routes.py:111-117` (`ShardInfo`)
- Test: `tests/unit/test_distributed_decode_step.py` (CREATE, first test)

**Interfaces:**
- Produces: shard dict gains `"kv_cache": None` (full-model-length `list[KVCache]` once filled, lazily by `decode_step` in Task 3). `ShardInfo` gains read-only `kv_offset: int` (0 when cache is None).
- Consumes: nothing new (Task 3 will read it).

- [ ] **Step 1: Write the failing test for `ShardInfo.kv_offset`**

Create `tests/unit/test_distributed_decode_step.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for /distributed/decode_step + /distributed/reset_cache (#630).

Fast validation/route tests — no model load. Real-model bit-exact coverage
lives in test_distributed_decode_step_e2e.py (this file's sibling convention
follows test_distributed_pipeline.py, but e2e is split out for clarity)."""
from __future__ import annotations

import pytest

pytest.importorskip("mlx.core")


def test_shard_info_exposes_kv_offset_zero_on_fresh_shard():
    """A freshly loaded shard has kv_cache=None → list_shards reports
    kv_offset=0."""
    from fusion_mlx.distributed.shard import ShardManager

    mgr = ShardManager()
    # Use a dummy shard entry to avoid a real model load in this unit test.
    mgr._shards["shard-fresh"] = {
        "shard_id": "shard-fresh",
        "model_id": "dummy",
        "shard_index": 0,
        "layer_range": [0, 4],
        "dtype": None,
        "num_layers": 16,
        "kv_cache": None,
    }
    from fusion_mlx.api.distributed_routes import ShardInfo

    info = ShardInfo(**mgr.list_shards()[0])
    assert info.kv_offset == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/unit/test_distributed_decode_step.py::test_shard_info_exposes_kv_offset_zero_on_fresh_shard -v`
Expected: FAIL — `ShardInfo` has no `kv_offset` field (`pydantic.ValidationError: extra fields not permitted` or the dict is missing it).

- [ ] **Step 3: Add `kv_offset` to `ShardInfo`**

In `fusion_mlx/api/distributed_routes.py`, add to the `ShardInfo` class (after `num_layers`):

```python
    kv_offset: int = 0
```

- [ ] **Step 4: Make `list_shards` compute `kv_offset`**

In `fusion_mlx/distributed/shard.py`, change `list_shards` (line 436) to include the computed offset (reads `cache[start].offset`, 0 when cache is None):

```python
    def list_shards(self) -> list[dict]:
        out = []
        for s in self._shards.values():
            start = s["layer_range"][0]
            cache = s.get("kv_cache")
            offset = cache[start].offset if cache is not None else 0
            row = dict(s)
            row["kv_offset"] = offset
            out.append(row)
        return out
```

- [ ] **Step 5: Add `"kv_cache": None` to the shard dict in `load_shard`**

In `load_shard` (line 204-211), add the field:

```python
        self._shards[shard_id] = {
            "shard_id": shard_id,
            "model_id": model_id,
            "shard_index": shard_index,
            "layer_range": [start, end],
            "dtype": dtype,
            "num_layers": total,
            "kv_cache": None,
        }
```

- [ ] **Step 6: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/unit/test_distributed_decode_step.py::test_shard_info_exposes_kv_offset_zero_on_fresh_shard -v`
Expected: PASS.

- [ ] **Step 7: Confirm existing distributed tests still green**

Run: `source .venv/bin/activate && pytest tests/unit/test_distributed_pipeline.py -v`
Expected: PASS (all). The new `kv_offset` field on `ShardInfo` and `kv_cache: None` in the dict must not break existing `list_shards` / `ShardInfo(**s)` callers (the field defaults to 0 and the dict key is new-but-harmless).

- [ ] **Step 8: Commit**

```bash
git add fusion_mlx/distributed/shard.py fusion_mlx/api/distributed_routes.py tests/unit/test_distributed_decode_step.py
git commit -m "feat(distributed): kv_cache registry field + ShardInfo.kv_offset (#630)"
```

---

## Task 3: Implement `decode_step()` on `ShardManager`

**Files:**
- Modify: `fusion_mlx/distributed/shard.py` (add `decode_step` method after `decode`)
- Test: `tests/unit/test_distributed_decode_step.py` (append validation tests)

**Interfaces:**
- Produces: `ShardManager.decode_step(shard_id, hidden_states_b64, input_ids, is_last_shard, temperature=None, top_p=None, return_logits=False) -> dict` returning `{"hidden_states"?, "shape", "dtype", "token_ids"?, "logits"?, "kv_offset"}`.

- [ ] **Step 1: Write the failing validation tests**

Append to `tests/unit/test_distributed_decode_step.py`:

```python
def _dummy_shard(mgr, shard_id="shard-x", start=0, end=4, total=16):
    """Register a dummy shard without loading a model (validation tests only)."""
    mgr._shards[shard_id] = {
        "shard_id": shard_id,
        "model_id": "dummy",
        "shard_index": 0,
        "layer_range": [start, end],
        "dtype": None,
        "num_layers": total,
        "kv_cache": None,
    }
    mgr._models["dummy"] = object()  # placeholder; validation fails before use
    return shard_id


def test_decode_step_rejects_both_input_modes():
    from fusion_mlx.distributed.shard import ShardError, ShardManager

    mgr = ShardManager()
    sid = _dummy_shard(mgr)
    with pytest.raises(ShardError):
        mgr.decode_step(sid, hidden_states_b64="AAAA", input_ids=[1])


def test_decode_step_rejects_neither_input_mode():
    from fusion_mlx.distributed.shard import ShardError, ShardManager

    mgr = ShardManager()
    sid = _dummy_shard(mgr)
    with pytest.raises(ShardError):
        mgr.decode_step(sid, hidden_states_b64=None, input_ids=None,
                        is_last_shard=False)


def test_decode_step_rejects_single_token_on_empty_cache():
    """Single-token input_ids with kv_cache=None is a decode call with no
    prefill — fail visibly (400), do not silently produce garbage attention."""
    from fusion_mlx.distributed.shard import ShardError, ShardManager

    mgr = ShardManager()
    sid = _dummy_shard(mgr)
    with pytest.raises(ShardError, match="prefill"):
        mgr.decode_step(sid, hidden_states_b64=None, input_ids=[42],
                        is_last_shard=True)


def test_decode_step_unknown_shard_404():
    from fusion_mlx.distributed.shard import ShardError, ShardManager

    mgr = ShardManager()
    with pytest.raises(ShardError):
        mgr.decode_step("shard-nope", hidden_states_b64=None,
                        input_ids=[1, 2], is_last_shard=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_distributed_decode_step.py -k "rejects or unknown_shard" -v`
Expected: FAIL — `ShardManager` has no `decode_step` attribute (`AttributeError`).

- [ ] **Step 3: Implement `decode_step()`**

In `fusion_mlx/distributed/shard.py`, add this method on `ShardManager` immediately AFTER the `decode` method. It mirrors `pipeline_step`'s embed/deserialize split + `decode`'s projection, threading the cache and building the mask from an in-slice cache:

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
        # Cache-aware forward + optional sample (#630). Serves prefill
        # (multi-token input_ids) and decode (single-token input) by input
        # length, no prefill flag. KV is in-process per shard, never
        # transported. is_last_shard=True: norm + lm_head on the LAST
        # position + sample; is_last_shard=False: return the outgoing
        # activation for the next shard. kv_offset reads cache[start].offset.
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask
        from mlx_lm.models.cache import KVCache

        shard = self._get_shard(shard_id)
        model = self._models[shard["model_id"]]
        inner = model.model
        start, end = shard["layer_range"]
        layers = inner.layers

        if hidden_states_b64 and input_ids:
            raise ShardError(
                "decode_step: provide exactly one of hidden_states / input_ids"
            )
        if not hidden_states_b64 and not input_ids:
            raise ShardError(
                "decode_step: needs hidden_states or input_ids"
            )

        cache = shard["kv_cache"]
        # Single-token input_ids on an empty cache is a decode call with no
        # prefill — wrong attention over nothing. Fail visibly. (The
        # hidden_states path cannot hit this: an intermediate shard always
        # receives multi-token activations on prefill and [1,hidden] on decode
        # only AFTER shard 0 prefilled and grew this shard's cache via the
        # decode loop — so the cache is never None when a [1,hidden] arrives.)
        if cache is None and input_ids is not None and len(input_ids) == 1:
            raise ShardError(
                "decode_step single-token input but KV empty — prefill first"
            )

        if hidden_states_b64:
            hidden = deserialize_activation(hidden_states_b64)
            if hidden.ndim == 1:
                hidden = hidden[None, None, :]  # [hidden] -> [1,1,hidden]
            elif hidden.ndim == 2:
                hidden = hidden[None, :, :]  # [seq,hidden] -> [1,seq,hidden]
        else:
            if len(input_ids) > _MAX_INPUT_IDS:
                raise ShardError(
                    f"input_ids length {len(input_ids)} exceeds cap {_MAX_INPUT_IDS}"
                )
            ids = mx.array(input_ids, dtype=mx.int32)
            hidden = inner.embed_tokens(ids[None, :])  # (1, seq, hidden)

        # Lazy-init the full-model-length cache list on first decode_step.
        if cache is None:
            cache = [KVCache() for _ in range(len(layers))]
            shard["kv_cache"] = cache
            logger.info(
                "distributed: decode_step lazy-init KV cache shard %s "
                "(%d layers)",
                shard_id,
                len(layers),
            )

        # Build the mask from a cache INSIDE this shard's slice (mirrors
        # LlamaModel.__call__'s create_attention_mask(h, cache[fa_idx])).
        # cache[start] is the first cache this shard touches; using a cache
        # outside [start,end) would read an empty offset=0 and build a wrong
        # mask. Validated bit-exact vs generate_step in the pre-plan probe.
        mask = create_attention_mask(hidden, cache[start])
        for i in range(start, end):
            hidden = layers[i](hidden, mask, cache=cache[i])
        mx.eval(hidden)

        kv_offset = int(cache[start].offset)

        if is_last_shard:
            # Sample from the LAST position only (prefill: position P-1;
            # decode: the single position).
            out = self._project_and_sample(
                model,
                hidden[:, -1:, :],
                temperature,
                top_p,
                return_logits,
            )
            out["hidden_states"] = None
            out["kv_offset"] = kv_offset
            logger.info(
                "distributed: decode_step %s (last) token=%s kv_offset=%d",
                shard_id,
                out["token_ids"],
                kv_offset,
            )
            return out

        out = {
            "hidden_states": serialize_activation(hidden),
            "shape": list(hidden.shape),
            "dtype": str(hidden.dtype),
            "token_ids": None,
            "kv_offset": kv_offset,
        }
        logger.debug(
            "distributed: decode_step %s -> shape=%s kv_offset=%d",
            shard_id,
            hidden.shape,
            kv_offset,
        )
        return out
```

- [ ] **Step 4: Run the validation tests — must pass**

Run: `source .venv/bin/activate && pytest tests/unit/test_distributed_decode_step.py -v`
Expected: PASS (all 5: the kv_offset test from Task 2 + 4 validation tests). Note: the `input_ids`-path single-token guard fires before any model use, so the `object()` placeholder model is never touched for those.

- [ ] **Step 5: Commit**

```bash
git add fusion_mlx/distributed/shard.py tests/unit/test_distributed_decode_step.py
git commit -m "feat(distributed): decode_step cache-aware forward+sample (#630)"
```

---

## Task 4: Implement `reset_cache()` + clear KV in `sync_weights`

**Files:**
- Modify: `fusion_mlx/distributed/shard.py` (add `reset_cache`; modify `sync_weights`)
- Test: `tests/unit/test_distributed_decode_step.py` (append reset tests)

**Interfaces:**
- Produces: `ShardManager.reset_cache(shard_id) -> {"shard_id", "kv_cleared", "prev_offset"}`. `sync_weights` clears `kv_cache` as a side effect (logged, response unchanged).

- [ ] **Step 1: Write the failing reset tests**

Append to `tests/unit/test_distributed_decode_step.py`:

```python
def test_reset_cache_idempotent_on_empty():
    """reset_cache on kv_cache=None is a no-op: prev_offset=0, no error."""
    from fusion_mlx.distributed.shard import ShardManager

    mgr = ShardManager()
    sid = _dummy_shard(mgr)
    out = mgr.reset_cache(sid)
    assert out == {"shard_id": sid, "kv_cleared": True, "prev_offset": 0}
    # idempotent
    out2 = mgr.reset_cache(sid)
    assert out2 == {"shard_id": sid, "kv_cleared": True, "prev_offset": 0}


def test_reset_cache_unknown_shard_404():
    from fusion_mlx.distributed.shard import ShardError, ShardManager

    mgr = ShardManager()
    with pytest.raises(ShardError):
        mgr.reset_cache("shard-nope")


def test_sync_weights_clears_kv_cache():
    """A weight swap invalidates cached K/V. sync_weights sets kv_cache=None
    (logged) even though its response is unchanged."""
    import base64

    from fusion_mlx.distributed import shard as shard_mod

    mgr = shard_mod.ShardManager()
    sid = _dummy_shard(mgr)
    # Simulate a populated cache (don't need a real model; just set the field).
    fake_cache = [type("C", (), {"offset": 5})() for _ in range(4)]
    mgr._shards[sid]["kv_cache"] = fake_cache
    # Build a minimal valid weights payload so sync_weights succeeds.
    import mlx.core as mx
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as fh:
        path = fh.name
    mx.savez(path, **{"model.layers.0.weight": mx.array([1.0])})
    try:
        with open(path, "rb") as fh:
            payload = base64.b64encode(fh.read()).decode("ascii")
    finally:
        os.unlink(path)
    # Need a real-ish model object whose load_weights won't crash on dummy.
    class _DummyModel:
        args = type("A", (), {"tie_word_embeddings": False})()
        def load_weights(self, items, strict=False):
            return None
    mgr._models["dummy"] = _DummyModel()
    out = mgr.sync_weights(sid, payload, None)
    assert out["params_updated"] == 1
    assert mgr._shards[sid]["kv_cache"] is None, "sync_weights must clear KV"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_distributed_decode_step.py -k "reset_cache or sync_weights_clears" -v`
Expected: FAIL — `ShardManager` has no `reset_cache` (`AttributeError`); `sync_weights` does not clear `kv_cache`.

- [ ] **Step 3: Implement `reset_cache()`**

In `fusion_mlx/distributed/shard.py`, add this method on `ShardManager` after `decode_step`:

```python
    def reset_cache(self, shard_id: str) -> dict:
        # Clear a shard's KV-cache so a new generation can prefill from
        # scratch without dropping/reloading the shard (weights stay on
        # GPU). Idempotent on kv_cache=None (prev_offset=0). The cache
        # list is full-model-length; read cache[start].offset, NOT
        # cache[0] — cache[0] is an unused empty cache for a shard whose
        # range starts >0.
        shard = self._get_shard(shard_id)
        start = shard["layer_range"][0]
        cache = shard["kv_cache"]
        prev = int(cache[start].offset) if cache is not None else 0
        shard["kv_cache"] = None
        logger.info(
            "distributed: reset KV cache shard %s (was offset %d)",
            shard_id,
            prev,
        )
        return {"shard_id": shard_id, "kv_cleared": True, "prev_offset": prev}
```

- [ ] **Step 4: Clear KV in `sync_weights`**

In `sync_weights` (line ~381-424), add the KV-clear just BEFORE the final `return` (after `logger.info(... synced ...)`). Insert:

```python
        shard["kv_cache"] = None
        logger.info(
            "distributed: cleared KV cache on shard %s after weight sync",
            shard_id,
        )
```

- [ ] **Step 5: Run tests — must pass**

Run: `source .venv/bin/activate && pytest tests/unit/test_distributed_decode_step.py -v`
Expected: PASS (all). Clean up the temp .npz in the sync test (the test already unlinks `path` after reading; the `with tempfile` + `os.unlink` pattern in the test handles cleanup — keep only final outputs + logs per project rule).

- [ ] **Step 6: Commit**

```bash
git add fusion_mlx/distributed/shard.py tests/unit/test_distributed_decode_step.py
git commit -m "feat(distributed): reset_cache endpoint + sync_weights KV clear (#630)"
```

---

## Task 5: Add `/decode_step` + `/reset_cache` routes

**Files:**
- Modify: `fusion_mlx/api/distributed_routes.py` (Pydantic models + routes)
- Test: `tests/unit/test_distributed_decode_step.py` (append route tests via TestClient)

**Interfaces:**
- Produces: `POST /distributed/decode_step` (DecodeStepRequest → DecodeStepResponse), `POST /distributed/reset_cache` (ResetCacheRequest → ResetCacheResponse). Reuses `_shard_error_response` + `Depends(verify_api_key)`.

- [ ] **Step 1: Write the failing route tests**

Append to `tests/unit/test_distributed_decode_step.py`:

```python
def _client_with_manager(mgr, monkeypatch):
    """Build a TestClient whose app uses the given ShardManager singleton.

    Two wiring points:
      1. ``shard_mod._manager = mgr`` — the routes call ``get_manager()``,
         which returns the module-global singleton. Override it so the
         router resolves OUR manager (with the dummy shard registered).
      2. ``monkeypatch`` the auth dependency to a no-op. ``verify_api_key``
         does NOT exempt loopback (#350 closed that bypass) and TestClient
         uses host ``"testclient"`` (not loopback), so without the override
         every route 401s before reaching the manager. This is the
         established pattern: ``tests/unit/test_audio_path_shaped_model.py``
         monkeypatches ``verify_api_key`` to ``lambda: None`` for route
         unit tests (no API key needed). Override on the FastAPI app's
         dependency container, not the module — cleaner and scoped."""
    import fusion_mlx.distributed.shard as shard_mod
    from fusion_mlx.api import distributed_routes as dr
    from fusion_mlx.middleware.auth import verify_api_key
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    shard_mod._manager = mgr
    app = FastAPI()
    app.include_router(dr.router)
    app.dependency_overrides[verify_api_key] = lambda: None
    return TestClient(app)


def test_decode_step_route_rejects_both_modes_400(monkeypatch):
    from fusion_mlx.distributed.shard import ShardManager

    mgr = ShardManager()
    sid = _dummy_shard(mgr)
    client = _client_with_manager(mgr, monkeypatch)
    r = client.post("/distributed/decode_step", json={
        "shard_id": sid, "hidden_states": "AAAA", "input_ids": [1],
        "is_last_shard": False,
    })
    assert r.status_code == 400


def test_decode_step_route_unknown_shard_404(monkeypatch):
    from fusion_mlx.distributed.shard import ShardManager

    mgr = ShardManager()
    client = _client_with_manager(mgr, monkeypatch)
    r = client.post("/distributed/decode_step", json={
        "shard_id": "shard-nope", "input_ids": [1, 2, 3],
        "is_last_shard": False,
    })
    assert r.status_code == 404


def test_reset_cache_route_idempotent_200(monkeypatch):
    from fusion_mlx.distributed.shard import ShardManager

    mgr = ShardManager()
    sid = _dummy_shard(mgr)
    client = _client_with_manager(mgr, monkeypatch)
    r = client.post("/distributed/reset_cache", json={"shard_id": sid})
    assert r.status_code == 200
    body = r.json()
    assert body["kv_cleared"] is True
    assert body["prev_offset"] == 0
    assert body["shard_id"] == sid
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_distributed_decode_step.py -k "route" -v`
Expected: FAIL — 404 (route does not exist yet on the router).

- [ ] **Step 3: Add the Pydantic models**

In `fusion_mlx/api/distributed_routes.py`, add after the `DecodeResponse` class (line ~95):

```python
class DecodeStepRequest(BaseModel):
    shard_id: str
    hidden_states: str | None = Field(
        None,
        description="base64 .npy of incoming hidden ([P,hidden] prefill or [1,hidden] decode)",
    )
    input_ids: list[int] | None = Field(
        None,
        description="token ids for the first shard (exactly one of this / hidden_states)",
    )
    is_last_shard: bool = Field(
        ..., description="true on the shard owning the final layers (samples)"
    )
    temperature: float | None = Field(None, description="sampling temp; 0/None = greedy")
    top_p: float | None = Field(None, description="nucleus top_p (with temp>0)")
    return_logits: bool = Field(False, description="include base64 .npy logits")


class DecodeStepResponse(BaseModel):
    hidden_states: str | None = Field(
        None, description="base64 .npy outgoing activation (intermediate shard)"
    )
    shape: list[int]
    dtype: str
    token_ids: list[int] | None = Field(
        None, description="sampled token (last shard); null on intermediate"
    )
    logits: str | None = Field(None, description="base64 .npy logits (if return_logits)")
    logits_shape: list[int] | None = None
    logits_dtype: str | None = None
    kv_offset: int = Field(..., description="shard cache length after this step")


class ResetCacheRequest(BaseModel):
    shard_id: str


class ResetCacheResponse(BaseModel):
    shard_id: str
    kv_cleared: bool
    prev_offset: int
```

- [ ] **Step 4: Add the routes**

In `fusion_mlx/api/distributed_routes.py`, add after the `decode` route (line ~189):

```python
@router.post("/decode_step", response_model=DecodeStepResponse)
async def decode_step(
    req: DecodeStepRequest,
    _auth: bool = Depends(verify_api_key),
) -> DecodeStepResponse:
    try:
        out = get_manager().decode_step(
            req.shard_id,
            req.hidden_states,
            req.input_ids,
            req.is_last_shard,
            req.temperature,
            req.top_p,
            req.return_logits,
        )
    except ShardError as exc:
        _shard_error_response(exc)
    return DecodeStepResponse(**out)


@router.post("/reset_cache", response_model=ResetCacheResponse)
async def reset_cache(
    req: ResetCacheRequest,
    _auth: bool = Depends(verify_api_key),
) -> ResetCacheResponse:
    try:
        out = get_manager().reset_cache(req.shard_id)
    except ShardError as exc:
        _shard_error_response(exc)
    return ResetCacheResponse(**out)
```

- [ ] **Step 5: Run route tests — must pass**

Run: `source .venv/bin/activate && pytest tests/unit/test_distributed_decode_step.py -v`
Expected: PASS (all). The route tests use the real ShardManager singleton override; no auth header needed because `verify_api_key` on TestClient defaults to loopback-trusted (confirm: if a 401 appears, set `FUSION_MLX_API_KEY` empty or add the `Authorization` header per the existing distributed tests — but the existing `test_distributed_pipeline.py` does not hit routes, so follow the auth convention in `tests/unit/test_cors_*` if needed).

- [ ] **Step 6: Commit**

```bash
git add fusion_mlx/api/distributed_routes.py tests/unit/test_distributed_decode_step.py
git commit -m "feat(api): /decode_step + /reset_cache routes (#630)"
```

---

## Task 6: Real-model bit-exact integration tests

**Files:**
- Create: `tests/unit/test_distributed_decode_step_e2e.py`
- Test: itself (real model, `@skip_no_model`)

**Interfaces:**
- Consumes: `ShardManager.decode_step`, `reset_cache`, `list_shards` from Tasks 2-4; `_project_and_sample` from Task 1.

- [ ] **Step 1: Write the one-shard bit-exact test**

Create `tests/unit/test_distributed_decode_step_e2e.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Real-model integration tests for /distributed/decode_step (#630).

Bit-exact vs mlx_lm.generate_step is the headline correctness gate: threading
the KVCache through decode_step must reproduce un-split generation. Follows
the test_distributed_pipeline.py convention (@skip_no_model, real LM from the
model dir)."""
from __future__ import annotations

import os

import pytest

pytest.importorskip("mlx.core")

_MODEL_CANDIDATES = ["models--mlx-community--Llama-3.2-1B-Instruct-4bit"]


def _find_small_lm() -> str | None:
    base = os.path.expanduser(
        os.environ.get("FUSION_MLX_MODEL_DIR", "~/.fusion-mlx/models")
    )
    for name in _MODEL_CANDIDATES:
        snap_root = os.path.join(base, name, "snapshots")
        if not os.path.isdir(snap_root):
            continue
        for snap in os.listdir(snap_root):
            snap_dir = os.path.join(snap_root, snap)
            if any(f.endswith(".safetensors") for f in os.listdir(snap_dir)):
                return snap_dir
    return None


_LM_PATH = _find_small_lm()
skip_no_model = pytest.mark.skipif(
    _LM_PATH is None, reason="no small LM with safetensors found in model dir"
)


def _ref_tokens(model, tok, prompt, n):
    """Greedy reference tokens from mlx_lm.generate_step."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    prompt_ids = tok.encode(prompt)
    gen = generate_step(
        mx.array(prompt_ids, dtype=mx.int32), model,
        max_tokens=n, sampler=make_sampler(temp=0.0),
    )
    return [int(t) for t, _ in zip(gen, range(n))]


@skip_no_model
def test_single_shard_decode_step_matches_generate_step():
    """One shard = whole model [0, total). Prefill via decode_step
    (input_ids, is_last_shard=True) -> token #1; loop single-token decode_step
    for the rest. Bit-exact vs generate_step (greedy)."""
    import mlx_lm

    from fusion_mlx.distributed.shard import ShardManager

    mgr = ShardManager()
    model, tok = mlx_lm.load(_LM_PATH)
    total = len(model.model.layers)
    info = mgr.load_shard(_LM_PATH, 0, [0, total])

    prompt = "The capital of France is"
    prompt_ids = tok.encode(prompt)
    n = 5

    # prefill (multi-token, last shard) -> token #1
    out = mgr.decode_step(info["shard_id"], None, prompt_ids, is_last_shard=True,
                          temperature=0.0)
    tok_id = out["token_ids"][0]
    gen = [tok_id]
    assert out["kv_offset"] == len(prompt_ids), (
        f"prefill kv_offset {out['kv_offset']} != prompt len {len(prompt_ids)}"
    )
    # decode loop
    for _ in range(n - 1):
        out = mgr.decode_step(info["shard_id"], None, [tok_id],
                              is_last_shard=True, temperature=0.0)
        tok_id = out["token_ids"][0]
        gen.append(tok_id)
    ref = _ref_tokens(model, tok, prompt, n)
    assert gen == ref, f"single-shard decode_step {gen} != generate_step {ref}"
    assert out["kv_offset"] == len(prompt_ids) + n - 1
```

- [ ] **Step 2: Run the one-shard test**

Run: `source .venv/bin/activate && pytest tests/unit/test_distributed_decode_step_e2e.py::test_single_shard_decode_step_matches_generate_step -v`
Expected: PASS. If it FAILS, do NOT patch the test — root-cause the mask/cache threading via systematic-debugging (the pre-plan probe proved the approach bit-exact, so a failure here is an implementation bug in Task 3's `decode_step`, likely the mask or the last-position slice).

- [ ] **Step 3: Write the two-shard bit-exact test**

Append to `tests/unit/test_distributed_decode_step_e2e.py`:

```python
@skip_no_model
def test_two_shard_decode_step_matches_generate_step():
    """Split at the midpoint. Prefill: shard A (input_ids, not last) -> shard
    B (hidden_states, last) -> token #1. Decode loop: A -> B per token.
    Bit-exact vs generate_step (greedy). Pins the boundary activation crossing
    is correct WITH cache."""
    import mlx_lm

    from fusion_mlx.distributed.shard import ShardManager

    mgr = ShardManager()
    model, tok = mlx_lm.load(_LM_PATH)
    total = len(model.model.layers)
    split = total // 2
    a = mgr.load_shard(_LM_PATH, 0, [0, split])
    b = mgr.load_shard(_LM_PATH, 1, [split, total])

    prompt = "The capital of France is"
    prompt_ids = tok.encode(prompt)
    n = 5

    # prefill: A embeds + [0,split) -> [P,hidden]; B [split,total) + sample
    out_a = mgr.decode_step(a["shard_id"], None, prompt_ids, is_last_shard=False)
    assert out_a["shape"][1] == len(prompt_ids)
    out_b = mgr.decode_step(b["shard_id"], out_a["hidden_states"], None,
                            is_last_shard=True, temperature=0.0)
    tok_id = out_b["token_ids"][0]
    gen = [tok_id]
    assert out_a["kv_offset"] == len(prompt_ids)
    assert out_b["kv_offset"] == len(prompt_ids)
    # decode loop: single-token A -> B
    for _ in range(n - 1):
        out_a = mgr.decode_step(a["shard_id"], None, [tok_id], is_last_shard=False)
        assert out_a["shape"][1] == 1
        out_b = mgr.decode_step(b["shard_id"], out_a["hidden_states"], None,
                                is_last_shard=True, temperature=0.0)
        tok_id = out_b["token_ids"][0]
        gen.append(tok_id)
    ref = _ref_tokens(model, tok, prompt, n)
    assert gen == ref, f"two-shard decode_step {gen} != generate_step {ref}"
    assert out_a["kv_offset"] == len(prompt_ids) + n - 1
    assert out_b["kv_offset"] == len(prompt_ids) + n - 1
```

- [ ] **Step 4: Run the two-shard test**

Run: `source .venv/bin/activate && pytest tests/unit/test_distributed_decode_step_e2e.py -v`
Expected: PASS (both). Same root-cause discipline if it fails.

- [ ] **Step 5: Write reset + no-auto-reset contract tests**

Append to `tests/unit/test_distributed_decode_step_e2e.py`:

```python
@skip_no_model
def test_reset_then_reuse_different_prompt():
    """generate, reset_cache on the shard, generate a DIFFERENT prompt ->
    correct (cache did not bleed across generations)."""
    import mlx_lm

    from fusion_mlx.distributed.shard import ShardManager

    mgr = ShardManager()
    model, tok = mlx_lm.load(_LM_PATH)
    total = len(model.model.layers)
    info = mgr.load_shard(_LM_PATH, 0, [0, total])

    p1 = "The capital of France is"
    p1_ids = tok.encode(p1)
    out = mgr.decode_step(info["shard_id"], None, p1_ids, is_last_shard=True,
                          temperature=0.0)
    gen1 = [out["token_ids"][0]]
    for _ in range(3):
        out = mgr.decode_step(info["shard_id"], None, gen1[-1:],
                              is_last_shard=True, temperature=0.0)
        gen1.append(out["token_ids"][0])

    # reset -> new generation from a different prompt
    reset = mgr.reset_cache(info["shard_id"])
    assert reset["kv_cleared"] is True
    assert reset["prev_offset"] == len(p1_ids) + 3
    p2 = "The largest planet is"
    p2_ids = tok.encode(p2)
    out = mgr.decode_step(info["shard_id"], None, p2_ids, is_last_shard=True,
                          temperature=0.0)
    gen2 = [out["token_ids"][0]]
    for _ in range(3):
        out = mgr.decode_step(info["shard_id"], None, gen2[-1:],
                              is_last_shard=True, temperature=0.0)
        gen2.append(out["token_ids"][0])

    ref2 = _ref_tokens(model, tok, p2, 4)
    assert gen2 == ref2, f"after reset, gen2 {gen2} != ref {ref2}"
    # sanity: the two generations differ (different prompts -> different output)
    assert gen1 != gen2


@skip_no_model
def test_no_auto_reset_appends_wrongly_documented():
    """Pins the contract: the server does NOT auto-reset between generations.
    Two generations WITHOUT reset on the same shard append into one cache ->
    the second generation sees the first prompt's KV and produces wrong output
    (not a crash). This documents the 'reset is the caller's responsibility'
    contract; it is NOT a correctness assertion that the output is right."""
    import mlx_lm

    from fusion_mlx.distributed.shard import ShardManager

    mgr = ShardManager()
    model, tok = mlx_lm.load(_LM_PATH)
    total = len(model.model.layers)
    info = mgr.load_shard(_LM_PATH, 0, [0, total])

    p1_ids = tok.encode("The capital of France is")
    mgr.decode_step(info["shard_id"], None, p1_ids, is_last_shard=True,
                    temperature=0.0)
    # Second prefill WITHOUT reset -> appends onto p1's KV. The output is
    # semantically wrong (attention over p1+p2 positions) but must NOT crash.
    p2_ids = tok.encode("The largest planet is")
    out = mgr.decode_step(info["shard_id"], None, p2_ids, is_last_shard=True,
                          temperature=0.0)
    assert out["token_ids"] is not None  # did not crash
    assert out["kv_offset"] == len(p1_ids) + len(p2_ids)  # appended, not reset
    # And the output differs from a clean-reset generation of p2:
    mgr.reset_cache(info["shard_id"])
    out_clean = mgr.decode_step(info["shard_id"], None, p2_ids, is_last_shard=True,
                                temperature=0.0)
    assert out["token_ids"] != out_clean["token_ids"], (
        "without reset, the appended-cache output should differ from a clean run"
    )
```

- [ ] **Step 6: Run all e2e tests**

Run: `source .venv/bin/activate && pytest tests/unit/test_distributed_decode_step_e2e.py -v`
Expected: PASS (all 4). Clean up: no process data left (tests use in-process ShardManager, no server started, no temp files beyond what the framework manages).

- [ ] **Step 7: Commit**

```bash
git add tests/unit/test_distributed_decode_step_e2e.py
git commit -m "test(distributed): real-model bit-exact decode_step e2e (#630)"
```

---

## Task 7: Update docs

**Files:**
- Modify: `docs/distributed-pipeline.md`

- [ ] **Step 1: Read the current doc's Out-of-scope + endpoint list**

Run: `grep -n "Out-of-scope\|KV-cache\|/decode\|/pipeline_step\|endpoint" docs/distributed-pipeline.md | head -30`
Identify the section that lists the surface (load_shard, pipeline_step, decode, sync_weights, shards) and the Out-of-scope list naming "KV-cache / attention-mask threading" and "Multi-token autoregressive loop".

- [ ] **Step 2: Add `decode_step` + `reset_cache` to the endpoint list**

Add a section documenting the two new endpoints: request/response shapes, the prefill-vs-decode-by-length rule, the no-KV-transport model, the `is_last_shard` role split, and the KV lifecycle (lazy init on first decode_step → grows per call → reset_cache clears → sync_weights clears → drop_shard frees). Include the concurrency limit (one generation per shard; KV is per-shard process-singleton state).

- [ ] **Step 3: Mark the two Out-of-scope items as LANDED**

Update the Out-of-scope list entries for "KV-cache / attention-mask threading across shards" and "Multi-token autoregressive loop" to note they are now landed via `decode_step` (reference this plan + the spec).

- [ ] **Step 4: Commit**

```bash
git add docs/distributed-pipeline.md
git commit -m "docs(distributed): decode_step + reset_cache + KV lifecycle (#630)"
```

---

## Task 8: Full test sweep + lint + type check

**Files:**
- none (verification only)

- [ ] **Step 1: Run the full distributed test suite**

Run: `source .venv/bin/activate && pytest tests/unit/test_distributed_pipeline.py tests/unit/test_distributed_decode_step.py tests/unit/test_distributed_decode_step_e2e.py -v`
Expected: PASS (all). The existing cache-less `pipeline_step`/`decode` tests must stay green alongside the new cache-aware tests.

- [ ] **Step 2: Run lint + type check**

Run: `source .venv/bin/activate && ruff check fusion_mlx/distributed/shard.py fusion_mlx/api/distributed_routes.py tests/unit/test_distributed_decode_step.py tests/unit/test_distributed_decode_step_e2e.py && black --check fusion_mlx/distributed/shard.py fusion_mlx/api/distributed_routes.py tests/unit/test_distributed_decode_step.py tests/unit/test_distributed_decode_step_e2e.py && mypy fusion_mlx/distributed/shard.py fusion_mlx/api/distributed_routes.py`
Expected: PASS. Fix any findings (4-space indent, no docstrings in code, type hints on all new signatures).

- [ ] **Step 3: Run a broader regression slice**

Run: `source .venv/bin/activate && pytest tests/unit/ -k "distributed or cors or public_api" -x -q`
Expected: PASS. Confirms the new `ShardInfo.kv_offset` field and `kv_cache` dict key did not break any public-API or CORS re-export test (the `public_api.Server is Server` identity trap from #648 must stay clean — no `importlib.reload` introduced anywhere).

- [ ] **Step 4: Report status**

Summarize: total tests added, bit-exact gate result (one-shard + two-shard vs generate_step), lint/type status. No push until the user approves. Branch `feat/630-multitoken-kv-decode` holds all commits.
