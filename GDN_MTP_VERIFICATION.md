# GDN hybrid + vendored MTP: verification result (2026-10-23)

Branch `research/gdn-spec-verify`. Question: is the conservative
"recurrent (ArraysCache) layers" gate in `engine_core` correct to block
speculative decode for the Qwen3.8-27B GDN hybrid (48 linear_attention +
16 full_attention layers), or does the vendored MTP path handle recurrent
state correctly?

## Experiment

A/B on port 11433, `mlx-community/Qwen3.8-27B-4bit`, `--text-only`,
greedy (temperature=0), same 4 prompts (code / math / story / Fibonacci),
max_tokens 300-600:

- **Run A (baseline)**: plain serve, no spec decode.
- **Run B**: `--enable-mtp --mtp-num-draft-tokens 2`.

## Results

Speed (server-side `completion_tps`, steady state after first-request
compile warmup):

| Mode | tok/s |
|---|---|
| baseline | 16.2 - 19.6 |
| MTP K=2 | 27.9 - 33.0 |

**+65-90% decode throughput.** One outlier (Fibonacci listing, 6.2 tok/s):
spec decode can thrash on highly repetitive formatting — draft proposes,
target rejects, per-cycle cost exceeds single-token decode. Known spec
characteristic, not GDN-specific.

Correctness — the decisive test: greedy spec decode is mathematically
equivalent to standard greedy decode when the verify path is correct. Any
recurrent-state corruption would diverge the outputs.

| Request | identical |
|---|---|
| merge-sorted-lists (2339 chars) | byte-for-byte |
| train average speed (1897 chars) | byte-for-byte |
| lighthouse story (1886 chars) | byte-for-byte |
| Fibonacci structure (1483 chars) | byte-for-byte |

**4/4 outputs byte-identical between MTP and baseline.** The vendored MTP
verify path handles the GDN `ArraysCache` state exactly.

## Interpretation

The `engine_core` gate comment is accurate as written: it blocks the
**draft-model** method (`SPEC_DRAFT_MODEL_ENABLED` path) because that
generic verify (`model([D1..DK], cache)`) was never audited for recurrent
layers. The **vendored MTP** path (`--enable-mtp`) is a separate mechanism
with its own verify and is NOT gated — and per this experiment it is exact
on the GDN hybrid.

Production gap: the default profile's spec-auto resolves `ddtree`/
`draft_model` (blocked by the recurrent gate) or n-gram (disabled by
per-model override on this VLM-served model), so a default VLM serve of the
27B runs with **no spec decode at all**. The +65-90% requires
`--text-only --enable-mtp` (MTP is additionally blocked on the VLM engine).

## Operational recommendation

For CC-style text workloads, serve the 27B with:

```
fusion-mlx serve mlx-community/Qwen3.8-27B-4bit --text-only --enable-mtp --mtp-num-draft-tokens 2
```

This trades VLM (vision) capability on that instance for +65-90% decode.
Vision requests would need a separate VLM instance or a serve-mode switch.

## Caveats

- Greedy equivalence tested; sampling (temperature>0) paths use the same
  verify with rejection sampling — expected to hold, not re-verified here.
- Single machine (Apple Silicon, ~72GB), single checkpoint
  (10c35caf snapshot). The recurrent gate for the generic draft-model
  method stays as-is — this result does not audit that path.
- K=2 chosen per the earlier ceiling result (K=4 accept-rate collapse,
  see AWSD phase-B notes in memory).
