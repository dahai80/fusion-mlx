# Speculative-Decoding Routing Benchmark — Production Routing-Decision Dataset

doc_id: fusion-mlx/docs/benchmarks/spec-route-benchmark
status: published
updated: 2026-11-12
dataset: [data/spec_route_decisions_anonymized.jsonl](data/spec_route_decisions_anonymized.jsonl)

## Summary

fusion-mlx's speculative-decoding auto-router logs every routing decision to
`spec_route_decisions.jsonl` in the runtime directory. This document publishes
an anonymized snapshot of that log — **45,753 routing decisions spanning 61
days of real production traffic** (September 12 – November 12, 2026) —
together with the analysis we derive from it. To our knowledge no peer
serving engine publishes comparable routing-accuracy data: llama.cpp
documents speculative flags but publishes no acceptance measurements, and
oMLX has no speculative decoding at all.

Headline result: on production traffic the dynamic decision-tree drafter
(`ddtree`) sustains a **mean 44.5% token acceptance rate** (p50 55%), the
n-gram suffix drafter (`suffix`) sustains **25.3%** (stable across all 8
weeks), and native MTP measures **5%** — below the router's 20% abandon
threshold, i.e. the router correctly stops routing workloads to MTP on
hybrid-attention models.

## What the router logs

One JSON line per routing decision, written by
`fusion_mlx/speculative/auto_router.py` (`_append_spec_route_decision`):

| Field | Meaning |
|---|---|
| `t_rel_s` | Seconds since the first decision in the snapshot (absolute timestamps removed) |
| `method` | Speculative strategy selected for the request (`suffix`, `ddtree`, `mtp`, `dspark`, `dflash2`, `medusa`, `eagle_fused`, `eagle3`) |
| `reason` | Router rule that produced the decision (e.g. `table_route hit`, `ngram fallback`, `mtp available`, `keep_accept`) |
| `candidates` | Strategies eligible for this request |
| `model_family` | Coarse model family (e.g. `qwen3_5`) when known |
| `prompt_tokens` | Prompt length in tokens |
| `accept_rate` | Rolling acceptance rate observed for the current method at decision time; `null` when no recent verified sample exists |
| `kv_pressure` | KV-cache pressure signal; not populated in this snapshot |

## Methodology

- **Snapshot**: production server log, exported verbatim after an
  anonymization pass. No prompt text, user identifiers, file paths, or host
  data exist in the source schema — the log records only the fields above —
  and absolute timestamps were shifted to a session-relative clock.
- **Acceptance rate semantics**: `accept_rate` is the rolling observed
  acceptance of a strategy over recent verified decode rounds, recorded at
  the moment the router makes a decision. It is present on 5.2% of records
  (2,397 of 45,753) because most decisions are made before any verified
  sample exists for that strategy.
- **Aggregation**: arithmetic mean / median / p90 per strategy over records
  with non-null `accept_rate`.

## Findings

### Method acceptance rates

| Method | n (with accept signal) | mean | p50 | p90 | typical prompt p50 |
|---|---|---|---|---|---|
| `suffix` (n-gram) | 1,483 | 0.253 | 0.100 | 0.600 | 15,605 |
| `ddtree` (decision tree) | 732 | 0.445 | 0.550 | 0.600 | 128 |
| `mtp` (native multi-token) | 182 | 0.050 | 0.050 | 0.050 | 100 |

Reading:

- **`ddtree` is the strongest drafter** on this traffic: median 55% of draft
  tokens accepted, mean 44.5%. Its decisions skew to short prompts (p50 128
  tokens), where the router's table hits route conversational traffic to the
  tree drafter.
- **`suffix` carries the long-context load** (p50 15.6k tokens): a 25%
  acceptance on 15k-token agent prompts still skips roughly one verify round
  in four, which is meaningful at that scale.
- **`mtp` at 5% is a negative result worth publishing**: on
  hybrid-attention (GDN) model families the native MTP heads accept almost
  nothing. The router's thresholds (abandon < 20%, keep >= 40%) make this
  self-correcting — `mtp` gets abandoned automatically, and its 182 recorded
  decision-time samples are consistent with the floor value before
  abandonment.

### Stability over time

`suffix` acceptance by week is essentially flat — 0.252, 0.256, 0.252,
—, 0.253, 0.252, 0.254, 0.238 (weeks 0-8, n=43-551/week). The n-gram
drafter's acceptance is a property of the workload mix, not of drift.

### Routing quality vs static configuration

The router's value over a static flag choice is visible in the decision
reasons: 26,372 `table_route hit` (workload-keyed table decisions), 12,887
`ngram fallback`, 2,663 `mtp available`, 2,436 `long_doc threshold`, 833
`sorted fallback`, and 558 `keep_accept` (method retained because its
rolling acceptance >= 40%). A static configuration would have pinned every
request to one strategy; the log shows strategies being selected per
workload shape (short conversational → `ddtree`, long agent prompts →
`suffix`) and abandoned automatically when acceptance decays (MTP).

## Limitations

- `accept_rate` is a decision-time rolling signal, not a per-request outcome
  measure; it is present on 5.2% of records.
- Latency saved per decision is not captured in this schema version — the
  log has no latency field. Adding per-decision latency deltas is planned
  for the next schema revision.
- `kv_pressure` was never populated in this snapshot; the pressure-aware
  routing rule therefore could not be evaluated.
- `model_family` is null on 42% of records (older schema versions).
- Single-host, single-tenant production workload; acceptance rates on other
  traffic mixes will differ.

## Reproducing

```bash
# rolling stats per method from the published dataset
python3 - <<'PY'
import json, statistics, collections
by = collections.defaultdict(list)
for line in open("docs/benchmarks/data/spec_route_decisions_anonymized.jsonl"):
    r = json.loads(line)
    if r["accept_rate"] is not None:
        by[r["method"]].append(r["accept_rate"])
for m, v in by.items():
    print(m, len(v), round(statistics.mean(v), 3), round(statistics.median(v), 3))
PY
```

The snapshot will be refreshed monthly with a fresh anonymized export to
build a longitudinal series.

