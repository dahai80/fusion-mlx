# VLM Architectural-Drawing OCR Benchmark (#1414)

Endpoint, models, per-category accuracy, structured-output success,
long-image slicing stability, and slice-size safety probes.

## mlx-community--Qwen2.5-VL-7B-Instruct-8bit

| category | exact-match | median sim | valid JSON | med latency (s) | p90 (s) | errors | truncated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| printed | 0.1667 | 0.1992 | 1.0 | 1.091 | 1.183 | 0 | 0 |
| dimensions | 1.0 | 1.0 | 1.0 | 1.289 | 1.289 | 0 | 0 |
| table | 0.0 | 0.1932 | 1.0 | 2.596 | 2.596 | 0 | 0 |
| handwriting | 1.0 | 1.0 | 1.0 | 0.966 | 0.973 | 0 | 0 |

Long image A1@200DPI: 12 slices @ 2048px (overlap 15%) — regions recovered 4/7, OOM/5xx 0, truncated 0, median 4.662s, max 5.067s.

| slice px | slices | OOM/5xx | truncated | median latency (s) |
| --- | --- | --- | --- | --- |
| 2048 | 12 | 0 | 0 | 1.244 |
| 2560 | 6 | 0 | 0 | 9.158 |
| 3072 | 6 | 0 | 1 | 29.994 |

## Recommendation

- Quant tier: **none-met-threshold**
- Max safe slice size: **2560 px**
- Rationale: mean exact-match >= 0.8 and valid-JSON >= 0.9 on category suite; slice probes up to 2560px had zero OOM/5xx and zero truncation. If no model met the accuracy threshold, drawing-parser M2 should use the cloud fallback path.
