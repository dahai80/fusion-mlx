# VLM Architectural-Drawing OCR Benchmark (#1414)

Endpoint, models, per-category accuracy, structured-output success,
long-image slicing stability, and slice-size safety probes.

## mlx-community--Qwen2.5-VL-7B-Instruct-4bit

| category | exact-match | median sim | valid JSON | med latency (s) | p90 (s) | errors | truncated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| printed | 0.1667 | 0.1992 | 1.0 | 0.977 | 1.218 | 0 | 0 |
| dimensions | 1.0 | 1.0 | 1.0 | 1.417 | 1.417 | 0 | 0 |
| table | 0.0 | 0.1932 | 1.0 | 1.993 | 1.993 | 0 | 0 |
| handwriting | 1.0 | 1.0 | 1.0 | 0.893 | 0.911 | 0 | 0 |

Long image A1@200DPI: 12 slices @ 2048px (overlap 15%) — regions recovered 4/7, OOM/5xx 0, truncated 1, median 4.701s, max 19.114s.

| slice px | slices | OOM/5xx | truncated | median latency (s) |
| --- | --- | --- | --- | --- |
| 2048 | 12 | 0 | 0 | 1.171 |
| 2560 | 6 | 0 | 1 | 11.299 |
| 3072 | 6 | 0 | 0 | 26.225 |

## Recommendation

- Quant tier: **none-met-threshold**
- Max safe slice size: **3072 px**
- Rationale: mean exact-match >= 0.8 and valid-JSON >= 0.9 on category suite; slice probes up to 3072px had zero OOM/5xx and zero truncation. If no model met the accuracy threshold, drawing-parser M2 should use the cloud fallback path.
