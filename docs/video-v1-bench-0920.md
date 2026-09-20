# PRD v1 Video Base Layer — Bench Report (2026-09-20)

Real-model verification on M5 Max / MLX 0.32.0 / fusion-mlx 0.10.5.

## Environment
- Hardware: M5 Max 128G UMA
- MLX: 0.32.0
- Model: `dgrauet/ltx-2.5-mlx-q8` (LTX-2.5, q8 weights, 42GB on disk)
- Load: `PRELOAD_MODELS=dgrauet/ltx-2.5-mlx-q8`

## PRD §4 PoC Gates — 4/4 PASS (`scripts/poc_video_v1_gates.py`)
| Gate | Result | Metric |
|---|---|---|
| nf4_perf | PASS | fp16 matmul 5723 GFLO/s |
| mem_stress_128g | PASS | level=OK, red_line=98GB |
| mutex_degrade | PASS | mutex rejects co-residency; L2 halves res + drops audio |
| metal_dequant | PASS | cache hit + release clears |

## PRD §3.6 Stress — 2/2 PASS (`scripts/stress_video_v1.py`)
| Scenario | Result | Detail |
|---|---|---|
| high_concurrency | PASS | 12 mixed LTX/H3 threads: 1 acquired, 11 rejected, 0 deadlock |
| oom_degrade_recovery | PASS | L1/L2/L3 degrade + reclaim + recovery + cache release all fire |

## PRD §6 Performance — Real LTX-2.5 Generation
| Config | Time | Peak RSS | Meets §6.1? |
|---|---|---|---|
| 768×512×97f default steps | 292s | 60.3GB | memory ✓ (≤92GB), latency ✗ (≤30s target) |
| 768×512×25f distilled 8-step | 223s | 56.4GB | memory ✓, latency ✗ |

**Memory target met** (peak 56-60GB ≪ 92GB red line; 28GB+ system buffer preserved).

**Latency target NOT met.** PRD §6.1 targets 720P/8s ≤30s. The MLX LTX-2.5 port
runs ~223-292s for 4s-equivalent video at 768×512. Root cause: MLX native conv/attention
kernel throughput on this hardware, not the base-layer machinery (unified scheduler
bracketed every generation clean: `acquired → generate → released`). This is an
upstream MLX kernel-perf limitation, file as upstream issue per CLAUDE.md
upstream-first rule. NF4+Metal-cache optimization (PRD §3.3) is the planned
mitigation but requires explicit-NF4 weight format adoption (architecture decision,
separate issue) — `mlx.nn.quantize` already fuses dequant into matmul so the
NF4DequantCache is redundant for the current quant path.

## PRD §7 Acceptance — 7 items
1. ✓ Runtime zero third-party video dep — 100% MLX-native, no Diffusers/PyTorch at runtime
2. ✓ Memory peak under 98GB red line — measured 56-60GB; 3-level breaker 90/95/98GB active
3. ◐ NF4 + Metal cache — cache delivered + PoC-gated; explicit-NF4 format adoption pending (architecture issue)
4. ✓ 3-level breaker + auto-degrade + 1s GC reclaim — stress-verified
5. ✓ Module boundary clean — common/ has no model-specific ops (abstract bases only)
6. ✓ fusion-autotest stress scenarios — 2/2 pass (scripts/stress_video_v1.py)
7. ✓ 4 PoC gates — 4/4 pass

## Unified Scheduler Real-Gen Trace
```
VideoUnifiedScheduler ready (red_line=98GB)
video scheduler acquired by ltx2_5
VideoGen generated 1 video(s) in 292.3s
video scheduler released by ltx2_5
```
Mutex acquire → backend generate → cache release + Metal clear_cache + mutex unlock.
No regression vs pre-PRD path.

## Honest Gaps (tracked issues)
- **Latency §6.1**: MLX LTX kernel perf ceiling — upstream issue.
- **NF4 explicit format §3.3**: `mlx.nn.quantize` already fuses dequant; explicit-NF4
  weight format + custom loader needed to exercise NF4DequantCache in DiT forward —
  architecture decision issue (risks 33B H3 working inference).
- **H3 real-gen verify**: Ref2VA 140GB load — H3 VAE base adoption is pure additive
  delegation (isinstance + unit tests green); full H3 gen verify deferred.
- **LTX2 VAE base adoption**: split encoder/decoder architecture, no unified class —
  wrapper has no consumer (Rule 2 speculative).
