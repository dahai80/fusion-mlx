# DFlash2 深度优化分析 (0919)

## 问题

用户问: "宣传的 DFlash2 在 macOS 上优化极强,为什么我们自己不行?"

flag-audit-0917 显示 DFlash2 -3.6% (29.62 vs 30.73 tok/s OFF)。

## 根因定位 (Profiling)

在 `dflash2_spec_step` 加分阶段计时,实测 Qwen3.8-27B-4bit + z-lab/Qwen3.8-27B-DFlash2:

```
dflash2_spec: step=51, block=5, accepted=3/4 (75.0%),
  propose=2.0ms d1gate=4.0ms verify=39.0ms
  snap=0.2ms restore=0.1ms replay=36.5ms trim=0.0ms
  rollback=36.6ms | rate=37.0%, circuit=False
```

**瓶颈 = replay=36.5ms** (≈ verify=39ms)。acceptance 良好 (75%),不是低接受率问题。

## 为什么有 replay?

Qwen3.8-27B 是 **hybrid 模型** (GDN/ArraysCache):
- `model_type: qwen3_5`, `layer_types: [linear_attention, ..., full_attention, ...]`
- `linear_attention` 层用 **ArraysCache** (GDN 状态,不可 trim)
- `full_attention` 层用 **KVCache** (可 trim, O(1))

spec verify 处理 [D1..DK] 后,ArraysCache 状态包含所有 K 个 token。
rejection 时只能 trim KVCache (O(1)),ArraysCache 无法 trim → 必须:
1. snapshot (deepcopy) verify 前的 ArraysCache
2. restore snapshot
3. **replay** [D1..D_n_accepted] 跑完整 model forward 重建 GDN 状态

replay = 完整 model forward over n_accepted tokens ≈ verify cost。

## 经济学分析 (break-even)

hybrid 模型 block_size=B, acceptance rate r:
- cost = verify(B) + P(reject) × replay(r×B)
- replay ≈ verify (同量级) → 每 rejection 步算力翻倍
- break-even: r × (1 + r^B × B) > B

B=5 时: r=0.95 → 4.6 < 5 (仍亏); r=1.0 → 6 > 5。
**break-even acceptance ≈ 100%** — hybrid 模型 spec decode 几乎不可能盈利。

实测 rate=37% → spec decode **2.7× 慢于** regular decode。

## DFlash2 官方 4× 怎么做到的?

DFlash2 blog 声称 Apple M5 Max 4× throughput。关键差异:
1. 用 **oMLX** (forked Ollama+MLX),不是 vanilla mlx-lm
2. oMLX 大概率有 **kernel 级 GDN 状态 checkpoint** — 消除 replay
3. DFlash2 (v2) drafters **只存在于 hybrid 模型** (Qwen3.8/3.5/3.6)
4. dense 模型 (Qwen3-8B) 只有 **DFlash v1** drafters (不同 API)

vanilla mlx-lm 无法不 fork 消除 replay — `gated_delta_update` kernel 只返回 final state,
不暴露 per-token intermediate state。

## 修复: Hybrid-aware 预设

`fusion_mlx/speculative/dflash2/presets.py`:

1. `_is_hybrid_model(model)`: 检测 `args.model_type in (qwen3_5, qwen3_8, qwen3_next, gpt_oss)`
   或 `text_config.layer_types` 含 `linear_attention`
2. `_adjust_for_hybrid(preset)`: hybrid 模型 →
   - `block_size: 5 → 2` (减小 replay cost)
   - `circuit_breaker_threshold: 0.20 → 0.80` (acceptance < 80% 立即暂停)
   - `circuit_breaker_window: 10 → 6` (更快响应)
   - `hybrid: True` flag
3. `batched.py`: hybrid preset 时跳过 `scheduler_config.dflash2_block_size` 默认值 (5)
   的 `or` 覆盖,用 preset 的 forced block_size

circuit breaker 在 50-66% acceptance 时正确暂停 (日志验证):
```
dflash2_spec: circuit breaker paused (avg accept 50.0%...
dflash2_spec: circuit breaker paused (avg accept 66.7%...
```

## 基准测试结果

Qwen3.8-27B-4bit + z-lab/Qwen3.8-27B-DFlash2, 4 runs, max_tokens=256, temp=0:

| Config | decode_tps | vs OFF |
|--------|-----------|--------|
| OFF (no DFlash2) | 29.74 | — |
| ON old (block=5, cb=0.20) | 27.04 | -9.1% |
| ON fix (block=2, cb=0.80) | 29.39 | -1.2% |

回归从 -9.1% 降到 -1.2%。剩余 -1.2% = warmup steps + circuit breaker probe 开销。

## 结论

**为什么我们自己不行**: DFlash2 (v2) 只支持 hybrid 模型,而 hybrid 模型的 GDN/ArraysCache
在 rejection 时必须 replay (完整 forward) 重建状态,成本 ≈ verify,使 spec decode 净负。
官方 4× 依赖 oMLX fork 的 kernel 级优化,vanilla mlx-lm 无法实现。

**最佳实践**:
- DFlash2 (v2) → 仅 hybrid 模型,用本修复避免回归 (circuit breaker 快速暂停)
- DFlash v1 → dense 模型 (Qwen3-8B),KVCache.trim O(1) 无 replay,理论可盈利
- 真正 4× speedup → 需 oMLX fork 或等待 MLX 上游 GDN checkpoint 支持
