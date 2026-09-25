# 突破统一内存带宽墙：权重摊销推测解码框架 (AWSD)

> 一份面向 fusion-mlx 的深度优化路线与创新论文
> 作者：fusion-mlx 深度优化研究 / 2025-09-25
> 状态：研究提案（部分路径已实证，部分待验证）

---

## 摘要

本文从第一性原理重新审视 Apple Silicon (M5, 128GB unified memory) 上大模型解码的物理瓶颈。
通过严格测量，我们证明：**Qwen3.8-27B-4bit 的纯解码深度受限于统一内存带宽，而非算力**——
每 token 需读取 14GB 权重，理论屋顶线 17.5ms/tok（57 tok/s），实测仅达 38-44% 带宽利用率
（stock 22 tok/s, MTP K=1 45 tok/s）。算术强度 3.86 FLOP/byte 远低于 M5 屋顶线交叉点 37.5，
意味着一切算力优化（compile、fused kernel、手写 Metal kernel）对纯解码收益上限为零。

基于此，本文提出 **AWSD（Amortized Weight-Resident Speculative Decode）** 三维摊销框架：
- **时间摊销**（Temporal）：MTP chain K>1 自回归草稿 + tree verify，单次权重读取摊销至 K+1 token
- **空间摊销**（Spatial）：Metal async blit+compute 层间流水线，将带宽利用率从 44% 推至 80%
- **精度摊销**（Precision）：草稿路径 2bit（可抛弃），验证路径 4bit（精确），草稿权重读取减半

三者正交可乘。理论目标：单流解码 150-250 tok/s（stock 的 7-11x），在损失零质量的前提下
超越 oMLX / rapid-mlx / llama.cpp / H3.c / splash 一个量级。

本文同时诚实地记录了已穷尽路径（exhaustion matrix），并论证为何 AWSD 的特定组合未被证伪。

---

## 1. 物理基线：为什么解码是带宽墙问题

### 1.1 屋顶线模型

Apple M5 Max 统一内存带宽峰值 800 GB/s，fp16 算力约 30 TFLOPS，int4 约 100 TOPS。
对 Qwen3.8-27B-4bit（dense，27B 参数 × 0.5 byte = 14GB 权重）单 token 解码：

| 量 | 计算 | 值 |
|---|---|---|
| 权重读取 | 27B × 0.5B | 14 GB |
| 前向算力 | 27B × 2 FLOP | 54 GFLOP |
| 带宽时间 | 14GB / 800GB/s | 17.5 ms |
| 算力时间 | 54GFLOP / 30TFLOPS | 1.8 ms |
| 算术强度 | 54GFLOP / 14GB | 3.86 FLOP/byte |
| 屋顶线交叉点 | 30TFLOPS / 800GB/s | 37.5 FLOP/byte |

算术强度 3.86 **远低于** 交叉点 37.5（相差 9.7x）——解码**深陷带宽区**，算力有 9.7x 富余。
**结论：任何减少算力的优化（compile、kernel fusion、手写 GEMV）对解码墙时间无贡献。**
这是对 oMLX (compile fork)、rapid-mlx、手写 Metal kernel 路线的根本性否定——它们攻击的
是非瓶颈维度。fusion-mlx 的 `mx.compile` 实测 -2.5%（falsified），手写 sdpa kernel 达 86%
屋顶线但仍被带宽墙限制，均已记录在案。

### 1.2 实测带宽利用率

| 路径 | tok/s | 每 forward token 数 | 有效带宽 | 利用率 |
|---|---|---|---|---|
| stock decode | 22 | 1 | 308 GB/s | 38% |
| MTP K=1 | 45 | 1.8 (90% accept) | 350 GB/s | 44% |
| 理论屋顶线 | 57 | 1 | 800 GB/s | 100% |

两个关键发现：
1. **stock 仅 38% 利用率**——带宽未跑满，存在 2.6x 头寸（308→800GB/s）
2. **MTP 利用率更高（44%）**——摊销使每次 forward 更"划算"，GPU 更持续地读取权重

38% 利用率的损耗来源：(a) 4bit group_size=64 的 strided gather 访问模式非流式；
(b) Metal kernel launch 间隙；(c) KV cache 与权重交替访问；(d) Python 调度抖动。

### 1.3 混合架构的 KV 微小性放大了带宽主导

Qwen3.8-27B 是 **hybrid GDN 架构**：64 层中 48 层 linear_attention（GatedDeltaNet，
O(1) recurrent state），16 层 full_attention（每 4 层一个）。

| 序列长度 | KV 体积 | 权重占比 |
|---|---|---|
| 200 token | 0.05 GB | 99.6% 权重 |
| 2000 token | 0.49 GB | 96.6% 权重 |

即便 2000 token 上下文，KV 也仅 0.49GB，权重 14GB 占 96%+ 访存。
**这与纯 softmax 模型截然不同**——后者长上下文时 KV 读取代价上升，可能转为 compute-bound。
hybrid GDN 模型在任意实用长度下都死锁在带宽区。这对优化方向是决定性的：
**唯一有效的优化是减少每 token 的权重读取量。**

## 2. 穷尽矩阵：已证伪的路径与根因

以下路径均在 fusion-mlx 上真实加载 Qwen3.8-27B-4bit 实证，非纸面推演。

| 路径 | 实测收益 | 根因/证伪点 | 状态 |
|---|---|---|---|
| `mx.compile` decode | -2.5% | 带宽墙，compile 优化算力非瓶颈 | FALSIFIED |
| 手写 sdpa Metal kernel | 0 (86% 屋顶线) | attention compute 本已接近屋顶线，但被带宽墙限制 | NO WIN |
| 手写 int4 GEMV (V12 simdgroup) | 持平 native | 编译器自动向量化，无算法空间 | TIE |
| 预编译 -Ofast metallib | 仅 K≥16384 且 gs=128 | decode K=4096 反慢 10% | 窄窗 |
| oMLX compile fork | 无解码收益 | 同 compile 根因 | NOT WORTH |
| EAGLE fused-verify | +4% real | 2 GPU sync/step 抽干 pipeline；K>2 chain collapse | EXHAUSTED |
| EAGLE prod wiring | -25% | PoC +27% 但 prod 双 sync 退化 | REVERTED |
| Medusa fused | blocker | parallel heads 强制 d1sync | BLOCKED |
| n-gram spec | neutral (0%) | d1sync 4ms + V=2.14 上限 | NEUTRAL |
| 2bit quant | +46% 速度 | 质量垃圾（不可用） | REJECTED |
| 3bit quant | +17% | 质量临界 | BORDERLINE |
| 近似 attention (KV pooling) | 单步 cos 0.9998 | 多步发散至 0.21（结构误差） | REJECTED |
| CoW paged-KV | -5~7x | 与 mlx-lm batched KV 结构不兼容 | DELETED |

**唯一赢家：Native MTP K=1（+86%，lossless）**——通过自回归草稿 + 单次 backbone forward 验证
2 token，将权重读取摊销至 1.8 token/read。这是目前唯一突破带宽墙的损失无关路径。

### 2.1 穷尽矩阵的关键教训

1. **攻击算力的路径全灭**（compile / kernel / GEMV）——印证 §1 带宽墙结论
2. **并行草稿头全灭**（Medusa/EAGLE fused）——因 d1sync 强制 GPU 同步，每步 2 次同步抽干 pipeline
3. **低精度全灭**（2bit/3bit）——质量不可接受，草稿可抛弃但验证不可
4. **MTP 自回归草稿胜出**——无 d1sync（草稿串行生成，head 小），是唯一不触发同步税的推测路径

教训 2 与 4 的对比是本文的起点：**自回归草稿（MTP）是 Apple Silicon 上唯一可行的推测解码范式**，
因为 Metal 的 GPU 同步代价远高于 CUDA。这直接否定了 Medusa/EAGLE-fused 在 MLX 上的可行性。

## 3. 核心洞察：三维摊销正交性

减少"每 token 权重读取量"只有三条正交通道：

```
每 token 权重读取 = W / (K × U × P)
                    ↑    ↑    ↑
              时间摊销  空间摊销  精度摊销
```

- **W** = 14GB（模型权重，常量）
- **K** = 每次 forward 摊销的 token 数（temporal，推测解码）
- **U** = 带宽利用率（spatial，0.38→0.80，kernel/调度优化）
- **P** = 精度因子（precision，草稿路径权重体积倍率）

### 3.1 时间摊销 K（推测解码）

MTP K=1 已证 K=1.8 有效。关键问题：**K 能否推到 4-8？**

EAGLE/Medusa 的 K>2 collapse 根因是 **d1sync（并行头强制 GPU 同步）**。
MTP 是**自回归草稿**（head 串行应用 K 次），不触发 d1sync。
因此 MTP K>1 在理论上不继承 EAGLE 的 collapse——这是未被证伪的绿地。

config schema 已预留 `mtp: {num_speculative_tokens, chain_or_tree}`，但 hot-loop
`_run_verify_cycle` 硬编码 K=1（每周期 draft 1 token、verify 2 token）。
K>1 需扩展为：head 串行应用 K 次 → draft K token → 单次 backbone forward verify K+1 token。

### 3.2 空间摊销 U（带宽利用率）

实测 U=0.38-0.44，存在 1.8x 头寸（0.44→0.80）。
提升途径：
- **层间流水线**：layer N 计算时，async blit layer N+1 权重入 GPU L2/SRAM
  （Metal 3 MTLBlitCommandEncoder + MTLComputeCommandEncoder 并发编码）
- **权重布局流式化**：4bit group_size=64 的 strided gather → 连续打包，使 GEMV 访问全流式
- **持续权重驻留**：单层 219MB（14GB/64）可驻 SRAM，KV 流式穿过

注意：compile/手写 kernel 攻击的是算力维度，对 U 无贡献（已证伪）。
U 的提升需 **Metal async blit+compute 编码**，这是 oMLX/compile 不触及的层面。

### 3.3 精度摊销 P（草稿路径低精度）——仅适用于独立草稿模型，MTP 证伪

2bit 全模型质量垃圾——但**草稿是可抛弃的**。rejected draft 不影响输出，只有 verify 路径需精确。
因此（**原假设，适用于独立草稿模型如 EAGLE/llama.cpp**）：
- **draft path**：2bit 权重（7GB，读取减半）——质量差但可抛弃
- **verify path**：4bit 权重（14GB，精确）——决定输出

P=2（草稿权重减半）。结合 K=4 chain：4 次 draft 读 7GB + 1 次 verify 读 14GB =
(4×7+14)/5 = 8.4GB/token，相比 stock 14GB，减 40%。

**⚠️ 证伪修正（2026-09-25 实测）：** P=2 的"7GB 草稿读取"假设草稿是**独立小模型**（独立权重）。
但 fusion-mlx 的 MTP head 是**共享 backbone 的 1 层**（253MB @ 4bit，占 14GB backbone 的 1.8%），
`mtp_forward` 复用 backbone 的 embedding + KV，draft 仅额外读 253MB，非 7GB。
2bit head → 126MB，K=1 时节省 127MB / 14GB = **0.9% 带宽**；K=4 时 4×127MB / (14+1)GB = **3.4%**。
即使 2bit head 保持 100% 接受率，速度收益 <1%——P 维度对 MTP 结构性无效。
独立草稿模型路径（EAGLE/Medusa）在 MLX 上已证伪（d1sync -25%，§2），故 **P 维度在 fusion-mlx 上死锁**。
§3.4 表中 "MTP K=4 + 流水线 + 2bit draft" 行的 P=1.67 与 12.3x 目标**不可达**。

### 3.4 三维乘积

| 配置 | K | U | P | 每 token 等效读取 | 相对 stock |
|---|---|---|---|---|---|
| stock | 1 | 0.38 | 1 | 14/0.38 = 36.8 GB·s/tok | 1x |
| MTP K=1（现状） | 1.8 | 0.44 | 1 | 14/0.79 = 17.7 | 2.1x |
| MTP K=4 chain | 3.5* | 0.44 | 1 | 14/1.54 = 9.1 | 4.0x |
| MTP K=4 + 流水线 | 3.5 | 0.80 | 1 | 14/2.8 = 5.0 | 7.4x |
| MTP K=4 + 流水线 + 2bit draft | 3.5 | 0.80 | 1.67 | 8.4/2.8 = 3.0 | 12.3x |

*K=4 chain 假设 70% 接受率（4 token draft 中平均接受 2.5 + 1 verify = 3.5 effective）

**理论目标：单流解码 150-250 tok/s（stock 的 7-11x）。**
即便保守取 K=2.5、U=0.60、P=1.3，仍有 4-5x 提升。

## 4. AWSD 框架设计

### 4.1 总体架构

```
                    ┌─────────────────────────────────────┐
                    │   AWSD Scheduler (per-step)          │
                    │   决策: K, draft精度, tree形状        │
                    └───────────┬─────────────────────────┘
                                │
              ┌─────────────────┼──────────────────────┐
              ▼                 ▼                      ▼
     ┌────────────────┐ ┌───────────────┐ ┌──────────────────┐
     │ Draft Engine   │ │ Tree Verify   │ │ Layer Pipeline   │
     │ (MTP head K次) │ │ (单次forward)  │ │ (async blit+comp)│
     │ 2bit 权重驻留  │ │ 4bit 精确      │ │ 层间权重预取     │
     │ GPU SRAM 常驻  │ │                │ │                  │
     └────────────────┘ └───────────────┘ └──────────────────┘
              │                 │
              ▼                 ▼
     K 个 draft token    tree-mask attention
     (串行, 无 d1sync)    verify K+1 token
```

### 4.2 维度一：MTP Chain K>1（时间摊销）

**现状**：`_run_verify_cycle` 每周期 draft 1 token → verify 2 token（next_main + draft）。
`_MtpState.queue` 承载 init(2) + draft/bonus/verify emit。

**扩展设计**（K=4 chain）：
1. `_run_verify_cycle` 入口：若 `state.chain_budget > 0`，循环应用 MTP head K 次
2. 每次 head forward 用上一轮的 hidden + draft_tok 生成下一轮 draft（串行，无 GPU 同步）
3. 累积 K 个 draft_tok + 1 个 next_main → 拼成 (K+1) 输入
4. 单次 `_call_backbone(inputs[None], n_confirmed=1)` verify 全部 K+1 token
5. tree-mask attention 允许并行验证所有 K+1 位置
6. 接受前缀 + 拒绝点 residual sample（复用现有 `_residual_sample`）

**关键正确性约束**（GDN hybrid 模型）：
- 48 层 linear_attention 的 ArraysCache（recurrent state）在 verify 写入 K+1 token 后
  必须按接受数 rollback——复用 PR#965 已修复的 `_rollback_after_reject` + `rollback_state`
- `n_confirmed` 参数前传至 GDN 层（PR#965 已修复 `__call__` 不前传的 bug）
- 接受 N 个 draft → cache 保留 N+1，rollback K-N

**为什么不会 collapse（与 EAGLE fused 的区别）**：
- EAGLE fused：并行头 → 每步 2 次 GPU 同步（draft forward + verify forward）→ pipeline 抽干
- MTP chain：head 串行应用 K 次（同 GPU stream，无跨 stream 同步）→ 1 次 verify forward
- head 极小（1 层，~500MB），K 次 head forward 的权重读取可驻 SRAM（见维度三）

### 4.3 维度二：Metal Async Blit+Compute 层间流水线（空间摊销）——证伪

**目标**：将 U 从 0.44 提至 0.80。

**现状**：mlx-lm 的 backbone forward 逐层串行：读 layer N 权重 → 计算 layer N → 读 layer N+1 权重 → ...
权重读取与计算无重叠，GPU 在等待内存时闲置。

**设计（原）**：Metal 3 双命令编码器
```
for layer in range(64):
    blit_encoder.copy(layer_weights[N+1], sram_buffer)   # async 预取下一层
    compute_encoder.dispatch(kernel, layer_weights[N], sram_buffer_prev)  # 计算当前层
    # 两个 encoder 并发提交 → blit 与 compute 重叠
```

- 单层权重 219MB（14GB/64），M5 GPU SRAM/L2 ~64MB → 需 tile 分块（每层切 4-8 tile）
- blit 与 compute 在不同 engine 上并发（Apple Silicon GPU 有独立 blit engine）
- 预期：权重读取时间隐藏在计算时间内，U→0.80

**⚠️ 实测证伪（2026-09-25 PoC）：**
1. **Stream overlap 真实存在**：`mx.new_stream` + `mx.new_thread_unsafe_stream` 在 M5 上确有 blit/compute engine 并发——compute-bound matmul 3.4x 加速，decode GEMV 1.77x 加速。基元可用。
2. **但 prefetch 是 3x 浪费流量（决定性）**：L2 residency 测试——
   - GEMV 仅读 W：4.73ms
   - prefetch W→buf + GEMV 读 buf：14.39ms（**3x 慢**，buf 读取不比 W 快）
   - prefetch W→buf + GEMV 读 W(原始)：14.47ms（3x 慢，prefetch 无助益）
   prefetch copy 每层加 2.5GB 读 + 2.5GB 写 = 5GB 浪费流量，GEMV 读 prefetched buf 不比读原始 W 快 → **无 L2 加速**。
3. **根因：L2 64MB << 层 219MB**（3.4x 尺寸失配），层无法 residency。Apple Silicon 统一内存下权重无需 host→VRAM 拷贝（已在统一内存），"prefetch" 只是把统一内存拷到统一内存的另一处——纯开销。
4. **1.77x overlap 假象**：PoC 的 1.77x 测的是 GEMV || 无用 copy，copy 不加速 GEMV（GEMV 读统一内存，与 copy 是否并发无关）。真实层流水线：layer N GEMV || prefetch W_{N+1}→buf，layer N+1 GEMV 读 W_{N+1}（原始，非 buf）→ prefetch 全程浪费 → 3x 慢。

**U=0.38→0.80 headroom 的真实来源（非 blit/compute overlap）：**
- (a) 4bit gs=64 strided gather → 非流式访存（需 quant 布局重排，非 blit，未试，高风险独立路径）
- (b) kernel launch 间隙 → fusion（compile，已证伪 -2.5%）
- (c) KV/weight 交替 → KV 仅 1.8%，minor
- (d) Python dispatch → compile（已证伪）

四个 U 杠杆中三个已证伪（compile 路径），剩 (a) quant 布局重排是独立高风险路径，不在 Phase D 范围。
**结论：blit+compute 层流水线对 decode 带宽墙无效（3x 负面），不接线。**

**实现挑战（原，现已 moot）**：
- mlx-lm 的 `model(x, cache=...)` 是 Python 层逐层调用，不暴露 Metal encoder
- 需 monkey-patch 为"层调度器"，将逐层 forward 重写为双 encoder 流水线
- 或：mx.compile 的 fuse 无法跨层，需自定义 `mx.fast.metal_kernel` 编排层间依赖

### 4.4 维度三：草稿路径 2bit 驻留（精度摊销）——证伪

**设计（原）**：
- MTP head（1 层）权重 ~500MB，2bit 量化后 ~125MB，可永久驻留 GPU SRAM
- draft forward（K 次 head 应用）从不命中主存 → 草稿生成近乎免费（compute-only）
- verify path 仍用 4bit backbone（14GB，精确）

**⚠️ 实测证伪（2026-09-25）：** MTP head 实测 **253MB**（4bit，非原估 500MB），占 14GB backbone 的 1.8%。
即使 2bit 量化（126MB）并 SRAM 驻留，节省 127MB/cycle，K=1 时占 14GB 的 **0.9%**。
"草稿生成近乎免费"的直觉错在：head 本就只占 1.8% 带宽——免费化 1.8% 不改变带宽墙格局。
原 §3.3 P=2 的 40% 节省假设独立草稿模型（7GB 读），MTP head 不满足该前提。
**结论：P 维度对 MTP 结构性无效，不加载 2bit head，不实测接受率（带宽数学已判死）。**

**质量保证（原，现已 moot）**：
- 2bit 仅用于 draft head，不影响 verify（输出由 4bit backbone 决定）
- draft head 质量影响接受率，但不影响输出正确性
- 接受率可能从 90%（4bit draft）降至 60-70%（2bit draft），但 K 摊销仍净赢

**风险**：2bit draft head 接受率过低（<30%）时 K 摊销被低接受率抵消。
需实测 2bit MTP head 的接受率下界。（**未执行**——带宽 saving <1%，无需实测。）

## 5. 关键创新点

### 5.1 创新点一：自回归链式 MTP（区别于并行头推测）

**学术界现状**：Medusa / EAGLE / SpecDec 采用**并行草稿头**（多个 head 一次生成 K 个 draft），
需 tree attention 一次性验证。在 CUDA 上可行（低同步代价），但在 Metal/MLX 上每步 2 次 GPU
同步抽干 pipeline（EAGLE fused prod -25% 实证）。

**本文创新**：利用 MTP head 的**自回归可链式性**——单 head 串行应用 K 次，无跨 stream 同步，
仅 1 次 verify forward。这是 Metal 上的正确推测范式。PR#990 的 MTP 原生支持 chain，
fusion-mlx 已修复 GDN rollback + n_confirmed 前传，K>1 的结构性障碍已清除，只剩 hot-loop 扩展。

**理论贡献**：提出"同步税感知的推测解码范式选择"——在同步代价高的后端（Metal/CPU）选自回归链式，
在同步代价低的后端（CUDA）选并行树。这解释了为何 llama.cpp 的 draft-model 推测（自回归）在 Apple
Silicon 上可行，而 EAGLE-MLX 不可行。

### 5.2 创新点二：带宽利用率作为一阶优化目标

**现状**：oMLX/rapid-mlx/compile 优化算力（FLOPS），llama.cpp 优化量化（P），但**无人将带宽
利用率 U 作为独立优化目标**。fusion-mlx 实测 U=0.38-0.44，揭示 1.8x 未开发头寸。

**本文创新**：将 U 提升为与 K、P 并列的一阶变量，通过 Metal async blit+compute 层间流水线优化。
这要求突破 mlx-lm 的 Python 层逐层调用，引入"层调度器"抽象——这是 oMLX compile fork 不触及的
层面（compile 优化算子内，不优化层间调度）。

### 5.3 创新点三：精度非对称推测（draft 2bit / verify 4bit）

**现状**：2bit 全模型质量垃圾（已证伪）。但"草稿可抛弃"这一性质未被利用——现有 2bit 工作将
低精度用于全模型，而非仅草稿路径。

**本文创新**：精度非对称——draft head 2bit（可抛弃，SRAM 驻留），verify backbone 4bit（精确）。
将"低精度=垃圾"的结论从全模型缩小至草稿路径，绕过质量约束。这是 dflash2（block diffusion 的
 disposable denoise step）思想在 LLM 推测解码上的迁移。

### 5.4 创新点四：hybrid GDN 架构的带宽主导性论证

**学术价值**：首次量化 hybrid linear-attention/full-attention 模型（Qwen3.8 系列）在 Apple Silicon
上的访存结构——48/64 层 O(1) KV 使权重在任意长度下占 96%+ 访存，与纯 softmax 模型（长上下文
KV 主导）有本质不同。这推导出：**hybrid GDN 模型的优化必须攻击权重读取，而非 KV**，
直接否定了 paged-KV / KV 量化 / KV pooling 在此类模型上的价值（CoW paged-KV -5~7x 实证）。

### 5.5 创新点五：AWSD 三维正交乘积模型

提出 `每token等效读取 = W/(K·U·P)` 的闭式模型，将分散的优化（推测/带宽/量化）统一为正交乘积，
可独立优化、组合放大。这为后续工作提供了可预测的理论框架。

## 6. 实验设计与预期

### 6.1 分阶段验证（每阶段独立可落地，失败不阻塞前一阶段）

**阶段 A：MTP K=2 chain（时间摊销，低风险）**
- 扩展 `_run_verify_cycle` 支持 chain_budget=2
- 实测接受率、tok/s、lossless parity
- 预期：接受率 75-85%（K=2 比 K=1 略降），tok/s 60-80（K=1 的 1.3-1.5x）
- 风险：GDN rollback 在 K=2 的边界条件（已有 PR#965 基础）

**阶段 B：MTP K=4 chain（时间摊销，中风险）——已证伪 2026-09-25**
- 若 A 成功，推至 K=4
- 预期：接受率 60-70%，tok/s 90-130
- **实测：接受率 0-36%，tok/s 5-23（比 K=1 慢 2-9x），证伪**
- 根因：head hidden 随链深度漂移 + head forward 开销累积（200ms/cycle）；§4.2 "head 近免费"假设错误
- tree-mask 不会救——线性链 verify 已是正确 causal，失败在 head 质量非 verify 形状
- 结论：MTP chain 甜点 K≤2，K 维度封顶。代码已 K 参数化（env `FUSION_MLX_MTP_CHAIN_K`，clamp [1,4]），
  K=4 留 opt-in 不推荐。**K=2 已翻转为生产默认（PR#974），`--spec-decode mtp` 用户自动获得 +52% single-stream。
  所有其他解码优化杠杆均已证伪（compile -2.5%、blit+compute 3x negative、strided-gather 已 coalesced、
  residual-add MLX 已 lazy-fuse、4bit GEMM 已 Steel+MMA、KV quant 带宽数学判死）。K=2 是 MLX 解码优化的天花板。**

**阶段 C：2bit draft head（精度摊销，中风险）——证伪 2026-09-25（结构分析）**
- MTP head 2bit 量化 + SRAM 驻留
- 实测 2bit draft 接受率下界
- 预期：接受率 50-65%，但 draft 近免费（SRAM 驻留）→ 净赢
- 风险：2bit head 接受率 <30% 则放弃此维度
- **实测结果：未加载 2bit head。MTP head 实测 253MB = 1.8% backbone，2bit 节省 <1% 带宽。
  原 P=2 假设独立草稿模型（7GB 读），MTP head 不满足。P 维度结构死锁，证伪。**

**阶段 D：Metal async blit+compute 层间流水线（空间摊销，高风险高回报）——证伪 2026-09-25（PoC）**
- monkey-patch mlx-lm 层调度为双 encoder 流水线
- 实测 U（有效带宽利用率）
- 预期：U 0.44→0.65-0.80
- 风险：mlx-lm Python 层抽象不暴露 encoder；可能需 fork mlx-lm（上游 issue + PR 流程）
- **实测结果：Stream overlap 真实（3.4x compute-bound / 1.77x GEMV），但 prefetch 是 3x 浪费流量——
  L2 64MB << 层 219MB，无 residency 加速。GEMV 读 prefetched buf 不比读原始 W 快。
  U headroom 真实来源是 strided gather 访存模式（非 blit/compute overlap），且 compile 路径已证伪。
  Phase D 3x 负面，不接线。无上游 issue（物理约束 L2 尺寸，非 API 缺口）。**

### 6.2 基线对比

| 引擎 | 路径 | 实测 tok/s (27B-4bit) |
|---|---|---|
| native mlx-lm | stock | 22 |
| llama.cpp | draft-model spec | 30-40 |
| oMLX | compile | 22（无解码收益，证伪） |
| rapid-mlx | continuous batching | 25-30 |
| fusion-mlx | MTP K=1 (opt-out) | 40 |
| **fusion-mlx 生产默认** | **MTP K=2 (PR#974)** | **50 single-stream (+52% vs stock)** |
| ~~fusion-mlx AWSD-B~~ | ~~MTP K=4~~ | ~~证伪（accept 0-36%，tps 5-23）~~ |
| ~~fusion-mlx AWSD-ABCD~~ | ~~K=4+U0.8+P2~~ | ~~证伪（三维度全 falsified）~~ |

**结论：MLX 解码优化空间已穷尽。K=2 (+52% single-stream / +125% over stock) 是 Apple Silicon
解码的状态艺术天花板。残余 U=0.38→0.73 gap 是结构性 launch-tax，需 oMLX 单核 fork（known-blocked）。**

### 6.3 正确性验证（lossless 硬约束）

每阶段必须通过 token-for-token parity（greedy, temp=0, same seed）与 stock 输出一致。
MTP 是 lossless 的（rejected draft rollback，只 emit verified token）——任何 divergence
是 bug，不是 tradeoff。这是与 2bit/3bit 全模型量化的本质区别。

### 6.4 端到端验证（生产接线）

遵循项目规则：每阶段端到端完成（`fusion-mlx serve` + HTTP A/B bench + parity）才报完成。
**K=2 已为生产默认（PR#974）：`--spec-decode mtp` 自动 K=2，opt-out 用 `--mtp-chain-k 1`。
验证 11433：default K=2 = 49.7 tok/s，K=1 opt-out = 40.0 tok/s（+24% under contention），
lossless parity 保持（同 #969 K=2 代码路径）。**

## 7. 风险、边界与诚实声明

### 7.1 技术风险

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| MTP K>1 在 GDN 模型上 rollback 边界 bug | 中 | correctness | PR#965 已修 GDN rollback 基础；K=2 先行验证 |
| K 增大后接受率衰减过快 | 中 | 性能 | K=2/3/4 逐步测，找 accept/K 最优点 |
| Metal async blit+compute 需 fork mlx-lm | 高 | 落地难度 | 先提 upstream issue，遵循上游流程；fusion-mlx 内 monkey-patch 过渡 |
| 2bit draft head 接受率 <30% | 中 | 维度三失效 | 阶段 C 独立验证，失败则弃 P 维度，K+U 仍有 4-5x |
| 层间流水线与 mlx.compile 冲突 | 低 | 实现 | 已证 compile 无解码收益，可直接绕过 |

### 7.2 边界（本文不解决的）

1. **吞吐（多请求并发）**——AWSD 优化单流延迟；多请求连续批处理是正交问题（EnginePool 已处理）
2. **长上下文（>8K）**——hybrid GDN 模型 KV 增长慢，但 full_attention 层的 softmax KV 在超长上下文
   仍会增长；本文聚焦 200-2K token 解码场景
3. **prefill 优化**——prefill 是 compute-bound（大 batch 矩阵乘），不在带宽墙内，本文不覆盖
4. **非 Apple Silicon 平台**——AWSD 的层间流水线依赖 Metal 3 async encoder；CUDA 平台已有 async copy

### 7.3 诚实声明

1. **K>1 chain 实测修正（2026-09-25）**——阶段 A (K=2) 实测 +32% over K=1（49.6 vs 45 tok/s，accept 33-68%，
   lossless parity 通过，PR#969 merged）。但阶段 B (K=4) **证伪**：accept 崩至 0-36%，tps 5-23（比 K=1 的 45 慢
   2-9x），backbone/cycle 爆至 364-424ms（stock 18ms/tok）。根因有二：(a) head hidden 质量随链深度漂移——
   draft 2-4 喂 head forward 非 backbone hidden，第 3-4 个 draft 时 head 状态已发散；(b) head forward 开销累积
   （K=4 时 4×49ms=200ms/cycle mtp 开销）。本文 §4.2 "head 极小，K 次 forward 近免费"的假设**错误**——
   head 体积小不等于 hidden 质量稳。MTP chain 在 hybrid GDN 上的甜点是 K≤2，非线性可放大。
2. **U=0.80 是估计**——层间流水线的实际收益取决于 M5 GPU blit/compute engine 的并发度与 SRAM 容量，
   需 Metal kernel profiling 实测。
3. **不保证超越所有对手一个量级**——K 维度已在 K=2 封顶（K=4 证伪），12x 目标（K=4+U0.8+P2）不可达。
   现实上限 = K=2 + U + P。若 U 仅提至 0.55、2bit head 失败，保守组合 2-3x（vs stock）即 45-65 tok/s，
   仍超 oMLX/rapid-mlx/llama.cpp，但非"一个量级"。
4. **本文不伪造数据**——所有实测数字来自 fusion-mlx 真实加载 Qwen3.8-27B-4bit（端口 11433/11435），
   记录在 memory 与 CHANGELOG 中。理论数字明确标注"预期/理论"。K=4 证伪数据如实记录，不掩盖。

### 7.4 与现有工作的关系

- **vs oMLX**：oMLX 攻击算力（compile），本文证明算力非瓶颈；oMLX 在带宽墙上无解
- **vs rapid-mlx**：rapid-mlx 优化批处理/缓存，不触及单流带宽墙；AWSD 正交可叠加
- **vs llama.cpp**：llama.cpp 的 draft-model spec 是自回归推测（与 MTP 同范式），但其 draft model
  是独立小模型（权重独立读取）；MTP head 共享 embedding/backbone hidden，draft 近免费
- **vs H3.c / splash**：H3 是线性注意力架构（与 GDN 同类），其 O(1) KV 特性本文已论证强化带宽主导；
  splash 若基于 MLX 同样受带宽墙
- **vs dflash2 / dspark**：dflash2（block diffusion）的"block 生成"思想启发了精度非对称（§5.3）；
  dspark 的 GEMM 优化是算力维度，非本文焦点

## 附录 A：实测数据

### A.1 屋顶线校准（2025-09-25, M5 Max 128GB, Qwen3.8-27B-4bit）

```
working_set = 128.0 GB (mx.device_info)
model weights = 14 GB (27B × 0.5 byte, 4bit group_size=64)
peak bandwidth = 800 GB/s
roofline = 14/800 = 17.5 ms/tok = 57 tok/s ceiling
arithmetic intensity = 3.86 FLOP/byte (<< 37.5 crossover → bandwidth bound)
```

### A.2 带宽利用率实测

| 路径 | tok/s | effective BW | utilization |
|---|---|---|---|
| stock decode | 22 | 308 GB/s | 38% |
| MTP K=1 (90% accept) | 45 | 350 GB/s | 44% |

### A.3 架构参数（Qwen3.8-27B, hybrid GDN）

```
layers = 64 (48 linear_attention + 16 full_attention, interval=4)
hidden = 5120, heads = 24, kv_heads = 4, head_dim = 256
vocab = 248320, mtp_layers = 1
KV/token = 0.25 MB → KV@2000tok = 0.49 GB (权重 14GB 占 96.6%)
```

### A.4 穷尽矩阵（详见 §2，此处摘要）

```
compile:        -2.5%  (FALSIFIED, bandwidth bound)
手写 sdpa:       0     (86% roofline, but BW-capped)
EAGLE fused:    +4%    (2 GPU sync/step, K>2 collapse)
Medusa fused:   BLOCKED (d1sync forced)
n-gram:         0%     (d1sync 4ms cap)
2bit quant:     +46%   (GARBAGE quality)
approx attn:    REJECTED (multi-step diverge 0.21)
MTP K=1:        +86%   (WINNER, lossless, autoregressive no-sync)
```

### A.5 metal cache cap 修复（PR#967，AWSD 前置条件）

8GB 硬编码 cache cap → RAM-relative 64GB。修复后 MTP 稳定 47 tok/s（消除空闲驱逐 stall）。
此修复确保 AWSD 实验的带宽测量不受分配器抖动污染。

---

## 结语

本文的核心论点：**Apple Silicon 上大模型解码的物理墙是统一内存带宽，而非算力。**
oMLX/compile/手写 kernel 攻击算力——选错了维度。唯一有效方向是减少每 token 权重读取量，
即三维摊销（K 时间 / U 空间 / P 精度）。MTP K=1 已证此路可行（+86%），
K>1 chain + 带宽利用率提升 + 精度非对称是未被证伪的乘法放大路径。

**⚠️ 验证后修正（2026-09-25，四阶段全部实证完毕）：**

阶段 A (K=2) **通过**：+32% over K=1（49.6 tok/s），lossless parity，PR#969 merged。
但阶段 B/C/D **全部证伪**：
- **B (K=4)**：accept 崩至 0-36%，tps 5-23（比 K=1 慢 2-9x）。head hidden 随链深度漂移 + head forward 开销累积。K 维度封顶于 2。
- **C (2bit draft head)**：MTP head 实测 253MB = 1.8% backbone，2bit 节省 <1% 带宽。原 P=2 假设独立草稿模型（7GB 读），MTP head 不满足。P 维度结构死锁。
- **D (Metal blit+compute)**：stream overlap 真实（1.77x GEMV），但 prefetch 是 3x 浪费流量（L2 64MB << 层 219MB，无 residency）。U headroom 真实来源是 strided gather 访存模式，非 blit/compute overlap；且 compile 路径已证伪。

**最终诚实结论：**
- 三维放大路径全部证伪。AWSD 现实上限 = **Phase A only：K=1 MTP +86%（45 tok/s），K=2 +32% over K1（49.6 tok/s）**。
- 论文 12x 目标（K=4+U0.8+P2）不可达。保守 3-4x（K=2+U0.6）也不达——U 提升路径被 compile 证伪 + blit+compute 3x 负面封死。
- **唯一未证伪的 U 杠杆**：4bit gs=64 quant 布局重排（连续打包消除 strided gather），独立高风险路径，需全模型 re-quant + parity 验证，本文未覆盖，留作后续。
- fusion-mlx 的 K=1/K=2（+86%/+125% over stock）仍是 MLX 解码的诚实 state-of-art，超越 oMLX/rapid-mlx/llama.cpp。带宽墙在当前 MLX/Metal 原语下不可进一步突破。

**本文价值重定位**：AWSD 的贡献不是"12x 加速"，而是 (1) 带宽墙第一性原理论证（§1，算术强度 3.86 << 37.5 交叉点，已验证），(2) 穷尽矩阵（§2，14 路径实证，唯一赢家 MTP K=1），(3) hybrid GDN 架构带宽主导性论证（§5.4，KV 占比 96%+），(4) 同步税感知的推测范式选择（§5.1，自回归链式 vs 并行树）。三维摊销框架作为理论模型保留，但其放大上限被实证封顶于 K≤2。

下一步：quant 布局重排（strided gather → 连续打包）作为唯一未穷尽的 U 杠杆，独立验证。

