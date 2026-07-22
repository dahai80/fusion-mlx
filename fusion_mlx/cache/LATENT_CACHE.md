# UMA Radix Latent Cache

fusion-mlx 的下一个主打特性:把 #178 的 DiffusionRadixCache 从 **文本 KV**
扩展到 **视频帧 latent**,在 Apple Silicon 统一内存(UMA)上做零拷贝复用。

## 为什么这是独创的

x86 + CUDA 栈的显存是离散的:latent 在 GPU 上,复用必须跨 PCIe 拷贝,
开销常常抵消复用收益。Apple Silicon UMA 下,CPU 和 GPU 共享同一块物理内存,
一个 VAE 编码出来的 `mx.array` 已经在 GPU 上,重复 I2V 请求直接复用同一个指针,
**跳过 VAE encoder 的模型加载 + forward**,零拷贝。

这是 CUDA 栈结构上无法复刻的优势 -- 不是把现有 runtime 移植到 Metal,
而是只有 UMA 才能成立的能力。

## Phase-1(已落地):输入图像 latent 复用

重复 I2V 请求(同一张输入图 + 同一分辨率)命中缓存,跳过 VAE-encode。

### 接入点

- **LTX-2** (`fusion_mlx/video/ltx2/generate.py`):4 条 I2V pipeline 分支
  (DISTILLED stage1/stage2、DEV、DEV_TWO_STAGE、DEV_TWO_STAGE_HQ)全部接入。
- **Wan2.2** (`fusion_mlx/video/wan2/generate.py`):TI2V-5B 单图编码分支接入
  (缓存 raw encode 输出 `[1,1,H_lat,W_lat,z_dim]`,命中时跳过
  `load_vae_encoder` + `encode`)。

### 缓存键

`image_latent_key(model_id, image_source, height, width, dtype)` 生成:

```
latent:{model_id}:{height}x{width}:{dtype}:{sha256(image)[:16]}
```

同一张图 + 同一分辨率 + 同一精度 = 同一个键 = 命中。

### 配置(环境变量)

| 变量 | 默认 | 说明 |
|------|------|------|
| `FUSION_LATENT_CACHE` | `1` | 开关(`0` 关闭,缓存工厂返回 `None`) |
| `FUSION_LATENT_CACHE_MAX_MB` | `2048` | 单模型 latent 缓存上限(MB),LRU 淘汰 |

### 统计

缓存通过 `DiffusionRadixCache` 的 WeakSet 注册到 `all_cache_stats()`,
经 `GET /v1/cache/stats` 暴露(label = `latent:<model_id>`),含
`entries` / `hits` / `size_bytes` / `max_bytes`。

## 设计原则

1. **复用 #178,不新建缓存类** -- `latent_cache.py` 是
   `DiffusionRadixCache` 的薄工厂,字节记账 / LRU / pin / radix 全部继承。
2. **miss 路径字节一致** -- 未命中时执行原始 encode 代码 + `put`,
   输出与无缓存时完全一致;命中时才跳过 load + encode。
3. **VAE 无关** -- 缓存的是任意 `mx.array`,与具体 VAE 类解耦。

## Phase 路线

| Phase | 内容 | 状态 |
|-------|------|------|
| 1 | 输入图像 latent 复用(跳过 VAE-encode) | ✅ 已落地 |
| 2 | 多镜头尾帧 -> 下一镜头首帧零拷贝复用 | validation-gated,未开始 |
| 3 | cached vs naive 基准对比 | 未开始 |

Phase-2 是真正的"多镜头零拷贝":捕获上一镜头去噪后的尾帧 latent,pin 住,
作为下一镜头首帧 conditioning 零拷贝复用。这是 PR #199 README 里
"Next milestone" 的完整形态。若复用导致画面不连贯,回退到重新编码。

## 验证状态

- `tests/unit/test_latent_cache.py`:15 个单测全绿(真实 `mx.array` 的
  hit/miss/eviction/pin/registry/`all_cache_stats` 字节记账)。
- ruff + black --fast 干净。
- **完整 HTTP I2V E2E 待模型重下**:本机 `wan22-ti2v-5b` 是指向
  `~/.cache/mlx-video-models/wan22-ti2v-5b` 的失效符号链接(目标已删),
  LTX-2-dev-bf16 已释放;两条接入路径目前无本机视频模型可跑端到端。
  需通过 https://hf-mirror.com 重下后补 E2E。

## 相关

- #178 DiffusionRadixCache(文本 KV,Phase-1/2 已合并 PR #183/#195)
- PR #199 README 原创性区块(顶部,把本特性列为 "Next milestone")
