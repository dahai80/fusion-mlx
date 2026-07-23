# UMA Radix Latent Cache

fusion-mlx 的主打特性:把 #178 的 DiffusionRadixCache 从 **文本 KV**
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

## Phase-2(已落地):多镜头尾帧→首帧零拷贝复用

多镜头 pipeline 中,镜头 N 去噪后的尾帧 latent 被 pin 住,
作为镜头 N+1 的首帧 conditioning 零拷贝复用,跳过 VAE decode→re-encode。

### 工作流程

1. Shot N 生成完成 → denoise 后、VAE decode 前捕获尾帧 latent:
   - LTX-2: `latents[:, :, -1:, :, :]` (5D)
   - Wan2: `latents[:, -1:, :, :]` (4D)
2. `put_session_tail(session_id, model_id, tail)` 写入独立缓存实例
3. Shot N+1 请求(同一 `session_id` + `image` 参数):
   - `get_session_tail` 命中 → 直接用作 `image_latent`,跳过 VAE encode
   - 未命中 → 回退到 Phase-1 `_encode_image_latent` 路径

### 接入点

- **LTX-2** (`fusion_mlx/video/ltx2/generate.py`):I2V 分支首帧复用 + 尾帧捕获
- **Wan2.2** (`fusion_mlx/video/wan2/generate.py`):mask-blend (TI2V-5B) 首帧复用 + 尾帧捕获
- **HTTP API** (`fusion_mlx/api/videos_routes.py`):`session_id` 字段透传
- **Backend** (`base.py`/`ltx2.py`/`wan2.py`):`session_id` 参数传递链

### 缓存键

`session_tail_key(session_id, model_id)` 生成:

```
session_tail:{model_id}:{session_id}
```

不同 session 隔离,不同 model 隔离。

### 配置(环境变量)

| 变量 | 默认 | 说明 |
|------|------|------|
| `FUSION_LATENT_CACHE` | `1` | 总开关(`0` 关闭所有 latent 缓存) |
| `FUSION_SESSION_TAIL_CACHE` | `0` | Phase-2 session tail 开关(需同时开启 `FUSION_LATENT_CACHE=1`) |
| `FUSION_LATENT_CACHE_MAX_MB` | `2048` | 单模型 latent 缓存上限(MB),LRU 淘汰 |

### HTTP 用法

```bash
# Shot 1: 生成视频并缓存尾帧
curl -X POST http://localhost:11434/v1/videos/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat walking", "model": "ltx-2", "image": "/path/to/img.png", "session_id": "my-session-1"}'

# Shot 2: 复用上一镜头尾帧作为首帧(跳过 VAE encode)
curl -X POST http://localhost:11434/v1/videos/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "the cat jumps", "model": "ltx-2", "image": "/path/to/img.png", "session_id": "my-session-1"}'
```

### 验证门控

若 shot-2 输出不连贯 → 将 `FUSION_SESSION_TAIL_CACHE` 保持默认 OFF,
文档记录为 NEGATIVE 结果(类似 #177 speculative denoise 的做法)。
Machinery 已落地,env-gated,default-OFF 直到 E2E 验证通过。

## 设计原则

1. **复用 #178,不新建缓存类** -- `latent_cache.py` 是
   `DiffusionRadixCache` 的薄工厂,字节记账 / LRU / pin / radix 全部继承。
2. **miss 路径字节一致** -- 未命中时执行原始 encode 代码 + `put`,
   输出与无缓存时完全一致;命中时才跳过 load + encode。
3. **VAE 无关** -- 缓存的是任意 `mx.array`,与具体 VAE 类解耦。
4. **session tail 独立实例** -- 不与 image latent 缓存共享,通过 WeakSet
   注册到 `all_cache_stats()`,label = `session_tail`。

## Phase 路线

| Phase | 内容 | 状态 |
|-------|------|------|
| 1 | 输入图像 latent 复用(跳过 VAE-encode) | ✅ 已落地 |
| 2 | 多镜头尾帧→首帧零拷贝复用(session_id) | ✅ 代码落地,env-gated 默认 OFF |
| 3 | LTX-2-dev E2E 验证 + cached vs naive 基准 | 🔄 进行中 |

## 验证状态

- `tests/unit/test_latent_cache.py`:15 个单测全绿(Phase-1)
- `tests/unit/test_latent_cache_session.py`:11 个单测全绿(Phase-2 session tail)
- ruff + black --fast 干净。
- conftest.py mlx skip-list 已更新。

## 相关

- #178 DiffusionRadixCache(文本 KV,Phase-1/2 已合并 PR #183/#195)
- #2 UMA Radix Latent cache(本特性的 issue tracker)
