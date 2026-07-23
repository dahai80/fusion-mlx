# Feature Gap: SDXL / ControlNet / IP-Adapter / AnimateDiff 落地计划

## 现状盘点

### fusion-mlx 图像生成架构

```
ImageGenEngine (engines/image_gen.py)
  └── mflux-fusion 0.18 (唯一绑定 Flux2Klein txt2img)
        ├── Flux1 variants: ControlNet, Depth, Fill, Kontext, Redux, In-Context
        ├── Flux2 variants: Klein (txt2img), Klein Edit
        ├── 其他: ErnieImage, Fibo, Ideogram4, Qwen Image, SeedVR2, Z-Image
        └── DepthPro (单目深度估计)

API 层: POST /v1/images/generate (仅 txt2img，仅 Flux2 Klein)
视频层: LTX-2 / Wan2 / SkyReels-V3 (无 AnimateDiff)
```

### 缺口矩阵

| 特性 | 底层实现 | fusion-mlx 接入 | API 路由 | 难度 |
|------|----------|-----------------|----------|------|
| SDXL 推理 | ❌ mlx 无公开 SDXL | ❌ | ❌ | 🔴 高(需移植 UNet+VAE+CLIP) |
| ControlNet-Canny | ✅ mflux flux_controlnet | ❌ 未桥接 | ❌ | 🟡 中(引擎层桥接) |
| ControlNet-Depth | ✅ mflux flux_depth | ❌ 未桥接 | ❌ | 🟡 中 |
| IP-Adapter (Redux) | ✅ mflux flux_redux | ❌ 未桥接 | ❌ | 🟡 中 |
| Kontext (图编辑) | ✅ mflux flux_kontext | ❌ 未桥接 | ❌ | 🟡 中 |
| Fill (inpaint) | ✅ mflux flux_fill | ❌ 未桥接 | ❌ | 🟡 中 |
| AnimateDiff | ❌ MLX 无实现 | ❌ | ❌ | 🔴 高(需从零移植) |

## 分阶段落地计划

### Phase-A: mflux 桥接层 (预计 3-4 天)

**目标**: 把 mflux 已有的 ControlNet/Redux/Kontext/Fill/Depth 通过 fusion-mlx 的
ImageGenEngine 和 API 暴露出来，不需要移植任何模型代码。

#### A1: ImageGenEngine 多变体支持

当前 `ImageGenEngine.__init__` 硬编码 `Flux2Klein`。改为根据 model config 动态选择:

```python
class ImageGenEngine(BaseNonStreamingEngine):
    def __init__(self, model_name: str, **kwargs):
        self._variant = kwargs.get("variant", "txt2img")
        self._controlnet_image = None
        self._redux_image = None
        self._kontext_image = None
        self._fill_mask = None

    async def start(self) -> None:
        variant_map = {
            "txt2img": ("mflux.models.flux2.variants.txt2img.flux2_klein", "Flux2Klein"),
            "controlnet_canny": ("mflux.models.flux.variants.controlnet.flux_controlnet", "FluxControlnet"),
            "controlnet_depth": ("mflux.models.flux.variants.depth.flux_depth", "FluxDepth"),
            "redux": ("mflux.models.flux.variants.redux.flux_redux", "FluxRedux"),
            "kontext": ("mflux.models.flux.variants.kontext.flux_kontext", "FluxKontext"),
            "fill": ("mflux.models.flux.variants.fill.flux_fill", "FluxFill"),
        }
        module_path, class_name = variant_map[self._variant]
```

#### A2: API 路由扩展

`/v1/images/generate` 扩展请求体:

```python
class ImageGenerateRequest(BaseModel):
    prompt: str
    # ...existing...
    variant: str | None = None          # "controlnet_canny", "redux", etc.
    control_image: str | None = None    # ControlNet 条件图路径/base64
    reference_image: str | None = None  # Redux/IP-Adapter 参考图
    edit_image: str | None = None       # Kontext 编辑输入图
    mask_image: str | None = None       # Fill inpaint mask
    strength: float = 1.0               # 条件强度
```

#### A3: ComfyUI 节点扩展

`fusion_mlx/integrations/comfyui.py` 注册新节点:
- `FusionControlNetNode` — 条件图 + prompt → 图片
- `FusionReduxNode` — 参考图 + prompt → 风格迁移
- `FusionKontextNode` — 编辑图 + prompt → 图编辑
- `FusionFillNode` — 图 + mask + prompt → inpaint

#### A4: UMA Radix Latent Cache 对接

mflux ControlNet/Redux 的 text encoder 复用与 #178 Phase-1 一致:
- `FluxControlnet` 走 `FluxPromptEncoder` → 复用 CLIP text cache
- `FluxRedux` 走 `ReduxPromptEncoder` → 新增 image-embedding cache key
- 条件图 VAE encode → 复用 Phase-1 image latent cache

#### A5: 测试 + CI

- 每个新 variant 3 个单测 (load/generate/unload)
- conftest mlx skip-list 更新
- `model_discovery.py` DIFFUSERS_PIPELINE_TASKS 更新

---

### Phase-B: SDXL MLX 移植 (预计 5-7 天)

**目标**: 在 fusion-mlx 中实现原生 SDXL 推理 pipeline。

#### B1: SDXL 架构组件

```
fusion_mlx/video/sdxl/
    __init__.py
    unet.py                      # SDXL UNet (dual text encoder cross-attention)
    vae.py                       # SDXL VAE
    text_encoder.py              # Dual CLIP (clip-vit-L + clip-vit-G)
    scheduler.py                 # Euler/AuraFlow scheduler
    generate.py                  # 主 pipeline (txt2img + img2img)
    convert.py                   # diffusers -> MLX safetensors 转换
```

#### B2: SDXL 独有挑战

- **Dual Text Encoder**: CLIP-ViT/L + CLIP-ViT/G → 154 tokens
  → #178 DiffusionRadixCache 需双 encoder 各自缓存
- **Time Embedding**: SDXL UNet 用 `original_size/crops_coords_top_left`
  → `ImageGenEngine.generate()` 需新增参数
- **Refiner**: 先只做 base，refiner 后续

#### B3: SDXL Backend 注册

```python
class SDXLBackend(ImageBackend):
    model_patterns = ["sdxl", "stable-diffusion-xl"]
    supports_i2i = True
    supports_controlnet = False  # Phase-C
```

---

### Phase-C: ControlNet for SDXL (预计 3-4 天，依赖 Phase-B)

SDXL ControlNet 与 Flux ControlNet 架构不同:
- Flux: DiT + ControlNet 独立 conditioning
- SDXL: UNet + ControlNet 注入到每个 down/mid/up block

实现:
1. 移植 `diffusers` 的 `SDXLControlNet` → `fusion_mlx/video/sdxl/controlnet.py`
2. SDXL UNet forward 中注入 ControlNet conditioning
3. API: `/v1/images/generate` 新增 `control_image` + `controlnet_type`

---

### Phase-D: AnimateDiff MLX 移植 (预计 7-10 天，最高难度)

#### D1: 依赖链

```
SD 1.5 MLX pipeline (新移植)
  → AnimateDiff Motion Module (temporal attention)
    → AnimateDiff T2V
      → AnimateDiff + ControlNet
        → AnimateDiff + IP-Adapter
```

#### D2: 架构

```
fusion_mlx/video/animatediff/
    __init__.py
    motion_module.py        # Temporal Transformer (核心)
    unet_animatediff.py     # SD1.5 UNet + Motion Module 注入
    generate.py             # T2V pipeline
    lora.py                 # Motion LoRA
```

#### D3: 与现有视频 pipeline 的关系

- LTX-2/Wan2/SkyReels: DiT 架构 (从头训练)
- AnimateDiff: UNet 架构 (图像模型 + temporal attention)
- **不共享 pipeline 代码**，但共享 VideoBackend + UMA Latent Cache + ComfyUI Stage API

---

## 优先级排序

| 优先级 | Phase | 收益/成本比 | 工期 |
|--------|-------|------------|------|
| **P0** | Phase-A (mflux 桥接) | 🟢 最高 — 零移植 | 3-4 天 |
| **P1** | Phase-B (SDXL) | 🟡 中 — 需移植 | 5-7 天 |
| **P2** | Phase-C (SDXL ControlNet) | 🟡 中 — 依赖 B | 3-4 天 |
| **P3** | Phase-D (AnimateDiff) | 🔴 低(成本高) | 7-10 天 |

## 建议执行顺序

```
Phase-A → Phase-B → Phase-C → Phase-D
 3-4天    5-7天     3-4天      7-10天
```

Phase-A 可立即开始，零风险。Phase-B/C 需模型下载+转换。
Phase-D 建议等 A/B 验证后再启动。

## 风险

1. **mflux API 不稳定**: variant 类可能版本间改名 → 锁定 mflux-fusion>=0.18
2. **SDXL 权重格式**: diffusers safetensors 与 MLX 不兼容 → 需 convert 脚本
3. **AnimateDiff 时间注意力**: MLX MFA 是否支持 5D 输入 → 需验证
4. **内存**: SDXL UNet ~5GB + Dual CLIP ~1.5GB + VAE ~0.5GB = 7GB 最小
