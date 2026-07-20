# PR 需求：`/v1/images/generate` 增加参考图条件生成（redux / in_context），支持跨生成保持角色一致性

> 本 PR 由需求方提出，目标结果与验收标准已写明，待实现方在此分支上补代码。
> 业务背景：ComfyUI4macos「鬼故事短剧工厂」流水线 P3.2「真实人物形象」轨道需要 `character_style=realistic` + `motion_mode=multi_pose`：对同一角色生成 N 个不同姿态关键帧再定格剪辑成 clip。当前 fusion-mlx 的图像生成是**纯 txt2img**，每次生成独立采样，相同文本 prompt + 相同 seed 也无法保证人脸/服饰/身份在多帧间一致，导致真实人物风格的多姿态剪辑出现「换脸」。底层 mflux **已具备**参考图条件生成能力（`Flux1Redux` / `Flux1InContextDev`），但 fusion-mlx 未暴露。

## 一、问题现象（current behavior）

`POST /v1/images/generate`（`fusion_mlx/api/images.py`）当前请求体：

```python
class ImageGenerateRequest(BaseModel):
    prompt: str
    n: int = 1
    width: int = 1024
    height: int = 1024
    steps: int = 4
    seed: int | None = None
    guidance: float = 4.0
    response_format: str = "url"
    model: str | None = None
```

无任何参考图字段。`ImageGenEngine.generate(prompt, width, height, steps, seed, guidance, n_images, output_format, **kwargs)` 的 `**kwargs` 被完全忽略；`start()` 只加载 `mflux.models.flux.variants.txt2img.flux.Flux1`（txt2img 变体）。

### 实测影响（ComfyUI4macos `MultiPoseStage`）

对同一角色连续生成 3 帧姿态关键帧：

| 帧 | prompt 后缀 | seed | 结果 |
|:---|:---|:---|:---|
| 1 | "facing forward, neutral pose" | 42 | 角色 A（黑发、青衣） |
| 2 | "turning to the side, mid-action pose" | 43 | 角色 B（发型/服饰漂移） |
| 3 | "reaching outward, dynamic action pose" | 44 | 角色 C（脸型变化） |

即便锁定 `scene_seed` 与 `char_appearance` 文本描述，txt2img 的随机噪声初始化仍使人脸/身份在帧间漂移。`character_style=realistic` 因此无法用于多姿态剪辑，被迫降级为 `cartoon`/`puppet`（卡通风格对身份漂移更宽容）。

## 二、根因定位（root-cause pointers，供实现方核实）

1. **路由层无参考图入参** — `fusion_mlx/api/images.py` 的 `ImageGenerateRequest` 没有 `reference_image` / `reference_strength` / `mode` 字段；`generate_image()` 路由只把 `prompt/width/height/steps/seed/guidance/n` 透传给 `engine.generate()`。

2. **引擎层只加载 txt2img 变体** — `fusion_mlx/engines/image_gen.py:start()` 写死：
   ```python
   from mflux.models.flux.variants.txt2img.flux import Flux1
   flux = Flux1(model_config=model_config, model_path=self._model_path)
   ```
   未根据请求或模型名加载 `Flux1Redux` / `Flux1InContextDev`。

3. **`generate()` 丢弃 `**kwargs`** — 即便路由层透传参考图参数，`generate()` 的 `**kwargs` 既不解析也不转发给 `flux.generate_image()`。

4. **mflux 能力已存在但未被桥接** — 实测 mflux 提供三类参考图条件生成（路径 `mflux/src/mflux/models/flux/variants/`）：

   - **redux**（`redux/flux_redux.py`，`Flux1Redux`）— IP-Adapter 风格身份保持，经 `SiglipVisionTransformer` + `ReduxEncoder` 编码参考图，签名：
     ```python
     def generate_image(self, seed, prompt, redux_image_paths: list[Path|str],
                        num_inference_steps=4, height=1024, width=1024, guidance=4.0,
                        redux_image_strengths: list[float]|None=None,
                        image_strength: float|None=None, scheduler="linear") -> GeneratedImage
     ```
     **首选**：身份保持最强，适合多姿态一致性。
   - **in_context dev**（`in_context/flux_in_context_dev.py`，`Flux1InContextDev`）— VAE 直接编码参考图，签名：
     ```python
     def generate_image(self, seed, prompt, num_inference_steps=4, height=1024, width=1024,
                        guidance=4.0, image_path: Path|str|None=None,
                        image_strength: float|None=None, scheduler="linear") -> GeneratedImage
     ```
     img2img 风格，结构保持强、身份保持弱于 redux。
   - **kontext**（`kontext/flux_kontext.py`）— 参考图编辑/变体，适合改图而非一致性生成。

   三者均要求 `ModelConfig.dev()`（schnell 不支持参考图条件）。

## 三、需求（target behavior）

### 3.1 HTTP API 扩展（`POST /v1/images/generate`）

`ImageGenerateRequest` 增加可选字段（全部向后兼容，缺省时退化为现有 txt2img 行为）：

```python
class ImageGenerateRequest(BaseModel):
    prompt: str
    n: int = Field(default=1, ge=1, le=4)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    steps: int = Field(default=4, ge=1, le=50)
    seed: int | None = None
    guidance: float = Field(default=4.0, ge=1.0, le=20.0)
    response_format: str = Field(default="url", pattern="^(url|b64_json)$")
    model: str | None = None
    # —— 新增：参考图条件生成 ——
    reference_image: str | None = None        # base64（无前缀）或 data:image/png;base64,... 或可下载 URL
    reference_strength: float = Field(default=0.6, ge=0.0, le=1.0)  # 参考图影响强度
    conditioning_mode: str = Field(default="redux", pattern="^(redux|in_context)$")  # 条件模式
```

语义：
- `reference_image` 缺省 → 走现有 txt2img 路径（`Flux1`），**零行为变化**。
- `reference_image` 提供 → 进入参考图条件路径：
  - `conditioning_mode="redux"` → `Flux1Redux`，参考图作为 `redux_image_paths=[ref]`，`redux_image_strengths=[reference_strength]`。
  - `conditioning_mode="in_context"` → `Flux1InContextDev`，参考图作为 `image_path=ref`，`image_strength=reference_strength`。

### 3.2 引擎层扩展（`ImageGenEngine`）

1. `start()` 仍按需惰性加载。新增按 `conditioning_mode` 选择 mflux 变体：
   - txt2img（无参考图）→ `Flux1`
   - redux → `Flux1Redux`（需 `ModelConfig.dev()`）
   - in_context → `Flux1InContextDev`（需 `ModelConfig.dev()`）

   **性能优先**：参考图变体与 txt2img 变体不要同时常驻。建议同一个 `ImageGenEngine` 实例在首次请求时按该请求是否带参考图决定加载哪个变体；若后续请求模式不同，按需 `stop()` 再加载（接受重载开销，换内存占用）。实现方可选更优策略，但不得同时持有两个 Flux 模型。

2. `generate()` 扩展签名（保持现有参数顺序与默认值不变）：
   ```python
   async def generate(self, prompt, width=1024, height=1024, steps=4, seed=None,
                      guidance=4.0, n_images=1, output_format="PNG",
                      reference_image: str | None = None,
                      reference_strength: float = 0.6,
                      conditioning_mode: str = "redux", **kwargs) -> list[bytes]
   ```
   - 解析 `reference_image`（base64 / data URL / URL → 本地 PIL.Image 或临时文件路径）。
   - 转发到对应 mflux 变体的 `generate_image()`。
   - **Fail visibly**：若 `conditioning_mode="redux"` 但当前加载的是 txt2img 变体且模型非 dev，抛明确错误（"redux conditioning requires Flux dev model"），不得静默退化为 txt2img。

3. **前向兼容**：现有调用方（ComfyUI4macos `ImageGenerateStage._generate_http` 等）不传新字段时，行为与今天完全一致。

### 3.3 安全与实现约束（不可违背）

- API key 鉴权沿用现有 `FUSION_MLX_API_KEY` 中间件，**不得**在 URL query 传 key，**不得**在日志/错误里打印参考图 base64 全文（仅日志记录 `reference_image_len` 与来源类型）。
- 参考图若为 URL，服务端拉取须设超时（≤30s）与大小上限（≤10MB），防 SSRF：禁止内网地址（127/10/172.16/169.254/::1 等）。
- 参考图解码失败 → HTTP 422 + 明确错误信息，不得 500。

## 四、验收标准（acceptance criteria）

实现方完成后，以下用例须全部通过（实现方可补单元测试）：

1. **回归零行为变化**：不带 `reference_image` 的请求，响应与 PR 合入前完全一致（同 seed/同 prompt 产出同图）。
2. **redux 身份一致性**（核心验收）：
   - 固定一张角色参考图 `ref.png` + 固定 `seed=42`。
   - 对 3 个不同姿态 prompt（"facing forward" / "turning to the side" / "reaching outward"）分别调用 `/v1/images/generate`（`reference_image=ref.png 的 base64`, `conditioning_mode="redux"`, `reference_strength=0.6`）。
   - 3 张产出图经人脸embedding余弦相似度两两 ≥ 0.85（或实现方采用等价的一致性度量并写明阈值），即「同一人物不同姿态」。
   - 对照：不带参考图、同 3 prompt + 同 seed，相似度显著低于该阈值（证明一致性来自参考图而非 prompt）。
3. **in_context 路径可用**：`conditioning_mode="in_context"` 同样产出图像，且参考图结构被保持（边缘/构图相似）。
4. **HTTP 契约**：
   - `response_format="b64_json"` 与 `"url"` 两种返回格式均正常。
   - 参考图 base64 非法 → 422；URL 不可达/超大/内网 → 422/4xx，不 500。
   - 响应 `created` 字段保留。
5. **Fail visibly**：对 schnell 模型请求 `conditioning_mode="redux"` 返回明确 4xx/5xx 错误信息，不静默退化为 txt2img。
6. **日志**：每次参考图条件生成记录 `mode / reference_strength / prompt_len / 耗时 / 模型名`，不记录参考图内容。

## 五、实现指引（pointers，非约束）

- mflux 入口：
  - `mflux/src/mflux/models/flux/variants/redux/flux_redux.py` → `Flux1Redux`
  - `mflux/src/mflux/models/flux/variants/in_context/flux_in_context_dev.py` → `Flux1InContextDev`
  - 两者构造签名与 `Flux1` 一致：`(model_config=ModelConfig.dev(), model_path=..., quantize=..., lora_paths=..., lora_scales=...)`。
- 参考 `fusion_mlx/engines/image_gen.py` 现有 `_load()` 的 executor/timeout 模式加载新变体。
- 路由改动集中在 `fusion_mlx/api/images.py`，引擎改动集中在 `fusion_mlx/engines/image_gen.py`，不要扩散。
- 模型权重：redux 需 `SiglipVisionTransformer` + `ReduxEncoder` 权重（mflux 的 `FluxInitializer.init_redux` 会处理下载/加载），确认 `mlx-community` 有对应 dev redux 权重；若无，实现方在 PR 描述中说明所需权重名。

## 六、对调用方（ComfyUI4macos）的预期对接

本 PR 落地后，ComfyUI4macos 侧将（由需求方自行实现，**不在本 PR 范围**）：

- `ImageGenerateStage._generate_http` 在 `character_style=realistic` 时，把首帧生成图作为后续多姿态帧的 `reference_image` 透传，`conditioning_mode="redux"`。
- `MultiPoseStage` 在 realistic 轨道下用同一参考图驱动 N 帧生成，实现真实人物身份跨姿态一致。

即：本 PR 只负责让 fusion-mlx「能接受参考图、能保持身份」，ComfyUI4macos 的接线是下游工作。

## 七、范围外（out of scope）

- kontext（编辑）变体——本 PR 不要求。
- ControlNet（canny/depth）——本 PR 不要求，但实现方若顺手暴露更优，可在 PR 描述中说明。
- 视频生成 / 真实 AI 运动模型——独立 PR。
- LLM JSON 确定性（`response_format=json_object` + `enable_thinking=false`）——见 `feat/json-mode-deterministic-pure-json` 分支的独立 PR。
