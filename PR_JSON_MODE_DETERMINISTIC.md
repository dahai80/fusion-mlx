# PR 需求：`response_format=json_object` + `enable_thinking=false` 必须确定性地产出纯 JSON（无 thinking）

> 本 PR 由需求方提出，目标结果与验收标准已写明，待实现方在此分支上补代码。
> 业务背景：ComfyUI4macos「鬼故事短剧工厂」流水线 `prompt_expand` 阶段依赖 fusion-mlx 的 `Qwen3.5-9B-4bit`（vlm 引擎）按 JSON schema 生成分镜。当前 LLM 调用非确定性地失败，导致流水线频繁回退到确定性「西游记」兜底剧本，使 LLM 驱动的分层故事结构（LLM 摘要 → 分集大纲 → 逐集场景）形同虚设。

## 一、问题现象（current behavior）

对 `Qwen3.5-9B-4bit`（`engine_type=vlm`，`thinking_default=true`）调用 `/v1/chat/completions`，同时设置：

- `chat_template_kwargs={"enable_thinking": false}`
- `response_format={"type": "json_object"}`
- system prompt 明确要求「直接输出 JSON，不要思考过程」

模型**仍然**在 JSON 之前先生成一段**纯文本** `Thinking Process:\n\n1.  **Analyze the Request:** ...` 链式思考，长度 7000–8500 tokens。这段思考是**纯文本**（未包裹在 Qwen3 的 think 标签内），因此不会被服务端的 think-strip 逻辑清除，最终原样进入 `choices[0].message.content`。

### 实测数据（同一 prompt，temp=0.75，json_mode=true，enable_thinking=false）

| 场景 | max_tokens | 输出 tokens | content 首字符 | 结果 |
|:---|:---|:---|:---|:---|
| 简单 prompt | 2048 | ~46 | `{` | ✅ 纯 JSON |
| 真实编剧 prompt（无 directive） | 4096 | 4099 | `T`(Thinking) | ❌ 思考占满，无 JSON |
| 真实编剧 prompt（无 directive） | 6144 | 6147 | `T` | ❌ 思考占满，无 JSON |
| + directive 前缀 | 4096 ×3 runs | 4099/4101/4099 | T/T/T(末run出JSON) | 2/3 ❌ 截断 |
| + directive 前缀 | 8192 ×3 runs | 8195/8220/8201 | `/T/T` | 3/3 含JSON，2/3 需剥思考 |
| 真实 prompt_expand 消息 | 8192 | 7166 | `{` | ✅ 偶发干净 JSON |
| 真实 prompt_expand 消息 | 8192 | 8211 | `T` | ⚠️ 思考+JSON，可能截断 |
| 真实 prompt_expand 消息 | 12288 | 8795 | `{` | ✅ 干净 JSON |

**结论**：相同请求在「干净 JSON / 思考前缀+JSON / 纯思考截断」三态间随机摆动。`max_tokens=16384` 时约 1/3 概率纯思考顶满 cap、完全不产 JSON。

### 影响

- `enable_thinking=false` 对该模型**实际无效**——不能阻止 CoT。
- `response_format=json_object` **不强制 JSON**——模型可先输出自由文本思考再吐 JSON。
- 生产流水线 `prompt_expand` 解析失败率约 33–66%，触发「西游记」兜底，LLM 驱动编剧失效。

## 二、根因定位（root-cause pointers，供实现方核实）

1. **VLM 引擎静默丢弃 `enable_thinking`** — `fusion_mlx/engines/vlm.py:357-386`（及 `:526-536` 视频路径）：

   ```python
   template_kwargs = {"tokenize": False, "add_generation_prompt": True}
   if self._enable_thinking is not None:
       template_kwargs["enable_thinking"] = self._enable_thinking
   try:
       prompt = template_target.apply_chat_template(messages, **template_kwargs)
   except TypeError:
       template_kwargs.pop("enable_thinking", None)        # ← 静默丢弃！
       prompt = template_target.apply_chat_template(messages, **template_kwargs)
   ```

   若 `Qwen3.5-9B-4bit` 的 chat template / processor 在接收 `enable_thinking` kwarg 时抛 `TypeError`（社区 4bit 量化的 template 签名可能与官方 Qwen3 不同），fusion-mlx 会**静默移除 `enable_thinking` 再重试**，思考抑制就此丢失，且无任何 warning/error。违反「Fail visibly」。

2. **VLM 引擎不合并请求级 `chat_template_kwargs`** — 对比 `fusion_mlx/engines/batched.py:438-439` 有 `template_kwargs.update(chat_template_kwargs)`，`vlm.py` 完全没有等价逻辑。客户端发送的 `chat_template_kwargs={"enable_thinking": false}` 是否真正到达 VLM template，取决于路由是否把它翻译成引擎级 `enable_thinking` 参数（见下条，需核实）。

3. **`response_format` 仅靠 prompt 注入，非受约束解码** — `fusion_mlx/routes/chat.py:1917-1926`：`response_format` 通过 `build_json_system_prompt()` 注入一段「请输出 JSON」的 system prompt，**不是** token 级 JSON grammar 约束。模型完全可以先输出思考文本再输出 JSON，json_mode 因此形同建议而非强制。

4. 路由侧已有 `_resolve_enable_thinking` + json_mode 自动注入 `chat_template_kwargs.enable_thinking=false` 的「auto-disable」逻辑（`routes/chat.py:1941-1986`），但因上面 1/2 两点在 VLM 引擎侧落空。

## 三、需求（target behavior）

提供一条**确定性**路径，让 vlm/llm 引擎在客户端要求 JSON 时产出**纯 JSON、零思考**：

1. **`response_format={"type":"json_object"}` 必须强制 JSON**：响应 `content` 经标准清理后必须是合法 JSON 对象（`json.loads` 成功，首字符为 `{`，无思考前缀、无 markdown 代码围栏）。**首选受约束解码**（JSON grammar token masking，如 outlines / lm-format-enforcer / mlx-lm 结构化生成），使模型物理上无法生成非 JSON token；次选服务端在返回前严格剥离「首个 `{` 之前的一切内容」并校验。受约束解码为强偏好方案——它把输出边界压缩到真实 JSON 体量（约 700–2200 tokens），消除 7–8k token 的思考浪费。

2. **`enable_thinking=false` 必须真正抑制思考**：对 Qwen3 系 vlm 引擎，关闭思考后输出不得包含任何 CoT（含纯文本 `Thinking Process:`）。**禁止静默丢弃**：若 template 拒收 `enable_thinking` kwarg（`TypeError`），必须 `logger.warning` 明确上报「template 不支持 enable_thinking，思考抑制不可用」，而非静默 pop。

3. **VLM 引擎应合并请求级 `chat_template_kwargs`**（与 batched.py 对齐），或路由必须可靠地把 `chat_template_kwargs.enable_thinking` 翻译为引擎 `enable_thinking` 参数并核实生效。

4. **向后兼容**：未设置 `response_format` / `enable_thinking` 的调用保持现行行为；`enable_thinking=true`（默认）仍正常输出思考。

## 四、验收标准（acceptance criteria）

下方「复现脚本」对 `Qwen3.5-9B-4bit` 连跑 10 次，须满足：

- **10/10** 响应 `content` 满足 `json.loads(content)` 成功 **且** `content[0] == '{'`（无思考前缀、无 markdown）。
- **10/10** 在 `max_tokens=4096` 内完成、无截断（证明思考已被抑制，仅生成 JSON）。
- 真实编剧 prompt（脚本中 `REAL_PROMPT`）同样 **10/10** 通过，且 `max_tokens=4096` 足够。
- 真实 prompt 的**中位延迟**从现状 ~140–180s（8k 思考 token）降至 **< 40s**（约 1–2k JSON token）。
- 对照组：`enable_thinking=true` 且无 `response_format` 的调用仍输出思考，行为不变。

## 五、复现脚本（repro）

```python
# 前置：fusion-mlx serve 已起，Qwen3.5-9B-4bit 已 loaded。
# 运行：python repro_json_mode.py  （需 httpx）
import os, time, json, statistics
import httpx

BASE = "http://127.0.0.1:11434"
KEY  = os.environ["FUSION_MLX_API_KEY"]
MODEL = "Qwen3.5-9B-4bit"

SIMPLE = [{"role":"system","content":"只输出JSON。"},
          {"role":"user","content":"输出 {\"scenes\":[{\"id\":1,\"audio_script\":\"月下孤舟渡亡魂\"}]}。"}]

REAL_PROMPT = [{"role":"system","content":
    "直接输出单个合法JSON对象，不要输出任何思考过程，不要使用markdown代码块。"
    "JSON第一个字符必须是{，最后一个必须是}。\n"
    "你是一位资深电视剧编剧。输出 schema："
    "{\"story_title\":str,\"global_style\":str,\"character_registry\":[{name,appearance,voice}],"
    "\"scenes\":[{scene_id,visual_prompt,audio_script,sound_effect,characters,duration_seconds}]}。"},
    {"role":"user","content":
    "故事种子：青溪渡阴\n剧集标题：朱砂破阵\n目标分镜数：2\n"
    "已有角色：书生 appearance=\"young scholar in blue robe\"\n"
    "请严格按 schema 输出 JSON，分镜数必须等于 2。"}]

def run(msgs, label, n=10, mt=4096):
    lats, ok = [], 0
    with httpx.Client(timeout=600, headers={"Authorization":f"Bearer {KEY}"}) as c:
        for i in range(n):
            t0 = time.time()
            r = c.post(f"{BASE}/v1/chat/completions", json={
                "model": MODEL, "messages": msgs, "temperature": 0.75,
                "max_tokens": mt, "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": False},
            })
            dt = time.time() - t0
            content = r.json()["choices"][0]["message"]["content"]
            lats.append(dt)
            good = False
            try:
                good = content.lstrip().startswith("{") and json.loads(content) is not None
            except Exception:
                good = False
            ok += int(good)
            print(f"  [{label}] run{i} dt={dt:.1f}s ok={good} head={content[:40]!r}")
    print(f"== {label}: {ok}/{n} ok, median={statistics.median(lats):.1f}s")
    return ok, lats

assert run(SIMPLE, "simple")[0] == 10
assert run(REAL_PROMPT, "real")[0] == 10
print("ACCEPTANCE PASSED")
```

当前该脚本在 fusion-mlx 现状下：`real` 用例远达不到 10/10（实测约 0–4/10），且单次延迟 140–180s。

## 六、建议实现方向（非约束，供实现方参考）

- **首选**：在 vlm/batched 引擎的生成路径接入 JSON grammar 受约束解码（mlx-lm 已具备 logits processor 扩展点；可引入 outlines 或自实现 JSON 状态机 token mask）。`response_format.type == "json_object"` 时启用，`json_schema` 时按 schema 约束（bonus）。
- **次选/兜底**：保留 prompt 注入，但在返回前对 `content` 做「定位首个 `{`、末个 `}`、切片、`json.loads` 校验」的强制清洗；失败则返回 4xx 明确错误（不得返回带思考前缀的 200）。
- **必做**：`vlm.py` 的 `except TypeError` 静默 pop 改为 `logger.warning` 上报；VLM 引擎合并请求级 `chat_template_kwargs`（与 batched 对齐）。
- **观测**：`/health` 或 metrics 暴露 `json_mode_constrained_decoding` 能力位，便于客户端探测。

## 七、非目标（non-goals）

- 不要求改变默认思考行为（`thinking_default=true` 保留）。
- 不要求支持任意复杂 JSON Schema（`json_object` 任意合法 JSON 对象即可；`json_schema` 为 bonus）。
- 不要求改 mlx-community 模型权重本身——纯服务端/推理路径修复。
