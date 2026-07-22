# SkyReels-V3 DiT 变体统一重构 (Tier 1+2)

## 目标
R2V / V2V / A2V 三个变体从"每变体一个独立文件、copy-paste"重构为"共享 Block + 共享 DiT 基类 + 薄子类"。
顺带关闭 #186（V2V/A2V 的 modulation 广播 bug + 补 forward_partial）。
**硬约束：权重加载 bit-identical、V2V lazy compile 3.3x 不退化、A2V audio 路径不破坏、不破坏已发布 1.0/2.0/3.0。**

## 证据（为何必须重构）
同一个 modulation 广播 bug，3 份 copy-paste 里只修了 1 份：
- `transformer_r2v.py:247` WanAttentionBlock - `[B,6,1,dim]` 已修 (Phase-2)
- `transformer_v2v.py:164` V2VAttentionBlock - `[B,6,dim]` (#186)
- `transformer_a2v.py:303` A2VAttentionBlock - `[B,6,dim]` (#186)
根因：独立类使 R2V 修复无法传播。#164/#168 的 norm 命名修复当时也是"R2V 改一遍、V2V/A2V 复刻一遍"。

## 现状分歧矩阵

| 维度 | R2V (SkyReelsR2VDiT) | V2V (SkyReelsV2VDiT) | A2V (SkyReelsA2VDiT) |
|---|---|---|---|
| Block 类 | WanAttentionBlock | V2VAttentionBlock | A2VAttentionBlock |
| has_temporal | False (可选) | True (强制) | True (强制) |
| temporal_window | -1 | 96 | 32 |
| audio cross-attn | 无 | 无 | norm_x + audio_cross_attn (真实权重) |
| text_dim 传入 block | 有 | 无 | 无 |
| __call__ 签名 | (x,t,context,...) | (x,t,context,...,temporal_len) | (x,t,audio_embeds,text_embeds,...) |
| context 来源 | 外部预融合 | 外部预融合 | 内部 audio_embedding 融合 |
| lazy _compiled_call | 无 | 有 (3.3x) | 有 (+ _compiled_blocks, 60 层整体编译劣化故仅 per-block+lazy) |
| forward_partial | 有 (#177) | 无 (#186) | 无 (#186) |
| num_layers 默认 | 40 | 40 | 60 |
| modulation reshape | [B,6,1,dim] | [B,6,dim] bug | [B,6,dim] bug |

共享原语已存在：`common.py`(PatchEmbed3D/mul_add/mul_add_add/maybe_compile/rope_params)、`attention.py`(WanSelfAttention/WanTemporalAttention/WanT2V/WanI2V CrossAttention/WAN_CROSSATTENTION_CLASSES/_sdpa)、`wan2/attention.py`(WanSelfAttention,9 callers 已共享)。
需迁移到共享位置（避免循环 import）：`Head`(现 transformer_r2v.py)、`AudioCrossAttention`+`AudioEmbedding`(现 transformer_a2v.py)。

## 设计

### Tier 1 - Block 统一
新文件 `fusion_mlx/video/skyreels_v3/blocks.py`：
- `SkyReelsDiTBlock`（统一 Block）：`__init__` 参数为三者的并集 - `has_temporal`、`temporal_window`、`has_audio_cross_attn`、`text_dim`。modulation 用 R2V 的修复版（`[B,6,1,dim]`）。`__call__` 签名为三者并集，`audio_ctx=None` 可选。前向顺序：modulation->self-attn->temporal(if has_temporal)->cross-attn->audio cross-attn(if has_audio_cross_attn and audio_ctx)->ffn。
- `AudioCrossAttention` 从 transformer_a2v.py 迁入（避免 blocks.py<->a2v 循环）。
- 保留薄子类名以兼容现有测试 + 权重 key：
  - `WanAttentionBlock(SkyReelsDiTBlock)` - 默认 has_temporal=False（R2V）
  - `V2VAttentionBlock(SkyReelsDiTBlock)` - 默认 has_temporal=True, temporal_window=96
  - `A2VAttentionBlock(SkyReelsDiTBlock)` - 默认 has_temporal=True, temporal_window=32, has_audio_cross_attn=True

权重安全：`load_weights` 按 key 名映射（`blocks.N.norm1/modulation/audio_cross_attn.*`），类身份无关。#168/#169 已对齐 norm 命名 -> key 一致 -> 统一类不改变 key 集。

### Tier 2 - DiT 基类统一
新文件 `fusion_mlx/video/skyreels_v3/dit_base.py`：
- `SkyReelsBaseDiT`：
  - 共享 `__init__` config 解析（dim/ffn_dim/num_heads/num_layers/patch_size/in_dim/out_dim/text_dim/text_len/freq_dim/window_size/qk_norm/cross_attn_norm/eps/cross_attn_type 路由/added_kv_proj_dim/temporal_window）- 三者逐字节相同。
  - 共享模块：patch_embedding/text_embedding/time_embedding/time_projection/freqs/head。
  - `_build_blocks(block_cls, **flags)` - 子类 override 选 Block。
  - 共享 `_unpatchify`（三者相同）。
  - 共享 `forward_partial`（#177 已验证的 layer-pruned draft 前向）- 三者皆得 -> 关 #186 的 spec 路径缺口。
  - 共享 `_forward_core(x, t, context, seq_lens, grid_sizes, ..., block_extra)`：patch->text->time->block 循环(透传 block_extra 给 audio_ctx/temporal_len)->head->unpatchify。
  - 共享 `_dispatch(raw_fn, *args)`：lazy `_compiled_call` 缓存模式（V2V/A2V 用）。
- 薄子类（仍在各自 transformer_*.py，保 import 路径兼容）：
  - `SkyReelsR2VDiT(SkyReelsBaseDiT)`：block=WanAttentionBlock(has_temporal=False, text_dim=...)，`use_lazy_compile=False`，`__call__`=直接 `_call_raw`->`_forward_core`。
  - `SkyReelsV2VDiT(SkyReelsBaseDiT)`：block=V2VAttentionBlock，`use_lazy_compile=True`，`__call__`=lazy dispatch->`_call_raw`->`_forward_core`(temporal_len)。保留 `context_window_size`（dead 但保兼容）。继承 `forward_partial`。
  - `SkyReelsA2VDiT(SkyReelsBaseDiT)`：block=A2VAttentionBlock，`use_lazy_compile=True`，`audio_embedding`+`audio_dim` 独有，`_call_raw` 先 `context=audio_embedding(audio_embeds,text_embeds); audio_ctx=audio_embeds` 再 `_forward_core`(audio_ctx)。继承 `forward_partial`。保留 `_compiled_blocks` 字段。

## #186 关闭
- modulation 修复随统一 Block 自动覆盖 V2V/A2V（一处修复）。
- `forward_partial` 随基类下发到 V2V/A2V（spec denoise 路径就绪，env 门控默认关，零生产风险）。

## 文件变更
- NEW `fusion_mlx/video/skyreels_v3/blocks.py`（统一 Block + AudioCrossAttention + 3 薄子类名）
- NEW `fusion_mlx/video/skyreels_v3/dit_base.py`（SkyReelsBaseDiT + _forward_core + _dispatch + forward_partial + _unpatchify）
- EDIT `transformer_r2v.py`：移除 WanAttentionBlock/Head 定义（迁 blocks.py/common）、SkyReelsR2VDiT 改薄子类，保 `__all__` 导出兼容
- EDIT `transformer_v2v.py`：移除 V2VAttentionBlock、SkyReelsV2VDiT 改薄子类 + 继承 forward_partial
- EDIT `transformer_a2v.py`：移除 A2VAttentionBlock/AudioCrossAttention/AudioEmbedding(迁共享)、SkyReelsA2VDiT 改薄子类 + 继承 forward_partial
- NEW `tests/unit/test_skyreels_dit_unified.py`：3 变体 weight-key parity + forward parity(旧 vs 新) + V2V/A2V modulation B>1 修复 + V2V/A2V forward_partial bit-identical(n_blocks==num_layers)
- EDIT `tests/unit/conftest.py`：新测试加入 _OPT_DEP_SUITES "mlx" skip 列表（CI Linux 收集期 mlx mock 崩溃 #178/#179 教训）
- EDIT `SPECULATIVE_DENOISE.md` / `README.md`：#186 forward_partial 已下发 V2V/A2V
- Memory 更新 + #186 close

## 验证策略（分层，坦白风险）
1. **R2V 真实 14B**（权重已下载 `~/.fusion-mlx/models/Skywork/SkyReels-V3-R2V-14B-MLX`）：
   - 重构前抓 golden：`load_dit_weights` -> 0-dropped 审计 + 一组随机输入的 forward 输出。
   - 重构后：0-dropped 不变 + `mx.allclose(new, golden, 1e-4)`。
2. **V2V/A2V**（真实权重未下载，坦白无法 0-dropped 验证）：
   - weight-key parity：旧类 vs 新薄子类，加载同一组 synthetic 权重，断言 `parameters()` key 集相同 + 值相同。
   - forward parity：同随机输入 -> `mx.allclose(old_out, new_out, 1e-4)`（`FUSION_DISABLE_COMPILE=1` 确定性）。
   - 现有 `test_skyreels_v2v_a2v_norm.py::test_v2v/a2v_block_norm_naming_affine` 通过。
3. **V2V lazy compile**：重构后跑 V2V `_call_raw` 经 `_compiled_call`，断言首步 eager、次步 compiled、输出与 eager 一致（compile 不改变数值）。
4. lint：所有触碰文件 `black --fast` + `ruff check --fix`（CI py3.11 + UP035/UP006/F401 教训）。

## 风险与缓解
- **A2V audio 路径破坏**：audio_embedding/audio_cross_attn 是真实权重路径。缓解：AudioCrossAttention 整体迁移不改实现；forward parity 测试覆盖 audio_ctx 分支。
- **V2V lazy compile 退化**：`mx.compile(self._call_raw)` 改为编译基类 `_call_raw`（同样捕获 self，同样算子图）-> 3.3x 应保持。缓解：lazy compile 数值 parity 测试。
- **循环 import**：blocks.py/dit_base.py 需 Head/AudioCrossAttention/AudioEmbedding -> 全部迁入共享模块，transformer_*.py 只 import。预先用 `python -c "import fusion_mlx.video.skyreels_v3.transformer_r2v"` 三文件验证。
- **V2V/A2V 无真实权重**：只能 synthetic parity，无法 0-dropped。坦诚记录在 PR + memory。
- **GateGuard**：新文件首次编辑触发，陈述 importers/API/schema/用户指令后重试；提交用 `git commit -F -` heredoc 绕文件创建门。

## 检查点（abort-friendly，每点验证通过再进下一步）
- CP1：blocks.py 统一 Block + 3 薄子类 -> R2V forward parity (golden) 通过 + V2V/A2V synthetic forward parity 通过 + 现有 norm 测试通过。**未通过则停。**
- CP2：dit_base.py 基类 + 3 DiT 薄子类 -> 三者 weight-key parity + forward parity + R2V 14B 0-dropped 不变。
- CP3：forward_partial 下发 V2V/A2V -> bit-identical(n_blocks==num_layers) 测试通过。
- CP4：V2V lazy compile 数值 parity + 新测试入 conftest skip 列表 + lint clean。
- CP5：commit + PR + #186 close + memory/README 更新。

## 不在范围
- 不动 wan2/ 底座（仅消费其共享原语）。
- 不改 pipelines/__init__.py 的调用方（薄子类保同名同签名，调用方零改动）。
- 不重构 ltx2/ltx_video_legacy/wan2 其它 transformer。
- Phase-3 comfyUI Stage API 仍 deferred（阻塞于真实加速路径 = 需 distilled draft，非 layer-prune）。

## token 预算
大重构（2 新文件 + 3 重写 + 测试 + 14B 验证 + lint + PR）。按 CP 分段，每 CP 后可 compact + 评估是否继续。若中途预算告急，CP1/CP2 已是独立可交付价值（#186 modulation 修复 + Block 统一），可提前收口。
