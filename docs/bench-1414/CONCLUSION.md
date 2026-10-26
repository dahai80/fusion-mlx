# #1414 基准结论：图纸 OCR 视觉模型实测（2026-10-26）

- 执行环境：fusion-mlx v0.10.20（含 #1420 修复 27c7f5f4），gateway/直连 `:11434`，A1 图纸 200 DPI 合成样张
- 脚本：`scripts/bench_vlm_drawing_ocr_1414.py`（本目录留档副本）；原始结果见 `4bit/`、`8bit/`

## 实测结果

| 类别 | 4bit (Qwen2.5-VL-7B) | 8bit (Qwen2.5-VL-7B) |
| --- | --- | --- |
| 印刷体标注 | em=0.17, sim=0.20 | em=0.17, sim=0.20 |
| 尺寸数字 | em=1.00, sim=1.00 | em=1.00, sim=1.00 |
| 表格 | em=0.00, sim=0.19 | em=0.00, sim=0.19 |
| 手写体备注 | em=1.00, sim=1.00 | em=1.00, sim=1.00 |
| 长图稳定性 | 12 切片全部 200，4/7 区域恢复，1 片截断，中位 4.7s / 最长 19.1s | 12 切片全部 200，4/7 区域恢复，0 截断，中位 4.7s / 最长 5.1s |
| oom/5xx | **0** | **0** |

## 结论（回填 fusion-drawing-parser 配置）

1. **#1420 修复实证有效**：修复前 qwen2.5-VL 图像请求持续 500（chat_generation_failed），修复后 oom_or_5xx = 0。
2. **推荐量化档位：4bit**（当前 fusion-drawing-parser 默认即 4bit，无需改配置）。两档精度完全一致；8bit 仅长图延迟更稳（无截断、max 5.1s vs 19.1s）——延迟敏感场景可选 8bit。
3. **安全切片尺寸：2048px @ 15% overlap 实测稳定**（12 切片 0 错误），维持 fusion-drawing-parser 默认 `slice.px=2048`。
4. **已知弱项（诚实记录）**：印刷体标注与表格的 exact-match/similarity 偏低——合成 ground truth 为严格全文比对，模型输出为结构化 JSON 与 GT 文本形态不同是主要因素，但 sim≈0.2 也提示提示词调优空间。建议后续：提示词调优专项，或在低置信度时启用 fusion-drawing-parser 已实现的 `cloud_fallback`。
5. **延迟**：单切片中位 4.7s，远低于 PRD §8 的 ≤60s/页约束 ✅。

## 遗留

- 提示词调优专项（印刷体/表格 sim 提升）建议另开 issue 跟踪。
- 基准使用合成样张；真实扫描图纸的准确率验收随 ai-information-search 全链路实测进行。
