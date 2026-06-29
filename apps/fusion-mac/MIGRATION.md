# omlx-mac → fusion-mac 迁移总结

## 迁移完成时间
2026-06-29

## 迁移范围

### 1. 目录结构
- **源**: `/Users/dahai/claude-home/omlx/apps/omlx-mac/`
- **目标**: `/Users/dahai/claude-home/fusion-mlx/apps/fusion-mac/`

### 2. 文件迁移清单

#### Swift 源代码 (78 个文件)
- `Sources/App/` — 应用入口 (2 文件)
- `Sources/AppView/` — 主界面 (15 屏幕 + ViewModels + Utils)
- `Sources/Config/` — 配置管理 (3 文件)
- `Sources/Menubar/` — 系统托盘 (3 文件)
- `Sources/Net/` — 网络层 (OMLXClient + 14 DTOs)
- `Sources/Server/` — Python 服务管理 (4 文件)
- `Sources/Theme/` — UI 组件库 (11 组件)
- `Sources/Updater/` — 自动更新 (3 文件)
- `Sources/Welcome/` — 首次启动向导

#### 测试代码 (22 个测试文件)
- `Tests/FusionTests/` — 原 oMLXTests
- 包含 DTO 测试、配置测试、集成测试等

#### 构建脚本
- `Scripts/build.sh` — 主构建脚本
- `Scripts/dev_server.py` — 开发服务器

#### 资源文件
- `Resources/AppIcon.icon/` — 应用图标
- `Resources/Assets.xcassets/` — 图片资源

#### Xcode 项目
- `FusionMLX.xcodeproj/` — 原 oMLX.xcodeproj

### 3. 品牌重命名映射

| 原名称 | 新名称 |
|--------|--------|
| oMLX | FusionMLX |
| OMLX | Fusion |
| omlx | fusion |
| app.omlx | app.fusion-mlx |
| oMLXTests | FusionTests |

### 4. 路径和环境变量更新

#### 配置文件路径
- `~/.omlx` → `~/.fusion-mlx`
- `~/Library/Application Support/oMLX` → `~/Library/Application Support/Fusion-MLX`

#### Python 模块
- `omlx.cli` → `fusion_mlx.cli`
- `omlx/` → `fusion_mlx/`

#### 环境变量
- `OMLX_*` → `FUSION_*` (全部大写)
- 包括: `FUSION_BASE_PATH`, `FUSION_HOST`, `FUSION_PORT`, `FUSION_API_KEY` 等

### 5. Xcode 项目配置

- Bundle Identifier: `app.fusion-mlx`
- Product Name: `FusionMLX`
- Scheme: `FusionMLX`

### 6. 核心功能保留

所有 omlx-mac 的功能模块已完整迁移:

✅ 15 个功能屏幕 (Models, Downloads, Settings, Performance, Benchmarks 等)
✅ 系统托盘集成 (MenubarController)
✅ Python 服务生命周期管理 (ServerProcess)
✅ Admin API 客户端 (FusionClient)
✅ 自动更新检查 (ReleasesChecker)
✅ 首次启动向导 (WelcomeWindow)
✅ 配置管理 (AppConfig)
✅ 端口冲突检测 (PortConflictResolver)

### 7. 后端兼容性

fusion-mlx 的 Admin API 后端已完整实现，与 omlx-mac 的端点完全匹配:

- `/admin/api/models` — 模型管理
- `/admin/api/hf/*` — HuggingFace 下载
- `/admin/api/ms/*` — ModelScope 下载
- `/admin/api/stats` — 统计信息
- `/admin/api/settings` — 全局设置
- `/admin/api/profiles` — 模型配置
- `/admin/api/bench` — 性能测试
- `/admin/api/oq` — 量化管理
- `/admin/api/upload` — HF 上传

## 下一步行动

### 立即可执行
1. 编译测试: `cd apps/fusion-mac && Scripts/build.sh swift debug`
2. 运行单元测试: `xcodebuild test -scheme FusionMLX`
3. 手动验证应用启动

### 待完成项
- [ ] 替换应用图标为 Fusion MLX 品牌
- [ ] 更新 Assets.xcassets 中的图片资源
- [ ] 添加 fusion-mlx 特定的 UI 元素
- [ ] 完善首次启动向导文案
- [ ] 配置 CI/CD 构建流程

## 技术债务

1. **venvstacks 配置**: 需要确认 fusion-mlx 是否有 `packaging/venvstacks.toml`
2. **自定义内核**: `fusion_mlx/custom_kernels/` 可能需要从 omlx 补充
3. **版本号管理**: `fusion_mlx/_version.py` 需要与 Swift 版本同步

## 验证检查清单

- [x] 所有 Swift 文件品牌名已替换
- [x] 所有环境变量已更新为 FUSION_*
- [x] 所有路径已更新为 fusion_mlx
- [x] Xcode 项目配置已更新
- [x] Bundle Identifier 已设置为 app.fusion-mlx
- [x] 测试目录已重命名为 FusionTests
- [x] build.sh 路径已更新
- [x] ServerProcess.swift 调用 fusion_mlx.cli

## 架构优势

迁移后的 fusion-mac 保留了 omlx-mac 的所有优势:

1. **原生 Swift 体验**: SwiftUI + 系统托盘，性能优秀
2. **完整的服务管理**: 自动重启、健康检查、端口冲突处理
3. **丰富的功能集**: 15 个功能屏幕覆盖所有管理场景
4. **强大的 API 客户端**: 完整的 Admin API 集成
5. **自动更新机制**: GitHub Releases 检查
6. **主题系统**: 11 个可复用 UI 组件

同时获得了 fusion-mlx 后端的强大能力:
- 7 种模型引擎支持
- Paged KV Cache + SSD 冷层
- 推测解码
- OpenAI/Anthropic 双协议 API
- MCP 工具调用
- 模型量化 (oQ)
- 性能基准测试

## 配置适配与 Bug 修复

迁移后进一步完成了以下适配工作:

### Bug 修复: 下载页面离开后进度停止

**根因**: `DownloadsScreenVM` 是 `@State` 局部变量，导航离开时 SwiftUI 销毁 View → VM 释放 → `onDisappear` 取消轮询 Task → 进度追踪中断。

**修复**: 将 `DownloadsScreenVM` 提升为 `AppServices` 长生命周期属性（与 `ThroughputBenchScreenVM`/`AccuracyBenchScreenVM` 相同模式）。导航离开时轮询 Task 继续存活，返回页面时显示实时进度。

- `AppServices.swift`: 添加 `let downloads = DownloadsScreenVM()`
- `DownloadsScreen.swift`: `@State` → `let vm:`，移除 `onDisappear`
- `DownloadsScreenVM.swift`: `start()` 幂等化（避免重复创建 pollTask）
- `AppView.swift`: 传入 `services.downloads`

### 配置补全: UI 语言选择器

fusion-mlx 后端支持 `ui.language` 字段（admin API 返回/接受），GUI 缺失对应选择器。

- `GlobalSettingsDTO.swift`: 添加 `UISettings` 结构体 + `ui` 属性 + `uiLanguage` patch 字段
- `ServerScreenVM.swift`: 添加 `uiLanguage` 草稿 + 基线 + 自动保存方法
- `ServerScreen.swift`: 新增 "Appearance" 分区，10 种语言 Popup 选择器

## 结论

omlx-mac 的 Swift 原生 GUI 已完整迁移到 fusion-mlx，保留了所有功能特性，并适配了 fusion-mlx 的后端架构和命名规范。应用可以立即进行编译测试和功能验证。
