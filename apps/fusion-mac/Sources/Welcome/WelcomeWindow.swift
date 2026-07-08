// First-run welcome wizard. Three-step flow: product intro, setup, and
// success. The setup step persists config and spawns the server.
//
// Architecture
//   • `WelcomeWindowController` is the AppKit owner of the NSWindow + the
//     SwiftUI `WelcomeView`. AppDelegate creates one on first run only —
//     returning users never see this window.
//   • `WelcomeViewModel` is a @MainActor ObservableObject holding the wizard
//     state across pages, the validation, and the "Start Server" action.
//
// First-run trigger lives in `AppDelegate` (PR 10 addition). When settings.json
// already exists (re-entry), the Welcome page is skipped via app boot flow.

import AppKit
import SwiftUI

// MARK: - Window controller

@MainActor
final class WelcomeWindowController: NSObject, NSWindowDelegate {
    static let willCloseNotification = Notification.Name("FusionWelcomeWillClose")

    private var window: NSWindow?
    private var vm: WelcomeViewModel?
    private weak var services: AppServices?
    private weak var server: ServerProcess?
    private let didFinish: (AppConfig, ServerProcess?) -> Void

    init(
        services: AppServices,
        server: ServerProcess?,
        didFinish: @escaping (AppConfig, ServerProcess?) -> Void
    ) {
        self.services = services
        self.server = server
        self.didFinish = didFinish
        super.init()
    }

    func show() {
        if let window {
            window.makeKeyAndOrderFront(self)
            NSApp.activate(ignoringOtherApps: true)
            return
        }
        guard let services else { return }

        let vm = WelcomeViewModel(services: services, server: server)
        vm.onFinish = { [weak self] config, server in
            guard let self else { return }
            self.didFinish(config, server)
        }
        vm.onOpenDashboard = { [weak self] in
            self?.close()
        }
        vm.onClose = { [weak self] in
            self?.close()
        }
        self.vm = vm

        let root = WelcomeView(vm: vm)
            .environment(services)

        let hosting = NSHostingController(rootView: root)
        hosting.view.frame = NSRect(x: 0, y: 0, width: 680, height: 620)

        let win = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 680, height: 620),
            styleMask: [.titled, .closable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        win.title = String(localized: "welcome.window.title",
                           defaultValue: "Welcome to FusionMLX",
                           comment: "Window title bar text for the Welcome wizard")
        win.titleVisibility = .hidden
        win.titlebarAppearsTransparent = true
        win.titlebarSeparatorStyle = .none
        win.backgroundColor = .windowBackgroundColor
        win.isMovableByWindowBackground = true
        win.contentViewController = hosting
        win.center()
        win.delegate = self
        win.isReleasedWhenClosed = false
        self.window = win

        win.makeKeyAndOrderFront(self)
        NSApp.activate(ignoringOtherApps: true)
    }

    func close() {
        window?.close()
    }

    // NSWindowDelegate

    nonisolated func windowWillClose(_ notification: Notification) {
        DispatchQueue.main.async {
            MainActor.assumeIsolated {
                self.handleWillClose()
            }
        }
    }

    /// Closing before Start Server is cancellation, not partial setup. Do not
    /// write settings.json or create the base path; AppDelegate will terminate
    /// so the next launch starts from the Welcome intro again.
    private func handleWillClose() {
        NotificationCenter.default.post(
            name: WelcomeWindowController.willCloseNotification,
            object: nil
        )
    }
}

// MARK: - View model

enum WelcomeStep: Equatable, Sendable {
    case intro
    case setup
    case hardwareDetect
    case modelSource
    case recommend
    case complete
}

@MainActor
final class WelcomeViewModel: ObservableObject {
    @Published var step: WelcomeStep = .intro
    @Published var basePath: String
    @Published var modelDir: String
    @Published var portText: String
    @Published var apiKey: String = ""
    @Published var lastError: String?
    @Published var isStarting: Bool = false
    @Published var startCompleted: Bool = false

    /// Model source mirror: huggingface | hf-mirror | modelscope
    @Published var modelSource: String = "huggingface"
    /// Use case: agent | coding | chat
    @Published var useCase: String = "agent"

    // MARK: Editable recommendation fields
    @Published var editMaxContext: Int = 65536
    @Published var editMaxTokens: Int = 4096
    @Published var editCacheEnabled: Bool = true
    @Published var editIdleTimeout: Int = 300
    @Published var editDflash: Bool = false
    @Published var editDspark: Bool = false
    @Published var editTurboquant: Bool = false
    @Published var validationWarning: String?

    // MARK: Recommended model download
    /// Download state per model repo ID: repoId -> status
    @Published var modelDownloads: [String: String] = [:]  // idle|downloading|done|error
    @Published var downloadErrors: [String: String] = [:]
    /// Which models the user has checked for download
    @Published var selectedModels: Set<String> = []

    struct ModelOption: Identifiable {
        let id: String          // repoId
        let displayName: String
        let reason: String
    }

    /// All recommended models for current use case + hardware
    var recommendedModels: [ModelOption] {
        let ramGB = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        let chip = detectChipName()
        let bw = gpuBandwidthFromChip(chip: chip) ?? 0
        switch useCase {
        case "coding":
            if ramGB >= 128 {
                return [
                    ModelOption(id: "Qwen/Qwen3.6-27B", displayName: "Qwen3.6-27B (Q4_K_M)", reason: "Best coding model on \(chip)"),
                    ModelOption(id: "deepseek-ai/DeepSeek-Coder-V2", displayName: "DeepSeek-Coder-V2", reason: "Strong alternative for code gen"),
                ]
            }
            if ramGB >= 64 {
                return [ModelOption(id: "Qwen/Qwen3.5-9B", displayName: "Qwen3.5-9B (4bit)", reason: "Balanced coding on \(Int(ramGB))GB RAM")]
            }
            return [ModelOption(id: "Qwen/Qwen3-0.6B", displayName: "Qwen3-0.6B (4bit)", reason: "Lightweight for \(Int(ramGB))GB RAM")]
        case "agent":
            if ramGB >= 128 {
                return [
                    ModelOption(id: "deepseek-ai/DeepSeek-V4-Flash", displayName: "DeepSeek-V4-Flash", reason: "Best agent throughput on \(chip) (\(bw)GB/s)"),
                    ModelOption(id: "Qwen/Qwen3.6-27B", displayName: "Qwen3.6-27B", reason: "Larger context for agent tasks"),
                ]
            }
            if ramGB >= 64 {
                return [
                    ModelOption(id: "Qwen/Qwen3.5-9B", displayName: "Qwen3.5-9B", reason: "Balanced agent on \(Int(ramGB))GB RAM"),
                    ModelOption(id: "unsloth/gpt-oss-120b-GGUF", displayName: "GPT-OSS-120B", reason: "High quality if RAM allows"),
                ]
            }
            return [ModelOption(id: "Qwen/Qwen3-0.6B", displayName: "Qwen3-0.6B (4bit)", reason: "Lightweight for \(Int(ramGB))GB RAM")]
        default: // chat
            if ramGB >= 64 {
                return [
                    ModelOption(id: "Qwen/Qwen3.5-9B", displayName: "Qwen3.5-9B", reason: "Best chat quality-speed on \(chip)"),
                    ModelOption(id: "google/gemma-4-31B-it", displayName: "Gemma-4-31B", reason: "Strong chat alternative"),
                ]
            }
            if ramGB >= 32 {
                return [ModelOption(id: "Qwen/Qwen3-0.6B", displayName: "Qwen3-0.6B", reason: "Chat-friendly for \(Int(ramGB))GB RAM")]
            }
            return [ModelOption(id: "Qwen/Qwen3-0.6B", displayName: "Qwen3-0.6B (4bit)", reason: "Lightweight for \(Int(ramGB))GB RAM")]
        }
    }

    func toggleModelSelection(_ repoId: String) {
        if selectedModels.contains(repoId) { selectedModels.remove(repoId) }
        else { selectedModels.insert(repoId) }
    }

    func selectAllModels() {
        selectedModels = Set(recommendedModels.map(\.id))
    }

    func downloadSelectedModels() {
        guard let services else { return }
        for repoId in selectedModels {
            guard modelDownloads[repoId] != "downloading" else { continue }
            modelDownloads[repoId] = "downloading"
            Task {
                do {
                    let resp = try await services.client.startHFDownload(repoId: repoId, hfToken: "")
                    if resp.success {
                        modelDownloads[repoId] = "done"
                    } else {
                        modelDownloads[repoId] = "error"
                        downloadErrors[repoId] = "Failed to start"
                    }
                } catch {
                    modelDownloads[repoId] = "error"
                    downloadErrors[repoId] = error.localizedDescription
                }
            }
        }
    }

    // MARK: Recommended settings (computed from hardware + use case)
    var recommendedMaxContext: Int {
        let ramGB = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        if useCase == "coding" { return ramGB >= 64 ? 131072 : 65536 }
        if useCase == "agent" { return ramGB >= 64 ? 65536 : 32768 }
        return ramGB >= 64 ? 32768 : 16384
    }
    var recommendedMaxTokens: Int {
        let ramGB = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        if useCase == "coding" { return ramGB >= 64 ? 8192 : 4096 }
        return ramGB >= 64 ? 4096 : 2048
    }
    var recommendedCacheEnabled: Bool {
        (try? FileManager.default.attributesOfFileSystem(forPath: NSHomeDirectory()))
            .flatMap { $0[.systemFreeSize] as? Int64 }
            .map { $0 > 100_000_000_000 } ?? false
    }
    var recommendedIdleTimeout: Int {
        let ramGB = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        if useCase == "agent" { return ramGB >= 64 ? 600 : 1200 }
        return ramGB >= 64 ? 300 : 600
    }

    var recommendedModelReason: String {
        let ramGB = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        let chip = detectChipName()
        if useCase == "coding" {
            return "Best for coding: large context (\(recommendedMaxContext) tokens) on \(chip) with \(Int(ramGB))GB unified memory"
        }
        if useCase == "agent" {
            return "Best for agents: balanced throughput on \(chip) (\(Int(ramGB))GB RAM, \(gpuBandwidthFromChip(chip: chip) ?? 0)GB/s bandwidth)"
        }
        return "Best for chat: quality-speed balance on \(chip) (\(Int(ramGB))GB RAM)"
    }
    var recommendedDflash: Bool { useCase == "agent" }
    var recommendedDspark: Bool { useCase == "coding" }
    var recommendedTurboquant: Bool {
        let ramGB = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        return ramGB >= 64
    }

    func validateRecommendedSettings() {
        let ramGB = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        if editMaxContext > 262144 {
            validationWarning = "Max context 262K+ may exceed \(Int(ramGB))GB RAM capacity"
        } else if editMaxContext > 131072 && ramGB < 64 {
            validationWarning = "Max context 128K+ recommended only for 64GB+ RAM (you have \(Int(ramGB))GB)"
        } else if editMaxTokens > 16384 {
            validationWarning = "Max tokens 16K+ unusual for most models"
        } else if editIdleTimeout < 60 {
            validationWarning = "Idle timeout <60s may cause frequent reloads"
        } else if editIdleTimeout > 3600 && ramGB < 32 {
            validationWarning = "Long timeout on \(Int(ramGB))GB RAM may cause memory pressure"
        } else {
            validationWarning = nil
        }
    }

    func loadRecommendedEdits() {
        editMaxContext = recommendedMaxContext
        editMaxTokens = recommendedMaxTokens
        editCacheEnabled = recommendedCacheEnabled
        editIdleTimeout = recommendedIdleTimeout
        editDflash = recommendedDflash
        editDspark = recommendedDspark
        editTurboquant = recommendedTurboquant
        validationWarning = nil
    }

    func detectChipName() -> String {
        var size = 0; sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        guard size > 0 else { return "Apple Silicon" }
        var buf = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &buf, &size, nil, 0)
        return String(cString: buf).trimmingCharacters(in: .whitespaces)
    }

    func gpuBandwidthFromChip(chip: String) -> Int? {
        let c = chip.uppercased()
        if c.contains("M5 MAX") { return 614 }
        if c.contains("M5 PRO") { return 307 }
        if c.contains("M5") { return 153 }
        if c.contains("M4 ULTRA") { return 819 }
        if c.contains("M4 MAX") { return 546 }
        if c.contains("M4 PRO") { return 273 }
        if c.contains("M4") { return 120 }
        if c.contains("M3 ULTRA") { return 800 }
        if c.contains("M3 MAX") { return 400 }
        if c.contains("M3 PRO") { return 150 }
        if c.contains("M3") { return 100 }
        if c.contains("M2 ULTRA") { return 800 }
        if c.contains("M2 MAX") { return 400 }
        if c.contains("M2 PRO") { return 200 }
        if c.contains("M2") { return 100 }
        if c.contains("M1 ULTRA") { return 800 }
        if c.contains("M1 MAX") { return 400 }
        if c.contains("M1 PRO") { return 200 }
        if c.contains("M1") { return 68 }
        return nil
    }

    var onFinish: ((AppConfig, ServerProcess?) -> Void)?
    var onOpenDashboard: (() -> Void)?
    var onClose: (() -> Void)?

    private weak var services: AppServices?
    private weak var server: ServerProcess?

    init(services: AppServices, server: ServerProcess?) {
        self.services = services
        self.server = server
        let cfg = services.config
        self.basePath = cfg.basePath.isEmpty ? AppConfig.defaultBasePath() : cfg.basePath
        self.modelDir = cfg.modelDir
        self.portText = String(cfg.port)
        self.apiKey = cfg.apiKey ?? ""
    }

    /// Single-page validation gate — runs Storage + API-key checks in
    /// sequence and surfaces the first failure into `lastError`.
    func validateSetup() -> Bool {
        validateStorage() && validateApiKey()
    }

    func beginSetup() {
        step = .setup
        lastError = nil
    }

    func backToIntro() {
        guard !isStarting else { return }
        step = .intro
        lastError = nil
    }

    func requestClose() {
        onClose?()
    }

    // MARK: Validation

    func validateStorage() -> Bool {
        let trimmedBase = basePath.trimmingCharacters(in: .whitespaces)
        guard !trimmedBase.isEmpty else {
            lastError = String(localized: "welcome.error.base_dir_required",
                               defaultValue: "Base directory is required.",
                               comment: "Welcome wizard validation: empty base path")
            return false
        }
        guard let port = Int(portText.trimmingCharacters(in: .whitespaces)),
              (1...65535).contains(port) else {
            lastError = String(localized: "welcome.error.port_out_of_range",
                               defaultValue: "Port must be a number between 1 and 65535.",
                               comment: "Welcome wizard validation: port not in valid range")
            return false
        }
        _ = port
        lastError = nil
        return true
    }

    func validateApiKey() -> Bool {
        let key = apiKey.trimmingCharacters(in: .whitespaces)
        guard key.count >= 4 else {
            lastError = String(localized: "welcome.error.key_too_short",
                               defaultValue: "API key must be at least 4 characters.",
                               comment: "Welcome wizard validation: api key below min length")
            return false
        }
        guard !key.contains(where: { $0.isWhitespace }) else {
            lastError = String(localized: "welcome.error.key_whitespace",
                               defaultValue: "API key must not contain whitespace.",
                               comment: "Welcome wizard validation: api key contains spaces")
            return false
        }
        guard key.unicodeScalars.allSatisfy({ $0.value >= 0x20 && $0.value < 0x7F }) else {
            lastError = String(localized: "welcome.error.key_non_ascii",
                               defaultValue: "API key must contain only printable ASCII.",
                               comment: "Welcome wizard validation: api key has non-printable or non-ASCII chars")
            return false
        }
        lastError = nil
        return true
    }

    // MARK: Folder picker

    func browseBaseDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = String(localized: "welcome.browse.prompt",
                              defaultValue: "Select",
                              comment: "NSOpenPanel button label for the Welcome wizard's folder pickers")
        panel.message = String(localized: "welcome.browse.base_message",
                               defaultValue: "Choose a parent folder. A .fusion-mlx directory will be created inside it.",
                               comment: "NSOpenPanel message when picking the Base Directory in Welcome wizard")
        if panel.runModal() == .OK, let url = panel.url {
            basePath = url.appendingPathComponent(".fusion-mlx", isDirectory: true).path
        }
    }

    func browseModelDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = String(localized: "welcome.browse.prompt",
                              defaultValue: "Select",
                              comment: "NSOpenPanel button label for the Welcome wizard's folder pickers")
        panel.message = String(localized: "welcome.browse.model_message",
                               defaultValue: "Choose the directory containing your model files.",
                               comment: "NSOpenPanel message when picking the Model Directory in Welcome wizard")
        if panel.runModal() == .OK, let url = panel.url {
            modelDir = url.path
        }
    }

    // MARK: Finish

    func startServer() async -> Bool {
        guard let services else { return false }
        isStarting = true
        defer { isStarting = false }

        // 1. Persist AppConfig.
        guard let port = Int(portText.trimmingCharacters(in: .whitespaces)) else {
            lastError = String(localized: "welcome.error.invalid_port",
                               defaultValue: "Invalid port.",
                               comment: "Welcome wizard: port field couldn't be parsed as an integer")
            return false
        }
        let trimmedKey = apiKey.trimmingCharacters(in: .whitespaces)
        let resolvedBase = ((basePath.trimmingCharacters(in: .whitespaces)
                             as NSString).expandingTildeInPath as NSString)
            .standardizingPath
        var config = services.config
        config.bindAddress = "127.0.0.1"
        config.basePath = resolvedBase
        config.port = port
        // modelDir is always a literal path. The wizard's "Reset" button
        // clears the field — interpret that as "use the default for the
        // basePath I just picked" rather than persisting an empty string.
        let trimmedDir = modelDir.trimmingCharacters(in: .whitespaces)
        let resolvedModelDir = trimmedDir.isEmpty
            ? AppConfig.defaultModelDir(forBasePath: resolvedBase)
            : trimmedDir
        config.setModelDirs([resolvedModelDir])
        // hf_endpoint is set later from Downloads → "HF Mirror" — we don't
        // touch the existing value here so a returning user's mirror choice
        // survives a re-entry into the wizard.
        config.apiKey = trimmedKey

        // Ensure the base directory exists before spawning the server. The
        // Python child creates `<base>/settings.json` on first start; if the
        // directory is missing, it bails with "Cannot create directory".
        do {
            try FileManager.default.createDirectory(
                at: URL(fileURLWithPath: resolvedBase),
                withIntermediateDirectories: true
            )
        } catch {
            lastError = String(localized: "welcome.error.mkdir_failed",
                               defaultValue: "Cannot create base directory: \(error.localizedDescription)",
                               comment: "Welcome wizard: mkdir on the base path failed; placeholder is the system error message")
            return false
        }

        // When the user kept the default ~/.fusion-mlx, clear every override.
        let isDefault = (resolvedBase == AppConfig.defaultBasePath())
        AppConfig.persistBasePath(isDefault ? nil : resolvedBase)

        do {
            try config.save()
        } catch {
            lastError = String(localized: "welcome.error.save_config_failed",
                               defaultValue: "Failed to save config: \(error.localizedDescription)",
                               comment: "Welcome wizard: writing settings.json failed; placeholder is the system error message")
            return false
        }
        services.updateConfig(config)

        // 2. Build a ServerProcess if AppDelegate didn't already pre-stage one
        // (first-run path defers spawning until the wizard finishes).
        let proc: ServerProcess
        if let existing = server {
            proc = existing
        } else {
            do {
                let runtime = try PythonRuntime.resolve()
                proc = ServerProcess(
                    runtime: runtime,
                    bindAddress: config.bindAddress,
                    port: config.port,
                    basePath: URL(fileURLWithPath: config.basePath, isDirectory: true)
                )
            } catch {
                lastError = String(localized: "welcome.error.python_runtime_failed",
                                   defaultValue: "Failed to locate Python runtime: \(error.localizedDescription)",
                                   comment: "Welcome wizard: PythonRuntime.resolve() threw; placeholder is the system error message")
                return false
            }
        }
        services.bind(server: proc)

        // 3. Start the server (port-conflict surfaces inline; user can edit
        // the port and tap again).
        do {
            switch try proc.start() {
            case .started, .alreadyRunning:
                break
            case .portConflict(let conflict):
                lastError = conflict.isFusion
                    ? String(localized: "welcome.error.port_in_use_fusion",
                             defaultValue: "Port \(String(config.port)) is already in use (FusionMLX server already running).",
                             comment: "Welcome wizard: bind() failed because another FusionMLX instance owns the port")
                    : String(localized: "welcome.error.port_in_use",
                             defaultValue: "Port \(String(config.port)) is already in use.",
                             comment: "Welcome wizard: bind() failed because some other process owns the port")
                return false
            }
        } catch {
            lastError = String(localized: "welcome.error.start_server_failed",
                               defaultValue: "Failed to start server: \(error.localizedDescription)",
                               comment: "Welcome wizard: ServerProcess.start() threw; placeholder is the system error message")
            return false
        }

        // 4. Best-effort post-start fix-ups: setup-api-key (or login if the
        // server already had one) + hf_endpoint patch. None of these are
        // fatal on first run — the user can re-do them in Security /
        // Server screens.
        // Give the server a beat to bind, then wait until the health-check
        // loop has confirmed /health 200 (cap 8s so a hung server doesn't
        // freeze the wizard).
        try? await Task.sleep(for: .milliseconds(500))
        await waitUntilHealthyOrTimeout(proc: proc, timeout: 8)

        _ = await setupServerApiKey(client: services.client, key: trimmedKey)

        startCompleted = true
        step = .complete
        onFinish?(config, proc)
        return true
    }

    /// Opens the admin dashboard after the setup step has started the server.
    @discardableResult
    func openWebDashboard() -> Bool {
        guard let services else { return false }
        // Use /admin/auto-login with API key so the browser gets a session cookie
        if let url = MenubarController.webAdminURL(
            host: services.config.host,
            port: services.config.port,
            apiKey: services.config.apiKey
        ) {
            NSWorkspace.shared.open(url)
            onOpenDashboard?()
            return true
        }
        // Fallback: try the dashboard directly (will show login page)
        guard let url = AppConfig.httpURL(
            host: services.config.host,
            port: services.config.port,
            path: "/admin/auto-login"
        ) else {
            return false
        }
        NSWorkspace.shared.open(url)
        onOpenDashboard?()
        return true
    }

    private func setupServerApiKey(client: FusionClient, key: String) async -> Bool {
        // Try setup-api-key (fresh install). When the server already has a
        // key set, the endpoint returns 400 — we swallow that and let
        // `FusionClient`'s 401 auto-login handle the next authenticated call.
        // The server is local-only on first run, so we don't need an
        // explicit login round-trip here.
        do {
            _ = try await client.setupApiKey(key, confirm: key)
            return true
        } catch {
            return false
        }
    }

    private func waitUntilHealthyOrTimeout(proc: ServerProcess, timeout: TimeInterval) async {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if case .running = proc.state { return }
            try? await Task.sleep(for: .milliseconds(200))
        }
    }
}

// MARK: - View

struct WelcomeView: View {
    @ObservedObject var vm: WelcomeViewModel
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let theme = scheme == .dark ? FusionTheme.dark : FusionTheme.light
        ZStack {
            WelcomeBackdrop()
                .ignoresSafeArea()

            VStack(spacing: 0) {
                Group {
                    switch vm.step {
                    case .intro:
                        WelcomeIntroBody()
                    case .setup:
                        WelcomeSetupBody(vm: vm)
                    case .hardwareDetect:
                        WelcomeHardwareDetectBody(vm: vm)
                    case .modelSource:
                        WelcomeModelSourceBody(vm: vm)
                    case .recommend:
                        WelcomeRecommendBody(vm: vm)
                    case .complete:
                        WelcomeCompleteBody(vm: vm)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                WelcomeFooter(vm: vm)
            }
        }
        .environment(\.fusionTheme, theme)
        .frame(width: 680, height: 620)
    }
}

// MARK: - Welcome redesign

private enum WelcomeStyle {
    static let bg = Color(nsColor: .windowBackgroundColor)
    static let panel = Color(nsColor: .controlBackgroundColor)
    static let panelBorder = Color(nsColor: .separatorColor)
    static let text = Color(nsColor: .labelColor)
    static let muted = Color(nsColor: .secondaryLabelColor)
    static let faint = Color(nsColor: .tertiaryLabelColor)
    static let fill = Color(nsColor: .quaternaryLabelColor).opacity(0.16)
    static let accent = Color.accentColor
}

private struct WelcomeBackdrop: View {
    var body: some View {
        WelcomeStyle.bg
    }
}

private struct WelcomeIntroBody: View {
    var body: some View {
        VStack(spacing: 24) {
            Spacer(minLength: 18)
            WelcomeIcon(size: 88)

            VStack(spacing: 14) {
                Text(String(localized: "welcome.header.title",
                            defaultValue: "FusionMLX",
                            comment: "Main heading shown on the Welcome wizard"))
                    .font(.fusionDisplay(48, weight: .semibold))
                    .foregroundStyle(WelcomeStyle.text)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                Text(String(localized: "welcome.header.subtitle",
                            defaultValue: "Local AI, no more waiting.",
                            comment: "Short tagline under the Welcome wizard's main heading"))
                    .font(.fusionDisplay(25, weight: .semibold))
                    .foregroundStyle(WelcomeStyle.text)
                    .multilineTextAlignment(.center)
                Text(String(localized: "welcome.header.tagline",
                            defaultValue: "macOS-native MLX server with smart caching.\nClaude Code, OpenClaw, and Cursor respond in 5 seconds, not 90.",
                            comment: "Sub-tagline under the Welcome wizard's main heading"))
                    .font(.fusionText(15))
                    .foregroundStyle(WelcomeStyle.muted)
                    .multilineTextAlignment(.center)
                    .lineSpacing(5)
                    .frame(maxWidth: 540)
                    .padding(.top, 6)
            }

            HStack(spacing: 10) {
                FeaturePill(icon: "lock.fill", title: "Localhost")
                FeaturePill(icon: "memorychip", title: "MLX")
                FeaturePill(icon: "bolt.fill", title: "Smart caching")
            }
            .padding(.top, 8)

            Text(String(localized: "welcome.header.meta",
                        defaultValue: "Apple Silicon · macOS-native · Apache 2.0",
                        comment: "Short metadata line on the Welcome intro page"))
                .font(.fusionText(12))
                .foregroundStyle(WelcomeStyle.faint)

            Spacer(minLength: 28)
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 54)
    }
}

private struct WelcomeSetupBody: View {
    @ObservedObject var vm: WelcomeViewModel
    @State private var keyVisible: Bool = false

    var body: some View {
        VStack(spacing: 22) {
            WelcomeIcon(size: 70)

            VStack(spacing: 6) {
                Text(String(localized: "welcome.setup.title",
                            defaultValue: "Set up your local server",
                            comment: "Heading on the Welcome setup page"))
                    .font(.fusionDisplay(24, weight: .semibold))
                    .foregroundStyle(WelcomeStyle.text)
                Text(String(localized: "welcome.intro",
                            defaultValue: "Choose where models live, pick a port, and create the API key you'll use from apps and the web dashboard.",
                            comment: "Intro paragraph at the top of the Welcome wizard's setup body"))
                    .font(.fusionText(13))
                    .foregroundStyle(WelcomeStyle.muted)
                    .multilineTextAlignment(.center)
                    .lineSpacing(2)
                    .frame(maxWidth: 480)
            }

            VStack(alignment: .leading, spacing: 18) {
                WelcomeNotice(
                    icon: "memorychip.fill",
                    title: String(localized: "welcome.setup.local.title",
                                  defaultValue: "Local server",
                                  comment: "Title for the local server setup card"),
                    text: String(localized: "welcome.setup.local.body",
                                 defaultValue: "FusionMLX binds to 127.0.0.1 on first run, so clients on this Mac can use it without exposing the server to your network.",
                                 comment: "Body for the local server setup card")
                )

                VStack(spacing: 0) {
                    SettingRow(
                        icon: "number",
                        title: String(localized: "welcome.storage.port.label",
                                      defaultValue: "Port",
                                      comment: "Row label for the server port field in Welcome wizard"),
                        subtitle: String(localized: "welcome.storage.port.sub",
                                         defaultValue: "Default 11435. Change this only if the port is already in use.",
                                         comment: "Sublabel for the port field with the recommended default")
                    ) {
                        TextInput(text: $vm.portText, mono: true, width: 96)
                    }

                    WelcomeDivider()

                    SettingRow(
                        icon: "folder",
                        title: String(localized: "welcome.storage.model_dir.label",
                                      defaultValue: "Model Directory",
                                      comment: "Row label for the Model Directory picker in Welcome wizard"),
                        subtitle: String(localized: "welcome.storage.model_dir.sub",
                                         defaultValue: "Where downloaded models are stored.",
                                         comment: "Sublabel explaining the model directory")
                    ) {
                        HStack(spacing: 8) {
                            Text(vm.modelDir.isEmpty
                                 ? AppConfig.defaultModelDir(forBasePath: vm.basePath)
                                 : vm.modelDir)
                                .font(.fusionMono(11))
                                .foregroundStyle(WelcomeStyle.muted)
                                .lineLimit(1)
                                .truncationMode(.middle)
                                .frame(width: 232, alignment: .trailing)
                            Button {
                                vm.browseModelDirectory()
                            } label: {
                                Image(systemName: "folder")
                                    .font(.system(size: 13, weight: .semibold))
                            }
                            .buttonStyle(.fusion(.normal, size: .small))
                            .disabled(vm.isStarting)
                            .help(String(localized: "welcome.button.browse",
                                         defaultValue: "Browse...",
                                         comment: "Folder picker trigger button in Welcome wizard"))
                        }
                    }

                    WelcomeDivider()

                    SettingRow(
                        icon: "key",
                        title: String(localized: "welcome.api_key.label",
                                  defaultValue: "API Key",
                                  comment: "Row label for the primary API key field in Welcome wizard"),
                        subtitle: String(localized: "welcome.api_key.sub",
                                         defaultValue: "This key is also your Web Dashboard login password.",
                                         comment: "Sublabel explaining API key usage")
                    ) {
                        HStack(spacing: 0) {
                            TextInput("welcome.api_key.placeholder", text: $vm.apiKey, placeholder: "sk-fusion-…", isSecure: !keyVisible, mono: true, width: 210)
                            Button {
                                keyVisible.toggle()
                            } label: {
                                Image(systemName: keyVisible ? "eye.slash" : "eye")
                                    .font(.system(size: 13, weight: .semibold))
                            }
                            .disabled(vm.isStarting)
                            .help(keyVisible
                                  ? String(localized: "welcome.api_key.hide",
                                           defaultValue: "Hide key",
                                           comment: "Tooltip on the eye-slash button that masks the API key field")
                                  : String(localized: "welcome.api_key.show",
                                           defaultValue: "Show key",
                                           comment: "Tooltip on the eye button that unmasks the API key field"))

                            Button {
                                vm.apiKey = APIKeyGenerator.random()
                                keyVisible = true
                            } label: {
                                Image(systemName: "arrow.triangle.2.circlepath")
                            }
                            .help(String(localized: "security.api_key.generate",
                                         defaultValue: "Generate a random key",
                                         comment: "Tooltip on the API key regenerate button"))
                        }
                        .buttonStyle(.fusion(.plain, size: .small))
                    }
                }
                .background(WelcomeStyle.panel)
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .strokeBorder(WelcomeStyle.panelBorder.opacity(0.45), lineWidth: 0.5)
                )

                Text(String(localized: "welcome.hint.settings_path",
                            defaultValue: "Settings are stored in ~/.fusion-mlx/settings.json.",
                            comment: "Hint line under the API key section pointing to settings.json"))
                    .font(.fusionText(11))
                    .foregroundStyle(WelcomeStyle.faint)
                    .frame(maxWidth: .infinity, alignment: .center)
            }
            .frame(width: 560)
        }
        .padding(.horizontal, 48)
        .padding(.top, 18)
        .padding(.bottom, 6)
        .disabled(vm.isStarting)
    }
}

private struct WelcomeCompleteBody: View {
    @ObservedObject var vm: WelcomeViewModel

    var body: some View {
        VStack(spacing: 24) {
            Spacer(minLength: 34)
            WelcomeIcon(size: 84)

            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 42, weight: .semibold))
                .foregroundStyle(WelcomeStyle.accent)
                .accessibilityHidden(true)

            VStack(spacing: 10) {
                Text(String(localized: "welcome.complete.title",
                            defaultValue: "All set!",
                            comment: "Heading on the Welcome completion page"))
                    .font(.fusionDisplay(30, weight: .semibold))
                    .foregroundStyle(WelcomeStyle.text)
                Text(String(localized: "welcome.complete.description",
                            defaultValue: "FusionMLX is running locally at http://127.0.0.1:\(vm.portText). Open the web dashboard to download your first model, manage settings, and connect your coding tools.",
                            comment: "Description on the Welcome completion page"))
                    .font(.fusionText(14))
                    .foregroundStyle(WelcomeStyle.muted)
                    .multilineTextAlignment(.center)
                    .lineSpacing(3)
                    .frame(maxWidth: 500)
                Text(String(localized: "welcome.complete.status",
                            defaultValue: "Server started. Dashboard is ready.",
                            comment: "Short success status on the Welcome completion page"))
                    .font(.fusionText(12, weight: .medium))
                    .foregroundStyle(WelcomeStyle.faint)
                    .padding(.top, 4)
            }
            Spacer(minLength: 40)
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 54)
    }
}

private struct WelcomeFooter: View {
    @ObservedObject var vm: WelcomeViewModel

    var body: some View {
        HStack(alignment: .center, spacing: 18) {
            if vm.step == .setup || vm.step == .hardwareDetect || vm.step == .modelSource || vm.step == .recommend {
                Button {
                    switch vm.step {
                    case .setup: vm.backToIntro()
                    case .hardwareDetect: vm.step = .setup
                    case .modelSource: vm.step = .hardwareDetect
                    case .recommend: vm.step = .modelSource
                    default: vm.backToIntro()
                    }
                } label: {
                    Label(String(localized: "common.back",
                                 defaultValue: "Back",
                                 comment: "Back button label"),
                          systemImage: "chevron.left")
                }
                .buttonStyle(.fusion(.plain))
                .disabled(vm.isStarting)
            } else {
                Spacer()
                    .frame(width: 90)
            }

            Spacer()

            if let error = vm.lastError {
                Text(error)
                    .font(.fusionText(11.5))
                    .foregroundStyle(Color(nsColor: .systemRed))
                    .lineLimit(2)
                    .multilineTextAlignment(.trailing)
                    .frame(maxWidth: 320, alignment: .trailing)
            }

            primaryButton
        }
        .padding(.horizontal, 28)
        .frame(height: 72)
        .background(WelcomeStyle.panel.opacity(0.68))
        .overlay(alignment: .top) {
            Rectangle()
                .fill(WelcomeStyle.panelBorder.opacity(0.5))
                .frame(height: 0.5)
        }
    }

    @ViewBuilder
    private var primaryButton: some View {
        switch vm.step {
        case .intro:
            WelcomeCTA(
                title: String(localized: "welcome.button.get_started",
                              defaultValue: "Get Started",
                              comment: "Primary footer button that advances from the intro to setup"),
                systemImage: "arrow.right",
                width: 142
            ) {
                vm.beginSetup()
            }

        case .setup:
            WelcomeCTA(
                title: String(localized: "welcome.button.continue",
                              defaultValue: "Continue",
                              comment: "Footer button that advances from setup to hardware detection"),
                systemImage: "arrow.right",
                width: 142
            ) {
                if vm.validateSetup() {
                    vm.step = .hardwareDetect
                }
            }

        case .hardwareDetect:
            WelcomeCTA(
                title: String(localized: "welcome.button.continue",
                              defaultValue: "Continue",
                              comment: "Footer button that advances from hardware detect to model source"),
                systemImage: "arrow.right",
                width: 142
            ) {
                vm.step = .modelSource
            }

        case .modelSource:
            WelcomeCTA(
                title: String(localized: "welcome.button.continue",
                              defaultValue: "Continue",
                              comment: "Footer button that advances from model source to recommendations"),
                systemImage: "arrow.right",
                width: 142
            ) {
                vm.step = .recommend
            }

        case .recommend:
            WelcomeCTA(
                title: vm.isStarting
                    ? "Starting Server..."
                    : "Start Server",
                systemImage: vm.isStarting ? nil : "arrow.right",
                isBusy: vm.isStarting,
                width: 160
            ) {
                Task {
                    guard vm.validateSetup() else { return }
                    _ = await vm.startServer()
                }
            }
            .disabled(vm.isStarting)

        case .complete:
            WelcomeCTA(
                title: String(localized: "welcome.button.open_dashboard",
                              defaultValue: "Open Web Dashboard",
                              comment: "Footer button that opens the local web dashboard"),
                systemImage: "arrow.up.right",
                width: 210
            ) {
                _ = vm.openWebDashboard()
            }
        }
    }
}

// MARK: - Model Source Selection

private struct WelcomeModelSourceBody: View {
    @ObservedObject var vm: WelcomeViewModel

    var body: some View {
        VStack(spacing: 22) {
            WelcomeIcon(size: 70)

            VStack(spacing: 6) {
                Text("Model Source")
                    .font(.fusionDisplay(24, weight: .semibold))
                    .foregroundStyle(WelcomeStyle.text)
                Text("Choose where to download models from.\nChinese users should select HF Mirror or ModelScope.")
                    .font(.fusionText(13))
                    .foregroundStyle(WelcomeStyle.muted)
                    .multilineTextAlignment(.center)
                    .lineSpacing(2)
                    .frame(maxWidth: 480)
            }

            VStack(spacing: 10) {
                SourceOption(
                    title: "HuggingFace",
                    subtitle: "hub.huggingface.co",
                    icon: "globe",
                    isSelected: vm.modelSource == "huggingface"
                ) { vm.modelSource = "huggingface" }

                SourceOption(
                    title: "HF Mirror",
                    subtitle: "hf-mirror.com",
                    icon: "arrow.triangle.branch",
                    isSelected: vm.modelSource == "hf-mirror"
                ) { vm.modelSource = "hf-mirror" }

                SourceOption(
                    title: "ModelScope",
                    subtitle: "modelscope.cn",
                    icon: "square.stack.3d.up",
                    isSelected: vm.modelSource == "modelscope"
                ) { vm.modelSource = "modelscope" }
            }
            .frame(width: 420)
        }
        .padding(.horizontal, 48)
        .padding(.top, 18)
    }
}

private struct SourceOption: View {
    let title: String
    let subtitle: String
    let icon: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 14) {
                Image(systemName: icon)
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(isSelected ? WelcomeStyle.accent : WelcomeStyle.muted)
                    .frame(width: 28)

                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.fusionText(14, weight: .semibold))
                        .foregroundStyle(WelcomeStyle.text)
                    Text(subtitle)
                        .font(.fusionText(11))
                        .foregroundStyle(WelcomeStyle.faint)
                }

                Spacer()

                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 20))
                        .foregroundStyle(WelcomeStyle.accent)
                }
            }
            .padding(14)
            .background(isSelected ? WelcomeStyle.accent.opacity(0.08) : WelcomeStyle.fill)
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(isSelected ? WelcomeStyle.accent.opacity(0.4) : .clear, lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Hardware Detection Body

private struct WelcomeHardwareDetectBody: View {
    @ObservedObject var vm: WelcomeViewModel

    var body: some View {
        VStack(spacing: 22) {
            WelcomeIcon(size: 70)
            VStack(spacing: 6) {
                Text("Your Mac Hardware")
                    .font(.fusionDisplay(24, weight: .semibold))
                    .foregroundStyle(WelcomeStyle.text)
                Text("Detected hardware configuration. This helps us recommend optimal settings.")
                    .font(.fusionText(13))
                    .foregroundStyle(WelcomeStyle.muted)
                    .multilineTextAlignment(.center)
                    .lineSpacing(2)
                    .frame(maxWidth: 480)
            }
            VStack(alignment: .leading, spacing: 0) {
                HwRow(label: "Chip", value: detectChip())
                WelcomeDivider()
                HwRow(label: "CPU Cores", value: "\(ProcessInfo.processInfo.processorCount)")
                WelcomeDivider()
                let chip = detectChip()
                let isAppleSilicon = chip.contains("Apple") || chip.contains("M")
                if isAppleSilicon {
                    HwRow(label: "GPU", value: chip)  // Apple Silicon = unified GPU
                    WelcomeDivider()
                    HwRow(label: "GPU Memory", value: String(format: "%.0f GB (unified)", Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824))
                    WelcomeDivider()
                    if let bw = gpuBandwidth(chip: chip) {
                        HwRow(label: "Memory BW", value: "\(bw) GB/s")
                        WelcomeDivider()
                    }
                }
                HwRow(label: "System RAM", value: String(format: "%.1f GB", Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824))
                WelcomeDivider()
                HwRow(label: "Disk Free", value: diskFree())
            }
            .background(WelcomeStyle.panel)
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(WelcomeStyle.panelBorder.opacity(0.45), lineWidth: 0.5))
            .frame(width: 500)
        }
        .padding(.horizontal, 48).padding(.top, 18).padding(.bottom, 6)
    }

    private func detectChip() -> String {
        var size = 0; sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        guard size > 0 else { return "Apple Silicon" }
        var buf = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &buf, &size, nil, 0)
        return String(cString: buf).trimmingCharacters(in: .whitespaces)
    }

    private func diskFree() -> String {
        guard let attrs = try? FileManager.default.attributesOfFileSystem(forPath: NSHomeDirectory()),
              let free = attrs[.systemFreeSize] as? Int64 else { return "—" }
        return String(format: "%.0f GB", Double(free) / 1_073_741_824)
    }

    /// Apple Silicon GPU memory bandwidth lookup (GB/s)
    private func gpuBandwidth(chip: String) -> Int? {
        let c = chip.uppercased()
        if c.contains("M5 MAX") { return 614 }
        if c.contains("M5 PRO") { return 307 }
        if c.contains("M5") { return 153 }
        if c.contains("M4 ULTRA") { return 819 }
        if c.contains("M4 MAX") { return 546 }
        if c.contains("M4 PRO") { return 273 }
        if c.contains("M4") { return 120 }
        if c.contains("M3 ULTRA") { return 800 }
        if c.contains("M3 MAX") { return 400 }
        if c.contains("M3 PRO") { return 150 }
        if c.contains("M3") { return 100 }
        if c.contains("M2 ULTRA") { return 800 }
        if c.contains("M2 MAX") { return 400 }
        if c.contains("M2 PRO") { return 200 }
        if c.contains("M2") { return 100 }
        if c.contains("M1 ULTRA") { return 800 }
        if c.contains("M1 MAX") { return 400 }
        if c.contains("M1 PRO") { return 200 }
        if c.contains("M1") { return 68 }
        return nil
    }
}

private struct HwRow: View {
    let label: String; let value: String
    var body: some View {
        HStack(spacing: 12) {
            Text(label).font(.fusionText(13, weight: .medium)).foregroundStyle(WelcomeStyle.text)
            Spacer()
            Text(value).font(.fusionMono(12)).foregroundStyle(WelcomeStyle.muted)
        }.padding(.horizontal, 16).padding(.vertical, 10)
    }
}

// MARK: - Recommendation Body

private struct WelcomeRecommendBody: View {
    @ObservedObject var vm: WelcomeViewModel

    var body: some View {
        ScrollView {
            VStack(spacing: 14) {
                WelcomeIcon(size: 45)
                VStack(spacing: 3) {
                    Text("Recommended Configuration").font(.fusionDisplay(22, weight: .semibold)).foregroundStyle(WelcomeStyle.text)
                    Text("Select your use case — recommendations adjust automatically.").font(.fusionText(12)).foregroundStyle(WelcomeStyle.muted).multilineTextAlignment(.center).frame(maxWidth: 520)
                }

                // Use case picker
                HStack(spacing: 8) {
                    UseCaseButton(title: "Agent", icon: "robot", desc: "OpenClaw agents", isSelected: vm.useCase == "agent") { vm.useCase = "agent"; vm.loadRecommendedEdits() }
                    UseCaseButton(title: "Coding", icon: "chevron.left.forwardslash.chevron.right", desc: "Code completion", isSelected: vm.useCase == "coding") { vm.useCase = "coding"; vm.loadRecommendedEdits() }
                    UseCaseButton(title: "Chat", icon: "message", desc: "General chat", isSelected: vm.useCase == "chat") { vm.useCase = "chat"; vm.loadRecommendedEdits() }
                }.padding(.horizontal, 8)

                // Recommended models list with checkboxes & download
                VStack(alignment: .leading, spacing: 4) {
                    Text("Recommended Models").font(.fusionText(11, weight: .semibold)).foregroundStyle(WelcomeStyle.faint)
                        .padding(.horizontal, 16).padding(.top, 10)
                    ForEach(vm.recommendedModels) { opt in
                        HStack(spacing: 10) {
                            let state = vm.modelDownloads[opt.id] ?? "idle"
                            let isSelected = vm.selectedModels.contains(opt.id)
                            Button {
                                vm.toggleModelSelection(opt.id)
                            } label: {
                                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                                    .font(.system(size: 16))
                                    .foregroundStyle(isSelected ? WelcomeStyle.accent : WelcomeStyle.faint)
                            }.buttonStyle(.plain)

                            VStack(alignment: .leading, spacing: 2) {
                                Text(opt.displayName).font(.fusionText(12, weight: .semibold)).foregroundStyle(WelcomeStyle.text)
                                Text(opt.reason).font(.fusionText(10)).foregroundStyle(WelcomeStyle.faint)
                            }

                            Spacer()

                            if state == "idle" {
                                Button("Download") {
                                    vm.selectedModels.insert(opt.id)
                                    vm.downloadSelectedModels()
                                }.buttonStyle(.fusion(.normal, size: .small)).font(.fusionText(10))
                            } else if state == "downloading" {
                                ProgressView().controlSize(.small).scaleEffect(0.7)
                            } else if state == "done" {
                                Image(systemName: "checkmark.circle.fill").font(.system(size: 14)).foregroundStyle(.green)
                            } else if state == "error" {
                                Button("Retry") {
                                    vm.modelDownloads[opt.id] = "idle"
                                    vm.selectedModels.insert(opt.id)
                                    vm.downloadSelectedModels()
                                }.buttonStyle(.fusion(.normal, size: .small)).foregroundStyle(.red)
                            }
                        }
                        .padding(.horizontal, 16).padding(.vertical, 6)
                        .background(vm.selectedModels.contains(opt.id) ? WelcomeStyle.accent.opacity(0.04) : .clear)
                        .clipShape(RoundedRectangle(cornerRadius: 6))

                        if opt.id != vm.recommendedModels.last?.id {
                            WelcomeDivider().padding(.leading, 52)
                        }
                    }

                    // Bulk download button
                    if !vm.recommendedModels.isEmpty {
                        HStack(spacing: 12) {
                            Button("Select All") { vm.selectAllModels() }
                                .buttonStyle(.fusion(.plain, size: .small)).font(.fusionText(10))

                            Button {
                                vm.downloadSelectedModels()
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: "arrow.down.circle").font(.system(size: 11))
                                    Text("Download (\(vm.selectedModels.count))")
                                }
                                .font(.fusionText(11, weight: .medium))
                                .foregroundStyle(.white)
                                .padding(.horizontal, 12).padding(.vertical, 5)
                                .background(vm.selectedModels.isEmpty ? WelcomeStyle.faint : WelcomeStyle.accent)
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                            }
                            .buttonStyle(.plain).disabled(vm.selectedModels.isEmpty)

                            Spacer()

                            Text("Start Server below after downloading")
                                .font(.fusionText(9)).foregroundStyle(WelcomeStyle.faint)
                        }
                        .padding(.horizontal, 16).padding(.bottom, 10)
                    }
                }
                .background(WelcomeStyle.panel).clipShape(RoundedRectangle(cornerRadius: 12))
                .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(WelcomeStyle.accent.opacity(0.3), lineWidth: 1))
                .frame(width: 560)

                // Editable parameters
                VStack(alignment: .leading, spacing: 0) {
                    Text("Server & Performance").font(.fusionText(11, weight: .semibold)).foregroundStyle(WelcomeStyle.faint)
                        .padding(.horizontal, 16).padding(.top, 10).padding(.bottom, 4)
                    EditRow(label: "Port", value: Binding(get: { vm.portText }, set: { vm.portText = $0 }), mono: true)
                    WelcomeDivider()
                    EditRow(label: "Max Context", value: Binding(get: { String(vm.editMaxContext) }, set: { if let v = Int($0) { vm.editMaxContext = v; vm.validateRecommendedSettings() } }), mono: true)
                    WelcomeDivider()
                    EditRow(label: "Max Tokens", value: Binding(get: { String(vm.editMaxTokens) }, set: { if let v = Int($0) { vm.editMaxTokens = v; vm.validateRecommendedSettings() } }), mono: true)
                    WelcomeDivider()
                    EditRow(label: "Idle Timeout", value: Binding(get: { "\(vm.editIdleTimeout)s" }, set: { vm.editIdleTimeout = Int($0.dropLast()) ?? Int($0) ?? 300; vm.validateRecommendedSettings() }), mono: true)
                    WelcomeDivider()
                    ToggleRow(label: "SSD Cache", isOn: Binding(get: { vm.editCacheEnabled }, set: { vm.editCacheEnabled = $0 }))
                }
                .background(WelcomeStyle.panel).clipShape(RoundedRectangle(cornerRadius: 12))
                .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(WelcomeStyle.panelBorder.opacity(0.45), lineWidth: 0.5))
                .frame(width: 560)

                // Advanced features
                VStack(alignment: .leading, spacing: 0) {
                    Text("Advanced Features").font(.fusionText(11, weight: .semibold)).foregroundStyle(WelcomeStyle.faint)
                        .padding(.horizontal, 16).padding(.top, 10).padding(.bottom, 4)
                    ToggleRow(label: "DFlash (speculative decoding)", isOn: Binding(get: { vm.editDflash }, set: { vm.editDflash = $0 }), hint: "Faster generation via draft model")
                    WelcomeDivider()
                    ToggleRow(label: "DSpark (distributed推理)", isOn: Binding(get: { vm.editDspark }, set: { vm.editDspark = $0 }), hint: "Multi-node inference")
                    WelcomeDivider()
                    ToggleRow(label: "TurboQuant (fast quantization)", isOn: Binding(get: { vm.editTurboquant }, set: { vm.editTurboquant = $0 }), hint: "On-the-fly quantization for speed")
                }
                .background(WelcomeStyle.panel).clipShape(RoundedRectangle(cornerRadius: 12))
                .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(WelcomeStyle.panelBorder.opacity(0.45), lineWidth: 0.5))
                .frame(width: 560)

                // Validation warning
                if let warn = vm.validationWarning {
                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 12)).foregroundStyle(.orange)
                        Text(warn).font(.fusionText(11)).foregroundStyle(.orange)
                    }.padding(10).frame(width: 560)
                        .background(.orange.opacity(0.08)).clipShape(RoundedRectangle(cornerRadius: 8))
                }
            }
            .padding(.horizontal, 30).padding(.top, 10).padding(.bottom, 4)
        }
    }
}

private struct UseCaseButton: View {
    let title: String; let icon: String; let desc: String
    let isSelected: Bool; let action: () -> Void
    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon).font(.system(size: 16))
                Text(title).font(.fusionText(12, weight: .semibold))
                Text(desc).font(.fusionText(9)).foregroundStyle(WelcomeStyle.faint)
            }.frame(maxWidth: .infinity).padding(.vertical, 10)
                .background(isSelected ? WelcomeStyle.accent.opacity(0.12) : WelcomeStyle.fill)
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(isSelected ? WelcomeStyle.accent.opacity(0.4) : .clear, lineWidth: 1))
        }.buttonStyle(.plain).frame(width: 170)
    }
}

private struct EditRow: View {
    let label: String; let value: Binding<String>; var mono: Bool = false
    var body: some View {
        HStack(spacing: 12) {
            Text(label).font(.fusionText(12, weight: .medium)).foregroundStyle(WelcomeStyle.text).frame(width: 100, alignment: .leading)
            Spacer()
            TextField("", text: value).textFieldStyle(.plain).font(mono ? .fusionMono(11) : .fusionText(12)).foregroundStyle(WelcomeStyle.muted)
                .frame(width: 160).padding(.horizontal, 8).padding(.vertical, 4).background(WelcomeStyle.fill).clipShape(RoundedRectangle(cornerRadius: 6))
        }.padding(.horizontal, 16).padding(.vertical, 8)
    }
}

private struct ToggleRow: View {
    let label: String; let isOn: Binding<Bool>; var hint: String? = nil
    var body: some View {
        HStack(spacing: 12) {
            Text(label).font(.fusionText(12, weight: .medium)).foregroundStyle(WelcomeStyle.text).frame(width: 200, alignment: .leading)
            if let hint { Text(hint).font(.fusionText(9)).foregroundStyle(WelcomeStyle.faint) }
            Spacer()
            Toggle("", isOn: isOn).labelsHidden().controlSize(.small)
        }.padding(.horizontal, 16).padding(.vertical, 8)
    }
}

private struct WelcomeCTA: View {
    let title: String
    var systemImage: String?
    var isBusy: Bool = false
    var width: CGFloat
    let action: () -> Void

    @Environment(\.isEnabled) private var isEnabled

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                if isBusy {
                    ProgressView()
                        .controlSize(.small)
                }
                Text(title)
                    .font(.fusionText(13, weight: .medium))
                    .lineLimit(1)
                    .minimumScaleFactor(0.86)
                if let systemImage {
                    Image(systemName: systemImage)
                        .font(.system(size: 12, weight: .semibold))
                }
            }
            .foregroundStyle(Color.white)
            .frame(width: width, height: 32)
            .background(WelcomeStyle.accent)
            .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
            .opacity(isEnabled ? 1.0 : 0.55)
        }
        .buttonStyle(.plain)
    }
}

private struct WelcomeIcon: View {
    let size: CGFloat

    var body: some View {
        Image("AppLogo")
            .resizable()
            .interpolation(.high)
            .frame(width: size, height: size)
            .clipShape(RoundedRectangle(cornerRadius: size * 0.22, style: .continuous))
            .shadow(color: Color.black.opacity(0.10), radius: 12, y: 6)
            .accessibilityLabel("FusionMLX")
    }
}

private struct FeaturePill: View {
    let icon: String
    let title: String

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .semibold))
            Text(title)
                .font(.fusionText(12, weight: .medium))
        }
        .foregroundStyle(WelcomeStyle.muted)
        .padding(.horizontal, 11)
        .padding(.vertical, 7)
        .background(WelcomeStyle.fill)
        .clipShape(Capsule())
    }
}

private struct WelcomeNotice: View {
    let icon: String
    let title: String
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(WelcomeStyle.accent)
                .frame(width: 24, height: 24)
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.fusionText(13, weight: .semibold))
                    .foregroundStyle(WelcomeStyle.text)
                Text(text)
                    .font(.fusionText(12))
                    .foregroundStyle(WelcomeStyle.muted)
                    .lineSpacing(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(14)
        .background(WelcomeStyle.fill)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}

private struct SettingRow<Content: View>: View {
    let icon: String
    let title: String
    let subtitle: String
    let content: () -> Content

    init(
        icon: String,
        title: String,
        subtitle: String,
        @ViewBuilder content: @escaping () -> Content
    ) {
        self.icon = icon
        self.title = title
        self.subtitle = subtitle
        self.content = content
    }

    var body: some View {
        HStack(alignment: .center, spacing: 14) {
            Image(systemName: icon)
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(WelcomeStyle.muted)
                .frame(width: 22)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.fusionText(13, weight: .medium))
                    .foregroundStyle(WelcomeStyle.text)
                Text(subtitle)
                    .font(.fusionText(11.5))
                    .foregroundStyle(WelcomeStyle.faint)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 16)
            content()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 13)
    }
}

private struct WelcomeDivider: View {
    var body: some View {
        Rectangle()
            .fill(WelcomeStyle.panelBorder.opacity(0.42))
            .frame(height: 0.5)
            .padding(.leading, 52)
    }
}
