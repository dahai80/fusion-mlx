// ShellEnvWriter owns the app-managed CLI shim.
//
// Default launch behavior must not edit shell rc files. The app first creates
// `~/.fusion-mlx/bin/fusion`, then tries to expose it through a safe public symlink.
// Shell rc edits are kept behind an explicit user prompt only.

import Foundation

enum ShellEnvWriter {
    static let variableName = "Fusion_BASE_PATH"
    nonisolated(unsafe) static var homeOverrideForTests: URL?
    nonisolated(unsafe) static var shellOverrideForTests: String?
    nonisolated(unsafe) static var publicBinDirsOverrideForTests: [URL]?
    nonisolated(unsafe) static var cliPathPrefsURLOverrideForTests: URL?

    enum CLISetupResult: Equatable {
        case publicCommandReady(path: String)
        case needsShellPathPrompt(reason: String)
    }

    private enum WriterError: LocalizedError {
        case cliWrapperNotExecutable(String)

        var errorDescription: String? {
            switch self {
            case .cliWrapperNotExecutable(let path):
                return "App-bundle CLI wrapper is not executable: \(path)"
            }
        }
    }

    private static let cliShimBeginMarker = "# FusionMLX: CLI shim path begin"
    private static let cliShimEndMarker = "# FusionMLX: CLI shim path end"
    // #1733: marker stamped inside app-managed shims. A shim without this
    // marker is user-managed and is never rewritten by the app.
    private static let shimMarker = "# FusionMLX managed shim"
    private static let publicBinCandidates = [
        "/opt/homebrew/bin",
        "/usr/local/bin",
    ]

    /// Install/update `~/.fusion-mlx/bin/fusion` so app-only installs still expose
    /// the same terminal command as pip/Homebrew installs.
    @discardableResult
    static func ensureCLIShim(appBundleURL: URL = Bundle.main.bundleURL) throws -> CLISetupResult {
        let shimDir = home()
            .appendingPathComponent(".fusion-mlx", isDirectory: true)
            .appendingPathComponent("bin", isDirectory: true)
        try FileManager.default.createDirectory(
            at: shimDir,
            withIntermediateDirectories: true
        )

        let bundleCLI = appBundleURL
            .appendingPathComponent("Contents", isDirectory: true)
            .appendingPathComponent("MacOS", isDirectory: true)
            .appendingPathComponent("fusion-cli")
        guard FileManager.default.isExecutableFile(atPath: bundleCLI.path) else {
            throw WriterError.cliWrapperNotExecutable(bundleCLI.path)
        }
        // #1733: a shim that resolves into a mounted DMG dies on eject.
        // When the bundle lives on an ephemeral volume, exec a durable
        // copy under ~/.fusion-mlx/bin instead of the bundle path.
        let execTarget = try durableCLIExecutable(bundleCLI: bundleCLI, shimDir: shimDir)
        let shimURL = shimDir.appendingPathComponent("fusion")
        let stamp = "\(shimMarker) target=\(execTarget.path) bundle-mtime=\(bundleMTimes(bundleCLI))"
        let script = """
        #!/bin/sh
        \(stamp)
        BOOTSTRAP="$HOME/Library/Application Support/FusionMLX/base-path"
        if [ -r "$BOOTSTRAP" ]; then
            IFS= read -r \(variableName) < "$BOOTSTRAP" || \(variableName)=""
            if [ -n "$\(variableName)" ]; then
                export \(variableName)
            fi
        fi
        exec \(shellQuote(execTarget.path)) "$@"
        """
        // #1733: skip the write when an existing shim is user-managed (no
        // marker) or already current (same target + bundle mtime), so a
        // relaunch never resurrects a path the user fixed manually.
        if let existing = try? String(contentsOf: shimURL, encoding: .utf8) {
            let isCurrent = existing.contains(stamp)
            let isUserManaged = !existing.contains(shimMarker)
            if isCurrent || isUserManaged {
                return try finishCLIShimSetup(shimURL: shimURL)
            }
        }
        try script.write(to: shimURL, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: shimURL.path
        )

        return try finishCLIShimSetup(shimURL: shimURL)
    }

    /// PATH/symlink resolution after the shim itself is in place.
    private static func finishCLIShimSetup(shimURL: URL) throws -> CLISetupResult {
        if let path = firstCLIPathInCurrentPath() {
            if isManagedCLI(path: path, shimURL: shimURL) {
                return .publicCommandReady(path: path.path)
            }
            // #1733: a standalone `fusion` binary already resolves on PATH
            // and is not ours — do not take the name over.
            return .needsShellPathPrompt(
                reason:
                    "\(path.path) already exists on PATH as a standalone fusion command; FusionMLX will not take it over."
            )
        }

        let symlinkResult = ensurePublicSymlink(to: shimURL)

        switch symlinkResult {
        case .installed(let path):
            if let first = firstCLIPathInCurrentPath(),
               !isManagedCLI(path: first, shimURL: shimURL) {
                return .needsShellPathPrompt(
                    reason: "\(first.path) appears before the FusionMLX app-managed command on PATH."
                )
            }
            return .publicCommandReady(path: path.path)
        case .failed(let reasons):
            return .needsShellPathPrompt(reason: reasons.joined(separator: "\n"))
        }
    }

    static func shouldSuppressCLIPathPrompt() -> Bool {
        readCLIPathPrefs().suppressShellPathPrompt
    }

    static func suppressCLIPathPromptForever() {
        var prefs = readCLIPathPrefs()
        prefs.suppressShellPathPrompt = true
        writeCLIPathPrefs(prefs)
    }

    static func ensureShellPathExport() throws {
        try ensureCLIPathExport()
    }

    // MARK: - File targets

    /// #1733: resolve the executable the shim should exec. A bundle under
    /// /Volumes (mounted DMG) or on an unwritable volume is ephemeral — copy
    /// fusion-cli to ~/.fusion-mlx/bin/fusion-cli and exec that durable copy
    /// instead. The copy is refreshed only when the bundled binary is newer.
    static func durableCLIExecutable(bundleCLI: URL, shimDir: URL) throws -> URL {
        let fm = FileManager.default
        let bundlePath = bundleCLI.path
        let ephemeral =
            bundlePath.hasPrefix("/Volumes/")
            || !fm.isWritableFile(atPath: bundleCLI.deletingLastPathComponent().path)
        if !ephemeral {
            return bundleCLI
        }
        let durableCopy = shimDir.appendingPathComponent("fusion-cli")
        let bundledMTime = bundleMTimes(bundleCLI)
        if fm.fileExists(atPath: durableCopy.path) {
            let copyMTime = fileMTimes(durableCopy)
            // Refresh only when the bundled CLI is newer than the copy.
            if bundledMTime <= copyMTime {
                return durableCopy
            }
            try fm.removeItem(at: durableCopy)
        }
        try fm.copyItem(at: bundleCLI, to: durableCopy)
        try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: durableCopy.path)
        return durableCopy
    }

    private static func bundleMTimes(_ url: URL) -> Double {
        fileMTimes(url)
    }

    private static func fileMTimes(_ url: URL) -> Double {
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        let date = attrs?[.modificationDate] as? Date
        return date?.timeIntervalSince1970 ?? 0
    }

    private static func home() -> URL {
        homeOverrideForTests ?? FileManager.default.homeDirectoryForCurrentUser
    }

    private static func candidateFiles() -> [URL] {
        let names = [
            ".zshrc", ".zprofile", ".zshenv",
            ".bashrc", ".bash_profile", ".profile",
        ]
        return names.map { home().appendingPathComponent($0) }
    }

    /// Prefer the rc file matching the user's `$SHELL`. zsh users get
    /// `.zshrc`, bash users get `.bashrc`, anyone else falls back below.
    private static func primaryFile() -> URL? {
        let shell = shellOverrideForTests ?? ProcessInfo.processInfo.environment["SHELL"] ?? ""
        if shell.contains("zsh") {
            return home().appendingPathComponent(".zshrc")
        }
        if shell.contains("bash") {
            // On macOS, GUI-launched terminals run login shells, so
            // .bash_profile is what gets sourced. Prefer it when it
            // already exists.
            let profile = home().appendingPathComponent(".bash_profile")
            if FileManager.default.fileExists(atPath: profile.path) {
                return profile
            }
            return home().appendingPathComponent(".bashrc")
        }
        return nil
    }

    // MARK: - File mutation

    private enum PublicSymlinkResult {
        case installed(URL)
        case failed([String])
    }

    private struct CLIPathPrefs: Codable {
        var suppressShellPathPrompt: Bool = false
    }

    private static func ensurePublicSymlink(to shimURL: URL) -> PublicSymlinkResult {
        let fm = FileManager.default
        var reasons: [String] = []

        for dir in publicBinDirs() {
            guard fm.fileExists(atPath: dir.path) else {
                reasons.append("\(dir.path) does not exist.")
                continue
            }

            let link = dir.appendingPathComponent("fusion")
            if fm.fileExists(atPath: link.path) {
                if isManagedCLI(path: link, shimURL: shimURL) {
                    return .installed(link)
                }
                reasons.append("\(link.path) already exists and is not managed by FusionMLX.")
                continue
            }
            guard fm.isWritableFile(atPath: dir.path) else {
                reasons.append("\(dir.path) is not writable.")
                continue
            }

            do {
                try fm.createSymbolicLink(at: link, withDestinationURL: shimURL)
                return .installed(link)
            } catch {
                reasons.append("Failed to create \(link.path): \(error.localizedDescription)")
            }
        }

        if reasons.isEmpty {
            reasons.append("No writable public PATH directory was available.")
        }
        return .failed(reasons)
    }

    private static func publicBinDirs() -> [URL] {
        if let override = publicBinDirsOverrideForTests {
            return override
        }

        var seen = Set<String>()
        var dirs: [URL] = []
        let pathParts = (getenv("PATH").map { String(cString: $0) } ?? "")
            .split(separator: ":")
            .map(String.init)

        for path in pathParts where publicBinCandidates.contains(path) {
            if seen.insert(path).inserted {
                dirs.append(URL(fileURLWithPath: path, isDirectory: true))
            }
        }
        for path in publicBinCandidates where seen.insert(path).inserted {
            dirs.append(URL(fileURLWithPath: path, isDirectory: true))
        }
        return dirs
    }

    private static func firstCLIPathInCurrentPath() -> URL? {
        let current = getenv("PATH").map { String(cString: $0) } ?? ""
        for part in current.split(separator: ":").map(String.init) {
            let candidate = URL(fileURLWithPath: part, isDirectory: true)
                .appendingPathComponent("fusion")
            guard FileManager.default.isExecutableFile(atPath: candidate.path) else {
                continue
            }
            return candidate
        }
        return nil
    }

    private static func isManagedCLI(path: URL, shimURL: URL) -> Bool {
        path.resolvingSymlinksInPath().standardizedFileURL.path
            == shimURL.resolvingSymlinksInPath().standardizedFileURL.path
    }

    private static func cliPathPrefsURL() -> URL {
        cliPathPrefsURLOverrideForTests
            ?? AppConfig.appSupportURL().appendingPathComponent("cli-path-prefs.json")
    }

    private static func readCLIPathPrefs() -> CLIPathPrefs {
        let url = cliPathPrefsURL()
        guard let data = try? Data(contentsOf: url),
              let prefs = try? JSONDecoder().decode(CLIPathPrefs.self, from: data)
        else {
            return CLIPathPrefs()
        }
        return prefs
    }

    private static func writeCLIPathPrefs(_ prefs: CLIPathPrefs) {
        let url = cliPathPrefsURL()
        guard let data = try? JSONEncoder().encode(prefs) else { return }
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try? data.write(to: url, options: [.atomic])
    }

    private static func ensureCLIPathExport() throws {
        let files = candidateFiles()
        for url in files where FileManager.default.fileExists(atPath: url.path) {
            let raw = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
            if raw.contains(cliShimBeginMarker) {
                return
            }
        }

        let target = primaryFile() ?? files.first(where: {
            FileManager.default.fileExists(atPath: $0.path)
        }) ?? files.first!
        let block = """
        \(cliShimBeginMarker)
        case ":$PATH:" in
          *":$HOME/.fusion-mlx/bin:"*) ;;
          *) export PATH="$HOME/.fusion-mlx/bin:$PATH" ;;
        esac
        \(cliShimEndMarker)
        """
        try appendRawBlock(to: target, block: block)
    }

    private static func appendRawBlock(to url: URL, block: String) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        var existing = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
        if !existing.hasSuffix("\n"), !existing.isEmpty {
            existing.append("\n")
        }
        existing.append(block)
        if !existing.hasSuffix("\n") {
            existing.append("\n")
        }
        try existing.write(to: url, atomically: true, encoding: .utf8)
    }

    private static func shellQuote(_ value: String) -> String {
        if value.isEmpty { return "''" }
        return "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }
}
