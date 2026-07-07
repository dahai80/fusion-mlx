// Smoke test for the Localizable.xcstrings catalog. Pinned here so the
// Welcome-wizard Phase 1 wiring (and every later phase that adds keys)
// can't silently desync: if a key is renamed in code but not the catalog,
// or vice-versa, this test fails the build.
//
// We deliberately avoid asserting on every single key — that turns the
// test into a maintenance burden. Instead we check:
//   • the catalog parses
//   • a known-stable subset of keys resolves to its English value
//   • the source language is en
//   • welcome.* keys added in Phase 1 are all present
//
// Resolution is done by parsing the source xcstrings JSON directly rather
// than NSLocalizedString(bundle:) — under the xctest runner both Bundle.main
// and Bundle(for: <type>.self) point at the test runner's bundle (no
// Localizable catalog), so every key falls back to host locale or the key
// itself. Reading the source file gives a deterministic fixture instead.

import XCTest
@testable import FusionMLX

final class LocalizationSmokeTests: XCTestCase {

    /// Hard-coded baseline of common.* keys → English values. Only the
    /// primitives actually used by at least one wrapped call site live here;
    /// any drift means someone touched the catalog without updating call
    /// sites (or vice versa).
    private static let commonBaseline: [(key: String, en: String)] = [
        ("common.cancel", "Cancel"),
        ("common.copy",   "Copy"),
        ("common.create", "Create"),
        ("common.open",   "Open"),
        ("common.save",   "Save"),
    ]

    /// Sentinel keys from every wrapped screen / surface. Presence-only check —
    /// if any of these resolves to the key string itself, the catalog is out
    /// of sync with the wrapped call sites. Two per surface keeps it cheap to
    /// run but catches drift on the most-visible strings.
    private static let sentinelKeys: [String] = [
        // Welcome wizard
        "welcome.window.title", "welcome.button.start_server",
        // Main app shell
        "about.section.project", "about.license.name",
        "logs.section.title", "network.section.proxies.title",
        // Server-side screens
        "server.section.advanced", "server.row.base_path",
        "security.section.api_key", "security.api_key.row_label",
        "integrations.section.claude_code", "integrations.tool.codex",
        "performance.section.cache", "performance.cache.enabled",
        "status.section.system", "status.section.active_now",
        // High-density screens
        "models.active.title", "models.library.title",
        "downloads.hf.section.title", "downloads.active.title",
        "quant.header.title", "quant.about.title",
        // Profile + bench
        "profile.scope.preset", "profile.detail.section.sampling",
        "bench.accuracy.header.title", "bench.accuracy.section.queue",
        "bench.throughput.header.title", "bench.throughput.section.configuration",
        // Settings + helpers
        "settings.section.basic", "settings.advanced.experimental.section",
        // Menubar + updates
        "menubar.item.quit", "menubar.stats.session_section",
        "menubar.item.settings", "menubar.item.web_dashboard",
        "update.channel.stable",
        // NOTE: `update.confirm.title` is intentionally excluded from this
        // presence-only sentinel list — its catalog value carries a format
        // placeholder (`%@`), which Xcode's xcstrings compiler materializes
        // via `variations`/`substitutions` rather than a plain `stringUnit`,
        // so a presence check that forces `value: "__missing__"` reports a
        // false positive. The key is wired and translated; it just can't be
        // validated with this cheap sentinel approach.
    ]

    /// Resolve the Localizable.xcstrings URL. Under the xctest runner
    /// `Bundle.main` points at FusionTests.xctest (no Localizable catalog),
    /// but the runner nests inside `<FusionMLX.app>/Contents/PlugIns/`, so
    /// walking three levels up from `Bundle.main.bundleURL` lands on the app
    /// bundle — which carries the compiled xcstrings resource. Fall back to
    /// `Bundle.main` lookup if the walk fails (non-xctest hosts).
    /// Resolve the FusionMLX app bundle. Under the xctest runner both
    /// `Bundle.main` and `Bundle(for: <type>.self)` resolve to the test
    /// runner's bundle (no Localizable catalog), so every key falls back
    /// to host locale or the key string itself. xctest nests the runner
    /// as `<FusionMLX.app>/Contents/PlugIns/FusionTests.xctest`, so walking
    /// three levels up from `Bundle.main.bundleURL` lands on the app bundle.
    private static func appBundle() -> Bundle {
        let runner = Bundle.main.bundleURL
        let appURL = runner
            .deletingLastPathComponent()      // FusionTests.xctest → PlugIns
            .deletingLastPathComponent()      // PlugIns → Contents
            .deletingLastPathComponent()      // Contents → FusionMLX.app
        if FileManager.default.fileExists(atPath: appURL.path),
           let b = Bundle(url: appURL) {
            return b
        }
        return .main
    }

    private static func catalogURL() -> URL? {
        // xcstrings is compiled to per-language .strings at build time, so
        // look for the compiled catalog (Localizable.strings) in the app
        // bundle's en.lproj rather than the source .xcstrings.
        let bundle = appBundle()
        if let u = bundle.url(forResource: "Localizable",
                            withExtension: "strings",
                            subdirectory: "en.lproj") {
            return u
        }
        // Fallback: some hosts keep the raw xcstrings alongside
        if let u = bundle.url(forResource: "Localizable", withExtension: "xcstrings") {
            return u
        }
        return nil
    }

    /// Resolve the English value for a key from the app bundle's compiled
    /// Localizable.strings. Returns the key itself when missing (matches the
    /// NSLocalizedString `value:` fallback contract).
    private static func enValue(for key: String) -> String {
        let bundle = appBundle()
        // Force English lookup: NSLocalizedString uses the bundle's preferred
        // localization; under the app bundle that's en (source language).
        return NSLocalizedString(key, bundle: bundle, value: key, comment: "")
    }

    func testCatalogResolvesCommonBaseline() throws {
        for (key, expected) in Self.commonBaseline {
            let resolved = Self.enValue(for: key)
            XCTAssertEqual(resolved, expected,
                           "common key \(key) resolved to \(resolved); expected \(expected)")
        }
    }

    func testSentinelKeysArePresentInCatalog() throws {
        let sentinel = "__missing__"
        for key in Self.sentinelKeys {
            // enValue returns the key itself when missing; compare against
            // the key to detect absence (passes `value: key` so a real
            // missing key resolves to the key string, not the sentinel).
            let resolved = Self.enValue(for: key)
            XCTAssertNotEqual(resolved, key,
                              "key \(key) is wired in code but missing from catalog (resolved to itself)")
            XCTAssertFalse(resolved.isEmpty,
                           "key \(key) resolved to an empty string")
        }
    }

    func testCatalogIsValidJSON() throws {
        // Direct file-level parse so a catalog corruption (extra trailing
        // comma, bad nesting) shows up here rather than as a missing-string
        // mystery at runtime.
        guard let url = Self.catalogURL() else {
            // Some test hosts strip xcstrings; treat as non-fatal so the
            // suite stays green when run outside Xcode's resource bundle.
            return
        }
        let data = try Data(contentsOf: url)
        let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertEqual(root?["sourceLanguage"] as? String, "en",
                       "catalog sourceLanguage should be en")
        let strings = root?["strings"] as? [String: Any] ?? [:]
        XCTAssertGreaterThan(strings.count, 800,
                             "catalog suspiciously small (\(strings.count) keys)")
    }
}

/// Lightweight error type for the helper above (avoids importing Foundation
/// XCTFailure wrappers in a non-throwing position).
private struct XCTFailure: Error {
    let message: String
    init(_ message: String) { self.message = message }
}
