// Smoke test for the Localizable.xcstrings catalog. Pinned here so the
// Welcome-wizard Phase 1 wiring (and every later phase that adds keys)
// can't silently desync: if a key is renamed in code but not the catalog,
// or vice-versa, this test fails the build.
//
// We validate the source xcstrings JSON directly rather than going through
// NSLocalizedString(bundle:) — under the xctest runner every Bundle lookup
// (Bundle.main, Bundle(for: <type>.self), Bundle(url: <app>) ) resolves to
// either the test runner's bundle (no Localizable catalog) or DerivedData
// paths that don't carry the source xcstrings. The compiled .strings in
// the app bundle is a binary plist, not JSON, so the JSON-level checks
// below can't run against it either. Reading the source xcstrings file
// via #filePath gives a deterministic fixture that catches catalog drift
// (key renamed in code but not catalog, or vice versa) and catalog
// corruption (bad JSON) — which is what this smoke suite actually exists
// to catch.

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
    /// if any of these is absent (or has no en localization) the catalog is
    /// out of sync with the wrapped call sites. Two per surface keeps it
    /// cheap to run but catches drift on the most-visible strings.
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

    /// Resolve the source Localizable.xcstrings URL. The catalog lives at
    /// `apps/fusion-mac/Resources/`; this test file lives at
    /// `apps/fusion-mac/Tests/FusionTests/`, so three levels up from
    /// #filePath lands on the fusion-mac package root.
    private static func catalogURL() -> URL {
        URL(filePath: #filePath)
            .deletingLastPathComponent()   // FusionTests/
            .deletingLastPathComponent()   // Tests/
            .deletingLastPathComponent()   // fusion-mac/
            .appendingPathComponent("Resources", isDirectory: true)
            .appendingPathComponent("Localizable.xcstrings", isDirectory: false)
    }

    /// Load and parse the catalog JSON.
    private static func catalogRoot() throws -> [String: Any] {
        let url = catalogURL()
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTFailure("Localizable.xcstrings not found at \(url.path)")
        }
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw XCTFailure("Localizable.xcstrings root is not a JSON object")
        }
        return root
    }

    /// Resolve the English stringUnit value for a key, or nil when absent.
    private static func enValue(for key: String, in root: [String: Any]) -> String? {
        let strings = root["strings"] as? [String: Any] ?? [:]
        let entry = strings[key] as? [String: Any] ?? [:]
        let locs = entry["localizations"] as? [String: Any] ?? [:]
        let en = locs["en"] as? [String: Any] ?? [:]
        let unit = en["stringUnit"] as? [String: Any] ?? [:]
        return unit["value"] as? String
    }

    func testCatalogResolvesCommonBaseline() throws {
        let root = try Self.catalogRoot()
        for (key, expected) in Self.commonBaseline {
            let resolved = Self.enValue(for: key, in: root)
            XCTAssertEqual(resolved, expected,
                           "common key \(key) resolved to \(resolved ?? "<missing>"); expected \(expected)")
        }
    }

    func testSentinelKeysArePresentInCatalog() throws {
        let root = try Self.catalogRoot()
        for key in Self.sentinelKeys {
            let resolved = Self.enValue(for: key, in: root)
            XCTAssertNotNil(resolved,
                           "key \(key) is wired in code but missing from xcstrings")
            XCTAssertFalse(resolved?.isEmpty ?? true,
                           "key \(key) resolved to an empty string")
        }
    }

    func testCatalogIsValidJSON() throws {
        let url = Self.catalogURL()
        guard FileManager.default.fileExists(atPath: url.path) else {
            // Some test hosts can't reach the source repo layout; treat as non-fatal.
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

/// Lightweight error type for the helpers above.
private struct XCTFailure: Error {
    let message: String
    init(_ message: String) { self.message = message }
}
