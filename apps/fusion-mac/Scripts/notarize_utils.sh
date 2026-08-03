#!/usr/bin/env bash
# notarize_utils.sh — Developer ID signing + Apple notarization helpers.
#
# Sourced by package_dmg.sh. Provides:
#   - _detect_signing_identity(): find Developer ID or fallback ad-hoc
#   - _sign_app_bundle(): sign with Developer ID (+ hardened runtime) or ad-hoc
#   - _notarize_dmg(): submit DMG for Apple notarization + staple
#
# Environment variables (optional, for CI):
#   APPLE_DEVELOPER_ID     — "Developer ID Application: Name (TEAMID)"
#   APPLE_TEAM_ID          — short team ID (10-char)
#   APPLE_ID               — Apple ID email for notarization
#   APPLE_APP_PASSWORD     — app-specific password
#   APPLE_KEYCHAIN_PROFILE — notarytool keychain profile (alternative to APPLE_ID+PASSWORD)
#
# When no Developer ID certificate is found, all functions fall back to
# ad-hoc signing (codesign --sign -) and notarization is skipped.

set -euo pipefail

# --- Detection ---

# Print the best available signing identity, or "-" for ad-hoc.
_detect_signing_identity() {
    # 1. Explicit env var wins
    if [ -n "${APPLE_DEVELOPER_ID:-}" ]; then
        echo "$APPLE_DEVELOPER_ID"
        return 0
    fi

    # 2. Search keychain for "Developer ID Application"
    local identity
    identity=$(security find-identity -v -p codesigning 2>/dev/null \
        | grep "Developer ID Application" \
        | head -1 \
        | sed -E 's/.*"(.*)".*/\1/' || true)

    if [ -n "$identity" ]; then
        echo "$identity"
        return 0
    fi

    # 3. Fallback: ad-hoc
    echo "-"
}

# Returns 0 if a Developer ID identity is available, 1 otherwise.
_has_developer_id() {
    local id
    id=$(_detect_signing_identity)
    [ "$id" != "-" ]
}

# Extract team ID from signing identity or env var.
_team_id() {
    if [ -n "${APPLE_TEAM_ID:-}" ]; then
        echo "$APPLE_TEAM_ID"
        return 0
    fi
    # Parse from identity string: "Developer ID Application: Name (TEAMID)"
    local id
    id=$(_detect_signing_identity)
    echo "$id" | sed -E 's/.*\(([A-Z0-9]+)\).*/\1/' || true
}

# --- Signing ---

# Sign the app bundle with the best available identity.
# When Developer ID is available, enables hardened runtime + entitlements.
_sign_app_bundle() {
    local app_bundle="$1"
    local python_dir="$2"
    local identity
    identity=$(_detect_signing_identity)

    if [ "$identity" = "-" ]; then
        log "No Developer ID found — using ad-hoc signing"
        # Sign embedded Mach-O files
        _sign_embedded_mach_o_files "$python_dir"
        codesign --force --sign - "$app_bundle/Contents/MacOS/fusion-cli" >/dev/null 2>&1 || true
        _verify_embedded_signatures "$python_dir"
        # Flat seal
        codesign --force --sign - "$app_bundle"
        xattr -dr com.apple.quarantine "$app_bundle" 2>/dev/null || true
        ok "Ad-hoc signed"
        return 0
    fi

    local team_id
    team_id=$(_team_id)
    local entitlements="$app_bundle/Contents/Resources/FusionMLX.entitlements"
    # If entitlements don't exist at this path, use the source copy
    if [ ! -f "$entitlements" ]; then
        local script_dir
        script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        entitlements="$script_dir/../Resources/FusionMLX.entitlements"
    fi

    log "Developer ID signing: $identity"
    # Sign embedded Mach-O files (these don't need hardened runtime)
    _sign_embedded_mach_o_files_developer_id "$python_dir" "$identity"
    codesign --force --sign "$identity" "$app_bundle/Contents/MacOS/fusion-cli" >/dev/null 2>&1 || true
    _verify_embedded_signatures "$python_dir"

    # Sign app bundle with hardened runtime + entitlements
    local sign_args=(--force --sign "$identity" --options runtime)
    if [ -n "$team_id" ]; then
        sign_args+=(--timestamp)
    fi
    if [ -f "$entitlements" ]; then
        sign_args+=(--entitlements "$entitlements")
        log "Using entitlements: $entitlements"
    fi
    codesign "${sign_args[@]}" "$app_bundle"
    ok "Developer ID signed"
}

# Sign embedded Mach-O with Developer ID (no hardened runtime for libraries).
_sign_embedded_mach_o_files_developer_id() {
    local root="$1"
    local identity="$2"
    local count=0
    local failed=0
    local path

    while IFS= read -r -d '' path; do
        _is_mach_o_file "$path" || continue
        if codesign --force --sign "$identity" "$path" >/dev/null 2>&1; then
            count=$((count + 1))
        else
            failed=$((failed + 1))
            warn "  ! codesign failed: ${path#"$root"/}"
        fi
    done < <(
        find "$root" \
            \( -path "*/.dSYM/*" -o -path "*/__pycache__/*" \) -prune -o \
            -type f \( \
                -name "*.so" -o \
                -name "*.dylib" -o \
                -name "*.bundle" -o \
                -perm -100 -o \
                -perm -010 -o \
                -perm -001 \
            \) -print0
    )

    if [ "$failed" -gt 0 ]; then
        die "$failed embedded Mach-O file(s) failed to codesign"
    fi
    ok "  + signed $count embedded Mach-O files (Developer ID)"
}

# --- Notarization ---

# Notarize a DMG file. Requires Developer ID + Apple credentials.
# Falls back to a warning if credentials are missing.
_notarize_dmg() {
    local dmg_path="$1"

    if ! _has_developer_id; then
        warn "No Developer ID — skipping notarization"
        warn "Users will need: xattr -cr /Applications/FusionMLX.app"
        return 0
    fi

    # Check for notarization credentials
    local has_creds=0
    if [ -n "${APPLE_KEYCHAIN_PROFILE:-}" ]; then
        has_creds=1
    elif [ -n "${APPLE_ID:-}" ] && [ -n "${APPLE_APP_PASSWORD:-}" ]; then
        has_creds=1
    fi

    if [ "$has_creds" -eq 0 ]; then
        warn "Developer ID found but no notarization credentials"
        warn "Set APPLE_KEYCHAIN_PROFILE or APPLE_ID+APPLE_APP_PASSWORD to enable notarization"
        warn "Users will need: xattr -cr /Applications/FusionMLX.app"
        return 0
    fi

    log "Submitting DMG for notarization…"
    local submit_uuid

    if [ -n "${APPLE_KEYCHAIN_PROFILE:-}" ]; then
        submit_uuid=$(xcrun notarytool submit "$dmg_path" \
            --keychain-profile "$APPLE_KEYCHAIN_PROFILE" \
            --wait \
            --timeout 30m 2>&1 | tee /dev/stderr | grep -E 'id: [0-9a-f-]+' | head -1 | awk '{print $2}' || true)
    else
        local team_id
        team_id=$(_team_id)
        submit_uuid=$(xcrun notarytool submit "$dmg_path" \
            --apple-id "$APPLE_ID" \
            --password "$APPLE_APP_PASSWORD" \
            --team-id "${team_id:-}" \
            --wait \
            --timeout 30m 2>&1 | tee /dev/stderr | grep -E 'id: [0-9a-f-]+' | head -1 | awk '{print $2}' || true)
    fi

    if [ -z "$submit_uuid" ]; then
        warn "Notarization submission failed — DMG is signed but not notarized"
        warn "Users will need: xattr -cr /Applications/FusionMLX.app"
        return 0
    fi

    # Staple the notarization ticket
    log "Stapling notarization ticket…"
    xcrun stapler staple "$dmg_path" 2>&1 || warn "Staple failed (non-fatal)"

    ok "Notarized + stapled: $dmg_path"
}
