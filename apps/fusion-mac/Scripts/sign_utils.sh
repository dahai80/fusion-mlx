#!/usr/bin/env bash
# sign_utils.sh - shared ad-hoc code signing for the macOS app bundle.
#
# Sourced by build.sh and package_dmg.sh. The bundle ships hundreds of
# nested Mach-O files (mlx / numpy / transformers .so + .dylib) that MUST
# each carry a valid ad-hoc signature: macOS SIGKILLs the server child
# (exit 9, "Code Signature Invalid" / "Invalid Page") the moment a stale-
# signed .so is dlopen'd - which is exactly the launch crash this fixes.
#
# `codesign --deep` is deprecated and does NOT reliably re-sign nested .so
# (it leaves stale page hashes), so we sign each Mach-O explicitly.
#
# Depends on caller-defined helpers: ok(), warn(), die().

_is_mach_o_file() {
    local path="$1"
    local type
    type="$(file -b "$path" 2>/dev/null || true)"
    [[ "$type" == *"Mach-O"* ]]
}

# Sign every embedded Mach-O under $1 (the Python layers dir). Idempotent.
_sign_embedded_mach_o_files() {
    local root="$1"
    local count=0
    local failed=0
    local path

    while IFS= read -r -d '' path; do
        _is_mach_o_file "$path" || continue
        if codesign --force --sign - "$path" >/dev/null 2>&1; then
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
        die "$failed embedded Mach-O file(s) failed to codesign - a stale sig would SIGKILL (exit 9) on launch"
    fi
    ok "  + signed $count embedded Mach-O files"
}

# Spot-check mlx.core - the first native module the server loads and the
# one that crashes on a stale signature. Die loud so a broken bundle never
# ships (Rule 12): a verify failure here means the server would SIGKILL.
_verify_embedded_signatures() {
    local site="$1"
    local mlx_core
    mlx_core=$(find "$site/mlx" -name 'core.cpython-*.so' -print -quit 2>/dev/null || true)
    if [ -z "$mlx_core" ]; then
        warn "  ! mlx.core .so not found under $site/mlx - skipping verify"
        return 0
    fi
    if ! codesign --verify "$mlx_core" >/dev/null 2>&1; then
        die "signature verify failed for $mlx_core - server would SIGKILL (exit 9) on startup"
    fi
    ok "  + verified mlx.core signature"
}
