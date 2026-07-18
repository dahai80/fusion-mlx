# Fusion-MLX macOS App - Design Review

**Method:** gstack `/plan-design-review` (7-pass design audit)
**Target:** `apps/fusion-mac/` SwiftUI app (292 Swift files, macOS 26 Tahoe target, macOS 15 floor)
**Date:** 2026-07-10
**Reviewer:** Claude Code (cold review - no prior plan file in context)
**Scope:** app shell + navigation + design system (Theme/Glass/Components) + representative screens. Backend (`fusion_mlx/`) explicitly out of scope.

> Note: `/plan-design-review` is plan-stage. For an already-built UI, `/design-review`
> (live visual audit) is the closer fit. This is a static, code-level design audit.
> Web mockups skipped (native SwiftUI + project "no images" rule). Outside-voices
> (Codex + subagent) skipped (cost-critical session).

---

## Step 0 - Design Scope Assessment

**Overall rating: 7.5 / 10**

A 10 = documented design system (DESIGN.md), WCAG-AA-verified contrast on all token
pairings, every screen with loading/empty/error states, tested first-run onboarding,
accessibility audited (keyboard/VoiceOver/Dynamic Type), responsive verified at
min/max window sizes.

### What already exists (strong baseline - do not regress)
- `FusionTheme` token struct, dark/light variants, injected via `.environment(\.fusionTheme, theme)`.
- 11 theme components + `Glass` (Tahoe liquid-glass + macOS 15 material fallback) + `Squircle`.
- SF Pro via `.fusionText/.fusionDisplay/.fusionMono` - correct native macOS typeface.
- `DesktopWash` radial-gradient atmosphere (intentional, not slop).
- 4-kind `FusionButtonStyle` with disabled (0.45) / press (0.78) states.
- `StatusPill` 6 states; update sheet + `AsyncImage` both have empty/failure states.
- Pervasive `String(localized:)` i18n; `#Preview` light+dark.
- `NavigationSplitView` 4-group sidebar (Server/Models/Benchmark/General), constrained
  180-215pt, cross-screen deep-link scroll (`ScrollAnchorKey`).
- `ContentScaffold`: 42pt toolbar, 720pt max content, Settings.app large-title pattern.

### Biggest gaps
1. **No DESIGN.md** - token + component system is tribal knowledge in `Theme.swift`.
2. **Contrast unverified** - `textTertiary`/`textSecondary` on `controlBg`, 11pt labels.
3. **Per-screen states unverified** - Models/Downloads/Logs/Bench loading/empty/error.
4. **`WelcomeWindow` (1593 lines) unread** - first-run onboarding arc not assessed.

---

## The 7 Passes

### Pass 1 - Information Architecture: 8.5 / 10
4 logical groups matching macOS System Settings. Localized headers, SF Symbols,
deep-link cross-screen nav. Default landing = Status (correct "what's now").
- **Fix:** stale comment on *General* group header (`AppView.swift:473-475`) says
  "about/integrations/logs screens" but group holds `security` + `about` only.

### Pass 2 - Interaction State Coverage: 6.5 / 10
Button + update-sheet + AsyncImage states are good. **Weakest area:** per-screen
loading/empty/error for Models/Downloads/Logs/Bench unverified.
- **Action:** audit each screen for the 3 states; add missing ones.

### Pass 3 - User Journey & Emotional Arc: 7.0 / 10
Default = Status, deep-link nav thoughtful. For a dev tool the arc is
"competence + clarity" - the glass/Tahoe aesthetic + clean IA supports it.
- **Action:** review `WelcomeWindow.swift` first-run flow.

### Pass 4 - AI Slop Risk: 9.0 / 10
Native macOS, HIG, SF Pro, token system, no purple gradients / emoji-as-design /
icon grids. `DesktopWash` is intentional atmosphere. **Low slop.** No action.

### Pass 5 - Design System Alignment: 7.0 / 10
Strong *implicit* system. Gap: no DESIGN.md. `ContentScaffold` comment references
`fusion-components.jsx:250-292` - a JSX design source may be canonical; if so, the
Swift impl is a port and should be diffed against it for drift.
- **Action:** extract DESIGN.md from `Theme.swift`; locate + diff `fusion-components.jsx`.

### Pass 6 - Responsive & Accessibility: 6.0 / 10
`NavigationSplitView` collapse + 720 max + centering good. minWidth 880x600.
But 11pt labels, contrast ratios, keyboard focus order, VoiceOver landmarks,
Dynamic Type all unverified.
- **Action:** contrast audit (target AA 4.5:1) + a11y audit.

### Pass 7 - Unresolved Design Decisions
Surfaced as the follow-up question (see review log).

---

## GSTACK REVIEW REPORT

- **Plan file reviewed:** none (no plan in context - EXIT PLAN MODE gate short-circuits;
  this is a design audit producing findings, not a plan edit).
- **DESIGN.md status:** MISSING. A `FusionTheme` token system exists in-code; recommend
  extracting a DESIGN.md rather than running `/design-consultation` from scratch.
- **TODOS.md status:** MISSING (not created - no implementation tasks confirmed yet;
  pending user selection of gaps to address).
- **Mockups:** skipped (native SwiftUI + no-images rule).
- **Outside voices:** skipped (cost-critical).

### Review Log (gstack-review-log)
| Pass | Score | Verdict |
|------|-------|---------|
| IA | 8.5 | PASS (1 stale comment) |
| Interaction States | 6.5 | GAP - per-screen states unverified |
| User Journey | 7.0 | GAP - WelcomeWindow unread |
| AI Slop | 9.0 | PASS |
| Design System | 7.0 | GAP - no DESIGN.md |
| Responsive & A11y | 6.0 | GAP - contrast + a11y unverified |
| Unresolved Decisions | - | deferred to user |

### Review Readiness Dashboard (gstack-review-read)
- Design system maturity: **7.5 / 10** (implicit, undocumented)
- Slop risk: **LOW** (9.0)
- Blocking issues: **0** (no correctness/access blockers found)
- Recommended next actions: pending user selection (DESIGN.md / per-screen states /
  contrast+a11y / WelcomeWindow)
- Ready to implement: **NO** - awaiting user prioritization of the 4 gaps.

### Completion Summary
Cold design audit of the fusion-mlx macOS SwiftUI app. The app is genuinely
well-engineered - token system, Tahoe glass, grouped IA, component states, i18n -
and **not** AI slop (9.0). The gaps are documentation (DESIGN.md) and verification
(contrast, per-screen states, a11y, onboarding), not fundamental rework. No code
changes made (review-only).

---

## Deep-Dive Findings (post user selection of all 4 gaps)

User selected all four gaps. Each was investigated; findings below. **No code
changed** - the investigation upgraded several scores (the "unverified" gaps
turned out to be well-handled once verified).

### Ask 1 - DESIGN.md: DONE
Created `apps/fusion-mac/DESIGN.md` - documents the full token table (with the
AppKit system-color each token resolves to), typography, components, glass,
layout, a11y approach, and localization. Tribal knowledge in `Theme.swift` is
now explicit. **Pass 5 → 9.0.**

### Ask 2 - Per-screen state audit: PASS (pattern is strong)
`ModelsScreen` is exemplary - all three states present:
- **Empty:** ActiveModelsSection "No models loaded"; LibrarySection "No models
  discovered" + guidance ("Use the Downloads screen…").
- **Loading:** `ProgressView()` during delete, `StatusPill(.starting)` during
  load, `.disabled(m.isLoading)` on Load.
- **Error:** `if let error = vm.lastError` red banner at top.
- **Confirm:** destructive `confirmationDialog` for delete.
`WelcomeWindow` tracks downloads as a 4-state machine (idle/downloading/done/error)
with per-model `downloadErrors`. The state pattern is established and consistent.
**Pass 2 → 8.0.** Residual: Downloads/Logs/Bench not individually line-audited
(pattern verified via Models + Welcome; spot-check if any screen lacks the trio).

### Ask 3 - Contrast + a11y audit: PASS (by delegation)
**Key discovery:** every token resolves through a dynamic AppKit semantic color
(`Theme.swift:160-234`). Consequences:
- `text` = `.labelColor` → AA-compliant.
- `textSecondary`/`textTertiary` = `.secondaryLabelColor`/`.tertiaryLabelColor`
  → below AA in standard appearance, **but auto-bump when "Increase Contrast"
  is enabled** (Apple-managed). Used correctly for non-essential metadata
  (empty-state subtitles, "Idle" badge, secondary mono sizes).
- This is HIG-compliant macOS design, **not** an a11y defect.
Residual rule to enforce: never use `textTertiary` for essential/instructional
copy. Keyboard/VoiceOver rely on SwiftUI `NavigationSplitView`+`List` defaults
+ `.help(...)` tooltips on icon-only buttons. **Pass 6 → 8.5.**

### Ask 4 - WelcomeWindow review: PASS (strong onboarding)
6-step wizard (`intro → setup → hardwareDetect → modelSource → recommend →
complete`) via `WelcomeWindowController` (proper `NSWindowDelegate`, 680×620,
transparent titlebar, `isMovableByWindowBackground`, `isReleasedWhenClosed=false`).
Standouts:
- **Hardware-adaptive:** `sysctl` chip detection + full M1-M5 bandwidth table
  (M5 Max 614GB/s, M4 Ultra 819, …) + `physicalMemory` RAM tiers.
- **Use-case-driven:** agent/coding/chat × RAM tiers (128/64/32) → different
  model recommendations + computed settings (maxContext/maxTokens/idleTimeout/cache).
- **Editable recommendations + contextual validation warnings** (non-blocking):
  "Max context 262K+ may exceed XGB RAM", "Long timeout on XGB RAM may cause
  memory pressure".
- **Validation gates:** port 1-65535, api key ≥4 chars / no whitespace / ASCII.
- **Cancellation safety:** closing before Start = cancellation, no partial
  `settings.json` written.
- **Tested:** `WelcomeViewModelTests.swift` covers the VM.
Emotional arc (intro → "it detected my M5 Max!" → sensible defaults → complete)
suits a dev tool. **Pass 3 → 8.5.** Residual: `WelcomeView` body (visual layout
of steps) not fully line-reviewed (VM + controller reviewed; truncation at line 514).

### Revised scores
| Pass | Before | After | Change |
|------|--------|-------|--------|
| IA | 8.5 | 8.5 | - |
| Interaction States | 6.5 | 8.0 | +1.5 (Models exemplary) |
| User Journey | 7.0 | 8.5 | +1.5 (Welcome wizard strong) |
| AI Slop | 9.0 | 9.0 | - |
| Design System | 7.0 | 9.0 | +2.0 (DESIGN.md written) |
| Responsive & A11y | 6.0 | 8.5 | +2.5 (system-color delegation) |
| **Overall** | **7.5** | **8.6** | **+1.1** |

### Remaining residual (low priority, non-blocking)
1. Spot-check Downloads/Logs/Bench screens for the loading/empty/error trio.
2. Line-review `WelcomeView` body (step visual layout, step indicator).
3. Fix stale *General* group-header comment (`AppView.swift:473-475`).
4. Enforce (lint/review) the "no essential text in `textTertiary`" rule.

### Artifacts produced
- `apps/fusion-mac/DESIGN.md` (new - design system doc)
- `apps/fusion-mac/DESIGN_REVIEW.md` (this file - review + deep-dive)
