# FusionMLX macOS App - Design System

Single source of truth for the SwiftUI app at `apps/fusion-mac/`.
Mirrors the code in `Sources/Theme/`. Update this file when tokens or
components change.

## Design principles

1. **System colors, not hardcoded hex.** Every token resolves through a
   dynamic AppKit semantic color (`.labelColor`, `.controlColor`, …). The
   app tracks System Settings across light/dark, accent color, vibrancy,
   and the "Increase Contrast" accessibility setting for free.
2. **SF Pro via `.system`.** `.fusionText/.fusionDisplay/.fusionMono` all
   build on `Font.system(...)` - macOS picks the correct SF variant by
   size. No custom font files.
3. **Token-injected theme.** `FusionTheme` (dark/light) is passed through
   `.environment(\.fusionTheme, theme)`. Descendants read `@Environment(\.fusionTheme)`,
   no prop-drilling. `View.fusionThemed()` binds it to `\.colorScheme` once
   at the shell.
4. **macOS HIG shapes.** `cornerRadius` 14 / `rowRadius` 10, `.continuous`
   curves, `NavigationSplitView` + grouped `List(.sidebar)`, Settings.app
   large-title pattern.
5. **Localize everything.** All user strings go through
   `String(localized:defaultValue:comment:)`.

## Color tokens

`FusionTheme` struct (`Theme.swift:12`). Two variants: `.light`, `.dark`.

### Surfaces
| Token | Light | Dark | Resolves to |
|---|---|---|---|
| `windowBg` / `sidebarBg` / `contentBg` / `toolbarBg` | `windowBackgroundColor` | `underPageBackgroundColor` | NS window bg |
| `groupBg` | `labelColor` @ 0.035 | `labelColor` @ 0.03 | subtle group fill |
| `groupBorder` / `sidebarBorder` / `toolbarBorder` / `rowSep` / `separator` / `inputBorder` | `separatorColor` | `separatorColor` | NS separator |
| `controlBg` | `controlColor` | `controlColor` | NS control |
| `controlBgHover` | controlBg @ 0.92 | controlBg @ 0.85 | hover fill |
| `inputBg` | `textBackgroundColor` | `textBackgroundColor` | NS text bg |
| `inputBorderFocus` | `controlAccentColor` | `controlAccentColor` | focus ring |
| `glassBg` | windowBgLight @ 0.70 | windowBg @ 0.70 | regular glass |
| `glassBgStrong` | `groupBg` (light) | `groupBg` (dark) | strong glass |
| `codeBg` | text @ 0.05 | text @ 0.07 | code block fill |

### Text
| Token | Resolves to | WCAG note |
|---|---|---|
| `text` | `labelColor` | AA-compliant (primary) |
| `textSecondary` | `secondaryLabelColor` | below AA std; auto-adapts to "Increase Contrast" |
| `textTertiary` | `tertiaryLabelColor` | below AA std; **use only for non-essential** metadata |

> Primary text meets AA 4.5:1. Secondary/tertiary are Apple's own tiers -
> they sit below AA in standard appearance but auto-bump when the user
> enables "Increase Contrast". This is HIG-compliant. Rule: never put
> essential/instructional copy in `textTertiary`.

### Accent + selection
| Token | Resolves to |
|---|---|
| `accent` / `blueDot` / `inputBorderFocus` | `controlAccentColor` (user's system accent) |
| `accentSoft` | accent @ 0.16 (light) / 0.22 (dark) |
| `accentText` | `alternateSelectedControlTextColor` |
| `selBg` | `unemphasizedSelectedContentBackgroundColor` |
| `hoverBg` | `quaternaryLabelColor` |

### Status
| Token | Resolves to | Used for |
|---|---|---|
| `greenDot` / `successText` | `systemGreen` | running / loaded / success |
| `amberDot` / `warningText` | `systemOrange` | starting / warning |
| `redDot` | `systemRed` | error / destructive |
| `blueDot` | `controlAccentColor` | info / starting |
| `successBg` / `warningBg` | green/orange @ 0.16-0.18 | tinted bg fills |

### Desktop wash
`desktopWashBase` = window bg; `desktopWashTopLeft`/`BottomRight` = `.clear`
in the token (the `DesktopWash` view composes two `RadialGradient`s over
the base - see Glass.swift). `ignoresSafeArea()`.

### Metrics
`cornerRadius` = 14, `rowRadius` = 10, `groupHighlightTopOpacity` =
0.35 (light) / 0.08 (dark), `groupShadowOpacity` = 0.06 (light) / 0.08 (dark).

## Typography

All `Font.system(...)` - SF Pro / SF Mono.

| Helper | Definition | Use |
|---|---|---|
| `fusionText(size, weight:.regular)` | `.system(size:weight:)` | body / labels |
| `fusionDisplay(size, weight:.semibold)` | `.system(size:weight:)` | headlines (auto-promotes ≥20pt) |
| `fusionMono(size, weight:.regular)` | `.system(size:weight:design:.monospaced)` | ids, sizes, code |

Common sizes: 11 / 11.5 (buttons, pills, secondary mono), 12 (subtitles),
13 (row titles, regular buttons). macOS uses smaller type than web; 11pt
is the standard secondary-label size.

Color helpers: `Color.black(_:)`, `Color.white(_:)` (sRGB white alpha),
`Color(rgb24:opacity:)` (packed 24-bit hex).

## Components (`Sources/Theme/Components/`)

| Component | Role |
|---|---|
| `FusionButtonStyle` | 4 kinds (primary/destructive/normal/plain) × 2 sizes (small/regular); disabled 0.45 / pressed 0.78; cornerRadius 6 |
| `StatusPill` | 6 states (running/starting/stopping/stopped/error/custom); dot + label, capsule |
| `ScreenScaffold` / `ContentScaffold` | 42pt toolbar, 720pt max content, 20/28/36 padding, Settings.app large-title |
| `Segmented` | segmented control |
| `TextInput` | text field with focus border (`inputBorderFocus`) |
| `ProgressBar` | determinate progress |
| `ListGroup` | grouped card (groupBg + groupBorder) |
| `Row` / `FreeRow` | list rows with `isLast` separator control |
| `CodeChip` | mono inline chip (codeBg) |
| `Popup` | popover |
| `SectionHeader` | title + optional subtitle |

Plus `Glass.swift` (liquid glass) and `Squircle.swift` (rounded icon
containers with `SquircleGradient` presets: models/downloads/integrations/update).

## Glass & material

`View.appGlass(_ strength:regular/strong, tint:)`:
- **macOS 26 (Tahoe):** `glassEffect(_:in:)` + `.glassEffectTransition(.identity)`.
- **macOS 15 fallback:** `.regularMaterial` / `.thickMaterial` (+ separate tint).

`DesktopWash`: `ZStack` of `desktopWashBase` + two `RadialGradient`s
(topLeading endRadius 720, bottomTrailing endRadius 680), `ignoresSafeArea()`.

## Layout system

- Shell: `NavigationSplitView { SettingsSidebar } detail: { ContentScaffold { screen } }`.
- Sidebar: 4 groups - **Server** (status/server/network/performance/logs),
  **Models** (models/downloads/integrations/quantization),
  **Benchmark** (throughput/accuracy), **General** (security/about).
  Width 180-215pt. `List(.sidebar)` + `NavigationLink`.
- Detail: `ContentScaffold` centers content in a 720pt column, scrolls,
  42pt nav title. `.logs` screen opts into `fillsContentArea` (grows with window).
- Cross-screen deep-link scroll via `ScrollAnchorKey` (e.g. "Edit on Server ->").
- Window: min 880×600, ideal 880×600. Welcome window: 680×620.
- Default landing: `.status`.

## Accessibility approach

- **Contrast by delegation.** Tokens resolve to AppKit semantic tiers;
  primary text is AA, secondary/tertiary auto-adapt to "Increase Contrast".
  Do not use `textTertiary` for essential copy.
- **Keyboard/VoiceOver.** Rely on SwiftUI `NavigationSplitView` + `List`
  defaults (full keyboard access, VoiceOver landmarks). Add `.help(...)`
  tooltips on icon-only buttons (see ModelsScreen eject/trash/chevron).
- **Dynamic Type.** `.system` fonts respect the user's text-size settings.

## Localization

`String(localized:defaultValue:comment:)` everywhere - the `defaultValue`
is the in-product English fallback; catalog is not load-bearing. Keys are
namespaced by screen (`sidebar.*`, `models.*`, `welcome.*`, `common.*`).
