// PR 3 — button styles for primary / destructive / plain / regular.
//
// Use:
//   Button("Save") { … }
//     .buttonStyle(.fusion(.primary))
//
// The plain kind is the JSX "kind=plain" — borderless action label, e.g. the
// chevron-only row buttons in screens.

import SwiftUI

struct FusionButtonStyle: ButtonStyle {
    enum Kind: Sendable { case primary, destructive, normal, plain }
    enum Size: Sendable { case small, regular }

    let kind: Kind
    let size: Size

    @Environment(\.fusionTheme) private var theme
    /// Without this, a `.disabled()` button keeps its enabled paint because the
    /// custom background ignores SwiftUI's default isEnabled tint. Users then
    /// click an enabled-looking primary button and see no press animation
    /// (SwiftUI suppresses `isPressed` on disabled buttons) — leading to the
    /// "Apply only animates the first time" report. Reading `isEnabled` here
    /// dims the whole label so disabled state is visually unmistakable.
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        let labelFont = Font.fusionText(size == .small ? 11.5 : 13, weight: .medium)
        let hPad: CGFloat = size == .small ? 10 : 12
        let vPad: CGFloat = size == .small ? 4 : 6

        return configuration.label
            .font(labelFont)
            .padding(.horizontal, hPad)
            .padding(.vertical, vPad)
            .foregroundStyle(foreground(configuration))
            .background(background(configuration))
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay(border(configuration))
            .opacity(opacity(configuration))
            .contentShape(Rectangle())
    }

    /// Disabled state wins over press feedback. 0.45 is the macOS-feeling
    /// "this control is inert" tint — tested against `.primary` (blue),
    /// `.destructive` (red), `.normal` (themed control bg), and `.plain`
    /// (transparent + dimmed text).
    private func opacity(_ cfg: Configuration) -> Double {
        guard isEnabled else { return 0.45 }
        return cfg.isPressed ? 0.78 : 1.0
    }

    @ViewBuilder
    private func background(_ cfg: Configuration) -> some View {
        switch kind {
        case .primary:
            theme.accent
        case .destructive:
            theme.redDot
        case .normal:
            theme.controlBg
        case .plain:
            cfg.isPressed ? theme.hoverBg : Color.clear
        }
    }

    private func foreground(_ cfg: Configuration) -> Color {
        switch kind {
        case .primary, .destructive: return theme.accentText
        case .normal: return theme.text
        case .plain:  return theme.text
        }
    }

    @ViewBuilder
    private func border(_ cfg: Configuration) -> some View {
        if kind == .normal {
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .strokeBorder(theme.inputBorder, lineWidth: 0.5)
        }
    }
}

extension ButtonStyle where Self == FusionButtonStyle {
    static func fusion(
        _ kind: FusionButtonStyle.Kind = .normal,
        size: FusionButtonStyle.Size = .regular
    ) -> FusionButtonStyle {
        FusionButtonStyle(kind: kind, size: size)
    }
}

#Preview("Buttons") {
    VStack(alignment: .leading, spacing: 14) {
        HStack(spacing: 8) {
            Button("Save") {}.buttonStyle(.fusion(.primary))
            Button("Save") {}.buttonStyle(.fusion(.normal))
            Button("Delete") {}.buttonStyle(.fusion(.destructive))
            Button("Cancel") {}.buttonStyle(.fusion(.plain))
        }
        HStack(spacing: 8) {
            Button("Load") {}.buttonStyle(.fusion(.primary, size: .small))
            Button("Unload") {}.buttonStyle(.fusion(.normal, size: .small))
            Button { } label: {
                Image(systemName: "trash")
            }.buttonStyle(.fusion(.plain, size: .small))
        }
    }
    .padding(24)
    .fusionThemed()
}
