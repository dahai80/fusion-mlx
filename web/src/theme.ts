import { sdk } from "./sdk";

// Shell theme sync (fusion-web-ui-spec-1015 §theme): the palette is
// class-based (tokens.css) — dark is the default, html.light flips the
// --fs-* tokens. The module must NOT detect prefers-color-scheme itself;
// it follows the host via sdk.theme (embedded: host push + cached pull;
// standalone: the SDK tracks prefers-color-scheme). Same convention as
// fusion-cowork / fusion-agent-studio module theme.ts.
const STORAGE_KEY = "fusion.shell.theme";

export function applyTheme(mode: "dark" | "light"): void {
    const root = document.documentElement;
    root.classList.toggle("light", mode === "light");
    root.classList.toggle("dark", mode === "dark");
    try {
        localStorage.setItem(STORAGE_KEY, mode);
    } catch (e) {
        console.warn("[theme] persist failed", e);
    }
}

// Subscribe once at module load. Embedded: every host push re-applies.
// Standalone: sdk.theme tracks prefers-color-scheme, so OS-level changes
// still flip the console. Seed from the stored value so the first paint
// already matches (the iframe-load re-push can arrive before the SDK
// finishes wiring listeners).
export function initTheme(): void {
    sdk.theme.subscribe((mode) => {
        if (mode === "dark" || mode === "light") {
            applyTheme(mode);
        }
    });
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved === "dark" || saved === "light") {
            applyTheme(saved);
        }
    } catch (e) {
        console.warn("[theme] storage unavailable", e);
    }
}
