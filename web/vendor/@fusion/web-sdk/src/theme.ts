import type { ThemeMode } from './types';

/**
 * Sync cache behind `theme.get()` (issue #77). The embedded SDK seeds it from
 * the module document, corrects it via a background host pull (the host's
 * initial push can land on about:blank before the module's listener exists)
 * and keeps it fresh from host push events. Same-value updates don't
 * re-notify, so a pull racing a push never double-fires listeners.
 */
export function createThemeState(initial: ThemeMode) {
    let current = initial;
    const listeners = new Set<(mode: ThemeMode) => void>();
    return {
        get(): ThemeMode {
            return current;
        },
        set(mode: ThemeMode): void {
            if (mode === current) return;
            current = mode;
            listeners.forEach((fn) => fn(mode));
        },
        subscribe(listener: (mode: ThemeMode) => void): () => void {
            listeners.add(listener);
            return () => listeners.delete(listener);
        },
    };
}
