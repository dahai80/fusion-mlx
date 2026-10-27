import { describe, expect, it } from 'bun:test';
import { createThemeState } from '../src/theme';

// Issue #77: theme.get() stays synchronous behind a cache. The embedded SDK
// seeds it from the module document, corrects it via a background host pull
// and keeps it fresh from host push events.
describe('createThemeState (issue #77)', () => {
    it('returns the initial mode', () => {
        expect(createThemeState('dark').get()).toBe('dark');
        expect(createThemeState('light').get()).toBe('light');
    });
    it('set updates the cached mode and notifies subscribers', () => {
        const state = createThemeState('dark');
        const seen: string[] = [];
        state.subscribe((m) => seen.push(m));
        state.set('light');
        expect(state.get()).toBe('light');
        expect(seen).toEqual(['light']);
    });
    it('same-value set does not re-notify (push/pull double-fire guard)', () => {
        const state = createThemeState('light');
        const seen: string[] = [];
        state.subscribe((m) => seen.push(m));
        state.set('light');
        expect(seen).toEqual([]);
    });
    it('unsubscribe stops notifications', () => {
        const state = createThemeState('dark');
        const seen: string[] = [];
        const off = state.subscribe((m) => seen.push(m));
        off();
        state.set('light');
        expect(seen).toEqual([]);
        expect(state.get()).toBe('light');
    });
});
