import { afterEach, beforeEach, describe, expect, it } from 'bun:test';
import { createShellSdk } from '../src/index';

// #1298: a shell deep link opened at top level (/shell/fusion-mlx/
// ?embedded=1#/models) has no parent frame. The SDK must still recognize
// the embedded flag, skip the standalone warning, wire the host APIs and
// degrade the host relay explicitly instead of timing out.

type MessageListener = (ev: MessageEvent) => void;

interface FakeWindow {
    parent: unknown;
    location: {
        search: string;
        hash: string;
        origin: string;
        pathname: string;
    };
    listeners: Array<{ type: string; fn: MessageListener }>;
    addEventListener(type: string, fn: MessageListener): void;
    removeEventListener(): void;
    postMessage(): void;
    matchMedia(query: string): { matches: boolean; addEventListener(): void };
}

const ORIGIN = 'http://127.0.0.1:11432';

function makeWindow(search: string, framed: boolean): FakeWindow {
    const win: FakeWindow = {
        parent: null,
        location: {
            search,
            hash: search.includes('embedded') ? '#/models' : '',
            origin: ORIGIN,
            pathname: '/shell/fusion-mlx/',
        },
        listeners: [],
        addEventListener(type, fn) {
            win.listeners.push({ type, fn });
        },
        removeEventListener() {
            // no-op: the tests never unsubscribe
        },
        postMessage() {
            // no-op: no host frame in these tests
        },
        matchMedia() {
            return { matches: false, addEventListener() {} };
        },
    };
    win.parent = framed ? { postMessage() {} } : win;
    return win;
}

interface FakeDocument {
    documentElement: {
        lang: string;
        classList: { contains(): boolean };
    };
    visibilityState: string;
    listeners: Array<{ type: string; fn: () => void }>;
    addEventListener(type: string, fn: () => void): void;
    removeEventListener(): void;
}

function makeDocument(): FakeDocument {
    const doc: FakeDocument = {
        documentElement: {
            lang: 'zh',
            classList: { contains: () => false },
        },
        visibilityState: 'visible',
        listeners: [],
        addEventListener(type, fn) {
            doc.listeners.push({ type, fn });
        },
        removeEventListener() {
            // no-op
        },
    };
    return doc;
}

function installDocument(): FakeDocument {
    const doc = makeDocument();
    (globalThis as unknown as { document?: unknown }).document = doc;
    return doc;
}

function installDom(search: string, framed: boolean): FakeWindow {
    const win = makeWindow(search, framed);
    const g = globalThis as unknown as { window?: unknown; document?: unknown };
    g.window = win;
    g.document = makeDocument();
    return win;
}

function uninstallDom(): void {
    const g = globalThis as unknown as { window?: unknown; document?: unknown };
    delete g.window;
    delete g.document;
}

const warnings: string[] = [];
const logs: string[] = [];
const originalWarn = console.warn;
const originalLog = console.log;
const originalFetch = globalThis.fetch;

beforeEach(() => {
    warnings.length = 0;
    logs.length = 0;
    console.warn = (...args: unknown[]) => {
        warnings.push(args.map(String).join(' '));
    };
    console.log = (...args: unknown[]) => {
        logs.push(args.map(String).join(' '));
    };
});

afterEach(() => {
    console.warn = originalWarn;
    console.log = originalLog;
    globalThis.fetch = originalFetch;
    uninstallDom();
});

function messageListeners(win: FakeWindow): MessageListener[] {
    return win.listeners.filter((l) => l.type === 'message').map((l) => l.fn);
}

describe('embedded param detection (#1298)', () => {
    it('recognizes ?embedded=1 at top level and skips the standalone warning', () => {
        const win = installDom('?embedded=1', false);
        createShellSdk();
        expect(warnings.some((w) => w.includes('standalone mode'))).toBe(false);
        expect(
            logs.some((l) => l.includes('embedded via ?embedded=1 without a parent frame'))
        ).toBe(true);
        expect(messageListeners(win).length).toBeGreaterThan(0);
    });

    it('drops host events when there is no parent frame', () => {
        const win = installDom('?embedded=1', false);
        const sdk = createShellSdk();
        const themes: string[] = [];
        sdk.theme.subscribe((mode) => themes.push(mode));
        const handler = messageListeners(win)[0];
        expect(handler).toBeDefined();
        handler({
            origin: ORIGIN,
            source: { postMessage() {} },
            data: { type: 'event', event: 'theme', payload: 'light' },
        } as unknown as MessageEvent);
        expect(themes).toEqual([]);
    });

    it('derives visibility from the document without a parent frame', () => {
        installDom('?embedded=1', false);
        const doc = installDocument();
        const sdk = createShellSdk();
        const seen: boolean[] = [];
        sdk.visibility.subscribe((visible) => seen.push(visible));
        const listener = doc.listeners.find((l) => l.type === 'visibilitychange');
        expect(listener).toBeDefined();
        doc.visibilityState = 'hidden';
        listener?.fn();
        expect(seen).toEqual([false]);
    });

    it('degrades http to a direct request without a parent frame', async () => {
        installDom('?embedded=1', false);
        const calls: string[] = [];
        globalThis.fetch = (async (input: string) => {
            calls.push(input);
            return new Response(JSON.stringify({ ok: 1 }), {
                status: 200,
                headers: { 'content-type': 'application/json' },
            });
        }) as unknown as typeof fetch;
        const sdk = createShellSdk();
        const res = await sdk.http.get<{ ok: number }>('/v1/health');
        expect(calls).toEqual(['/v1/health']);
        expect(res.data).toEqual({ ok: 1 });
    });

    it('keeps standalone posture without the flag or a parent frame', () => {
        const win = installDom('', false);
        createShellSdk();
        expect(warnings.some((w) => w.includes('standalone mode'))).toBe(true);
        expect(messageListeners(win)).toHaveLength(0);
    });

    it('still treats a framed module as embedded without the flag', () => {
        const win = installDom('', true);
        createShellSdk();
        expect(warnings.some((w) => w.includes('standalone mode'))).toBe(false);
        expect(
            logs.some((l) => l.includes('embedded via ?embedded=1 without a parent frame'))
        ).toBe(false);
        expect(messageListeners(win).length).toBeGreaterThan(0);
    });
});
