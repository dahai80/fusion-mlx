import { afterAll, beforeEach, describe, expect, it } from 'bun:test';
import { createShellSdk } from '../src/index';

// #20 contract test (standalone posture): JSON bodies must carry a
// content-type header or FastAPI backends reject with 422. Bun test runs
// without window, so createShellSdk() always takes the standalone path.
const calls: Array<{ input: string; init?: RequestInit }> = [];
const originalFetch = globalThis.fetch;

beforeEach(() => {
    calls.length = 0;
    globalThis.fetch = (async (input: string, init?: RequestInit) => {
        calls.push({ input, init });
        return new Response(JSON.stringify({ ok: 1 }), {
            status: 200,
            headers: { 'content-type': 'application/json' },
        });
    }) as typeof fetch;
});

afterAll(() => {
    globalThis.fetch = originalFetch;
});

describe('standalone http content-type (issue #20)', () => {
    it('defaults content-type to application/json on JSON bodies', async () => {
        const sdk = createShellSdk();
        await sdk.http.post('/api/workflows', { name: 'w1' });
        const headers = calls[0]!.init?.headers as Record<string, string>;
        expect(headers['content-type']).toBe('application/json');
        expect(calls[0]!.init?.body).toBe(JSON.stringify({ name: 'w1' }));
    });
    it('allows caller content-type override', async () => {
        const sdk = createShellSdk();
        await sdk.http.request('/api/x', {
            method: 'PUT',
            body: { a: 1 },
            headers: { 'content-type': 'text/plain' },
        });
        const headers = calls[0]!.init?.headers as Record<string, string>;
        expect(headers['content-type']).toBe('text/plain');
    });
    it('sends no content-type or body on GET', async () => {
        const sdk = createShellSdk();
        await sdk.http.get('/api/x');
        const init = calls[0]!.init;
        expect(init?.body).toBeUndefined();
        expect(
            (init?.headers as Record<string, string> | undefined)?.['content-type']
        ).toBeUndefined();
    });
});
