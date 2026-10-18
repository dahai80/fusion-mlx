import { describe, expect, it } from 'bun:test';
// Issue #19 regression: success path must clear `loading`, else pages gating
// on `loading` (data ? content : skeleton) sit on the skeleton forever.
import { GlobalRegistrator } from '@happy-dom/global-registrator';
import { renderHook, waitFor } from '@testing-library/react';
import { useAutoRetry } from '../src/components/useAutoRetry';

GlobalRegistrator.register();

describe('useAutoRetry loading lifecycle (issue #19)', () => {
    it('clears loading on success', async () => {
        const { result } = renderHook(() => useAutoRetry(async () => 'data', []));
        expect(result.current.loading).toBe(true);
        await waitFor(() => expect(result.current.loading).toBe(false));
        expect(result.current.data).toBe('data');
        expect(result.current.error).toBeNull();
    });
    it('clears loading after exhausting retries', async () => {
        const { result } = renderHook(() =>
            useAutoRetry(
                async () => {
                    throw new Error('boom');
                },
                [],
                { retries: 1, backoffMs: 10 }
            )
        );
        await waitFor(() => expect(result.current.error).not.toBeNull());
        expect(result.current.loading).toBe(false);
    });
    it('sets loading again during manual retry and clears on success', async () => {
        let fail = true;
        const { result } = renderHook(() =>
            useAutoRetry(
                async () => {
                    if (fail) throw new Error('boom');
                    return 'ok';
                },
                [],
                { retries: 0, backoffMs: 10 }
            )
        );
        await waitFor(() => expect(result.current.error).not.toBeNull());
        fail = false;
        result.current.manualRetry();
        await waitFor(() => expect(result.current.data).toBe('ok'));
        expect(result.current.loading).toBe(false);
        expect(result.current.error).toBeNull();
    });
});
