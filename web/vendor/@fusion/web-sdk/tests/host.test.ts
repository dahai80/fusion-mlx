import { describe, expect, it } from 'bun:test';
import { isHttpErrorResult } from '../src/host';

// Issue #75: `case 'theme'` returns a bare string, and the old
// `'__httpError' in (result as object)` guard threw a TypeError on it —
// the module never received its theme response.
describe('isHttpErrorResult (issue #75)', () => {
    it('bare string results (theme pull) are not http errors', () => {
        expect(isHttpErrorResult('light')).toBe(false);
        expect(isHttpErrorResult('dark')).toBe(false);
    });
    it('null, undefined and primitives are not http errors', () => {
        expect(isHttpErrorResult(null)).toBe(false);
        expect(isHttpErrorResult(undefined)).toBe(false);
        expect(isHttpErrorResult(42)).toBe(false);
        expect(isHttpErrorResult(false)).toBe(false);
    });
    it('plain object results are not http errors', () => {
        expect(isHttpErrorResult({ status: 200, ok: true, data: {} })).toBe(false);
        expect(isHttpErrorResult({ viewport: 'desktop', mode: null })).toBe(false);
        expect(isHttpErrorResult({ ok: true })).toBe(false);
    });
    it('__httpError envelopes are http errors', () => {
        expect(isHttpErrorResult({ __httpError: true, status: 502, code: 'x', message: 'y' })).toBe(
            true
        );
    });
});
