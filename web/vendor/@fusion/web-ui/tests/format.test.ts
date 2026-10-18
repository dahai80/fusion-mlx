import { describe, expect, it } from 'bun:test';
import {
    formatBytes,
    formatDuration,
    formatLatency,
    formatPercent,
    formatRelativeTime,
    formatTokenCount,
} from '../src/format';

describe('formatBytes (binary)', () => {
    it('scales binary units', () => {
        expect(formatBytes(0)).toBe('0 B');
        expect(formatBytes(512)).toBe('512 B');
        expect(formatBytes(1024)).toBe('1.0 KiB');
        expect(formatBytes(1536 * 1024)).toBe('1.5 MiB');
        expect(formatBytes(3 * 1024 ** 3)).toBe('3.0 GiB');
    });
    it('handles invalid input', () => {
        expect(formatBytes(Number.NaN)).toBe('-');
    });
});

describe('formatTokenCount', () => {
    it('abbreviates k/M', () => {
        expect(formatTokenCount(999)).toBe('999');
        expect(formatTokenCount(1234)).toBe('1.2k');
        expect(formatTokenCount(2_500_000)).toBe('2.5M');
    });
});

describe('formatLatency', () => {
    it('switches unit at 1s', () => {
        expect(formatLatency(12.4)).toBe('12ms');
        expect(formatLatency(1500)).toBe('1.50s');
    });
});

describe('formatDuration', () => {
    it('formats h/m/s', () => {
        expect(formatDuration(59_000)).toBe('59s');
        expect(formatDuration(61_000)).toBe('1m01s');
        expect(formatDuration(3 * 3600_000 + 120_000)).toBe('3h02m');
    });
});

describe('formatPercent', () => {
    it('formats ratio', () => {
        expect(formatPercent(0.256, 1)).toBe('25.6%');
    });
});

describe('formatRelativeTime', () => {
    it('renders recent times', () => {
        const now = Date.now();
        expect(formatRelativeTime(now - 10_000, now)).toContain('前');
        expect(formatRelativeTime(now - 3 * 60_000, now)).toContain('分钟前');
    });
});
