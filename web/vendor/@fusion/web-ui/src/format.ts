/** Shared formatters (spec §9.2) — single source; modules must not hand-roll. */

const RELATIVE_UNITS: Array<[Intl.RelativeTimeFormatUnit, number]> = [
    ['minute', 60 * 1000],
    ['hour', 60 * 60 * 1000],
    ['day', 24 * 60 * 60 * 1000],
];

const rtf = new Intl.RelativeTimeFormat('zh-CN', { numeric: 'auto' });

export function formatRelativeTime(ts: number | Date, now: number = Date.now()): string {
    const t = ts instanceof Date ? ts.getTime() : ts;
    const diff = t - now;
    const abs = Math.abs(diff);
    if (abs < 45 * 1000) return rtf.format(Math.round(diff / 1000), 'second');
    for (const [unit, ms] of RELATIVE_UNITS) {
        if (abs < ms * 45) return rtf.format(Math.round(diff / ms), unit);
    }
    return rtf.format(Math.round(diff / (24 * 60 * 60 * 1000)), 'day');
}

/** Binary units (KiB/MiB/GiB/TiB), spec §9.2. */
export function formatBytes(bytes: number, digits = 1): string {
    if (!Number.isFinite(bytes)) return '-';
    if (Math.abs(bytes) < 1024) return `${Math.round(bytes)} B`;
    const units = ['KiB', 'MiB', 'GiB', 'TiB', 'PiB'];
    let value = bytes;
    let unit = 'B';
    for (const u of units) {
        value /= 1024;
        unit = u;
        if (Math.abs(value) < 1024) break;
    }
    return `${value.toFixed(digits)} ${unit}`;
}

/** Token counts with k/M abbreviation. */
export function formatTokenCount(n: number): string {
    if (!Number.isFinite(n)) return '-';
    if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
    if (Math.abs(n) >= 1000) return `${(n / 1000).toFixed(1)}k`;
    return String(Math.round(n));
}

/** Latency: <1000ms in ms, otherwise seconds with unit. */
export function formatLatency(ms: number): string {
    if (!Number.isFinite(ms)) return '-';
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
}

export function formatDuration(ms: number): string {
    if (!Number.isFinite(ms) || ms < 0) return '-';
    const s = Math.floor(ms / 1000);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    if (h > 0) return `${h}h${String(m).padStart(2, '0')}m`;
    if (m > 0) return `${m}m${String(sec).padStart(2, '0')}s`;
    return `${sec}s`;
}

export function formatPercent(ratio: number, digits = 0): string {
    if (!Number.isFinite(ratio)) return '-';
    return `${(ratio * 100).toFixed(digits)}%`;
}
