// Retrospective usage store (#1278): local snapshot ring of engine counters
// (requests/failed/prompt/completion) sampled at the page's poll cadence and
// persisted to localStorage so 2h/24h windows survive reloads. Frontend-only
// aggregation over /metrics counters — no new backend surface (issue: "window
// switch renders without backend changes").

export interface UsageSample {
    t: number;
    requests: number;
    failed: number;
    prompt: number;
    completion: number;
    tpsSum?: number;
    tpsCount?: number;
}

const KEY = "fusion-mlx-usage-samples-v1";
const MAX_AGE_MS = 25 * 3600_000; // keep ≥24h window + margin
const MAX_COUNT = 4000;

export function loadSamples(): UsageSample[] {
    try {
        const raw = localStorage.getItem(KEY);
        if (!raw) return [];
        const arr = JSON.parse(raw) as UsageSample[];
        if (!Array.isArray(arr)) return [];
        return arr.filter(
            (s) =>
                Number.isFinite(s?.t) &&
                Number.isFinite(s?.requests) &&
                Number.isFinite(s?.failed) &&
                Number.isFinite(s?.prompt) &&
                Number.isFinite(s?.completion),
        );
    } catch {
        return [];
    }
}

// Append one sample and persist. Downsample gate: skip writing when a sample
// for the same 30s bucket already exists (live state still gets every poll).
export function appendSample(s: UsageSample): void {
    try {
        const arr = loadSamples();
        const bucket = Math.floor(s.t / 30000);
        const last = arr[arr.length - 1];
        if (last && Math.floor(last.t / 30000) === bucket)
            arr[arr.length - 1] = s;
        else arr.push(s);
        const cutoff = s.t - MAX_AGE_MS;
        let start = 0;
        while (start < arr.length && arr[start].t < cutoff) start++;
        if (arr.length - start > MAX_COUNT) start = arr.length - MAX_COUNT;
        const keep = arr.slice(start);
        localStorage.setItem(KEY, JSON.stringify(keep));
    } catch (e) {
        // storage full / unavailable — retrospective windows degrade to live
        // samples only; never break the live page over persistence.
        console.warn("[usage] persist failed", e);
    }
}

export const WINDOWS: Array<{ key: string; label: string; ms: number }> = [
    { key: "5m", label: "5m", ms: 5 * 60_000 },
    { key: "30m", label: "30m", ms: 30 * 60_000 },
    { key: "2h", label: "2h", ms: 2 * 3600_000 },
    { key: "24h", label: "24h", ms: 24 * 3600_000 },
];

// Window slice + per-minute rate series + totals (counter deltas over the
// window). Deltas clamped ≥0 — engine restart resets counters mid-window.
export function windowSeries(
    samples: UsageSample[],
    windowMs: number,
    now: number,
): {
    points: Array<{ t: number; v: number }>;
    totals: { req: number; failed: number; tok: number; tps: number | null };
} {
    const cutoff = now - windowMs;
    const inWin = samples.filter((s) => s.t >= cutoff);
    const first = inWin[0];
    const lastS = inWin[inWin.length - 1];
    if (!first || !lastS || lastS === first) {
        return {
            points: [],
            totals: { req: 0, failed: 0, tok: 0, tps: null },
        };
    }
    const dReq = Math.max(0, lastS.requests - first.requests);
    const dFail = Math.max(0, lastS.failed - first.failed);
    const dTok =
        Math.max(0, lastS.completion - first.completion) +
        Math.max(0, lastS.prompt - first.prompt);
    let tps: number | null = null;
    const dTpsSum = (lastS.tpsSum ?? 0) - (first.tpsSum ?? 0);
    const dTpsCount = (lastS.tpsCount ?? 0) - (first.tpsCount ?? 0);
    if (dTpsCount > 0) {
        tps = dTpsSum / dTpsCount;
    }
    return {
        points: perMinutePoints(inWin),
        totals: { req: dReq, failed: dFail, tok: dTok, tps },
    };
}

// Per-minute request counts: pairwise counter deltas bucketed by minute.
function perMinutePoints(
    inWin: UsageSample[],
): Array<{ t: number; v: number }> {
    const out: Array<{ t: number; v: number }> = [];
    let curMin = -1;
    let curV = 0;
    for (let i = 1; i < inWin.length; i++) {
        const dt = inWin[i].t - inWin[i - 1].t;
        if (dt <= 0) continue;
        const d = Math.max(0, inWin[i].requests - inWin[i - 1].requests);
        const m = Math.floor(inWin[i].t / 60000);
        if (m !== curMin) {
            if (curMin >= 0) out.push({ t: curMin * 60000, v: curV });
            curMin = m;
            curV = 0;
        }
        curV += d;
    }
    if (curMin >= 0) out.push({ t: curMin * 60000, v: curV });
    return out;
}
