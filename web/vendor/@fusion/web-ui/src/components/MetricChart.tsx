import { useMemo, useState } from 'react';

export interface MetricPoint {
    t: number;
    v: number;
}

export interface MetricSeries {
    label: string;
    color?: string;
    points: MetricPoint[];
}

const RANGES: Array<[string, number]> = [
    ['5m', 5 * 60_000],
    ['30m', 30 * 60_000],
    ['2h', 2 * 60 * 60_000],
];

export function MetricChart({
    series,
    height = 160,
    unit,
}: { series: MetricSeries[]; height?: number; unit?: string }) {
    const [rangeIdx, setRangeIdx] = useState(0);
    const range = RANGES[rangeIdx]![1];
    const primary = series[0];
    const points = useMemo(() => {
        const cutoff = Date.now() - range;
        return (primary?.points ?? []).filter((p) => p.t >= cutoff);
    }, [primary, range]);
    const hasData = points.length > 0;
    const yMinRaw = hasData ? Math.min(...points.map((p) => p.v)) : 0;
    const yMaxRaw = hasData ? Math.max(...points.map((p) => p.v)) : 1;
    const span = Math.max(yMaxRaw - yMinRaw, 1e-9);
    const yMin = yMinRaw - span * 0.1;
    const yMax = yMaxRaw + span * 0.1;
    const tMin = hasData ? points[0]!.t : 0;
    const tMax = hasData ? points[points.length - 1]!.t : 1;
    const toXY = (p: MetricPoint): [number, number] => {
        const x = tMax > tMin ? ((p.t - tMin) / (tMax - tMin)) * 100 : 50;
        const y = 100 - ((p.v - yMin) / (yMax - yMin)) * 100;
        return [x, y];
    };
    const path = points
        .map(
            (p, i) =>
                `${i === 0 ? 'M' : 'L'}${toXY(p)
                    .map((n) => n.toFixed(2))
                    .join(' ')}`
        )
        .join(' ');
    const last = hasData ? toXY(points[points.length - 1]!) : null;
    return (
        <div className="rounded-card border border-border bg-bg-elevated p-3">
            <div className="mb-2 flex items-center justify-between">
                <span className="text-xs text-text-secondary">{primary?.label ?? ''}</span>
                <div className="flex gap-1">
                    {RANGES.map(([label], i) => (
                        <button
                            key={label}
                            type="button"
                            onClick={() => setRangeIdx(i)}
                            className={`rounded px-2 py-0.5 text-xs ${i === rangeIdx ? 'bg-brand text-white' : 'text-text-muted hover:bg-bg-muted'}`}
                        >
                            {label}
                        </button>
                    ))}
                </div>
            </div>
            <svg
                viewBox="0 0 100 100"
                preserveAspectRatio="none"
                style={{ height }}
                className="w-full"
                role="img"
                aria-label={primary?.label ?? 'metric chart'}
            >
                <path
                    d={path}
                    fill="none"
                    stroke="var(--fs-running)"
                    strokeWidth="1"
                    vectorEffect="non-scaling-stroke"
                />
                {last ? (
                    <circle cx={last[0]} cy={last[1]} r="0.8" fill="var(--fs-running)" />
                ) : null}
            </svg>
            <div className="mt-1 flex justify-between font-mono text-[10px] tabular-nums text-text-muted">
                <span>{hasData ? yMax.toFixed(1) : '-'}</span>
                <span>{unit ?? ''}</span>
                <span>{hasData ? yMin.toFixed(1) : '-'}</span>
            </div>
            {!hasData ? (
                <div className="pb-2 text-center text-xs text-text-muted">暂无数据</div>
            ) : null}
        </div>
    );
}
