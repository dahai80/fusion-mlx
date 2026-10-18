import { formatPercent } from '../format';

/** Single-value metric card: monospace tabular-nums value + optional trend. */
export function MetricTile({
    label,
    value,
    unit,
    trend,
    trendLabel,
}: { label: string; value: number | string; unit?: string; trend?: number; trendLabel?: string }) {
    const trendText =
        trend === undefined ? null : `${trend >= 0 ? '↑' : '↓'} ${formatPercent(Math.abs(trend))}`;
    const trendUp = trend === undefined ? false : trend >= 0;
    return (
        <div className="rounded-card border border-border bg-bg-elevated p-4">
            <div className="text-xs text-text-muted">{label}</div>
            <div className="mt-1 font-mono text-xl tabular-nums text-text-primary">
                {value}
                {unit ? <span className="ml-1 text-sm text-text-secondary">{unit}</span> : null}
            </div>
            {trendText ? (
                <div
                    className={`mt-1 text-xs tabular-nums ${trendUp ? 'text-success' : 'text-error'}`}
                >
                    {trendText}
                    {trendLabel ? <span className="ml-1 text-text-muted">{trendLabel}</span> : null}
                </div>
            ) : null}
        </div>
    );
}
