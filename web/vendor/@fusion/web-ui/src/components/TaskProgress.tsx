import { formatDuration } from '../format';

export interface TaskProgressProps {
    stage: string;
    current: number;
    total: number;
    startedAt?: number;
    etaMs?: number;
    errorLog?: string;
    onOpenErrorLog?: () => void;
}

/** Long-task progress: stage + steps + ETA + elapsed (spec §7.2). */
export function TaskProgress({
    stage,
    current,
    total,
    startedAt,
    etaMs,
    errorLog,
    onOpenErrorLog,
}: TaskProgressProps) {
    const ratio = total > 0 ? Math.min(1, current / total) : 0;
    const failed = ratio >= 1 && errorLog;
    return (
        <div className="rounded-card border border-border bg-bg-elevated p-4">
            <div className="flex items-center justify-between text-sm">
                <span className="text-text-primary">{stage}</span>
                <span className="font-mono tabular-nums text-text-secondary">
                    {current}/{total}
                </span>
            </div>
            <div className="mt-2 h-2 overflow-hidden rounded-full bg-bg-muted">
                <div
                    className={`h-full rounded-full ${failed ? 'bg-error' : 'bg-running'} transition-all`}
                    style={{ width: `${(ratio * 100).toFixed(1)}%` }}
                />
            </div>
            <div className="mt-2 flex items-center justify-between font-mono text-xs tabular-nums text-text-muted">
                <span>{(ratio * 100).toFixed(1)}%</span>
                <span>
                    {etaMs !== undefined ? `ETA ${formatDuration(etaMs)}` : ''}
                    {etaMs !== undefined && startedAt !== undefined ? ' · ' : ''}
                    {startedAt !== undefined
                        ? `已耗时 ${formatDuration(Date.now() - startedAt)}`
                        : ''}
                </span>
            </div>
            {failed && onOpenErrorLog ? (
                <button
                    type="button"
                    onClick={onOpenErrorLog}
                    className="mt-2 text-xs text-error underline"
                >
                    查看失败日志（最后 200 行）
                </button>
            ) : null}
        </div>
    );
}
