export type StatusKind = 'success' | 'running' | 'warning' | 'error' | 'offline';

const KIND_TEXT: Record<StatusKind, string> = {
    success: 'text-success',
    running: 'text-running',
    warning: 'text-warning',
    error: 'text-error',
    offline: 'text-offline',
};

const KIND_DOT: Record<StatusKind, string> = {
    success: 'bg-success',
    running: 'bg-running',
    warning: 'bg-warning',
    error: 'bg-error',
    offline: 'bg-offline',
};

const PULSE: Partial<Record<StatusKind, boolean>> = { running: true, warning: true };

/** Four-color status semantics (spec §3.1): success/running/warning/error/offline. */
export function StatusBadge({ status, label }: { status: StatusKind; label?: string }) {
    return (
        <span
            className={`inline-flex items-center gap-1.5 rounded-full border border-border bg-bg-elevated px-2 py-0.5 text-xs ${KIND_TEXT[status]}`}
        >
            <span
                className={`h-2 w-2 rounded-full ${KIND_DOT[status]} ${PULSE[status] ? 'animate-pulse' : ''}`}
            />
            {label ?? status}
        </span>
    );
}
