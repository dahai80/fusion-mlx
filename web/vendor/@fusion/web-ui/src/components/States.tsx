import type { ReactNode } from 'react';
/** Loading state: skeleton block (spec §6.1 — no spinner overlay). */
export function Skeleton({ className = '' }: { className?: string }) {
    return <div className={`animate-pulse rounded-card bg-bg-muted ${className}`} />;
}

export function SkeletonLines({ rows = 3 }: { rows?: number }) {
    return (
        <div className="flex flex-col gap-2" aria-busy="true">
            {Array.from({ length: rows }, (_, i) => (
                <Skeleton key={i} className="h-4 w-full" />
            ))}
        </div>
    );
}

/** Empty state: icon + one-line explanation + primary action (spec §6.1). */
export function EmptyState({
    title,
    description,
    action,
}: { title: string; description?: string; action?: ReactNode }) {
    return (
        <div className="flex flex-col items-center justify-center gap-2 px-4 py-12 text-center">
            <div className="text-2xl" aria-hidden>
                ∅
            </div>
            <div className="text-sm text-text-primary">{title}</div>
            {description ? (
                <div className="max-w-sm text-xs text-text-muted">{description}</div>
            ) : null}
            {action ? <div className="mt-2">{action}</div> : null}
        </div>
    );
}

/** Error state: gateway envelope message + retry (spec §6.1). */
export function ErrorState({
    message,
    retryLabel = '重试',
    onRetry,
    retrying,
}: {
    message: string;
    retryLabel?: string;
    onRetry?: () => void;
    /** Active auto-retry attempt; shows "第 N 次重试中" and hides manual button (spec §6.1). */
    retrying?: number;
}) {
    return (
        <div
            className="flex flex-col items-center justify-center gap-2 px-4 py-12 text-center"
            role="alert"
        >
            <div className="text-sm text-error">{message}</div>
            {retrying ? (
                <div className="text-xs text-text-muted">第 {retrying} 次重试中…</div>
            ) : null}
            {onRetry && !retrying ? (
                <button
                    type="button"
                    onClick={onRetry}
                    className="rounded-md border border-border px-4 py-2 text-sm text-text-primary hover:bg-bg-muted"
                >
                    {retryLabel}
                </button>
            ) : null}
        </div>
    );
}
