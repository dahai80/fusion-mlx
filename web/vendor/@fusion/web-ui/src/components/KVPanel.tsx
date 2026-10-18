import type { ReactNode } from 'react';
import { useState } from 'react';

/**
 * Key-value detail panel with built-in JSON preview (spec §5.1).
 * Pairs is the canonical form; object form is converted for convenience.
 */
export function KVPanel({
    pairs,
    object,
    title,
}: { pairs?: Array<[string, ReactNode]>; object?: Record<string, unknown>; title?: string }) {
    const [showJson, setShowJson] = useState(false);
    const items: Array<[string, ReactNode]> =
        pairs ?? Object.entries(object ?? {}).map(([k, v]) => [k, String(v)]);
    const jsonText = object ? JSON.stringify(object, null, 4) : null;
    return (
        <div className="rounded-card border border-border bg-bg-elevated p-4">
            {title ? <div className="mb-2 text-sm text-text-primary">{title}</div> : null}
            <dl className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                {items.map(([k, v]) => (
                    <div
                        key={k}
                        className="flex min-w-0 items-baseline justify-between gap-2 border-b border-border/40 py-1"
                    >
                        <dt className="shrink-0 text-xs text-text-muted">{k}</dt>
                        <dd className="min-w-0 truncate text-right font-mono text-sm tabular-nums text-text-primary">
                            {v}
                        </dd>
                    </div>
                ))}
            </dl>
            {jsonText ? (
                <div className="mt-3">
                    <button
                        type="button"
                        onClick={() => setShowJson((v) => !v)}
                        className="text-xs text-brand"
                    >
                        {showJson ? '收起 JSON' : 'JSON 预览'}
                    </button>
                    {showJson ? (
                        <pre className="mt-2 max-h-72 overflow-auto rounded-md bg-bg p-3 font-mono text-xs leading-5 text-text-secondary">
                            {jsonText}
                        </pre>
                    ) : null}
                </div>
            ) : null}
        </div>
    );
}
