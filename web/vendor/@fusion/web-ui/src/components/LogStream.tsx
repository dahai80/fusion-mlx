import { useEffect, useRef, useState } from 'react';

export interface LogLine {
    ts: number | string;
    level: 'debug' | 'info' | 'warn' | 'error';
    msg: string;
}

const LEVEL_COLOR: Record<LogLine['level'], string> = {
    debug: 'text-text-muted',
    info: 'text-text-secondary',
    warn: 'text-warning',
    error: 'text-error',
};

const LEVEL_TAG: Record<LogLine['level'], string> = {
    debug: 'DBG',
    info: 'INFO',
    warn: 'WARN',
    error: 'ERR',
};

/**
 * Monospace log viewer: autoscroll-lock (follows only when pinned to bottom),
 * level coloring, built-in download (spec §5.1).
 */
export function LogStream({ lines, maxLines = 2000 }: { lines: LogLine[]; maxLines?: number }) {
    const boxRef = useRef<HTMLDivElement>(null);
    const [pinned, setPinned] = useState(true);
    const visible = lines.slice(-maxLines);

    // biome-ignore lint/correctness/useExhaustiveDependencies: visible.length is the intentional trigger (auto-scroll when lines arrive)
    useEffect(() => {
        const el = boxRef.current;
        if (el && pinned) el.scrollTop = el.scrollHeight;
    }, [visible.length, pinned]);

    const download = () => {
        const text = visible.map((l) => `[${l.ts}] ${LEVEL_TAG[l.level]} ${l.msg}`).join('\n');
        const url = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
        const a = document.createElement('a');
        a.href = url;
        a.download = 'fusion-log.txt';
        a.click();
        URL.revokeObjectURL(url);
        console.log('[web-ui] log downloaded, lines:', visible.length);
    };

    return (
        <div className="flex h-full flex-col rounded-card border border-border bg-bg-elevated">
            <div className="flex items-center justify-between border-b border-border px-3 py-2">
                <span className="text-xs text-text-muted">
                    {visible.length} 行{pinned ? ' · 跟随中' : ' · 已锁定'}
                </span>
                <button
                    type="button"
                    onClick={download}
                    className="rounded px-2 py-1 text-xs text-text-secondary hover:bg-bg-muted"
                >
                    下载
                </button>
            </div>
            <div
                ref={boxRef}
                onScroll={(e) => {
                    const el = e.currentTarget;
                    setPinned(el.scrollHeight - el.scrollTop - el.clientHeight < 24);
                }}
                className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-3 py-2 font-mono text-xs leading-5"
            >
                {visible.map((l, i) => (
                    <div
                        key={i}
                        className={`whitespace-pre-wrap break-all ${LEVEL_COLOR[l.level]}`}
                    >
                        <span className="text-text-muted">{LEVEL_TAG[l.level]}</span> {l.msg}
                    </div>
                ))}
            </div>
        </div>
    );
}
