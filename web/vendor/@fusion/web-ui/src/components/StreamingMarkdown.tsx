import { Fragment, type ReactNode, useEffect, useRef, useState } from 'react';

/**
 * Streaming markdown: incremental text rendering, follow-scroll lock +
 * jump-to-bottom button (spec §7.1). No external markdown dep.
 */
export function StreamingMarkdown({ text, className = '' }: { text: string; className?: string }) {
    const boxRef = useRef<HTMLDivElement>(null);
    const [pinned, setPinned] = useState(true);

    // biome-ignore lint/correctness/useExhaustiveDependencies: text is the intentional trigger (auto-scroll when content streams in)
    useEffect(() => {
        const el = boxRef.current;
        if (el && pinned) el.scrollTop = el.scrollHeight;
    }, [text, pinned]);

    const blocks = parseBlocks(text);

    return (
        <div className={`relative ${className}`}>
            <div
                ref={boxRef}
                onScroll={(e) => {
                    const el = e.currentTarget;
                    setPinned(el.scrollHeight - el.scrollTop - el.clientHeight < 24);
                }}
                className="max-h-[70vh] overflow-y-auto overscroll-contain text-sm leading-6 text-text-primary"
            >
                {blocks.map((b, i) => (
                    <BlockView key={i} block={b} />
                ))}
            </div>
            {!pinned ? (
                <button
                    type="button"
                    onClick={() => {
                        setPinned(true);
                        const el = boxRef.current;
                        if (el) el.scrollTop = el.scrollHeight;
                    }}
                    className="absolute bottom-2 right-2 rounded-md border border-border bg-bg-elevated px-3 py-1.5 text-xs shadow-overlay"
                >
                    回到底部
                </button>
            ) : null}
        </div>
    );
}

interface Block {
    kind: 'code' | 'para' | 'list';
    text: string;
}

function parseBlocks(text: string): Block[] {
    const blocks: Block[] = [];
    const lines = text.split('\n');
    let buf: string[] = [];
    let mode: Block['kind'] | null = null;
    const flush = () => {
        if (buf.length > 0 && mode) blocks.push({ kind: mode, text: buf.join('\n') });
        buf = [];
    };
    for (const line of lines) {
        if (line.startsWith('```')) {
            flush();
            mode = mode === 'code' ? null : 'code';
            continue;
        }
        if (mode === 'code') {
            buf.push(line);
            continue;
        }
        if (/^[-*] /.test(line) && mode !== 'list') {
            flush();
            mode = 'list';
            buf.push(line.slice(2));
            continue;
        }
        if (/^[-*] /.test(line) && mode === 'list') {
            buf.push(line.slice(2));
            continue;
        }
        if (line.trim() === '') {
            flush();
            mode = null;
            continue;
        }
        if (mode === null) mode = 'para';
        buf.push(line);
    }
    flush();
    return blocks;
}

function BlockView({ block }: { block: Block }) {
    if (block.kind === 'code') {
        return <CodeBlock text={block.text} />;
    }
    if (block.kind === 'list') {
        return (
            <ul className="my-1 list-disc pl-5">
                {block.text.split('\n').map((line, i) => (
                    <li key={i}>{renderInline(line)}</li>
                ))}
            </ul>
        );
    }
    return <p className="my-1 whitespace-pre-wrap">{renderInline(block.text)}</p>;
}

function CodeBlock({ text }: { text: string }) {
    const [copied, setCopied] = useState(false);
    return (
        <div className="my-2 overflow-hidden rounded-md border border-border">
            <div className="flex items-center justify-between bg-bg-muted px-3 py-1">
                <span className="font-mono text-[10px] text-text-muted">code</span>
                <button
                    type="button"
                    onClick={() => {
                        void navigator.clipboard.writeText(text).then(() => {
                            setCopied(true);
                            setTimeout(() => setCopied(false), 1500);
                        });
                    }}
                    className="text-xs text-text-secondary hover:text-text-primary"
                >
                    {copied ? '已复制' : '复制'}
                </button>
            </div>
            <pre className="overflow-x-auto bg-bg p-3 font-mono text-xs leading-5 text-text-secondary">
                {text}
            </pre>
        </div>
    );
}

function renderInline(text: string): ReactNode {
    // Minimal inline: `code` and **bold**.
    const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g);
    return parts.map((part, i) => {
        if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
            return (
                <code key={i} className="rounded bg-bg-muted px-1 font-mono text-xs">
                    {part.slice(1, -1)}
                </code>
            );
        }
        if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
            return (
                <strong key={i} className="text-text-primary">
                    {part.slice(2, -2)}
                </strong>
            );
        }
        return <Fragment key={i}>{part}</Fragment>;
    });
}
